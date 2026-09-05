#!/usr/bin/env python3

"""Run linear-kernel replicates 1--4 in eight independent GPU jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from score_model import score_model, scoring_metadata


GROUP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GROUP_DIR.parents[2]
TEMP_DIR = PROJECT_ROOT/"val"/"temp"/"coag"/"test_linear"
OUTPUT_ROOT = PROJECT_ROOT/"val"/"logs"/"coagulation"/"test_linear"
RESULT_ROOT = OUTPUT_ROOT/"multiseed"
REPLICATES = range(1, 5)

# Longest-processing-time partition based on the completed seed-0 wall times.
# Each job owns its models exclusively and runs replicates 1--4 sequentially.
JOB_MODELS = {
    1: [
        "1e+6n_1e+1k_5e-3b",
    ],
    2: [
        "1e+6n_2e+1k_1e-2b",
        "1e+6n_2e+2k_8e-2b",
    ],
    3: [
        "1e+6n_2e+2k_5e-3b",
        "1e+6n_2e+1k_8e-2b",
        "1e+6n_1e+2k_8e-2b",
    ],
    4: [
        "1e+6n_1e+1k_1e-2b",
        "1e+6n_1e+2k_2e-2b",
        "1e+6n_2e+1k_4e-2b",
    ],
    5: [
        "1e+6n_1e+2k_5e-3b",
        "1e+6n_1e+1k_2e-2b",
        "1e+6n_2e+2k_4e-2b",
        "1e+6n_5e+1k_8e-2b",
    ],
    6: [
        "1e+6n_2e+1k_5e-3b",
        "1e+6n_2e+2k_2e-2b",
        "1e+6n_2e+1k_2e-2b",
        "1e+6n_5e+1k_4e-2b",
    ],
    7: [
        "1e+6n_5e+1k_5e-3b",
        "1e+6n_1e+2k_1e-2b",
        "1e+6n_5e+1k_2e-2b",
        "1e+6n_1e+2k_4e-2b",
    ],
    8: [
        "1e+6n_5e+1k_1e-2b",
        "1e+6n_2e+2k_1e-2b",
        "1e+6n_1e+1k_8e-2b",
        "1e+6n_1e+1k_4e-2b",
    ],
}

SCORE_FIELDS = (
    "tv_distance", "js_distance", "w1_log10_mass_dex", "cdf_sup_distance",
    "mass_relative_error", "number_ratio", "number_relative_error",
    "m2_over_m1_ratio", "m2_over_m1_relative_error", "minimum_mass", "maximum_mass",
)
CONTROLLER_FIELDS = (
    "bath_count", "continuation_launches", "activity_overshoots",
    "distribution_overshoots", "persistent_overshoots",
    "minimum_duration", "maximum_duration", "minimum_limit_scale",
    "maximum_f", "maximum_e", "maximum_touched", "maximum_events",
    "maximum_g", "maximum_g_upper", "maximum_d_bath",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def seed_streams(replicate: int) -> dict[str, int]:
    seed = 2*replicate + 1
    return {"initialization": replicate, "position": seed, "collision": seed}


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), GROUP_DIR/"score_model.py", GROUP_DIR/"Makefile", PROJECT_ROOT/"Makefile",
        *sorted((GROUP_DIR/"test_common").glob("*")),
        *sorted(GROUP_DIR.glob("*/flags.mk")),
        *sorted((PROJECT_ROOT/"src").rglob("*")),
        *sorted((PROJECT_ROOT/"inc").rglob("*")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def campaign_fingerprint(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def remove_raw_output(output_dir: Path, remove_variables: bool = False) -> None:
    for pattern in ("particle_*.dat", "collision_chain_*.json"):
        for path in output_dir.glob(pattern):
            path.unlink()
    if remove_variables:
        path = output_dir/"variables.txt"
        if path.exists():
            path.unlink()


def statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "p16": float(np.percentile(array, 16.0)),
        "p84": float(np.percentile(array, 84.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def job_seed_path(job: int, replicate: int) -> Path:
    return RESULT_ROOT/"jobs"/f"job_{job:02d}"/f"seed_{replicate:03d}.json"


def load_job_seed(job: int, replicate: int, fingerprint: str) -> dict[str, object]:
    path = job_seed_path(job, replicate)
    if path.exists():
        with path.open() as file:
            result = json.load(file)
        if result["campaign_fingerprint"] != fingerprint:
            raise RuntimeError(f"refusing to mix changed campaign sources with {path}")
        return result
    return {
        "schema": 1,
        "suite": "coag/test_linear",
        "job": job,
        "replicate": replicate,
        "seeds": seed_streams(replicate),
        "campaign_fingerprint": fingerprint,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "models": [],
    }


def run_new_replicate(job: int, models: list[str], replicate: int, fingerprint: str) -> None:
    streams = seed_streams(replicate)
    seed_result = load_job_seed(job, replicate, fingerprint)
    seed_result.update(status="running", finished_utc=None)
    records = {record["model"]: record for record in seed_result["models"]}

    for model in models:
        if records.get(model, {}).get("status") == "passed":
            remove_raw_output(OUTPUT_ROOT/model)
            continue

        executable = TEMP_DIR/"bin"/model/"gamedev"
        output_dir = OUTPUT_ROOT/model
        output_dir.mkdir(parents=True, exist_ok=True)
        remove_raw_output(output_dir, remove_variables=True)
        record: dict[str, object] = {
            "model": model,
            "status": "running",
            "seeds": streams,
            "started_utc": utc_now(),
            "executable": str(executable),
        }
        records[model] = record
        seed_result["models"] = [records[name] for name in models if name in records]
        atomic_json(job_seed_path(job, replicate), seed_result)

        print(f"\n=== seed {replicate:03d} run {model} ===", flush=True)
        environment = os.environ.copy()
        environment["COAG_INITIALIZATION_SEED"] = str(streams["initialization"])
        environment["COAG_POSITION_SEED"] = str(streams["position"])
        environment["COAG_COLLISION_SEED"] = str(streams["collision"])
        start_ns = time.perf_counter_ns()
        result = subprocess.run([str(executable)], cwd=PROJECT_ROOT, env=environment, check=False)
        record["wall_seconds"] = (time.perf_counter_ns() - start_ns)/1.0e9
        record["return_code"] = result.returncode
        record["finished_utc"] = utc_now()

        if result.returncode != 0:
            record["status"] = "run_failed"
            seed_result.update(status="failed", finished_utc=utc_now())
            atomic_json(job_seed_path(job, replicate), seed_result)
            raise SystemExit(result.returncode)

        try:
            scored = score_model(output_dir)
            if scored["runtime_seeds"] != streams:
                raise RuntimeError("saved runtime seeds do not match the requested streams")
            record.update(scored)
        except Exception:
            record["status"] = "score_failed"
            seed_result.update(status="failed", finished_utc=utc_now())
            atomic_json(job_seed_path(job, replicate), seed_result)
            raise

        record["status"] = "passed"
        atomic_json(job_seed_path(job, replicate), seed_result)
        remove_raw_output(output_dir)

    seed_result.update(status="passed", finished_utc=utc_now())
    atomic_json(job_seed_path(job, replicate), seed_result)


def run_job(job: int) -> None:
    models = JOB_MODELS[job]
    hashes = source_hashes()
    fingerprint = campaign_fingerprint(hashes)
    job_root = RESULT_ROOT/"jobs"/f"job_{job:02d}"
    manifest_path = job_root/"manifest.json"
    manifest: dict[str, object] = {
        "schema": 1,
        "suite": "coag/test_linear",
        "job": job,
        "status": "building",
        "started_utc": utc_now(),
        "finished_utc": None,
        "campaign_fingerprint": fingerprint,
        "models": models,
        "replicates": list(REPLICATES),
    }
    atomic_json(manifest_path, manifest)

    for model in models:
        print(f"\n=== build {model} ===", flush=True)
        build = subprocess.run(["make", "-C", str(GROUP_DIR), f"MODEL={model}"], check=False)
        if build.returncode != 0:
            manifest.update(status="build_failed", failed_model=model, finished_utc=utc_now())
            atomic_json(manifest_path, manifest)
            raise SystemExit(build.returncode)

    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)
    for completed, replicate in enumerate(REPLICATES, 1):
        run_new_replicate(job, models, replicate, fingerprint)
        manifest["replicates_completed"] = completed
        atomic_json(manifest_path, manifest)

    manifest.update(status="passed", finished_utc=utc_now(), replicates_completed=len(REPLICATES))
    atomic_json(manifest_path, manifest)
    print(f"\nJob results: {job_root}")


def write_aggregate(models: list[str], fingerprint: str) -> None:
    collected: dict[str, list[dict[str, object]]] = {model: [] for model in models}
    for replicate in REPLICATES:
        with (RESULT_ROOT/f"seed_{replicate:03d}.json").open() as file:
            seed_result = json.load(file)
        for record in seed_result["models"]:
            collected[record["model"]].append(record)

    aggregate_models = []
    for model in models:
        records = collected[model]
        aggregate_models.append({
            "model": model,
            "parameters": records[0]["parameters"],
            "replicates": len(records),
            "wall_seconds": statistics([float(record["wall_seconds"]) for record in records]),
            "scores": {
                field: statistics([float(record["score"][field]) for record in records])
                for field in SCORE_FIELDS
            },
            "controller": {
                field: statistics([float(record["controller"]["total"][field]) for record in records])
                for field in CONTROLLER_FIELDS
            },
        })

    atomic_json(RESULT_ROOT/"summary.json", {
        "schema": 1,
        "suite": "coag/test_linear",
        "campaign_fingerprint": fingerprint,
        "replicates": len(REPLICATES),
        "primary_score": "mean final-snapshot total-variation distance; report seed scatter",
        "models": aggregate_models,
    })


def finalize() -> None:
    models = sorted(path.parent.name for path in GROUP_DIR.glob("*/flags.mk"))
    partitioned = [model for job_models in JOB_MODELS.values() for model in job_models]
    if len(partitioned) != len(set(partitioned)) or sorted(partitioned) != models:
        raise RuntimeError("JOB_MODELS must contain every model exactly once")

    hashes = source_hashes()
    fingerprint = campaign_fingerprint(hashes)
    for job in JOB_MODELS:
        manifest_path = RESULT_ROOT/"jobs"/f"job_{job:02d}"/"manifest.json"
        with manifest_path.open() as file:
            job_manifest = json.load(file)
        if job_manifest["status"] != "passed":
            raise RuntimeError(f"job {job} is not complete")
        if job_manifest["campaign_fingerprint"] != fingerprint:
            raise RuntimeError(f"job {job} used different campaign sources")

    for replicate in REPLICATES:
        records: list[dict[str, object]] = []
        for job in JOB_MODELS:
            with job_seed_path(job, replicate).open() as file:
                shard = json.load(file)
            if shard["status"] != "passed" or shard["campaign_fingerprint"] != fingerprint:
                raise RuntimeError(f"job {job}, replicate {replicate} is not a compatible completed shard")
            records.extend(shard["models"])
        record_models = [record["model"] for record in records]
        if len(record_models) != len(set(record_models)) or sorted(record_models) != models:
            raise RuntimeError(f"replicate {replicate} does not contain every model exactly once")
        atomic_json(RESULT_ROOT/f"seed_{replicate:03d}.json", {
            "schema": 1,
            "suite": "coag/test_linear",
            "replicate": replicate,
            "seeds": seed_streams(replicate),
            "campaign_fingerprint": fingerprint,
            "status": "passed",
            "models": sorted(records, key=lambda record: record["model"]),
        })

    write_aggregate(models, fingerprint)
    atomic_json(RESULT_ROOT/"manifest.json", {
        "schema": 1,
        "suite": "coag/test_linear",
        "status": "passed",
        "finished_utc": utc_now(),
        "source_sha256": hashes,
        "campaign_fingerprint": fingerprint,
        "replicates_expected": len(REPLICATES),
        "models_per_replicate": len(models),
        "models": models,
        "jobs": {str(job): job_models for job, job_models in JOB_MODELS.items()},
        "seed_policy": {
            "replicate_labels": list(REPLICATES),
            "mapping": "initialization = replicate; position = collision = 2*replicate + 1",
        },
        "output_policy": (
            "particle and collision-chain files are deleted only after the model score is "
            "atomically recorded; variables.txt is retained; failed-model output is retained"
        ),
        "scoring": scoring_metadata(),
    })
    print(f"\nMultiseed results: {RESULT_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=int, choices=JOB_MODELS)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        run_job(args.job)


if __name__ == "__main__":
    main()
