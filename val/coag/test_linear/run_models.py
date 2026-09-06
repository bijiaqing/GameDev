#!/usr/bin/env python3

"""Run and reduce the independent 10-seed linear-kernel campaign."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from score_model import score_model, scoring_metadata


GROUP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GROUP_DIR.parents[2]
MODELS_DIR = GROUP_DIR/"models"
TEMP_DIR = PROJECT_ROOT/"val"/"temp"/"coag"/"test_linear"
OUTPUT_ROOT = PROJECT_ROOT/"out"/"test_linear"/"multiseed"
ARCHIVE_ROOT = PROJECT_ROOT/"out"/"test_linear"/"seed_000"
RESULT_ROOT = GROUP_DIR/"multiseed"
SEED_COUNT = 10

SCORE_FIELDS = (
    "tv_distance", "js_distance", "w1_log10_mass_dex", "cdf_sup_distance",
    "mass_relative_error", "number_ratio", "number_relative_error",
    "m2_over_m1_ratio", "m2_over_m1_relative_error",
    "minimum_mass", "maximum_mass",
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
        *sorted((MODELS_DIR/"models_common").glob("*")),
        *sorted(MODELS_DIR.glob("*/flags.mk")),
        *sorted((PROJECT_ROOT/"src").rglob("*")),
        *sorted((PROJECT_ROOT/"inc").rglob("*")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def remove_raw_output(output_dir: Path, remove_variables: bool = False) -> None:
    for pattern in ("particle_*.dat", "collision_chain_*.json"):
        for path in output_dir.glob(pattern):
            path.unlink()
    if remove_variables:
        variable_path = output_dir/"variables.txt"
        if variable_path.exists():
            variable_path.unlink()


def archive_seed_zero(output_dir: Path, archive_dir: Path) -> None:
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    if archive_dir.exists():
        raise RuntimeError(f"refusing to overwrite seed-0 output in {archive_dir}")
    output_dir.replace(archive_dir)


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


def write_aggregate(models: list[str]) -> None:
    collected: dict[str, list[dict[str, object]]] = {model: [] for model in models}
    for replicate in range(SEED_COUNT):
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
        "replicates": SEED_COUNT,
        "primary_score": "mean final-snapshot total-variation distance; report seed scatter",
        "models": aggregate_models,
    })


def run_replicate(models: list[str], replicate: int, seed_path: Path, fingerprint: str) -> None:
    streams = seed_streams(replicate)
    if seed_path.exists():
        with seed_path.open() as file:
            seed_result = json.load(file)
        if seed_result["campaign_fingerprint"] != fingerprint:
            raise RuntimeError(f"refusing to mix changed campaign sources with {seed_path}")
    else:
        seed_result = {
            "schema": 1,
            "suite": "coag/test_linear",
            "replicate": replicate,
            "seeds": streams,
            "campaign_fingerprint": fingerprint,
            "status": "running",
            "started_utc": utc_now(),
            "finished_utc": None,
            "models": [],
        }
    seed_result.update(status="running", finished_utc=None)
    records = {record["model"]: record for record in seed_result["models"]}

    for model in models:
        if records.get(model, {}).get("status") == "passed":
            if replicate > 0:
                remove_raw_output(OUTPUT_ROOT/model)
            continue

        if replicate == 0 and records.get(model, {}).get("status") == "scored":
            output_dir = OUTPUT_ROOT/model
            archive_dir = ARCHIVE_ROOT/model
            if archive_dir.exists() and output_dir.exists():
                raise RuntimeError(f"both rolling and archived seed-0 output exist for {model}")
            if not archive_dir.exists():
                archive_seed_zero(output_dir, archive_dir)
            records[model]["status"] = "passed"
            atomic_json(seed_path, seed_result)
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
        atomic_json(seed_path, seed_result)

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
            atomic_json(seed_path, seed_result)
            raise SystemExit(result.returncode)

        try:
            scored = score_model(output_dir)
            if scored["runtime_seeds"] != streams:
                raise RuntimeError("saved runtime seeds do not match the requested streams")
            record.update(scored)
        except Exception:
            record["status"] = "score_failed"
            seed_result.update(status="failed", finished_utc=utc_now())
            atomic_json(seed_path, seed_result)
            raise

        record["status"] = "scored" if replicate == 0 else "passed"
        atomic_json(seed_path, seed_result)
        if replicate == 0:
            archive_seed_zero(output_dir, ARCHIVE_ROOT/model)
            record["status"] = "passed"
            atomic_json(seed_path, seed_result)
        else:
            remove_raw_output(output_dir)

    seed_result.update(status="passed", finished_utc=utc_now())
    atomic_json(seed_path, seed_result)


def main() -> None:
    models = sorted(path.parent.name for path in MODELS_DIR.glob("*/flags.mk"))
    manifest_path = RESULT_ROOT/"manifest.json"
    hashes = source_hashes()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    for seed_path in RESULT_ROOT.glob("seed_*.json"):
        with seed_path.open() as file:
            previous = json.load(file)
        if previous["campaign_fingerprint"] != fingerprint:
            raise RuntimeError(f"refusing to mix changed campaign sources with {seed_path}")
    manifest: dict[str, object] = {
        "schema": 1,
        "suite": "coag/test_linear",
        "status": "building",
        "started_utc": utc_now(),
        "finished_utc": None,
        "source_sha256": hashes,
        "campaign_fingerprint": fingerprint,
        "replicates_expected": SEED_COUNT,
        "models_per_replicate": len(models),
        "models": models,
        "seed_policy": {
            "master_replicates": list(range(SEED_COUNT)),
            "mapping": "initialization = replicate; position = collision = 2*replicate + 1",
            "replicate_zero_matches_previous_initialization_0_position_1_collision_1": True,
        },
        "output_policy": (
            "replicate-zero output is moved intact to out/test_linear/seed_000 only after "
            "scoring; later particle and collision-chain files are deleted only after the "
            "model score is atomically recorded; failed-model output is retained"
        ),
        "scoring": scoring_metadata(),
    }
    atomic_json(manifest_path, manifest)

    for model in models:
        print(f"\n=== build {model} ===", flush=True)
        build = subprocess.run([
            "make", "-C", str(GROUP_DIR), f"MODEL={model}", f"OUTPUT_ROOT={OUTPUT_ROOT}",
        ], check=False)
        if build.returncode != 0:
            manifest.update(status="build_failed", failed_model=model, finished_utc=utc_now())
            atomic_json(manifest_path, manifest)
            raise SystemExit(build.returncode)

    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)

    for replicate in range(SEED_COUNT):
        seed_path = RESULT_ROOT/f"seed_{replicate:03d}.json"
        try:
            run_replicate(models, replicate, seed_path, fingerprint)
        except BaseException:
            manifest.update(status="failed", failed_replicate=replicate, finished_utc=utc_now())
            atomic_json(manifest_path, manifest)
            raise
        manifest["replicates_completed"] = replicate + 1
        atomic_json(manifest_path, manifest)

    write_aggregate(models)
    manifest.update(status="passed", finished_utc=utc_now(), replicates_completed=SEED_COUNT)
    atomic_json(manifest_path, manifest)
    print(f"\nMultiseed results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
