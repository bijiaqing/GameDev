#!/usr/bin/env python3

"""Run linear-kernel replicates 5--9 in eight independent GPU jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

import run_models as campaign


REPLICATES = range(5, 10)
JOBS_ROOT = campaign.RESULT_ROOT/"jobs_005_009"

# Preserve the seed-0 longest-processing-time partition used for replicates 1--4.
JOB_MODELS = {
    1: ["1e+6n_1e+1k_5e-3b"],
    2: ["1e+6n_2e+1k_1e-2b", "1e+6n_2e+2k_8e-2b"],
    3: ["1e+6n_2e+2k_5e-3b", "1e+6n_2e+1k_8e-2b", "1e+6n_1e+2k_8e-2b"],
    4: ["1e+6n_1e+1k_1e-2b", "1e+6n_1e+2k_2e-2b", "1e+6n_2e+1k_4e-2b"],
    5: ["1e+6n_1e+2k_5e-3b", "1e+6n_1e+1k_2e-2b", "1e+6n_2e+2k_4e-2b", "1e+6n_5e+1k_8e-2b"],
    6: ["1e+6n_2e+1k_5e-3b", "1e+6n_2e+2k_2e-2b", "1e+6n_2e+1k_2e-2b", "1e+6n_5e+1k_4e-2b"],
    7: ["1e+6n_5e+1k_5e-3b", "1e+6n_1e+2k_1e-2b", "1e+6n_5e+1k_2e-2b", "1e+6n_1e+2k_4e-2b"],
    8: ["1e+6n_5e+1k_1e-2b", "1e+6n_2e+2k_1e-2b", "1e+6n_1e+1k_8e-2b", "1e+6n_1e+1k_4e-2b"],
}


def source_hashes() -> dict[str, str]:
    hashes = campaign.source_hashes()
    path = Path(__file__).resolve()
    hashes[str(path.relative_to(campaign.PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def fingerprint(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def job_seed_path(job: int, replicate: int) -> Path:
    return JOBS_ROOT/f"job_{job:02d}"/f"seed_{replicate:03d}.json"


def run_job(job: int) -> None:
    models = JOB_MODELS[job]
    hashes = source_hashes()
    current_fingerprint = fingerprint(hashes)
    job_root = JOBS_ROOT/f"job_{job:02d}"
    manifest_path = job_root/"manifest.json"

    if manifest_path.exists():
        with manifest_path.open() as file:
            manifest = json.load(file)
        if manifest["campaign_fingerprint"] != current_fingerprint:
            raise RuntimeError(f"refusing to mix changed campaign sources with {manifest_path}")
    else:
        manifest = {
            "schema": 1,
            "suite": "coag/test_linear",
            "job": job,
            "started_utc": campaign.utc_now(),
        }
    manifest.update(
        status="building",
        finished_utc=None,
        source_sha256=hashes,
        campaign_fingerprint=current_fingerprint,
        models=models,
        replicates=list(REPLICATES),
    )
    campaign.atomic_json(manifest_path, manifest)

    for model in models:
        print(f"\n=== build {model} ===", flush=True)
        result = subprocess.run([
            "make", "-C", str(campaign.GROUP_DIR), f"MODEL={model}",
            f"OUTPUT_ROOT={campaign.OUTPUT_ROOT}",
        ], check=False)
        if result.returncode != 0:
            manifest.update(status="build_failed", failed_model=model, finished_utc=campaign.utc_now())
            campaign.atomic_json(manifest_path, manifest)
            raise SystemExit(result.returncode)

    manifest["status"] = "running"
    campaign.atomic_json(manifest_path, manifest)
    completed = []
    for replicate in REPLICATES:
        try:
            campaign.run_replicate(models, replicate, job_seed_path(job, replicate), current_fingerprint)
        except BaseException:
            manifest.update(status="failed", failed_replicate=replicate, finished_utc=campaign.utc_now())
            campaign.atomic_json(manifest_path, manifest)
            raise
        completed.append(replicate)
        manifest["replicates_completed"] = completed
        campaign.atomic_json(manifest_path, manifest)

    manifest.update(status="passed", finished_utc=campaign.utc_now())
    campaign.atomic_json(manifest_path, manifest)
    print(f"\nJob results: {job_root}")


def load_completed_seed(path: Path, models: list[str]) -> dict[str, object]:
    with path.open() as file:
        seed = json.load(file)
    replicate = int(path.stem.removeprefix("seed_"))
    names = [record["model"] for record in seed["models"]]
    if (
        seed["status"] != "passed"
        or seed["replicate"] != replicate
        or seed["seeds"] != campaign.seed_streams(replicate)
        or sorted(names) != models
        or len(names) != len(set(names))
    ):
        raise RuntimeError(f"incomplete seed result: {path}")
    return seed


def finalize() -> None:
    models = sorted(path.parent.name for path in campaign.MODELS_DIR.glob("*/flags.mk"))
    partitioned = [model for job_models in JOB_MODELS.values() for model in job_models]
    if sorted(partitioned) != models or len(partitioned) != len(set(partitioned)):
        raise RuntimeError("JOB_MODELS must contain every model exactly once")

    for replicate in range(5):
        load_completed_seed(campaign.RESULT_ROOT/f"seed_{replicate:03d}.json", models)

    hashes = source_hashes()
    current_fingerprint = fingerprint(hashes)
    for job in JOB_MODELS:
        manifest_path = JOBS_ROOT/f"job_{job:02d}"/"manifest.json"
        with manifest_path.open() as file:
            manifest = json.load(file)
        if manifest["status"] != "passed" or manifest["campaign_fingerprint"] != current_fingerprint:
            raise RuntimeError(f"job {job} is not a compatible completed job")

    for replicate in REPLICATES:
        records = []
        for job in JOB_MODELS:
            seed = load_completed_seed(job_seed_path(job, replicate), sorted(JOB_MODELS[job]))
            if seed["campaign_fingerprint"] != current_fingerprint:
                raise RuntimeError(f"job {job}, seed {replicate} used changed campaign sources")
            records.extend(seed["models"])
        names = [record["model"] for record in records]
        if sorted(names) != models or len(names) != len(set(names)):
            raise RuntimeError(f"seed {replicate} does not contain every model exactly once")
        campaign.atomic_json(campaign.RESULT_ROOT/f"seed_{replicate:03d}.json", {
            "schema": 1,
            "suite": "coag/test_linear",
            "replicate": replicate,
            "seeds": campaign.seed_streams(replicate),
            "campaign_fingerprint": current_fingerprint,
            "status": "passed",
            "models": sorted(records, key=lambda record: record["model"]),
        })

    seeds = [
        load_completed_seed(campaign.RESULT_ROOT/f"seed_{replicate:03d}.json", models)
        for replicate in range(campaign.SEED_COUNT)
    ]
    campaign.write_aggregate(models)
    campaign.atomic_json(campaign.RESULT_ROOT/"manifest.json", {
        "schema": 1,
        "suite": "coag/test_linear",
        "status": "passed",
        "finished_utc": campaign.utc_now(),
        "replicates_expected": campaign.SEED_COUNT,
        "replicates_completed": campaign.SEED_COUNT,
        "models_per_replicate": len(models),
        "models": models,
        "additional_source_sha256": hashes,
        "campaign_fingerprints_by_replicate": {
            str(replicate): seed["campaign_fingerprint"] for replicate, seed in enumerate(seeds)
        },
        "jobs_005_009": {str(job): job_models for job, job_models in JOB_MODELS.items()},
        "seed_policy": {
            "master_replicates": list(range(campaign.SEED_COUNT)),
            "mapping": "initialization = replicate; position = collision = 2*replicate + 1",
        },
        "output_policy": (
            "replicates 5--9 use rolling output deleted only after each model score is "
            "atomically recorded; failed-model output is retained"
        ),
        "scoring": campaign.scoring_metadata(),
    })
    print(f"\nMultiseed results: {campaign.RESULT_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=int, choices=JOB_MODELS)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    finalize() if args.finalize else run_job(args.job)


if __name__ == "__main__":
    main()
