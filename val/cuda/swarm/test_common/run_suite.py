#!/usr/bin/env python3

"""Run the analytical swarm verification matrix"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

VAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VAL_ROOT/"tool"))

from val_config import (
    PUBLICATION_TIER,
    SWARM_CHAIN_MODELS,
    SWARM_ENDPOINT_RESOLUTION,
    SWARM_FIXED_RESOLUTION,
    SWARM_GROUPS,
    swarm_metric_tiers,
    swarm_models,
    swarm_resolution_tiers,
)


CUDA_GROUPS = {
    "chain": SWARM_CHAIN_MODELS,
}


def collision_runtime_model(model: str) -> bool:
    """Identify production-runtime cases that do not use resolution sweeps"""

    return model.startswith("test_colchain_")


def utc_now() -> str:
    """Return a stable UTC timestamp for the aggregate manifest"""

    return datetime.now(timezone.utc).isoformat()


def capture(command: list[str], cwd: Path) -> str:
    """Capture optional environment information without making it a prerequisite"""

    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def write_manifest(path: Path, manifest: dict) -> None:
    """Update the aggregate state after every model"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", choices=("all", *SWARM_GROUPS, *CUDA_GROUPS), default="all"
    )
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--rebuild-manifest", action="store_true",
        help="reconstruct this group manifest from copied component manifests without rerunning",
    )
    parser.add_argument("--target", default=os.environ.get("CUDA_ARCH", "sm_80"))
    args = parser.parse_args()
    if args.build_only and args.rebuild_manifest:
        parser.error("--build-only and --rebuild-manifest cannot be combined")

    resolutions = args.res[:2] if args.quick else args.res
    run_environment = os.environ.copy()
    run_environment["CUDA_ARCH"] = args.target
    run_environment["VAL_SCOPE"] = args.group
    root = Path(__file__).resolve().parents[1]
    project_root = root.parents[2]
    out_root = project_root/"val"/"logs"/"swarm"/"cuda"
    scope_root = out_root if args.group == "all" else out_root/"groups"/args.group
    manifest_path = scope_root/("manifest_all.json" if args.group == "all" else "manifest.json")
    environment_path = scope_root/("environment_all.json" if args.group == "all" else "environment.json")

    # preserve the publication order from isolated trajectories through coupled collisions
    models = CUDA_GROUPS[args.group] if args.group in CUDA_GROUPS else swarm_models(args.group)

    entries = []
    for model in models:
        chain_model = collision_runtime_model(model)
        fixed_resolution = model in SWARM_FIXED_RESOLUTION
        model_resolutions = [] if chain_model \
            else (resolutions[:1] if fixed_resolution else (
                [resolutions[0], resolutions[-1]]
                if model in SWARM_ENDPOINT_RESOLUTION and len(resolutions) > 1
                else resolutions
            ))
        if model == "test_knn":
            output = "test_knn/suite_manifest.json"
        else:
            output = f"{model}/manifest.json"
        entries.append({
            "model": model,
            "resolutions": model_resolutions,
            "tier": PUBLICATION_TIER,
            "resolution_tiers": [] if chain_model \
                else swarm_resolution_tiers(model, model_resolutions),
            "output": None if args.build_only else output,
            "status": "pending",
        })

    if args.group in CUDA_GROUPS:
        publication_metrics = sum(
            1 if collision_runtime_model(entry["model"]) else len(entry["resolutions"])
            for entry in entries
        )
        metric_tiers = {PUBLICATION_TIER: publication_metrics}
    else:
        metric_tiers = swarm_metric_tiers(models, resolutions)
    included_tiers = [PUBLICATION_TIER]

    manifest = {
        "schema": 1,
        "suite": "swarm",
        "backend": "cuda",
        "group": args.group,
        "environment": environment_path.name,
        "requested_resolutions": args.res,
        "effective_resolutions": resolutions,
        "quick": args.quick,
        "build_only": args.build_only,
        "reconstructed": args.rebuild_manifest,
        "gpu_target": args.target,
        "campaign_tier": PUBLICATION_TIER,
        "included_tiers": included_tiers,
        "metric_tiers": metric_tiers,
        "models_expected": len(models),
        "models_completed": 0,
        "analytical_builds_expected": sum(
            len(entry["resolutions"]) for entry in entries
            if entry["model"] != "test_knn"
            and not collision_runtime_model(entry["model"])
        ),
        "knn_standalone_builds_expected": 4 if "test_knn" in models else 0,
        "knn_production_links_expected": 0,
        "status": "running",
        "passed": None,
        "started_utc": utc_now(),
        "finished_utc": None,
        "models": entries,
    }
    if args.rebuild_manifest:
        for index, entry in enumerate(entries):
            component_path = scope_root/entry["output"]
            try:
                component = json.loads(component_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                entry["status"] = "failed"
                entry["error"] = f"invalid component manifest: {error}"
                manifest["status"] = "failed"
                manifest["passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                raise SystemExit(f"cannot rebuild suite manifest: {component_path}: {error}")
            if component.get("passed") is not True:
                raise SystemExit(f"cannot rebuild suite manifest: {component_path} did not pass")
            entry["status"] = "passed"
            entry["return_code"] = 0
            manifest["models_completed"] = index + 1
        manifest["status"] = "passed"
        manifest["passed"] = True
        manifest["finished_utc"] = utc_now()
        write_manifest(manifest_path, manifest)
        print(f"SWARM TEST SUITE: MANIFEST PASS  manifest={manifest_path}")
        return

    manifest_path.unlink(missing_ok=True)
    write_manifest(manifest_path, manifest)
    environment = {
        "gpu_target": args.target,
        "nvcc": capture(["nvcc", "--version"], project_root),
        "nvidia_smi": capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            project_root,
        ),
        "python": sys.version,
    }
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    for idx_model, model in enumerate(models):
        model_resolutions = entries[idx_model]["resolutions"]
        command = [sys.executable, str(root/model/"run.py")]
        if not collision_runtime_model(model):
            command.extend(("--res", *(str(value) for value in model_resolutions)))
        if args.build_only:
            command.append("--build-only")
        if model == "test_knn":
            command.extend(("--arch", args.target))
        print(f"\n=== {model} ===", flush=True)
        entries[idx_model]["status"] = "running"
        write_manifest(manifest_path, manifest)
        try:
            result = subprocess.run(command, check=False, env=run_environment)
        except KeyboardInterrupt:
            entries[idx_model]["status"] = "interrupted"
            manifest["status"] = "interrupted"
            manifest["finished_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            print("\nSWARM TEST SUITE: INTERRUPTED", flush=True)
            raise
        if result.returncode != 0:
            entries[idx_model]["status"] = "failed"
            entries[idx_model]["return_code"] = result.returncode
            manifest["status"] = "failed"
            manifest["passed"] = False
            manifest["finished_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            print(f"\nSWARM TEST SUITE: FAIL at {model}", flush=True)
            raise SystemExit(result.returncode)
        if not args.build_only:
            component_path = scope_root/entries[idx_model]["output"]
            try:
                component = json.loads(component_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                entries[idx_model]["status"] = "failed"
                entries[idx_model]["error"] = f"invalid component manifest: {error}"
                manifest["status"] = "failed"
                manifest["passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                print(f"\nSWARM TEST SUITE: FAIL at {model} manifest", flush=True)
                raise SystemExit(1)
            if component.get("passed") is not True:
                entries[idx_model]["status"] = "failed"
                entries[idx_model]["error"] = "component manifest does not report passed=true"
                manifest["status"] = "failed"
                manifest["passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                print(f"\nSWARM TEST SUITE: FAIL at {model} manifest", flush=True)
                raise SystemExit(1)
        entries[idx_model]["status"] = "built" if args.build_only else "passed"
        entries[idx_model]["return_code"] = 0
        manifest["models_completed"] = idx_model + 1
        write_manifest(manifest_path, manifest)

    manifest["status"] = "passed"
    manifest["passed"] = True
    manifest["finished_utc"] = utc_now()
    write_manifest(manifest_path, manifest)
    label = "BUILD PASS" if args.build_only else "PASS"
    print(
        f"\nSWARM TEST SUITE: {label}  "
        f"models={len(models)}/{len(models)}  "
        f"analytical builds={manifest['analytical_builds_expected']}  "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
