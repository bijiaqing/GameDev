#!/usr/bin/env python3

"""Run the analytical swarm verification matrix"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path


VAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VAL_ROOT))
from val_config import model_output

from val_config import BACKEND, TARGET_ENV, DEFAULT_TARGET, backend_environment

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


BACKEND_GROUPS = {
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
        "--group", choices=("all", *SWARM_GROUPS, *BACKEND_GROUPS), default="all"
    )
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--rebuild-manifest", action="store_true",
        help="reconstruct the aggregate manifest from downloaded component manifests without rerunning",
    )
    parser.add_argument(
        "--target", default=os.environ.get(TARGET_ENV, DEFAULT_TARGET),
        help="AMDGPU target passed to hipcc",
    )
    args = parser.parse_args()
    if args.build_only and args.rebuild_manifest:
        parser.error("--build-only and --rebuild-manifest cannot be combined")

    run_environment = os.environ.copy()
    run_environment["GPU_BACKEND"] = BACKEND
    run_environment[TARGET_ENV] = args.target
    run_environment["VAL_SCOPE"] = args.group

    resolutions = args.res[:2] if args.quick else args.res
    root = Path(__file__).resolve().parents[1]/"mod"
    project_root = VAL_ROOT.parent
    scope_root = model_output(project_root/"val", "swarm", "_suite", BACKEND, scope=args.group)
    manifest_path = scope_root/("manifest_all.json" if args.group == "all" else "manifest.json")
    environment_path = scope_root/("environment_all.json" if args.group == "all" else "environment.json")

    # preserve the publication order from isolated trajectories through coupled collisions
    models = BACKEND_GROUPS[args.group] if args.group in BACKEND_GROUPS else swarm_models(args.group)

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

    if args.group in BACKEND_GROUPS:
        publication_metrics = sum(
            1 if collision_runtime_model(entry["model"]) else len(entry["resolutions"])
            for entry in entries
        )
        metric_tiers = {PUBLICATION_TIER: publication_metrics}
    else:
        metric_tiers = swarm_metric_tiers(models, resolutions)
    included_tiers = [PUBLICATION_TIER]
    campaign_tier = PUBLICATION_TIER

    manifest = {
        "schema": 1,
        "suite": "swarm",
        "backend": BACKEND,
        "group": args.group,
        "environment": environment_path.name,
        "requested_resolutions": args.res,
        "effective_resolutions": resolutions,
        "quick": args.quick,
        "build_only": args.build_only,
        "reconstructed": args.rebuild_manifest,
        "gpu_target": args.target,
        "campaign_tier": campaign_tier,
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
        for idx_model, entry in enumerate(entries):
            component_path = model_output(project_root/"val", "swarm", entry["model"], BACKEND, scope=args.group)/Path(entry["output"]).relative_to(entry["model"])
            try:
                component = json.loads(component_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                entry["status"] = "failed"
                entry["error"] = f"invalid component manifest: {error}"
                manifest["status"] = "failed"
                manifest["passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                raise SystemExit(f"cannot rebuild aggregate manifest: {component_path}: {error}")
            if component.get("passed") is not True:
                entry["status"] = "failed"
                entry["error"] = "component manifest does not report passed=true"
                manifest["status"] = "failed"
                manifest["passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                raise SystemExit(f"cannot rebuild aggregate manifest: {component_path} did not pass")
            entry["status"] = "passed"
            entry["return_code"] = 0
            manifest["models_completed"] = idx_model + 1

        manifest["status"] = "passed"
        manifest["passed"] = True
        manifest["finished_utc"] = utc_now()
        write_manifest(manifest_path, manifest)
        print(
            f"SWARM TEST SUITE: MANIFEST PASS  models={len(models)}/{len(models)}  "
            f"manifest={manifest_path}",
            flush=True,
        )
        return

    manifest_path.unlink(missing_ok=True)
    write_manifest(manifest_path, manifest)
    environment = {**backend_environment(capture, project_root), "gpu_target": args.target, "python": sys.version}
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    for idx_model, model in enumerate(models):
        model_resolutions = entries[idx_model]["resolutions"]
        if collision_runtime_model(model) or model == "test_knn":
            command = [sys.executable, str(root/model/"run.py"), "--target", args.target]
        else:
            command = [
                sys.executable, str(root/model/"run.py"),
                "--res", *(str(value) for value in model_resolutions),
            ]
        if args.build_only:
            command.append("--build-only")
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
            component_path = model_output(project_root/"val", "swarm", model, BACKEND, scope=args.group)/Path(entries[idx_model]["output"]).relative_to(model)
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
