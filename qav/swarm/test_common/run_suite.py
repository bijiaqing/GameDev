#!/usr/bin/env python3

"""Run the analytical swarm verification matrix"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path


GROUPS = {
    "grid": ["test_grid_1d", "test_grid_2d", "test_grid_3d"],
    "transport": ["test_orbit_1d", "test_drag_1d", "test_viscflow_1d", "test_orbit_2d", "test_drag_2d"],
    "diffusion": ["test_diffusion_1d", "test_diffusion_2d", "test_diffusion_3d"],
    "radiation": ["test_radiation_1d", "test_prdrag_1d", "test_radiation_2d", "test_prdrag_2d"],
    "boundary": ["test_boundary_1d", "test_boundary_2d", "test_boundary_3d", "test_boundary_half"],
    "collision": ["test_collision_1d", "test_import_1d", "test_collision_2d", "test_collision_3d"],
    "knn": ["test_knn"],
}

RADIAL_MODELS = [
    "test_grid_1d",
    "test_orbit_1d",
    "test_drag_1d",
    "test_viscflow_1d",
    "test_diffusion_1d",
    "test_radiation_1d",
    "test_prdrag_1d",
    "test_boundary_1d",
    "test_collision_1d",
    "test_import_1d",
    "test_knn",
]

# Rebuilding one-step algebra checks at several TEST_RES values adds no
# coverage.  Spatial, temporal, and ensemble tests retain every requested N.
FIXED_RESOLUTION = {
    "test_drag_1d",
    "test_viscflow_1d",
    "test_radiation_1d",
    "test_prdrag_1d",
    "test_collision_1d",
    "test_import_1d",
    "test_drag_2d",
    "test_radiation_2d",
    "test_prdrag_2d",
    "test_collision_2d",
    "test_collision_3d",
    "test_boundary_1d",
    "test_boundary_2d",
    "test_boundary_3d",
    "test_boundary_half",
    "test_knn",
}


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
    parser.add_argument("--group", choices=("all", "radial", *GROUPS), default="all")
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--knn-full", action="store_true")
    args = parser.parse_args()

    resolutions = args.res[:2] if args.quick else args.res
    root = Path(__file__).resolve().parents[1]
    project_root = root.parents[1]
    out_root = root/"out"
    manifest_path = out_root/"manifest.json"
    environment_path = out_root/"environment.json"

    # Preserve the order in GROUPS so terminal output follows the progression
    # from grid primitives to coupled physical operators.
    if args.group == "all":
        models = [model for values in GROUPS.values() for model in values]
    elif args.group == "radial":
        models = RADIAL_MODELS
    else:
        models = GROUPS[args.group]

    entries = []
    for model in models:
        model_resolutions = resolutions[:1] if model in FIXED_RESOLUTION else resolutions
        if model == "test_knn":
            output = "test_knn/radial/suite_manifest.json" if args.group == "radial" \
                else "test_knn/suite_manifest.json"
        else:
            output = f"{model}/manifest.json"
        entries.append({
            "model": model,
            "resolutions": model_resolutions,
            "output": None if args.build_only else output,
            "status": "pending",
        })

    manifest = {
        "suite": "swarm",
        "group": args.group,
        "environment": "environment.json",
        "requested_resolutions": args.res,
        "effective_resolutions": resolutions,
        "quick": args.quick,
        "knn_full": args.knn_full,
        "build_only": args.build_only,
        "models_expected": len(models),
        "models_completed": 0,
        "analytical_builds_expected": sum(
            len(entry["resolutions"]) for entry in entries
            if entry["model"] != "test_knn"
        ),
        "knn_standalone_builds_expected": (
            2 if args.group == "radial" else 4
        ) if "test_knn" in models else 0,
        "knn_production_links_expected": (
            2 if args.group == "radial" else 6
        ) if "test_knn" in models else 0,
        "status": "running",
        "suite_passed": None,
        "started_utc": utc_now(),
        "finished_utc": None,
        "models": entries,
    }
    manifest_path.unlink(missing_ok=True)
    write_manifest(manifest_path, manifest)
    environment = {
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
        command = [sys.executable, str(root/model/"run.py"), "--res", *(str(value) for value in model_resolutions)]
        if args.build_only:
            command.append("--build-only")
        if model == "test_knn" and args.group == "radial":
            command.append("--radial-only")
        if model == "test_knn" and args.knn_full:
            command.append("--full")
        print(f"\n=== {model} ===", flush=True)
        entries[idx_model]["status"] = "running"
        write_manifest(manifest_path, manifest)
        try:
            result = subprocess.run(command, check=False)
        except KeyboardInterrupt:
            entries[idx_model]["status"] = "interrupted"
            manifest["status"] = "interrupted"
            manifest["finished_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            print("\nSWARM TEST SUITE: INTERRUPTED", flush=True)
            raise
        if result.returncode != 0:
            entries[idx_model]["status"] = "failed"
            entries[idx_model]["returncode"] = result.returncode
            manifest["status"] = "failed"
            manifest["suite_passed"] = False
            manifest["finished_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            print(f"\nSWARM TEST SUITE: FAIL at {model}", flush=True)
            raise SystemExit(result.returncode)
        if not args.build_only:
            component_path = out_root/entries[idx_model]["output"]
            try:
                component = json.loads(component_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                entries[idx_model]["status"] = "failed"
                entries[idx_model]["error"] = f"invalid component manifest: {error}"
                manifest["status"] = "failed"
                manifest["suite_passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                print(f"\nSWARM TEST SUITE: FAIL at {model} manifest", flush=True)
                raise SystemExit(1)
            if component.get("passed") is not True:
                entries[idx_model]["status"] = "failed"
                entries[idx_model]["error"] = "component manifest does not report passed=true"
                manifest["status"] = "failed"
                manifest["suite_passed"] = False
                manifest["finished_utc"] = utc_now()
                write_manifest(manifest_path, manifest)
                print(f"\nSWARM TEST SUITE: FAIL at {model} manifest", flush=True)
                raise SystemExit(1)
        entries[idx_model]["status"] = "built" if args.build_only else "passed"
        entries[idx_model]["returncode"] = 0
        manifest["models_completed"] = idx_model + 1
        write_manifest(manifest_path, manifest)

    manifest["status"] = "passed"
    manifest["suite_passed"] = True
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
