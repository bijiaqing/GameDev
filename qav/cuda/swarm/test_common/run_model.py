#!/usr/bin/env python3

"""Build, execute, and validate one swarm test model"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from validate_case import analyze as default_analyze

QAV_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QAV_ROOT/"tool"))

from qav_config import model_analyzer, model_executable, swarm_model_tier, swarm_resolution_tiers


def orders(errors: list[float]) -> list[float]:
    """Calculate pairwise orders while tolerating an exactly zero error"""

    result = []
    for coarse, fine in zip(errors[:-1], errors[1:]):
        result.append(math.log(coarse/fine, 2.0) if coarse > 0.0 and fine > 0.0 else float("nan"))
    return result


def capture(command: list[str], cwd: Path) -> str:
    """Capture optional cluster diagnostics without making them test requirements"""

    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def clean_results(out_dir: Path) -> None:
    """Remove every artifact owned by one analytical model before a new run"""

    for pattern in ("*.dat", "meta_N*.json", "meta_N*.txt", "metrics_N*.json"):
        for path in out_dir.glob(pattern):
            path.unlink()
    for name in ("environment.json", "environment.txt", "manifest.json"):
        (out_dir/name).unlink(missing_ok=True)


def print_record(record: dict) -> None:
    """Print the errors and statistical ratios that actually determine PASS"""

    labels = []
    for name, values in record.get("errors", {}).items():
        labels.append(
            f"{name}: L2={values['l2']:.6e} Linf={values['linf']:.6e}"
        )
    if "mass_relative_change" in record:
        labels.append(f"mass={record['mass_relative_change']:.6e}")
    for name, values in record.get("statistics", {}).items():
        mean_fraction = abs(values["mean_error"]) / values["mean_limit"]
        variance_fraction = abs(values["variance_error"]) / values["variance_limit"]
        labels.append(
            f"{name}: mean/limit={mean_fraction:.3f} variance/limit={variance_fraction:.3f}"
        )
    print(f"N={record['resolution']:4d}  " + ", ".join(labels))


def run(model: str) -> None:
    """Compile every requested resolution, validate it, and write one model manifest"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    analyze = model_analyzer(project_root, "swarm", model, default_analyze)
    swarm_root = Path(__file__).resolve().parents[1]
    model_dir = swarm_root/model
    archive_root = project_root/"qav"/"logs"/"swarm"/"cuda"
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    out_dir = scope_root/model
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.build_only:
        clean_results(out_dir)

        # keep compiler, GPU, and driver information beside the numerical metrics
        # this is especially important for stochastic tests and CUDA regressions,
        # whose performance and last-bit results can depend on the environment
        environment = {
            "gpu_target": os.environ.get("CUDA_ARCH", "sm_80"),
            "nvcc": capture(["nvcc", "--version"], project_root),
            "nvidia_smi": capture(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
            ),
        }
        (out_dir/"environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )

    resolution_tiers = swarm_resolution_tiers(model, args.res)
    tier_by_resolution = {
        int(record["resolution"]): str(record["tier"]) for record in resolution_tiers
    }
    records = []
    target = os.environ.get("CUDA_ARCH", "sm_80")
    for resolution in args.res:
        # every test constant is compiled into CUDA code; cleaning prevents an
        # object built for a previous TEST_RES value from being reused
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda",
            f"GPU_TARGET={target}", f"QAV_SCOPE={scope}", "clean",
        ], check=True)
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda",
            f"GPU_TARGET={target}", f"QAV_SCOPE={scope}", f"RES={resolution}",
        ], check=True)
        if args.build_only:
            continue
        executable = model_executable(project_root, model, "cuda", "swarm")
        subprocess.run([str(executable)], cwd=project_root, check=True)

        # the CUDA driver writes raw values only; all expected values and pass
        # thresholds are constructed independently by validate_case.py
        record = analyze(out_dir, resolution)
        record["tier"] = tier_by_resolution[resolution]
        expected_case = model.removeprefix("test_")
        if record["case"] != expected_case or record["resolution"] != resolution:
            raise RuntimeError(
                f"{model} produced case={record['case']} resolution={record['resolution']}; "
                f"expected case={expected_case} resolution={resolution}"
            )
        (out_dir/f"metrics_N{resolution}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)

    if not records:
        return

    convergence_field = records[0].get("convergence_field")
    convergence_orders = []
    convergence_passed = True
    minimum_order = float(records[0].get("minimum_order", 0.0))
    if convergence_field and len(records) > 1:
        convergence_errors = [
            float(record["errors"][convergence_field]["l1"])
            for record in records
        ]
        if all(error > 0.0 for error in convergence_errors):
            convergence_orders = orders(convergence_errors)
            convergence_passed = convergence_orders[-1] >= minimum_order
        else:
            convergence_passed = False
    passed = all(record["passed"] for record in records) and convergence_passed
    manifest = {
        "schema": 1,
        "model": model,
        "case": records[0]["case"],
        "backend": "cuda",
        "gpu_target": target,
        "tier": swarm_model_tier(model),
        "resolution_tiers": resolution_tiers,
        "resolutions": [record["resolution"] for record in records],
        "files": [f"metrics_N{record['resolution']}.json" for record in records],
        "environment": "environment.json",
        "assessment": {
            "convergence_field": convergence_field,
            "minimum_order": minimum_order,
            "orders": convergence_orders,
            "convergence_passed": convergence_passed,
        },
        "passed": passed,
    }
    (out_dir/"manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"\n{model}: {'PASS' if passed else 'FAIL'}")
    for record in records:
        print_record(record)
    if convergence_orders:
        print(
            f"{convergence_field} L1 orders:",
            " ".join(f"{value:.4f}" for value in convergence_orders),
        )
    if not passed:
        raise SystemExit(1)
