#!/usr/bin/env python3

"""Compile, execute, and analyze one verification model at several resolutions

Every model-local ``run.py`` imports :func:`run` from this file and supplies its
own directory name.  This shared implementation keeps the build flags, output
layout, environment capture, JSON records, and convergence table identical
across all verification models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from validate_case import analyze as default_analyze

QAV_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QAV_ROOT/"tool"))

from qav_config import fluid_case_tier, model_analyzer, model_executable


def observed_orders(errors: list[float]) -> list[float]:
    """Return pairwise convergence orders for successive factor-of-two grids"""

    import math

    # if E(h) is proportional to h**p and the next grid has h/2, then
    # p = log_2(E(h)/E(h/2))
    return [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]


def capture(command: list[str], cwd: Path) -> str:
    """Run a diagnostic command and return its combined standard output"""

    try:
        # check=False is intentional: missing GPU diagnostics must not prevent a
        # simulation from running; stderr is folded into stdout so the saved
        # environment file contains either version information or the error
        result = subprocess.run(command, cwd=cwd, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def environment_record(project_root: Path, math_mode: str = "fast") -> dict[str, str]:
    """Record the CUDA toolchain and NVIDIA device used by one verification run"""

    return {
        "cuda_math": math_mode,
        "gpu_target": os.environ.get("CUDA_ARCH", "sm_80"),
        "nvcc": capture(["nvcc", "--version"], project_root),
        "nvidia_smi": capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
        ),
    }


def output_tag(model: str, cfl: float, power: float, shift: float) -> str:
    """Name parameter variants that would otherwise overwrite one another"""

    # only parameters varied by run_suite.py enter directory and metric names
    # the empty string keeps single-configuration model outputs one level higher
    if model in {"test_optdepth", "test_attenuation_2d"}:
        return f"p{power:+g}"
    if model == "test_x_transport_2d":
        return f"shift{shift:g}"
    if model in {"test_y_transport_cyl", "test_y_transport_sph", "test_z_transport_3d"}:
        return f"cfl{cfl:g}"
    return ""


def selected_sweep() -> str:
    """Return the fluid line-kernel implementation selected for this run"""

    sweep = os.environ.get("FLUID_SWEEP", "thread")
    if sweep not in {"thread", "block"}:
        raise SystemExit("FLUID_SWEEP must be thread or block")
    return sweep


def clean_results(out_dir: Path, data_dir: Path, variant: str) -> None:
    """Remove stale records owned by one model variant before rerunning it"""

    for path in data_dir.glob("*.dat"):
        path.unlink()
    for path in data_dir.glob("meta_N*.json"):
        path.unlink()
    (data_dir/"environment.json").unlink(missing_ok=True)

    for path in out_dir.glob("metrics_N*.json"):
        owned = path.name.endswith(f"_{variant}.json") if variant else bool(
            re.fullmatch(r"metrics_N\d+\.json", path.name)
        )
        if owned:
            path.unlink()
    (out_dir/f"manifest_{variant or 'default'}.json").unlink(missing_ok=True)


def assess_records(
    model: str, records: list[dict], shift: float, power: float,
) -> dict[str, object]:
    """Apply broad regression gates to one analytical convergence sequence"""

    import math

    scalar_values = []
    for record in records:
        for errors in record["errors"].values():
            scalar_values.extend(errors.values())
        if "mass_relative_change" in record:
            scalar_values.append(record["mass_relative_change"])
    finite = all(math.isfinite(float(value)) for value in scalar_values)

    if records and all("passed" in record for record in records):
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
                convergence_orders = observed_orders(convergence_errors)
                convergence_passed = convergence_orders[-1] >= minimum_order
            else:
                convergence_passed = False
        return {
            "finite": finite,
            "validator_owned": True,
            "convergence_field": convergence_field,
            "minimum_order": minimum_order,
            "orders": convergence_orders,
            "convergence_passed": convergence_passed,
            "passed": finite and convergence_passed
                and all(record["passed"] is True for record in records),
        }
    mass_max = max((record.get("mass_relative_change", 0.0) for record in records), default=0.0)
    mass_passed = mass_max <= 1.0e-8

    primary = "optdepth" if records[0]["case"] == "optdepth" else "density"
    errors = [record["errors"][primary]["l1"] for record in records]
    exact_case = (
        (model == "test_x_transport_2d" and abs(shift - round(shift)) <= 1.0e-12)
        or (model == "test_optdepth" and abs(power) <= 1.0e-12)
    )
    if model == "test_source_drag":
        accuracy_passed = max(
            values["linf"] for values in records[0]["errors"].values()
        ) <= 1.0e-12
        convergence_orders: list[float] = []
        convergence_passed = True
    elif exact_case:
        accuracy_passed = errors[-1] <= 1.0e-10
        convergence_orders = []
        convergence_passed = True
    else:
        accuracy_passed = errors[-1] <= 2.0e-2
        convergence_orders = observed_orders(errors) \
            if len(errors) > 1 and all(error > 0.0 for error in errors) else []
        minimum_order = 1.5 if len(records) >= 4 else 0.75
        convergence_passed = len(records) == 1 or (
            bool(convergence_orders) and convergence_orders[-1] >= minimum_order
        )

    return {
        "finite": finite,
        "mass_max": mass_max,
        "mass_tolerance": 1.0e-8,
        "mass_passed": mass_passed,
        "primary_field": primary,
        "finest_l1": errors[-1],
        "accuracy_tolerance": 1.0e-12 if model == "test_source_drag" else (1.0e-10 if exact_case else 2.0e-2),
        "accuracy_passed": accuracy_passed,
        "orders": convergence_orders,
        "convergence_passed": convergence_passed,
        "passed": finite and mass_passed and accuracy_passed and convergence_passed,
    }


def run(model: str) -> None:
    """Build and run one named verification model for all requested resolutions"""

    parser = argparse.ArgumentParser()
    default_resolutions = [8] if model == "test_source_drag" else [32, 64, 128, 256]
    parser.add_argument("--res", nargs="+", type=int, default=default_resolutions)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=-1.0)
    parser.add_argument("--shift", type=float, default=3.25)
    parser.add_argument(
        "--math-mode", choices=("fast", "precise"),
        default=os.environ.get("CUDA_MATH", "fast"),
        help="select the CUDA arithmetic mode without mixing its output archive",
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    # backend runners live one level below qav/cuda so parents[4] is the repository root
    project_root = Path(__file__).resolve().parents[4]
    analyze = model_analyzer(project_root, "fluid", model, default_analyze)
    test_root = Path(__file__).resolve().parents[1]
    model_dir = test_root / model
    sweep = selected_sweep()
    target = os.environ.get("CUDA_ARCH", "sm_80")
    archive_sweep = sweep if args.math_mode == "fast" else f"{sweep}_precise"
    archive_root = project_root/"qav"/"logs"/"fluid"/"cuda"/archive_sweep
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    out_dir = scope_root/model
    if not model_dir.is_dir():
        raise SystemExit(f"Unknown model directory: {model_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # parameter variants receive separate data directories, for example
    # out/thread/test_x_transport_2d/shift3.25, while their metrics JSON files
    # remain in the model's top-level output directory with the same tag in the name
    variant = output_tag(model, args.cfl, args.power, args.shift)
    tier_arguments: tuple[str, ...] = ()
    if model == "test_x_transport_2d":
        tier_arguments = ("--shift", str(args.shift))
    elif model in {"test_y_transport_cyl", "test_y_transport_sph", "test_z_transport_3d"}:
        tier_arguments = ("--cfl", str(args.cfl))
    elif model in {"test_optdepth", "test_attenuation_2d"}:
        tier_arguments = ("--power", str(args.power))
    evidence_tier = fluid_case_tier(model, tier_arguments)
    data_dir = out_dir / variant if variant else out_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        clean_results(out_dir, data_dir, variant)

    # record the compiler, GPU, and driver associated with these results; the
    # information is written once per parameter variant rather than once per N
    environment = environment_record(project_root, args.math_mode)
    (data_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    # one record is the dictionary returned by validate_case.analyze for one
    # resolution; the complete list is later used to print convergence orders
    records = []
    for resolution in args.res:
        # grid sizes and verification parameters are compile-time constants in
        # the CUDA tests; cleaning before every resolution prevents an object
        # compiled with an earlier N or parameter value from being reused
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda", f"FLUID_SWEEP={sweep}",
            f"GPU_TARGET={target}", f"CUDA_MATH={args.math_mode}",
            f"QAV_SWEEP={archive_sweep}", f"QAV_SCOPE={scope}", "clean"
        ], check=True)
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda", f"FLUID_SWEEP={sweep}",
            f"GPU_TARGET={target}", f"CUDA_MATH={args.math_mode}", f"QAV_SWEEP={archive_sweep}",
            f"QAV_SCOPE={scope}", f"RES={resolution}",
            f"CFL={args.cfl:.17g}", f"POWER={args.power:.17g}", f"SHIFT={args.shift:.17g}",
            f"OUT_TAG={variant}",
        ], check=True)
        if args.build_only:
            continue

        # the executable writes binary fields and a small JSON metadata file into
        # data_dir through PATH_OUT; analyze then constructs the analytical
        # cell averages and compares them with those files
        executable = model_executable(project_root, model, "cuda", "fluid")
        subprocess.run([str(executable)], cwd=project_root, check=True)
        record = analyze(data_dir, resolution)
        record["tier"] = evidence_tier

        # keep a machine-readable result per resolution so cluster output can be
        # downloaded and reassessed without rerunning the CUDA executable
        tag = f"N{resolution}"
        if variant:
            tag += f"_{variant}"
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)

    if args.build_only or not records:
        return

    assessment = assess_records(model, records, args.shift, args.power)
    passed = assessment["passed"] is True
    manifest = {
        "schema": 1,
        "model": model,
        "case": records[0]["case"],
        "backend": "cuda",
        "sweep": archive_sweep,
        "variant": variant,
        "tier": evidence_tier,
        "resolutions": [record["resolution"] for record in records],
        "files": [
            f"metrics_N{record['resolution']}{f'_{variant}' if variant else ''}.json"
            for record in records
        ],
        "environment": f"{variant + '/' if variant else ''}environment.json",
        "assessment": assessment,
        "passed": passed,
    }
    (out_dir/f"manifest_{variant or 'default'}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # optical depth is the primary field only for the zero-step quadrature test;
    # every dynamical test reports density in the compact terminal table
    field = records[0].get(
        "primary_field", "optdepth" if records[0]["case"] == "optdepth" else "density"
    )
    print(f"\n{records[0]['case']}: {field} convergence")
    print(f"{'N':>8} {'L1':>14} {'L2':>14} {'Linf':>14} {'mass rel':>14}")
    l1_errors = []
    for record in records:
        error = record["errors"][field]
        l1_errors.append(error["l1"])
        mass = record.get("mass_relative_change", float("nan"))
        print(f"{record['resolution']:8d} {error['l1']:14.6e} {error['l2']:14.6e} "
              f"{error['linf']:14.6e} {mass:14.6e}")
    if len(l1_errors) > 1:
        # orders are printed only for L1 here; l2 and Linf remain available in
        # every metrics JSON file for more detailed post-processing
        print("L1 orders:", " ".join(f"{order:.4f}" for order in observed_orders(l1_errors)))
    print(f"{model}{f'/{variant}' if variant else ''}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    # direct invocation cannot infer a model; a model-local run.py supplies it
    # explicitly and is therefore the supported entry point
    raise SystemExit("Call run(model_name) from a model-local run.py wrapper")
