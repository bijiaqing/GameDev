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
import subprocess
from pathlib import Path

from validate_case import analyze


def observed_orders(errors: list[float]) -> list[float]:
    """Return pairwise convergence orders for successive factor-of-two grids"""

    import math

    # If E(h) is proportional to h**p and the next grid has h/2, then
    # p = log_2(E(h)/E(h/2)).
    return [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]


def capture(command: list[str], cwd: Path) -> str:
    """Run a diagnostic command and return its combined standard output"""

    try:
        # check=False is intentional: missing GPU diagnostics must not prevent a
        # simulation from running.  stderr is folded into stdout so the saved
        # environment file contains either version information or the error.
        result = subprocess.run(command, cwd=cwd, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def output_tag(model: str, cfl: float, power: float, shift: float) -> str:
    """Name parameter variants that would otherwise overwrite one another"""

    # Only parameters varied by run_suite.py enter directory and metric names.
    # The empty string keeps single-configuration model outputs one level higher.
    if model == "verify_optdepth":
        return f"p{power:+g}"
    if model == "verify_x_transport_2d":
        return f"shift{shift:g}"
    if model in {"verify_y_transport_cyl", "verify_y_transport_sph", "verify_z_transport_3d"}:
        return f"cfl{cfl:g}"
    return ""


def selected_sweep() -> str:
    """Return the fluid line-kernel implementation selected for this run"""

    sweep = os.environ.get("FLUID_SWEEP", "thread")
    if sweep not in {"thread", "block"}:
        raise SystemExit("FLUID_SWEEP must be thread or block")
    return sweep


def run(model: str) -> None:
    """Build and run one named verification model for all requested resolutions"""

    parser = argparse.ArgumentParser()
    default_resolutions = [8] if model == "verify_source_drag" else [32, 64, 128, 256]
    parser.add_argument("--res", nargs="+", type=int, default=default_resolutions)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=-1.0)
    parser.add_argument("--shift", type=float, default=3.25)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    # Starting from verify_common/run_model.py, parents[3] is the repository root, where the
    # Makefile lives, and parents[1] is tst/fluid/, where the models live.
    project_root = Path(__file__).resolve().parents[3]
    test_root = Path(__file__).resolve().parents[1]
    model_dir = test_root / model
    sweep = selected_sweep()
    out_dir = test_root / "out" / sweep / model
    if not model_dir.is_dir():
        raise SystemExit(f"Unknown model directory: {model_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parameter variants receive separate data directories, for example
    # out/thread/verify_x_transport_2d/shift3.25, while their metrics JSON files
    # remain in the model's top-level output directory with the same tag in the name.
    variant = output_tag(model, args.cfl, args.power, args.shift)
    data_dir = out_dir / variant if variant else out_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Record the compiler, GPU, and driver associated with these results.  The
    # information is written once per parameter variant rather than once per N.
    environment = [
        "nvcc:\n" + capture(["nvcc", "--version"], project_root),
        "nvidia-smi:\n" + capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
        ),
    ]
    (data_dir / "environment.txt").write_text("\n\n".join(environment) + "\n")

    # One record is the dictionary returned by validate_case.analyze for one
    # resolution.  The complete list is later used to print convergence orders.
    records = []
    for resolution in args.res:
        # Grid sizes and verification parameters are compile-time constants in
        # the CUDA tests.  Cleaning before every resolution prevents an object
        # compiled with an earlier N or parameter value from being reused.
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", f"FLUID_SWEEP={sweep}", "clean"
        ], check=True)
        subprocess.run([
            "make", "-C", str(project_root), f"MODEL={model}", f"FLUID_SWEEP={sweep}",
            f"RES={resolution}",
            f"CFL={args.cfl:.17g}", f"POWER={args.power:.17g}", f"SHIFT={args.shift:.17g}",
            f"OUT_TAG={variant}",
        ], check=True)
        if args.build_only:
            continue

        # The executable writes binary fields and a small metadata file into
        # data_dir through PATH_OUT.  analyze then constructs the analytical
        # cell averages and compares them with those files.
        subprocess.run([str(model_dir / "gamedev")], cwd=project_root, check=True)
        record = analyze(data_dir, resolution)

        # Keep a machine-readable result per resolution so cluster output can be
        # downloaded and reassessed without rerunning the CUDA executable.
        tag = f"N{resolution}"
        if variant:
            tag += f"_{variant}"
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)

    if args.build_only or not records:
        return

    # Optical depth is the primary field only for the zero-step quadrature test;
    # every dynamical test reports density in the compact terminal table.
    field = "optdepth" if records[0]["case"] == "optdepth" else "density"
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
        # Orders are printed only for L1 here; L2 and Linf remain available in
        # every metrics JSON file for more detailed post-processing.
        print("L1 orders:", " ".join(f"{order:.4f}" for order in observed_orders(l1_errors)))


if __name__ == "__main__":
    # Direct invocation cannot infer a model.  A model-local run.py supplies it
    # explicitly and is therefore the supported entry point.
    raise SystemExit("Call run(model_name) from a model-local run.py wrapper")
