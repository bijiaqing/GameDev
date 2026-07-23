#!/usr/bin/env python3
"""Build, run, validate, and tabulate one rpi_fluid verification model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validate_case import analyze


def observed_orders(errors: list[float]) -> list[float]:
    import math
    return [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]


def capture(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def run(model: str) -> None:
    parser = argparse.ArgumentParser()
    default_resolutions = [8] if model == "verify_source_drag" else [32, 64, 128, 256]
    parser.add_argument("--res", nargs="+", type=int, default=default_resolutions)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=-1.0)
    parser.add_argument("--shift", type=float, default=3.25)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    model_dir = root / "mod" / model
    out_dir = root / "out" / model
    if not model_dir.is_dir():
        raise SystemExit(f"Unknown model directory: {model_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    environment = [
        "nvcc:\n" + capture(["nvcc", "--version"], root),
        "nvidia-smi:\n" + capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], root),
    ]
    (out_dir / "environment.txt").write_text("\n\n".join(environment) + "\n")

    records = []
    for resolution in args.res:
        subprocess.run(["make", "-C", str(root), f"MODEL={model}", "clean"], check=True)
        subprocess.run([
            "make", "-C", str(root), f"MODEL={model}", f"RES={resolution}",
            f"CFL={args.cfl:.17g}", f"POWER={args.power:.17g}", f"SHIFT={args.shift:.17g}",
        ], check=True)
        if args.build_only:
            continue

        subprocess.run([str(model_dir / "graffiti")], cwd=root, check=True)
        record = analyze(out_dir, resolution)
        tag = f"N{resolution}"
        if record["case"] == "optdepth":
            tag += f"_p{args.power:+g}"
        if record["case"] == "x_transport":
            tag += f"_shift{args.shift:g}"
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)

    if args.build_only or not records:
        return

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
        print("L1 orders:", " ".join(f"{order:.4f}" for order in observed_orders(l1_errors)))


if __name__ == "__main__":
    raise SystemExit("Call run(model_name) from a model-local run.py wrapper")
