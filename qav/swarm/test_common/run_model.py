#!/usr/bin/env python3

"""Build, execute, and validate one swarm test model"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from validate_case import analyze


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

    for pattern in ("*.dat", "meta_N*.txt", "metrics_N*.json"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    swarm_root = Path(__file__).resolve().parents[1]
    model_dir = swarm_root/model
    out_dir = swarm_root/"out"/model
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.build_only:
        clean_results(out_dir)

        # Keep compiler, GPU, and driver information beside the numerical metrics.
        # This is especially important for stochastic tests and CUDA regressions,
        # whose performance and last-bit results can depend on the environment.
        environment = {
            "nvcc": capture(["nvcc", "--version"], project_root),
            "nvidia_smi": capture(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
            ),
        }
        (out_dir/"environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )

    records = []
    for resolution in args.res:
        # Every test constant is compiled into CUDA code.  Cleaning prevents an
        # object built for a previous TEST_RES value from being reused.
        subprocess.run(["make", "-C", str(project_root), f"MODEL={model}", "clean"], check=True)
        subprocess.run(["make", "-C", str(project_root), f"MODEL={model}", f"RES={resolution}"], check=True)
        if args.build_only:
            continue
        subprocess.run([str(model_dir/"gamedev")], cwd=project_root, check=True)

        # The CUDA driver writes raw values only.  All expected values and pass
        # thresholds are constructed independently by validate_case.py.
        record = analyze(out_dir, resolution)
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

    passed = all(record["passed"] for record in records)
    manifest = {
        "model": model,
        "case": records[0]["case"],
        "resolutions": [record["resolution"] for record in records],
        "files": [f"metrics_N{record['resolution']}.json" for record in records],
        "environment": "environment.json",
        "passed": passed,
    }
    (out_dir/"manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"\n{model}: {'PASS' if passed else 'FAIL'}")
    for record in records:
        print_record(record)
    if records[0]["case"] in {"orbit_1d", "orbit_2d"} and len(records) > 1:
        values = [record["errors"]["state"]["l2"] for record in records]
        print("observed orders:", " ".join(f"{value:.4f}" for value in orders(values)))
