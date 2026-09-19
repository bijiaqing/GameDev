#!/usr/bin/env python3

"""Build, execute, and validate one swarm test model"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
import math
import subprocess
from pathlib import Path


from validate_case import analyze as default_analyze

VAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VAL_ROOT))
from val_config import model_output

from val_config import BACKEND, TARGET_ENV, DEFAULT_TARGET, backend_environment

from val_config import PUBLICATION_TIER, model_analyzer, model_executable, swarm_resolution_tiers

COLPHYS_MODELS = {"test_colphys_code", "test_colphys_cgs", "test_colphys_3d"}


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


def combine_search_records(search_records: dict[str, dict]) -> dict:
    """Retain one metric record while requiring both physical search paths to pass"""

    record = dict(search_records["morton"])
    record["collision_search"] = "both"
    record["searches"] = search_records
    record["passed"] = all(item["passed"] for item in search_records.values())
    record["maximum_relative_error"] = max(
        item["maximum_relative_error"] for item in search_records.values()
    )
    record["maximum_measure_relative_error"] = max(
        item["maximum_measure_relative_error"] for item in search_records.values()
    )
    record["activation"] = dict(record["activation"])
    record["activation"]["both_searches_passed"] = record["passed"]
    record["errors"] = {
        name: {
            norm: max(item["errors"][name][norm] for item in search_records.values())
            for norm in ("l1", "l2", "linf")
        }
        for name in record["errors"]
    }
    return record


def run(model: str) -> None:
    """Compile every requested resolution, validate it, and write one model manifest"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    analyze = model_analyzer(project_root, "swarm", model, default_analyze)
    swarm_root = Path(__file__).resolve().parents[1]
    model_dir = swarm_root/"mod"/model
    scope = os.environ.get("VAL_SCOPE", "manual")
    out_dir = model_output(project_root/"val", "swarm", model, BACKEND, scope=scope)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.build_only:
        clean_results(out_dir)

        # keep compiler, GPU, and driver information beside the numerical metrics
        # this is especially important for stochastic tests and CUDA regressions,
        # whose performance and last-bit results can depend on the environment
        environment = {**backend_environment(capture, project_root), "gpu_target": os.environ.get(TARGET_ENV, DEFAULT_TARGET)}
        (out_dir/"environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )

    resolution_tiers = swarm_resolution_tiers(model, args.res)
    tier_by_resolution = {
        int(record["resolution"]): str(record["tier"]) for record in resolution_tiers
    }
    cleaned_output_dirs = {out_dir} if not args.build_only else set()
    records = []
    target = os.environ.get(TARGET_ENV, DEFAULT_TARGET)
    for resolution in args.res:
        searches = ("kdtree", "morton") if model in COLPHYS_MODELS else (None,)
        search_records = {}
        record = None
        for search in searches:
            run_out_dir = out_dir if search is None else out_dir/search
            if not args.build_only:
                run_out_dir.mkdir(parents=True, exist_ok=True)
                if run_out_dir not in cleaned_output_dirs:
                    clean_results(run_out_dir)
                    cleaned_output_dirs.add(run_out_dir)

            make_options = [
                f"MODEL={model}", f"GPU_BACKEND={BACKEND}", f"GPU_TARGET={target}",
                f"VAL_SCOPE={scope}",
            ]
            if search is not None:
                make_options.extend((f"COLLISION_SEARCH={search}", f"OUT_TAG={search}"))

            # every test constant is compiled into CUDA code; cleaning prevents an
            # object built for a previous TEST_RES value from being reused
            subprocess.run([
                "make", "-C", str(project_root), *make_options, "clean",
            ], check=True)
            subprocess.run([
                "make", "-C", str(project_root), *make_options, f"RES={resolution}",
            ], check=True)
            if args.build_only:
                continue
            executable = model_executable(project_root, model, BACKEND, "swarm")
            subprocess.run([str(executable)], cwd=project_root, check=True)

            # the CUDA driver writes raw values only; all expected values and pass
            # thresholds are constructed independently by validate_case.py
            current = analyze(run_out_dir, resolution)
            expected_case = model.removeprefix("test_")
            if current["case"] != expected_case or current["resolution"] != resolution:
                raise RuntimeError(
                    f"{model} produced case={current['case']} resolution={current['resolution']}; "
                    f"expected case={expected_case} resolution={resolution}"
                )
            if search is None:
                record = current
            else:
                current["collision_search"] = search
                search_records[search] = current

        if args.build_only:
            continue
        if search_records:
            record = combine_search_records(search_records)
        record["tier"] = tier_by_resolution[resolution]
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
        "backend": BACKEND,
        "gpu_target": target,
        "tier": PUBLICATION_TIER,
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
