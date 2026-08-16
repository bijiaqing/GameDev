#!/usr/bin/env python3

"""Compare archived CUDA and ROCm verification metrics without comparing RNG streams"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_FLUID_METRICS = 85
EXPECTED_SWARM_METRICS = 51
STOCHASTIC_SWARM_CASES = {"diffusion_1d", "diffusion_2d", "diffusion_3d"}


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject malformed archive entries"""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def component_output(root: Path, component: str, backend: str) -> Path:
    """Resolve one backend's canonical component archive"""

    return root/"logs"/component/backend


def metric_files(root: Path, component: str, backend: str, sweep: str) -> dict[str, Path]:
    """Return stable relative names for one backend's analytical metric archive"""

    if component == "fluid":
        base = component_output(root, component, backend)/sweep
        paths = base.glob("test_*/metrics_*.json") if base.is_dir() else ()
    else:
        base = component_output(root, component, backend)
        paths = base.glob("test_*/metrics_N*.json") if base.is_dir() else ()
    return {str(path.relative_to(base)): path for path in paths}


def compare_value(
    cuda: Any,
    rocm: Any,
    location: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    mismatches: list[dict[str, Any]],
) -> None:
    """Recursively compare one deterministic metric record with numerical tolerances"""

    if isinstance(cuda, bool) or isinstance(rocm, bool):
        if type(cuda) is not type(rocm) or cuda != rocm:
            mismatches.append({"path": location, "cuda": cuda, "rocm": rocm})
        return
    if isinstance(cuda, (int, float)) and isinstance(rocm, (int, float)):
        cuda_number = float(cuda)
        rocm_number = float(rocm)
        equal = (math.isnan(cuda_number) and math.isnan(rocm_number)) or math.isclose(
            cuda_number, rocm_number,
            rel_tol=relative_tolerance, abs_tol=absolute_tolerance,
        )
        if not equal:
            mismatches.append({
                "path": location,
                "cuda": cuda,
                "rocm": rocm,
                "absolute_difference": abs(cuda_number - rocm_number),
            })
        return
    if isinstance(cuda, dict) and isinstance(rocm, dict):
        cuda_keys = set(cuda)
        rocm_keys = set(rocm)
        if cuda_keys != rocm_keys:
            mismatches.append({
                "path": location,
                "cuda_only_keys": sorted(cuda_keys - rocm_keys),
                "rocm_only_keys": sorted(rocm_keys - cuda_keys),
            })
        for key in sorted(cuda_keys & rocm_keys):
            compare_value(
                cuda[key], rocm[key], f"{location}.{key}",
                relative_tolerance, absolute_tolerance, mismatches,
            )
        return
    if isinstance(cuda, list) and isinstance(rocm, list):
        if len(cuda) != len(rocm):
            mismatches.append({
                "path": location, "cuda_length": len(cuda), "rocm_length": len(rocm),
            })
        for index, (cuda_item, rocm_item) in enumerate(zip(cuda, rocm)):
            compare_value(
                cuda_item, rocm_item, f"{location}[{index}]",
                relative_tolerance, absolute_tolerance, mismatches,
            )
        return
    if type(cuda) is not type(rocm) or cuda != rocm:
        mismatches.append({"path": location, "cuda": cuda, "rocm": rocm})


def compare_metrics(
    cuda_root: Path,
    rocm_root: Path,
    component: str,
    cuda_sweep: str,
    rocm_sweep: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    allow_partial: bool,
    fluid_field_report: Path | None,
) -> dict[str, Any]:
    """Compare the common analytical archive for one physical representation"""

    cuda_files = metric_files(cuda_root, component, "cuda", cuda_sweep)
    rocm_files = metric_files(rocm_root, component, "rocm", rocm_sweep)
    if component == "swarm":
        # restart_2d deliberately validates a vendor-native hipRAND checkpoint and has no CUDA peer
        rocm_files = {
            name: path for name, path in rocm_files.items()
            if not name.startswith("test_restart_2d/")
        }
    common = sorted(set(cuda_files) & set(rocm_files))
    missing_cuda = sorted(set(rocm_files) - set(cuda_files))
    missing_rocm = sorted(set(cuda_files) - set(rocm_files))
    expected = EXPECTED_FLUID_METRICS if component == "fluid" else EXPECTED_SWARM_METRICS
    mismatches: list[dict[str, Any]] = []
    stochastic_validations = 0
    direct_field_metrics: set[str] = set()
    direct_field_validation: dict[str, Any] | None = None

    if component == "fluid" and fluid_field_report is not None and fluid_field_report.is_file():
        field_report = load_json(fluid_field_report)
        field_records = field_report.get("records", [])
        valid_records = (
            field_report.get("passed") is True
            and field_report.get("model") == "test_z_transport_3d"
            and isinstance(field_records, list)
            and len(field_records) == 8
            and all(
                isinstance(record, dict)
                and record.get("passed") is True
                and record.get("metadata_equal") is True
                and isinstance(record.get("metric"), str)
                for record in field_records
            )
        )
        if valid_records:
            direct_field_metrics = {
                f"test_z_transport_3d/{record['metric']}" for record in field_records
            }
        direct_field_validation = {
            "path": str(fluid_field_report.resolve()),
            "passed": bool(valid_records),
            "records": len(field_records) if isinstance(field_records, list) else 0,
            "cuda_sweep": field_report.get("cuda_sweep"),
            "rocm_sweep": field_report.get("rocm_sweep"),
        }

    for name in common:
        cuda = load_json(cuda_files[name])
        rocm = load_json(rocm_files[name])
        case = str(cuda.get("case", ""))
        if component == "fluid" and name in direct_field_metrics:
            # The raw-field gate is better conditioned than comparing two
            # separately reduced truncation-error norms near compact-support
            # vacuum.  Retain exact agreement requirements for run metadata.
            for key in ("case", "resolution", "time", "steps", "cfl"):
                if cuda.get(key) != rocm.get(key):
                    mismatches.append({
                        "path": f"{name}.{key}", "cuda": cuda.get(key), "rocm": rocm.get(key),
                    })
            continue
        if component == "swarm" and case in STOCHASTIC_SWARM_CASES:
            stochastic_validations += 1
            if cuda.get("passed") is not True or rocm.get("passed") is not True:
                mismatches.append({
                    "path": name,
                    "reason": "stochastic analytical validation did not pass on both backends",
                    "cuda_passed": cuda.get("passed"),
                    "rocm_passed": rocm.get("passed"),
                })
            for key in ("case", "resolution"):
                if cuda.get(key) != rocm.get(key):
                    mismatches.append({
                        "path": f"{name}.{key}", "cuda": cuda.get(key), "rocm": rocm.get(key),
                    })
            continue
        compare_value(
            cuda, rocm, name, relative_tolerance, absolute_tolerance, mismatches,
        )

    if allow_partial:
        # A partial run treats the CUDA archive as the requested metric subset
        # and requires every one of those records to have a ROCm counterpart.
        # Extra ROCm records are expected when, for example, only precise-math
        # CUDA polar transport was rerun from the complete ROCm archive.
        complete = bool(cuda_files) and len(common) == len(cuda_files) and not missing_rocm
    else:
        complete = (
            len(cuda_files) == expected
            and len(rocm_files) == expected
            and len(common) == expected
            and not missing_cuda
            and not missing_rocm
        )
    return {
        "component": component,
        "expected_metrics": expected,
        "cuda_metrics": len(cuda_files),
        "rocm_metrics": len(rocm_files),
        "common_metrics": len(common),
        "stochastic_validation_only": stochastic_validations,
        "direct_field_validation": direct_field_validation,
        "missing_from_cuda": missing_cuda,
        "missing_from_rocm": missing_rocm,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
        "mismatches_truncated": len(mismatches) > 200,
        "partial_comparison": allow_partial,
        "complete": complete,
        "passed": complete and not mismatches,
    }


def compare_knn(cuda_root: Path, rocm_root: Path) -> dict[str, Any]:
    """Compare KNN coverage manifests while leaving vendor timing outside correctness"""

    relative = Path("test_knn/suite_manifest.json")
    cuda_path = component_output(cuda_root, "swarm", "cuda")/relative
    rocm_path = component_output(rocm_root, "swarm", "rocm")/relative
    missing = [
        label for label, path in (("cuda", cuda_path), ("rocm", rocm_path)) if not path.is_file()
    ]
    if missing:
        return {"component": "knn", "missing_backends": missing, "passed": False}

    cuda = load_json(cuda_path)
    rocm = load_json(rocm_path)
    compared = {
        "scope": (cuda.get("scope"), rocm.get("scope")),
        "ordinary_cases": (
            cuda.get("ordinary", {}).get("cases"), rocm.get("ordinary", {}).get("cases")
        ),
        "edge_cases": (cuda.get("edge", {}).get("cases"), rocm.get("edge", {}).get("cases")),
        "periodic_cases": (
            cuda.get("periodic", {}).get("cases"), rocm.get("periodic", {}).get("cases")
        ),
        "wedge_cases": (cuda.get("wedge", {}).get("cases"), rocm.get("wedge", {}).get("cases")),
        "production_links": (
            cuda.get("production_links", {}).get("cases"),
            rocm.get("production_links", {}).get("cases"),
        ),
    }
    coverage_equal = all(cuda_value == rocm_value for cuda_value, rocm_value in compared.values())
    both_passed = cuda.get("passed") is True and rocm.get("passed") is True
    return {
        "component": "knn",
        "cuda_manifest": str(cuda_path),
        "rocm_manifest": str(rocm_path),
        "coverage": {
            name: {"cuda": values[0], "rocm": values[1]} for name, values in compared.items()
        },
        "coverage_equal": coverage_equal,
        "both_native_suites_passed": both_passed,
        "passed": coverage_equal and both_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare archived CUDA and ROCm analytical verification metrics",
    )
    script = Path(__file__).resolve()
    qav_root = script.parents[1]
    parser.add_argument(
        "--cuda-qav", type=Path, default=qav_root,
        help="qav root containing the CUDA archive",
    )
    parser.add_argument("--rocm-qav", type=Path, default=qav_root)
    parser.add_argument("--component", choices=("all", "fluid", "swarm", "knn"), default="all")
    parser.add_argument(
        "--cuda-sweep",
        choices=("thread", "block", "thread_precise", "block_precise"),
        default="thread",
    )
    parser.add_argument("--rocm-sweep", choices=("thread", "block"), default="thread")
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-11)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="compare every CUDA metric present while allowing extra ROCm records",
    )
    parser.add_argument(
        "--fluid-field-report", type=Path,
        help="passed raw polar-field report that supersedes derived polar error-norm comparison",
    )
    parser.add_argument("--output", type=Path, default=qav_root/"logs"/"backend_comparison.json")
    args = parser.parse_args()

    if not args.cuda_qav.is_dir():
        parser.error(f"CUDA qav directory does not exist: {args.cuda_qav}")
    if not args.rocm_qav.is_dir():
        parser.error(f"ROCm qav directory does not exist: {args.rocm_qav}")

    print(f"CUDA archive: {args.cuda_qav.resolve()}")
    print(f"ROCm archive: {args.rocm_qav.resolve()}")

    fluid_field_report = args.fluid_field_report
    if fluid_field_report is None:
        default_field_report = args.rocm_qav/"logs"/"backend_field_comparison_z.json"
        if default_field_report.is_file():
            fluid_field_report = default_field_report

    selected = ("fluid", "swarm", "knn") if args.component == "all" else (args.component,)
    components = []
    for component in selected:
        if component == "knn":
            result = compare_knn(args.cuda_qav, args.rocm_qav)
        else:
            result = compare_metrics(
                args.cuda_qav, args.rocm_qav, component,
                args.cuda_sweep, args.rocm_sweep,
                args.relative_tolerance, args.absolute_tolerance,
                args.allow_partial,
                fluid_field_report,
            )
        components.append(result)
        print(
            f"{component}: {'PASS' if result['passed'] else 'FAIL'}"
            f"  common={result.get('common_metrics', 'n/a')}"
            f"  mismatches={result.get('mismatch_count', 'n/a')}",
            flush=True,
        )

    report = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_qav": str(args.cuda_qav.resolve()),
        "rocm_qav": str(args.rocm_qav.resolve()),
        "cuda_fluid_sweep": args.cuda_sweep,
        "rocm_fluid_sweep": args.rocm_sweep,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "components": components,
        "passed": all(component["passed"] for component in components),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"backend comparison: {'PASS' if report['passed'] else 'FAIL'}  report={args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
