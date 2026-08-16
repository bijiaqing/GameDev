#!/usr/bin/env python3

"""Compare matched CUDA and ROCm fluid fields before analytical norm reduction"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CONSERVED_FIELDS = (
    "dustdens_initial",
    "dustdens_final",
    "dustmomx_final",
    "dustmomy_final",
    "dustmomz_final",
)
VELOCITY_FIELDS = (
    "dustvelx_final",
    "dustvely_final",
    "dustvelz_final",
)


def fluid_output(root: Path, backend: str, sweep: str) -> Path:
    """Resolve one backend's canonical fluid sweep archive"""

    return root/"logs"/"fluid"/backend/sweep


def load_json(path: Path) -> dict[str, object]:
    """Read one JSON object and reject malformed records"""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def variant_name(metric: dict[str, object]) -> str:
    """Reconstruct the parameter subdirectory selected by the fluid QA runner"""

    case = str(metric["case"])
    if case == "optdepth":
        return f"p{float(metric['power']):+g}"
    if case == "x_transport":
        return f"shift{float(metric['shift']):g}"
    if case in {"y_transport_cyl", "y_transport_sph", "z_transport"}:
        return f"cfl{float(metric['cfl']):g}"
    return ""


def read_field(directory: Path, name: str, resolution: int, count: int) -> np.ndarray:
    """Load one header-free float64 field and verify its element count"""

    path = directory/f"{name}_N{resolution}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"missing raw field: {path}")
    values = np.fromfile(path, dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{path}: found {values.size} values; expected {count}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite value found: {path}")
    return values


def difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    """Measure direct differences relative to the CUDA reference field"""

    difference = candidate - reference
    difference_l2 = float(np.linalg.norm(difference))
    reference_l2 = float(np.linalg.norm(reference))
    reference_linf = float(np.max(np.abs(reference))) if reference.size else 0.0
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    return {
        "count": int(reference.size),
        "max_abs": max_abs,
        "relative_l2": difference_l2/reference_l2 if reference_l2 > 0.0 else difference_l2,
        "relative_linf": max_abs/reference_linf if reference_linf > 0.0 else max_abs,
    }


def density_weighted_relative_l2(
    reference: np.ndarray,
    candidate: np.ndarray,
    density: np.ndarray,
) -> float:
    """Measure velocity disagreement in proportion to represented dust mass"""

    difference_sq = (candidate - reference)**2
    numerator = float(np.sum(density*difference_sq))
    denominator = float(np.sum(density*reference**2))
    return math.sqrt(numerator/denominator) if denominator > 0.0 else math.sqrt(numerator)


def compare_record(
    cuda_model: Path,
    rocm_model: Path,
    metric_path: Path,
    l2_tolerance: float,
    linf_tolerance: float,
    absolute_tolerance: float,
    velocity_weighted_tolerance: float,
) -> dict[str, object]:
    """Compare every stored field for one resolution and parameter variant"""

    cuda_metric = load_json(metric_path)
    rocm_metric_path = rocm_model/metric_path.name
    rocm_metric = load_json(rocm_metric_path)
    resolution = int(cuda_metric["resolution"])
    variant = variant_name(cuda_metric)
    cuda_data = cuda_model/variant if variant else cuda_model
    rocm_data = rocm_model/variant if variant else rocm_model
    cuda_meta = load_json(cuda_data/f"meta_N{resolution}.json")
    rocm_meta = load_json(rocm_data/f"meta_N{resolution}.json")

    metadata_keys = ("case", "nx", "ny", "nz", "resolution", "steps", "time", "cfl")
    metadata_equal = all(cuda_meta.get(key) == rocm_meta.get(key) for key in metadata_keys)
    nx, ny, nz = (int(cuda_meta[key]) for key in ("nx", "ny", "nz"))
    count = nx*ny*nz

    cuda_fields = {
        name: read_field(cuda_data, name, resolution, count)
        for name in CONSERVED_FIELDS + VELOCITY_FIELDS
    }
    rocm_fields = {
        name: read_field(rocm_data, name, resolution, count)
        for name in CONSERVED_FIELDS + VELOCITY_FIELDS
    }

    density_scale = max(
        float(np.max(np.abs(cuda_fields["dustdens_final"]))),
        float(np.max(np.abs(rocm_fields["dustdens_final"]))),
    )
    density_floor = 1.0e-12*density_scale
    velocity_mask = (
        (cuda_fields["dustdens_final"] > density_floor)
        & (rocm_fields["dustdens_final"] > density_floor)
    )
    velocity_density = 0.5*(
        cuda_fields["dustdens_final"][velocity_mask]
        + rocm_fields["dustdens_final"][velocity_mask]
    )

    fields: dict[str, dict[str, float | int | bool]] = {}
    for name in CONSERVED_FIELDS + VELOCITY_FIELDS:
        reference = cuda_fields[name]
        candidate = rocm_fields[name]
        if name in VELOCITY_FIELDS:
            reference = reference[velocity_mask]
            candidate = candidate[velocity_mask]
        metrics = difference_metrics(reference, candidate)
        metrics["byte_equal"] = reference.tobytes() == candidate.tobytes()
        if name == "dustdens_initial":
            metrics["passed"] = metrics["byte_equal"]
        elif name in VELOCITY_FIELDS:
            weighted_l2 = density_weighted_relative_l2(
                reference, candidate, velocity_density,
            )
            metrics["density_weighted_relative_l2"] = weighted_l2
            metrics["passed"] = weighted_l2 <= velocity_weighted_tolerance
        else:
            metrics["passed"] = (
                metrics["relative_l2"] <= l2_tolerance
                and (
                    metrics["relative_linf"] <= linf_tolerance
                    or metrics["max_abs"] <= absolute_tolerance
                )
            )
        fields[name] = metrics

    return {
        "metric": metric_path.name,
        "variant": variant,
        "resolution": resolution,
        "metadata_equal": metadata_equal,
        "density_floor": density_floor,
        "velocity_cells": int(np.count_nonzero(velocity_mask)),
        "fields": fields,
        "passed": metadata_equal and all(bool(value["passed"]) for value in fields.values()),
    }


def main() -> None:
    qav_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="compare raw CUDA and ROCm fluid fields for one QA model",
    )
    parser.add_argument("--cuda-qav", type=Path, required=True)
    parser.add_argument("--rocm-qav", type=Path, required=True)
    parser.add_argument("--cuda-sweep", default="thread_precise")
    parser.add_argument("--rocm-sweep", default="thread")
    parser.add_argument("--model", default="test_z_transport_3d")
    parser.add_argument("--l2-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--linf-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--velocity-weighted-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--expected-records", type=int,
        help="reject an incomplete model archive instead of comparing only the files present",
    )
    parser.add_argument(
        "--output", type=Path, default=qav_root/"logs"/"backend_field_comparison.json"
    )
    args = parser.parse_args()

    cuda_model = fluid_output(args.cuda_qav, "cuda", args.cuda_sweep)/args.model
    rocm_model = fluid_output(args.rocm_qav, "rocm", args.rocm_sweep)/args.model
    if not cuda_model.is_dir():
        parser.error(f"CUDA model archive does not exist: {cuda_model}")
    if not rocm_model.is_dir():
        parser.error(f"ROCm model archive does not exist: {rocm_model}")

    metric_paths = sorted(cuda_model.glob("metrics_N*.json"))
    if not metric_paths:
        parser.error(f"no CUDA metric records found: {cuda_model}")
    if args.expected_records is not None and len(metric_paths) != args.expected_records:
        parser.error(
            f"CUDA archive contains {len(metric_paths)} metric records; "
            f"expected {args.expected_records}"
        )

    rocm_metric_paths = sorted(rocm_model.glob("metrics_N*.json"))
    if args.expected_records is not None and len(rocm_metric_paths) != args.expected_records:
        parser.error(
            f"ROCm archive contains {len(rocm_metric_paths)} metric records; "
            f"expected {args.expected_records}"
        )

    records = [
        compare_record(
            cuda_model,
            rocm_model,
            path,
            args.l2_tolerance,
            args.linf_tolerance,
            args.absolute_tolerance,
            args.velocity_weighted_tolerance,
        )
        for path in metric_paths
    ]
    report = {
        "schema": 1,
        "cuda_qav": str(args.cuda_qav.resolve()),
        "rocm_qav": str(args.rocm_qav.resolve()),
        "cuda_sweep": args.cuda_sweep,
        "rocm_sweep": args.rocm_sweep,
        "model": args.model,
        "l2_tolerance": args.l2_tolerance,
        "linf_tolerance": args.linf_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "velocity_weighted_tolerance": args.velocity_weighted_tolerance,
        "expected_records": args.expected_records,
        "compared_records": len(records),
        "records": records,
        "passed": all(bool(record["passed"]) for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"{'record':34s} {'field':20s} {'relative L2':>14s} {'relative Linf':>14s} "
        f"{'weighted L2':>14s} {'max abs':>14s}"
    )
    for record in records:
        for name, metrics in record["fields"].items():
            weighted_l2 = metrics.get("density_weighted_relative_l2")
            weighted_text = f"{weighted_l2:14.6e}" if weighted_l2 is not None else f"{'-':>14s}"
            print(
                f"{record['metric']:34s} {name:20s} "
                f"{metrics['relative_l2']:14.6e} {metrics['relative_linf']:14.6e} "
                f"{weighted_text} {metrics['max_abs']:14.6e}"
            )
    print(f"direct field comparison: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
