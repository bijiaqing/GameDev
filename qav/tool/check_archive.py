#!/usr/bin/env python3

"""Verify that one native QAV archive is complete and ready to transfer"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from qav_config import (
    EXPECTED_FLUID_METRICS,
    EXPECTED_PUBLICATION_FLUID_METRICS,
    EXPECTED_PUBLICATION_SWARM_METRICS,
    EXPECTED_SWARM_METRICS,
    QAV_TIERS,
    fluid_cases,
    swarm_models,
)


FLUID_RAW_FIELDS = (
    "dustdens_initial", "dustdens_final", "dustmomx_final", "dustmomy_final",
    "dustmomz_final", "dustvelx_final", "dustvely_final", "dustvelz_final",
)


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject malformed records"""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def variant_name(metric: dict[str, Any]) -> str:
    """Reconstruct the parameter directory associated with one fluid metric"""

    case = str(metric["case"])
    if case in {"optdepth", "attenuation_2d"}:
        return f"p{float(metric['power']):+g}"
    if case == "x_transport":
        return f"shift{float(metric['shift']):g}"
    if case in {"y_transport_cyl", "y_transport_sph", "z_transport"}:
        return f"cfl{float(metric['cfl']):g}"
    return ""


def variant_from_arguments(model: str, arguments: list[str]) -> str:
    """Resolve the manifest suffix for one suite case entry"""

    values = {arguments[index]: arguments[index + 1] for index in range(0, len(arguments), 2)}
    if model in {"test_optdepth", "test_attenuation_2d"}:
        return f"p{float(values['--power']):+g}"
    if model == "test_x_transport_2d":
        return f"shift{float(values['--shift']):g}"
    if model in {"test_y_transport_cyl", "test_y_transport_sph", "test_z_transport_3d"}:
        return f"cfl{float(values['--cfl']):g}"
    return "default"


def finite_numbers(value: Any) -> bool:
    """Return whether every floating-point value in a JSON-compatible object is finite"""

    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return False


def valid_metric_tiers(value: Any, metric_count: int) -> bool:
    """Require nonnegative integer tier counts that exactly partition an archive"""

    return (
        isinstance(value, dict)
        and all(
            tier in QAV_TIERS and isinstance(count, int) and count >= 0
            for tier, count in value.items()
        )
        and sum(value.values()) == metric_count
    )


def check_manifest(path: Path, component: str, backend: str) -> list[str]:
    """Validate one full-suite manifest and return any problems"""

    if not path.is_file():
        return [f"missing full-suite manifest: {path}"]
    try:
        record = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid full-suite manifest {path}: {error}"]

    problems = []
    expected = {"suite": component, "backend": backend, "group": "all", "passed": True}
    for key, value in expected.items():
        if record.get(key) != value:
            problems.append(f"{path}: expected {key}={value!r}; found {record.get(key)!r}")
    cases = record.get("cases") if component == "fluid" else record.get("models")
    completed = record.get("cases_completed") if component == "fluid" else record.get("models_completed")
    expected_count = record.get("cases_expected") if component == "fluid" else record.get("models_expected")
    canonical_count = len(fluid_cases("all")) if component == "fluid" \
        else len(swarm_models("all"))
    if not isinstance(cases, list) or completed != expected_count or completed != len(cases):
        problems.append(f"{path}: incomplete case accounting")
    elif len(cases) != canonical_count:
        problems.append(f"{path}: found {len(cases)} cases; expected {canonical_count}")
    elif any(not isinstance(entry, dict) for entry in cases):
        problems.append(f"{path}: one or more case entries are not JSON objects")
    elif any(entry.get("status") != "passed" for entry in cases):
        problems.append(f"{path}: one or more cases do not report status=passed")
    elif any(entry.get("tier") not in QAV_TIERS for entry in cases):
        problems.append(f"{path}: one or more cases have an invalid evidence tier")
    if record.get("campaign_tier") not in QAV_TIERS:
        problems.append(f"{path}: campaign_tier is missing or invalid")
    included_tiers = record.get("included_tiers")
    if not isinstance(included_tiers, list) or any(tier not in QAV_TIERS for tier in included_tiers):
        problems.append(f"{path}: included_tiers is missing or invalid")
    resolutions = record.get("effective_resolutions")
    if not isinstance(resolutions, list) or not resolutions:
        problems.append(f"{path}: effective_resolutions is missing or empty")
    return problems


def check_raw_fields(
    data_dir: Path, resolution: int, fields: tuple[str, ...], problems: list[str], label: str,
) -> list[str]:
    """Require every binary field to exist with the element count stated by its metadata"""

    missing = []
    meta_path = data_dir/f"meta_N{resolution}.json"
    if not meta_path.is_file():
        problems.append(f"missing {label} metadata: {meta_path}")
        return [str(data_dir/f"{field}_N{resolution}.dat") for field in fields]
    try:
        meta = load_json(meta_path)
        count = int(meta["nx"])*int(meta["ny"])*int(meta["nz"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        problems.append(f"invalid {label} metadata {meta_path}: {error}")
        return [str(data_dir/f"{field}_N{resolution}.dat") for field in fields]

    expected_bytes = 8*count
    for field in fields:
        path = data_dir/f"{field}_N{resolution}.dat"
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size != expected_bytes:
            problems.append(
                f"{label} field has {path.stat().st_size} bytes; "
                f"expected {expected_bytes}: {path}"
            )
    return missing


def check_fluid(
    qav_root: Path, backend: str, sweep: str, require_cross_fields: bool, allow_partial: bool,
) -> dict[str, Any]:
    """Check analytical fluid metrics plus the raw polar fields used cross-backend"""

    base = qav_root/"logs"/"fluid"/backend/sweep
    metrics = sorted(base.glob("test_*/metrics_*.json")) if base.is_dir() else []
    problems = check_manifest(base/"manifest_all.json", "fluid", backend)
    invalid_metrics = []
    raw_missing = []
    model_manifests = []

    suite_path = base/"manifest_all.json"
    if suite_path.is_file():
        suite = load_json(suite_path)
        for entry in suite.get("cases", []):
            model = str(entry.get("model", ""))
            arguments = entry.get("arguments", [])
            if not isinstance(arguments, list):
                problems.append(f"{suite_path}: invalid arguments for {model}")
                continue
            variant = variant_from_arguments(model, [str(value) for value in arguments])
            path = base/model/f"manifest_{variant}.json"
            model_manifests.append(str(path.relative_to(base)))
            if not path.is_file():
                problems.append(f"missing or failed fluid model manifest: {path}")
                continue
            model_record = load_json(path)
            if (
                model_record.get("passed") is not True
                or model_record.get("backend") != backend
                or model_record.get("model") != model
                or model_record.get("tier") != entry.get("tier")
            ):
                problems.append(f"invalid or failed fluid model manifest: {path}")
            environment = model_record.get("environment")
            if not isinstance(environment, str) or not (path.parent/environment).is_file():
                problems.append(f"missing fluid model environment referenced by {path}")
            files = model_record.get("files")
            if not isinstance(files, list) or any(
                not isinstance(name, str) or not (path.parent/name).is_file()
                for name in files
            ):
                problems.append(f"missing fluid metric referenced by {path}")

    for path in metrics:
        try:
            record = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"invalid metric {path}: {error}")
            continue
        if not finite_numbers(record):
            invalid_metrics.append(str(path.relative_to(base)))
        if record.get("tier") not in QAV_TIERS:
            problems.append(f"fluid metric has missing or invalid tier: {path}")
        if path.parent.name == "test_z_transport_3d":
            resolution = int(record["resolution"])
            variant = variant_name(record)
            data_dir = path.parent/variant if variant else path.parent
            missing = check_raw_fields(
                data_dir, resolution, FLUID_RAW_FIELDS, problems, "polar",
            )
            raw_missing.extend(
                str(Path(path).relative_to(base)) for path in missing
            )

    if (not allow_partial and len(metrics) != EXPECTED_FLUID_METRICS) or (allow_partial and not metrics):
        problems.append(
            f"fluid metric count is {len(metrics)}; expected "
            f"{'a nonzero partial archive' if allow_partial else EXPECTED_FLUID_METRICS}"
        )
    metric_tiers = load_json(suite_path).get("metric_tiers", {}) if suite_path.is_file() else {}
    if not valid_metric_tiers(metric_tiers, len(metrics)):
        problems.append("fluid tier counts do not match the archived metric count")
    elif not allow_partial and metric_tiers.get("publication") != EXPECTED_PUBLICATION_FLUID_METRICS:
        problems.append(
            f"fluid publication metric count is {metric_tiers.get('publication')}; "
            f"expected {EXPECTED_PUBLICATION_FLUID_METRICS}"
        )
    if invalid_metrics:
        problems.append(f"{len(invalid_metrics)} fluid metrics contain non-finite values")
    if raw_missing:
        problems.append(f"{len(raw_missing)} cross-backend raw fields are missing")

    precise_polar_metrics = 0
    if backend == "cuda" and require_cross_fields:
        manifest_path = base/"manifest_all.json"
        resolutions = load_json(manifest_path).get("effective_resolutions", []) \
            if manifest_path.is_file() else []
        base_sweep = sweep.removesuffix("_precise")
        precise_sweep = f"{base_sweep}_precise"
        precise_model = qav_root/"logs"/"fluid"/"cuda"/precise_sweep/"test_z_transport_3d"
        precise_paths = sorted(precise_model.glob("metrics_N*.json"))
        precise_polar_metrics = len(precise_paths)
        expected_precise = 2*len(resolutions)
        if precise_polar_metrics != expected_precise:
            problems.append(
                f"precise polar metric count is {precise_polar_metrics}; expected {expected_precise}"
            )
        for path in precise_paths:
            record = load_json(path)
            resolution = int(record["resolution"])
            variant = variant_name(record)
            data_dir = precise_model/variant
            missing = check_raw_fields(
                data_dir, resolution, FLUID_RAW_FIELDS, problems, "precise polar",
            )
            problems.extend(f"missing precise polar field: {path}" for path in missing)
    return {
        "component": "fluid",
        "sweep": sweep,
        "metrics": len(metrics),
        "expected_metrics": EXPECTED_FLUID_METRICS,
        "invalid_metrics": invalid_metrics,
        "missing_raw_fields": raw_missing,
        "model_manifests": model_manifests,
        "precise_polar_metrics": precise_polar_metrics,
        "problems": problems,
        "passed": not problems,
    }


def check_swarm(qav_root: Path, backend: str, allow_partial: bool) -> dict[str, Any]:
    """Check analytical swarm records and the complete KNN manifest"""

    base = qav_root/"logs"/"swarm"/backend
    metrics = sorted(base.glob("test_*/metrics_N*.json")) if base.is_dir() else []
    problems = check_manifest(base/"manifest_all.json", "swarm", backend)
    environment_path = base/"environment_all.json"
    if not environment_path.is_file():
        problems.append(f"missing full-suite environment record: {environment_path}")
    failed_metrics = []
    suite_path = base/"manifest_all.json"
    if suite_path.is_file():
        suite = load_json(suite_path)
        for entry in suite.get("models", []):
            if not isinstance(entry, dict):
                continue
            output = entry.get("output")
            model = entry.get("model")
            if not isinstance(output, str):
                problems.append(f"{suite_path}: missing component output for {model}")
                continue
            component_path = base/output
            if not component_path.is_file():
                problems.append(f"missing swarm component manifest: {component_path}")
                continue
            component = load_json(component_path)
            if component.get("passed") is not True:
                problems.append(f"swarm component manifest did not pass: {component_path}")
            if component.get("tier") != entry.get("tier"):
                problems.append(f"swarm component tier disagrees with {suite_path}: {component_path}")
            if model != "test_knn" and (
                component.get("backend") != backend or component.get("model") != model
            ):
                problems.append(f"invalid swarm component identity: {component_path}")
            if model != "test_knn" and component.get("resolution_tiers") \
                    != entry.get("resolution_tiers"):
                problems.append(
                    f"swarm component resolution tiers disagree with {suite_path}: "
                    f"{component_path}"
                )
            if model != "test_knn":
                environment = component.get("environment")
                if not isinstance(environment, str) \
                        or not (component_path.parent/environment).is_file():
                    problems.append(
                        f"missing swarm model environment referenced by {component_path}"
                    )
                files = component.get("files")
                if not isinstance(files, list) or any(
                    not isinstance(name, str) or not (component_path.parent/name).is_file()
                    for name in files
                ):
                    problems.append(f"missing swarm metric referenced by {component_path}")
    for path in metrics:
        try:
            record = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"invalid metric {path}: {error}")
            continue
        if record.get("passed") is not True:
            failed_metrics.append(str(path.relative_to(base)))
        if record.get("tier") not in QAV_TIERS:
            problems.append(f"swarm metric has missing or invalid tier: {path}")

    if (not allow_partial and len(metrics) != EXPECTED_SWARM_METRICS) or (allow_partial and not metrics):
        problems.append(
            f"swarm metric count is {len(metrics)}; expected "
            f"{'a nonzero partial archive' if allow_partial else EXPECTED_SWARM_METRICS}"
        )
    metric_tiers = load_json(suite_path).get("metric_tiers", {}) if suite_path.is_file() else {}
    if not valid_metric_tiers(metric_tiers, len(metrics)):
        problems.append("swarm tier counts do not match the archived metric count")
    elif not allow_partial and metric_tiers.get("publication") != EXPECTED_PUBLICATION_SWARM_METRICS:
        problems.append(
            f"swarm publication metric count is {metric_tiers.get('publication')}; "
            f"expected {EXPECTED_PUBLICATION_SWARM_METRICS}"
        )
    if failed_metrics:
        problems.append(f"{len(failed_metrics)} swarm metrics did not pass")

    knn_path = base/"test_knn"/"suite_manifest.json"
    knn_passed = False
    if knn_path.is_file():
        try:
            knn_record = load_json(knn_path)
            knn_passed = (
                knn_record.get("passed") is True
                and knn_record.get("tier") == "publication"
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"invalid KNN suite manifest {knn_path}: {error}")
    if not knn_passed:
        problems.append(f"KNN suite manifest is missing or did not pass: {knn_path}")
    return {
        "component": "swarm",
        "metrics": len(metrics),
        "expected_metrics": EXPECTED_SWARM_METRICS,
        "failed_metrics": failed_metrics,
        "knn_manifest": str(knn_path),
        "knn_passed": knn_passed,
        "problems": problems,
        "passed": not problems,
    }


def main() -> None:
    qav_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="check one native QAV archive before transfer")
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--component", choices=("all", "fluid", "swarm"), default="all")
    parser.add_argument(
        "--fluid-sweep",
        choices=("thread", "block", "thread_precise", "block_precise"),
        default="thread",
    )
    parser.add_argument("--qav-root", type=Path, default=qav_root)
    parser.add_argument(
        "--skip-cross-fields", action="store_true",
        help="do not require CUDA's additional precise polar archive",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="accept a passed quick matrix with fewer than the full metric count",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = ("fluid", "swarm") if args.component == "all" else (args.component,)
    components = [
        check_fluid(
            args.qav_root, args.backend, args.fluid_sweep, not args.skip_cross_fields,
            args.allow_partial,
        )
        if component == "fluid" else check_swarm(args.qav_root, args.backend, args.allow_partial)
        for component in selected
    ]
    report = {
        "schema": 1,
        "tier": "qualification",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "partial_archive": args.allow_partial,
        "qav_root": str(args.qav_root.resolve()),
        "components": components,
        "passed": all(component["passed"] for component in components),
    }
    output = args.output or args.qav_root/"logs"/f"archive_check_{args.backend}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for component in components:
        print(
            f"{component['component']}: {'PASS' if component['passed'] else 'FAIL'}  "
            f"metrics={component['metrics']}/{component['expected_metrics']}"
        )
        for problem in component["problems"]:
            print(f"  - {problem}")
    print(f"archive check: {'PASS' if report['passed'] else 'FAIL'}  report={output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
