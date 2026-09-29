#!/usr/bin/env python3

"""verify that one native validation archive is complete and ready to transfer"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from val_config import (
    EXPECTED_FLUID_METRICS,
    EXPECTED_PUBLICATION_FLUID_METRICS,
    EXPECTED_PUBLICATION_SWARM_METRICS,
    EXPECTED_SWARM_METRICS,
    SWARM_CHAIN_MODELS,
    VAL_TIERS,
    fluid_archive_sweep,
    fluid_cases,
    model_output,
    swarm_models,
    val_output_path,
)


def load_json(path: Path) -> dict[str, Any]:
    """read one JSON object and reject malformed records"""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def variant_from_arguments(model: str, arguments: list[str]) -> str:
    """resolve the manifest suffix for one suite case entry"""

    values = {arguments[index]: arguments[index + 1] for index in range(0, len(arguments), 2)}
    if model in {"test_optdepth", "test_attenuation_2d"}:
        return f"p{float(values['--power']):+g}"
    if model == "test_x_transport_2d":
        return f"shift{float(values['--shift']):g}"
    if model == "test_diffusion_poslimit" and "--direction" in values:
        return values["--direction"]
    if model in {"test_y_transport_cyl", "test_y_transport_sph", "test_z_transport_3d"}:
        return f"cfl{float(values['--cfl']):g}"
    return "default"


def finite_numbers(value: Any) -> bool:
    """return whether every floating-point value in a JSON-compatible object is finite"""

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
    """require nonnegative integer tier counts that exactly partition an archive"""

    return (
        isinstance(value, dict)
        and all(
            tier in VAL_TIERS and isinstance(count, int) and count >= 0
            for tier, count in value.items()
        )
        and sum(value.values()) == metric_count
    )


def check_manifest(path: Path, component: str, backend: str) -> list[str]:
    """validate one full-suite manifest and return any problems"""

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
    completed = (
        record.get("cases_completed") if component == "fluid" else record.get("models_completed")
    )
    expected_count = (
        record.get("cases_expected") if component == "fluid" else record.get("models_expected")
    )
    canonical_count = len(fluid_cases("all")) if component == "fluid" else len(swarm_models("all"))
    if not isinstance(cases, list) or completed != expected_count or completed != len(cases):
        problems.append(f"{path}: incomplete case accounting")
    elif len(cases) != canonical_count:
        problems.append(f"{path}: found {len(cases)} cases; expected {canonical_count}")
    elif any(not isinstance(entry, dict) for entry in cases):
        problems.append(f"{path}: one or more case entries are not JSON objects")
    elif any(entry.get("status") != "passed" for entry in cases):
        problems.append(f"{path}: one or more cases do not report status=passed")
    elif any(entry.get("tier") not in VAL_TIERS for entry in cases):
        problems.append(f"{path}: one or more cases have an invalid evidence tier")
    if record.get("campaign_tier") not in VAL_TIERS:
        problems.append(f"{path}: campaign_tier is missing or invalid")
    included_tiers = record.get("included_tiers")
    if not isinstance(included_tiers, list) or any(
        tier not in VAL_TIERS for tier in included_tiers
    ):
        problems.append(f"{path}: included_tiers is missing or invalid")
    resolutions = record.get("effective_resolutions")
    if not isinstance(resolutions, list) or not resolutions:
        problems.append(f"{path}: effective_resolutions is missing or empty")
    return problems


def check_fluid(
    val_root: Path,
    backend: str,
    sweep: str,
    allow_partial: bool,
) -> dict[str, Any]:
    """check the publication fluid metrics and their manifests"""

    base = model_output(val_root, "fluid", "_suite", backend, sweep)
    metrics = sorted((val_root / "fluid" / "out").glob(f"test_*/{backend}/{sweep}/metrics_*.json"))
    problems = check_manifest(base / "manifest_all.json", "fluid", backend)
    invalid_metrics = []
    model_manifests = []

    suite_path = base / "manifest_all.json"
    if suite_path.is_file():
        suite = load_json(suite_path)
        for entry in suite.get("cases", []):
            model = str(entry.get("model", ""))
            arguments = entry.get("arguments", [])
            if not isinstance(arguments, list):
                problems.append(f"{suite_path}: invalid arguments for {model}")
                continue
            variant = variant_from_arguments(model, [str(value) for value in arguments])
            path = (
                model_output(val_root, "fluid", model, backend, sweep) / f"manifest_{variant}.json"
            )
            model_manifests.append(str(path.relative_to(val_root)))
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
            if not isinstance(environment, str) or not (path.parent / environment).is_file():
                problems.append(f"missing fluid model environment referenced by {path}")
            files = model_record.get("files")
            if not isinstance(files, list) or any(
                not isinstance(name, str) or not (path.parent / name).is_file() for name in files
            ):
                problems.append(f"missing fluid metric referenced by {path}")

    for path in metrics:
        try:
            record = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"invalid metric {path}: {error}")
            continue
        if not finite_numbers(record):
            invalid_metrics.append(str(path.relative_to(val_root)))
        if record.get("tier") not in VAL_TIERS:
            problems.append(f"fluid metric has missing or invalid tier: {path}")

    if (not allow_partial and len(metrics) != EXPECTED_FLUID_METRICS) or (
        allow_partial and not metrics
    ):
        problems.append(
            f"fluid metric count is {len(metrics)}; expected "
            f"{'a nonzero partial archive' if allow_partial else EXPECTED_FLUID_METRICS}"
        )
    metric_tiers = load_json(suite_path).get("metric_tiers", {}) if suite_path.is_file() else {}
    if not valid_metric_tiers(metric_tiers, len(metrics)):
        problems.append("fluid tier counts do not match the archived metric count")
    elif (
        not allow_partial and metric_tiers.get("publication") != EXPECTED_PUBLICATION_FLUID_METRICS
    ):
        problems.append(
            f"fluid publication metric count is {metric_tiers.get('publication')}; "
            f"expected {EXPECTED_PUBLICATION_FLUID_METRICS}"
        )
    if invalid_metrics:
        problems.append(f"{len(invalid_metrics)} fluid metrics contain non-finite values")
    return {
        "component": "fluid",
        "sweep": sweep,
        "metrics": len(metrics),
        "expected_metrics": EXPECTED_FLUID_METRICS,
        "invalid_metrics": invalid_metrics,
        "model_manifests": model_manifests,
        "problems": problems,
        "passed": not problems,
    }


def check_swarm(val_root: Path, backend: str, allow_partial: bool) -> dict[str, Any]:
    """check analytical swarm records and the complete KNN manifest"""

    base = model_output(val_root, "swarm", "_suite", backend)
    metrics = sorted((val_root / "swarm" / "out").glob(f"test_*/{backend}/metrics_N*.json"))
    problems = check_manifest(base / "manifest_all.json", "swarm", backend)
    environment_path = base / "environment_all.json"
    if not environment_path.is_file():
        problems.append(f"missing full-suite environment record: {environment_path}")
    failed_metrics = []
    suite_path = base / "manifest_all.json"
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
            component_path = model_output(val_root, "swarm", model, backend) / Path(
                output
            ).relative_to(model)
            if not component_path.is_file():
                problems.append(f"missing swarm component manifest: {component_path}")
                continue
            component = load_json(component_path)
            if component.get("passed") is not True:
                problems.append(f"swarm component manifest did not pass: {component_path}")
            if component.get("tier") != entry.get("tier"):
                problems.append(
                    f"swarm component tier disagrees with {suite_path}: {component_path}"
                )
            if model != "test_knn" and (
                component.get("backend") != backend or component.get("model") != model
            ):
                problems.append(f"invalid swarm component identity: {component_path}")
            if model != "test_knn" and component.get("resolution_tiers") != entry.get(
                "resolution_tiers"
            ):
                problems.append(
                    f"swarm component resolution tiers disagree with {suite_path}: {component_path}"
                )
            if model != "test_knn":
                environment = component.get("environment")
                if (
                    not isinstance(environment, str)
                    or not (component_path.parent / environment).is_file()
                ):
                    problems.append(
                        f"missing swarm model environment referenced by {component_path}"
                    )
                files = component.get("files")
                if not isinstance(files, list) or any(
                    not isinstance(name, str) or not (component_path.parent / name).is_file()
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
            failed_metrics.append(str(path.relative_to(val_root)))
        if record.get("tier") not in VAL_TIERS:
            problems.append(f"swarm metric has missing or invalid tier: {path}")

    if (not allow_partial and len(metrics) != EXPECTED_SWARM_METRICS) or (
        allow_partial and not metrics
    ):
        problems.append(
            f"swarm metric count is {len(metrics)}; expected "
            f"{'a nonzero partial archive' if allow_partial else EXPECTED_SWARM_METRICS}"
        )
    metric_tiers = load_json(suite_path).get("metric_tiers", {}) if suite_path.is_file() else {}
    if not valid_metric_tiers(metric_tiers, len(metrics)):
        problems.append("swarm tier counts do not match the archived metric count")
    elif (
        not allow_partial and metric_tiers.get("publication") != EXPECTED_PUBLICATION_SWARM_METRICS
    ):
        problems.append(
            f"swarm publication metric count is {metric_tiers.get('publication')}; "
            f"expected {EXPECTED_PUBLICATION_SWARM_METRICS}"
        )
    if failed_metrics:
        problems.append(f"{len(failed_metrics)} swarm metrics did not pass")

    knn_path = model_output(val_root, "swarm", "test_knn", backend) / "suite_manifest.json"
    knn_passed = False
    if knn_path.is_file():
        try:
            knn_record = load_json(knn_path)
            knn_passed = (
                knn_record.get("passed") is True and knn_record.get("tier") == "publication"
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"invalid KNN suite manifest {knn_path}: {error}")
    if not knn_passed:
        problems.append(f"KNN suite manifest is missing or did not pass: {knn_path}")

    chain_manifests = []
    for model in SWARM_CHAIN_MODELS:
        path = model_output(val_root, "swarm", model, backend, scope="chain") / "manifest.json"
        valid = False
        if path.is_file():
            try:
                record = load_json(path)
                variants = record.get("variants")
                valid = (
                    record.get("model") == model
                    and record.get("backend") == backend
                    and record.get("tier") == "publication"
                    and record.get("passed") is True
                    and isinstance(variants, dict)
                    and bool(variants)
                    and all(
                        isinstance(variant, dict) and variant.get("passed") is True
                        for variant in variants.values()
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
        chain_manifests.append({"model": model, "path": str(path), "valid": valid})
        if not valid:
            problems.append(f"collision-chain manifest is missing or did not pass: {path}")
    return {
        "component": "swarm",
        "metrics": len(metrics),
        "expected_metrics": EXPECTED_SWARM_METRICS,
        "failed_metrics": failed_metrics,
        "knn_manifest": str(knn_path),
        "knn_passed": knn_passed,
        "collision_chain_manifests": chain_manifests,
        "collision_chain_passed": all(record["valid"] for record in chain_manifests),
        "problems": problems,
        "passed": not problems,
    }


def main() -> None:
    val_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="check one native validation archive before transfer"
    )
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--component", choices=("all", "fluid", "swarm"), default="all")
    parser.add_argument(
        "--fluid-sweep",
        choices=("thread", "block", "thread_precise", "block_precise"),
        default="thread",
        help="fluid line solver or archive directory; CUDA solvers map to their _precise archives",
    )
    parser.add_argument("--val-root", type=Path, default=val_root)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept a passed quick matrix with fewer than the full metric count",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.fluid_sweep.endswith("_precise"):
        args.fluid_sweep = fluid_archive_sweep(args.backend, args.fluid_sweep)
    elif args.backend != "cuda":
        parser.error(f"--fluid-sweep {args.fluid_sweep} names a CUDA archive")

    selected = ("fluid", "swarm") if args.component == "all" else (args.component,)
    components = [
        check_fluid(args.val_root, args.backend, args.fluid_sweep, args.allow_partial)
        if component == "fluid"
        else check_swarm(args.val_root, args.backend, args.allow_partial)
        for component in selected
    ]
    report = {
        "schema": 1,
        "tier": "publication",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "partial_archive": args.allow_partial,
        "val_root": str(args.val_root.resolve()),
        "components": components,
        "passed": all(component["passed"] for component in components),
    }
    try:
        output = val_output_path(
            args.val_root,
            args.output,
            f"check_archive_{args.backend}_{args.fluid_sweep.removesuffix('_precise')}.json",
        )
    except ValueError as error:
        parser.error(str(error))
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
