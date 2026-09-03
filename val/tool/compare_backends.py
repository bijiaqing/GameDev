#!/usr/bin/env python3

"""Compare archived CUDA and ROCm verification metrics without comparing RNG streams"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True

from val_config import (
    EXPECTED_FLUID_METRICS,
    EXPECTED_SWARM_METRICS,
    SWARM_CHAIN_MODELS,
    val_output_path,
)

STOCHASTIC_SWARM_CASES = {
    "diffusion_1d",
    "diffusion_2d",
    "diffusion_3d",
}

# The production fluid initializer uses cuRAND or hipRAND for its azimuthal
# perturbation.  Its absolute mass therefore differs between vendors even
# though the normalized startup-balance diagnostic is backend-independent.
BACKEND_DEPENDENT_FLUID_FIELDS = {
    "startup_3d": {"initial_mass"},
}


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


def suite_manifest(root: Path, component: str, backend: str, sweep: str) -> Path:
    """Resolve the full common-matrix manifest for one native archive"""

    base = component_output(root, component, backend)
    return (base/sweep if component == "fluid" else base)/"manifest_all.json"


def compare_collision_chains(cuda_root: Path, rocm_root: Path) -> dict[str, Any]:
    """Require the same passing frozen-bath publication matrix on both backends"""

    records = []
    for model in SWARM_CHAIN_MODELS:
        paths = {
            "cuda": component_output(cuda_root, "swarm", "cuda")
                /"groups"/"chain"/model/"manifest.json",
            "rocm": component_output(rocm_root, "swarm", "rocm")
                /"groups"/"chain"/model/"manifest.json",
        }
        manifests = {}
        for backend, path in paths.items():
            manifests[backend] = load_json(path) if path.is_file() else None
        variants = {}
        valid = True
        for backend, manifest in manifests.items():
            backend_variants = manifest.get("variants") if isinstance(manifest, dict) else None
            variants[backend] = {
                name: {
                    "search": value.get("search"),
                    "event_cap": value.get("event_cap"),
                    "passed": value.get("passed"),
                }
                for name, value in backend_variants.items()
                if isinstance(name, str) and isinstance(value, dict)
            } if isinstance(backend_variants, dict) else None
            valid = valid and isinstance(manifest, dict) \
                and manifest.get("model") == model \
                and manifest.get("backend") == backend \
                and manifest.get("tier") == "publication" \
                and manifest.get("integrator") == "frozen_bath" \
                and manifest.get("passed") is True
        coverage_equal = variants.get("cuda") == variants.get("rocm")
        records.append({
            "model": model,
            "cuda_manifest": str(paths["cuda"]),
            "rocm_manifest": str(paths["rocm"]),
            "coverage_equal": coverage_equal,
            "passed": valid and coverage_equal,
        })
    return {
        "models": records,
        "expected_models": len(SWARM_CHAIN_MODELS),
        "passed": all(record["passed"] for record in records),
    }


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
) -> dict[str, Any]:
    """Compare the common analytical archive for one physical representation"""

    cuda_files = metric_files(cuda_root, component, "cuda", cuda_sweep)
    rocm_files = metric_files(rocm_root, component, "rocm", rocm_sweep)
    common = sorted(set(cuda_files) & set(rocm_files))
    missing_cuda = sorted(set(rocm_files) - set(cuda_files))
    missing_rocm = sorted(set(cuda_files) - set(rocm_files))
    expected = EXPECTED_FLUID_METRICS if component == "fluid" else EXPECTED_SWARM_METRICS
    mismatches: list[dict[str, Any]] = []
    stochastic_validations = 0
    native_manifests: dict[str, dict[str, Any]] = {}
    native_tiers: dict[str, Any] = {}
    collision_chains = compare_collision_chains(cuda_root, rocm_root) \
        if component == "swarm" else None

    for backend, root, sweep in (
        ("cuda", cuda_root, cuda_sweep), ("rocm", rocm_root, rocm_sweep),
    ):
        path = suite_manifest(root, component, backend, sweep)
        valid = False
        if path.is_file():
            record = load_json(path)
            valid = (
                record.get("suite") == component
                and record.get("backend") == backend
                and record.get("group") == "all"
                and record.get("passed") is True
                and record.get("campaign_tier") == "publication"
                and record.get("included_tiers") == ["publication"]
            )
            native_tiers[backend] = record.get("metric_tiers")
        native_manifests[backend] = {"path": str(path), "valid": valid}

    tier_metadata_equal = (
        len(native_tiers) == 2
        and native_tiers.get("cuda") == native_tiers.get("rocm")
    )

    for name in common:
        cuda = load_json(cuda_files[name])
        rocm = load_json(rocm_files[name])
        case = str(cuda.get("case", ""))
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

        if component == "fluid" and case in BACKEND_DEPENDENT_FLUID_FIELDS:
            excluded = BACKEND_DEPENDENT_FLUID_FIELDS[case]
            cuda = {key: value for key, value in cuda.items() if key not in excluded}
            rocm = {key: value for key, value in rocm.items() if key not in excluded}
        compare_value(
            cuda, rocm, name, relative_tolerance, absolute_tolerance, mismatches,
        )

    if allow_partial:
        # a partial run treats the CUDA archive as the requested metric subset
        # and requires every one of those records to have a ROCm counterpart
        # extra records are acceptable when only a selected publication case
        # was rerun on one backend
        complete = (
            bool(cuda_files)
            and len(common) == len(cuda_files)
            and not missing_rocm
            and all(record["valid"] for record in native_manifests.values())
            and tier_metadata_equal
        )
    else:
        complete = (
            len(cuda_files) == expected
            and len(rocm_files) == expected
            and len(common) == expected
            and not missing_cuda
            and not missing_rocm
            and all(record["valid"] for record in native_manifests.values())
            and tier_metadata_equal
        )
    if collision_chains is not None:
        complete = complete and collision_chains["passed"]
    return {
        "component": component,
        "expected_metrics": expected,
        "cuda_metrics": len(cuda_files),
        "rocm_metrics": len(rocm_files),
        "common_metrics": len(common),
        "stochastic_validation_only": stochastic_validations,
        "native_manifests": native_manifests,
        "native_metric_tiers": native_tiers,
        "collision_chains": collision_chains,
        "tier_metadata_equal": tier_metadata_equal,
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
        "tier": (cuda.get("tier"), rocm.get("tier")),
        "ordinary_cases": (
            cuda.get("ordinary", {}).get("cases"), rocm.get("ordinary", {}).get("cases")
        ),
        "edge_cases": (cuda.get("edge", {}).get("cases"), rocm.get("edge", {}).get("cases")),
        "periodic_cases": (
            cuda.get("periodic", {}).get("cases"), rocm.get("periodic", {}).get("cases")
        ),
        "wedge_cases": (cuda.get("wedge", {}).get("cases"), rocm.get("wedge", {}).get("cases")),
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


def compare_provenance(
    cuda_root: Path, rocm_root: Path, ignore_fingerprint: bool,
) -> dict[str, Any]:
    """Require the archived CUDA and ROCm campaigns to use one source snapshot"""

    paths = {
        "cuda": cuda_root/"logs"/"run_all_cuda.json",
        "rocm": rocm_root/"logs"/"run_all_rocm.json",
    }
    records = {
        backend: load_json(path) for backend, path in paths.items() if path.is_file()
    }
    cuda_hash = records.get("cuda", {}).get("source_sha256")
    rocm_hash = records.get("rocm", {}).get("source_sha256")
    available = bool(
        len(records) == 2 and isinstance(cuda_hash, str) and isinstance(rocm_hash, str)
    )
    matching = bool(available and cuda_hash and cuda_hash == rocm_hash)
    return {
        "available": available,
        "ignored": ignore_fingerprint,
        "cuda_campaign": str(paths["cuda"]),
        "rocm_campaign": str(paths["rocm"]),
        "cuda_source_sha256": cuda_hash,
        "rocm_source_sha256": rocm_hash,
        "matching": matching,
        "passed": ignore_fingerprint or matching,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare archived CUDA and ROCm analytical verification metrics",
    )
    script = Path(__file__).resolve()
    val_root = script.parents[1]
    parser.add_argument(
        "--cuda-root", dest="cuda_root", type=Path, default=val_root,
        help="validation root containing logs/fluid/cuda and logs/swarm/cuda",
    )
    parser.add_argument(
        "--rocm-root", dest="rocm_root", type=Path, default=val_root,
        help="validation root containing logs/fluid/rocm and logs/swarm/rocm",
    )
    parser.add_argument("--component", choices=("all", "fluid", "swarm", "knn"), default="all")
    parser.add_argument(
        "--cuda-sweep",
        choices=("thread", "block", "thread_precise", "block_precise"),
        default="thread",
    )
    parser.add_argument("--rocm-sweep", choices=("thread", "block"), default="thread")
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-11)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="compare every CUDA metric present while allowing extra ROCm records",
    )
    parser.add_argument(
        "--ignore-source-fingerprint", action="store_true",
        help="compare archives without matching complete campaign source fingerprints",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.cuda_root.is_dir():
        parser.error(f"CUDA validation root does not exist: {args.cuda_root}")
    if not args.rocm_root.is_dir():
        parser.error(f"ROCm validation root does not exist: {args.rocm_root}")

    print(f"CUDA validation root: {args.cuda_root.resolve()}")
    print(f"ROCm validation root: {args.rocm_root.resolve()}")

    selected = ("fluid", "swarm", "knn") if args.component == "all" else (args.component,)
    provenance = compare_provenance(
        args.cuda_root, args.rocm_root, args.ignore_source_fingerprint,
    )
    components = []
    for component in selected:
        if component == "knn":
            result = compare_knn(args.cuda_root, args.rocm_root)
        else:
            result = compare_metrics(
                args.cuda_root, args.rocm_root, component,
                args.cuda_sweep, args.rocm_sweep,
                args.relative_tolerance, args.absolute_tolerance,
                args.allow_partial,
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
        "tier": "publication",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_root": str(args.cuda_root.resolve()),
        "rocm_root": str(args.rocm_root.resolve()),
        "cuda_fluid_sweep": args.cuda_sweep,
        "rocm_fluid_sweep": args.rocm_sweep,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "provenance": provenance,
        "components": components,
        "passed": provenance["passed"] and all(component["passed"] for component in components),
    }
    try:
        output = val_output_path(val_root, args.output, "backend_comparison.json")
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if provenance["available"]:
        print(
            f"source fingerprint: {'PASS' if provenance['matching'] else 'FAIL'}"
            f"{' (ignored)' if provenance['ignored'] else ''}"
        )
    else:
        print(
            "source fingerprint: unavailable"
            f"{' (ignored)' if provenance['ignored'] else ' — FAIL'}"
        )
    print(f"backend comparison: {'PASS' if report['passed'] else 'FAIL'}  report={output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
