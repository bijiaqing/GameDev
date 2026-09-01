#!/usr/bin/env python3

"""Exercise one ROCm collision-chain model through the production runtime"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

QAV_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QAV_ROOT/"tool"))

from qav_config import model_executable


EXPECTED_PARTICLES = 2048
PARTICLE_FIELDS = 8
SIZE_INDEX = 6
NUMBER_INDEX = 7


def utc_now() -> str:
    """Return the manifest timestamp in UTC"""

    return datetime.now(timezone.utc).isoformat()


def read_particle(path: Path) -> array:
    """Read one production swarm checkpoint as native double values"""

    values = array("d")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if len(values) % PARTICLE_FIELDS != 0:
        raise RuntimeError(f"invalid particle checkpoint length: {path}")
    return values


def represented_mass(values: array) -> float:
    """Sum the size-cubed mass proxy whose constant material factor cancels"""

    return math.fsum(
        values[index + NUMBER_INDEX]*values[index + SIZE_INDEX]**3
        for index in range(0, len(values), PARTICLE_FIELDS)
    )


def digest(path: Path) -> str:
    """Hash one exact artifact for repeat-determinism evidence"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def controller_equivalent(first: dict, second: dict) -> bool:
    """Compare controller schedules while allowing atomic-sum roundoff"""

    if first.keys() != second.keys():
        return False
    for key in first:
        value_first = first[key]
        value_second = second[key]
        if isinstance(value_first, list):
            if not isinstance(value_second, list) or len(value_first) != len(value_second):
                return False
            if not all(
                controller_equivalent(item_first, item_second)
                for item_first, item_second in zip(value_first, value_second)
            ):
                return False
        elif isinstance(value_first, dict):
            if not isinstance(value_second, dict) \
                or not controller_equivalent(value_first, value_second):
                return False
        elif isinstance(value_first, (int, float)) and not isinstance(value_first, bool):
            if not isinstance(value_second, (int, float)) \
                or isinstance(value_second, bool) \
                or not math.isclose(
                    float(value_first), float(value_second), rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                return False
        elif value_first != value_second:
            return False
    return True


def validate_controller(path: Path, require_continuation: bool) -> dict:
    """Validate the archived adaptive schedule independently of terminal output"""

    data = json.loads(path.read_text())
    baths = data.get("baths", [])
    numeric = (
        "minimum_duration", "maximum_duration", "minimum_limit_scale", "maximum_f",
        "maximum_e", "maximum_touched", "maximum_events", "maximum_g",
        "maximum_g_upper", "maximum_d_bath",
    )
    finite = all(math.isfinite(float(data.get(key, math.nan))) for key in numeric)
    complete = data.get("schema") == 1 \
        and data.get("bath_count") == len(baths) \
        and data.get("bath_count", 0) > 0 \
        and data.get("operator_count", 0) > 0 \
        and data.get("persistent_overshoots") == 0
    continuation = data.get("continuation_launches", 0) >= data.get("bath_count", 0)
    if require_continuation:
        continuation = data.get("continuation_launches", 0) > data.get("bath_count", 0)
    bath_numeric = (
        "duration", "limit_before", "limit_after", "max_f", "max_e", "max_touched",
        "max_events", "max_g", "max_g_upper", "d_bath",
    )
    records_valid = all(
        record.get("operator", 0) > 0
        and record.get("bath", 0) > 0
        and record.get("merged_bins", 0) > 0
        and record.get("continuation_launches", 0) > 0
        and record.get("duration", 0.0) > 0.0
        and 0.25 <= record.get("limit_before", 0.0) <= 1.0
        and 0.25 <= record.get("limit_after", 0.0) <= 1.0
        and all(math.isfinite(float(record.get(key, math.nan))) for key in bath_numeric)
        and all(record.get(key, -1.0) >= 0.0 for key in bath_numeric if key != "limit_before")
        and not record.get("persistent_overshoot", True)
        for record in baths
    )
    return {
        "finite": finite,
        "complete": complete,
        "continuation_exercised": continuation,
        "records_valid": records_valid,
        "passed": finite and complete and continuation and records_valid,
        "sha256": digest(path),
        "bath_count": data.get("bath_count"),
        "continuation_launches": data.get("continuation_launches"),
        "activity_overshoots": data.get("activity_overshoots"),
        "distribution_overshoots": data.get("distribution_overshoots"),
        "comparison": data,
    }


def run_once(
    executable: Path,
    project_root: Path,
    out_dir: Path,
    search: str,
    event_cap: int,
    fragmentation: bool,
) -> dict:
    """Run one fresh realization and return physical and controller diagnostics"""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    subprocess.run([str(executable)], cwd=project_root, check=True)
    initial_path = out_dir/"particle_00000.dat"
    final_path = out_dir/"particle_00001.dat"
    rng_path = out_dir/"rngstate_00001.dat"
    variable_path = out_dir/"variables.txt"
    controller_path = out_dir/"collision_chain_00001.json"
    for path in (initial_path, final_path, rng_path, variable_path, controller_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing collision-chain artifact: {path}")

    initial = read_particle(initial_path)
    final = read_particle(final_path)
    if len(initial) != len(final):
        raise RuntimeError("initial and final particle checkpoints have different lengths")
    if len(final) != EXPECTED_PARTICLES*PARTICLE_FIELDS:
        raise RuntimeError(
            f"particle checkpoint contains {len(final) // PARTICLE_FIELDS} records; "
            f"expected {EXPECTED_PARTICLES}"
        )
    finite = all(math.isfinite(value) for value in final)
    positive = all(
        final[index + SIZE_INDEX] > 0.0 and final[index + NUMBER_INDEX] > 0.0
        for index in range(0, len(final), PARTICLE_FIELDS)
    )
    changed = sum(
        final[index + SIZE_INDEX] != initial[index + SIZE_INDEX]
        for index in range(0, len(final), PARTICLE_FIELDS)
    )
    shrunk = sum(
        final[index + SIZE_INDEX] < initial[index + SIZE_INDEX]
        for index in range(0, len(final), PARTICLE_FIELDS)
    )
    mass_initial = represented_mass(initial)
    mass_final = represented_mass(final)
    mass_relative = abs(mass_final - mass_initial) / mass_initial
    variables = variable_path.read_text()
    provenance = "COLLISION_INTEGRATOR = frozen_bath" in variables \
        and f"COLLISION_SEARCH = {search}" in variables \
        and f"COL_EVENT_CAP  = {event_cap}" in variables \
        and "COL_CONTROLLER_AUDIT = path_integrated" in variables
    controller = validate_controller(controller_path, event_cap == 1)
    return {
        "particle_count": len(final) // PARTICLE_FIELDS,
        "finite": finite,
        "positive_species": positive,
        "changed_particles": changed,
        "shrunk_particles": shrunk,
        "fragmentation_exercised": (shrunk > 0) if fragmentation else None,
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_relative_error": mass_relative,
        "provenance": provenance,
        "controller": controller,
        "initial_sha256": digest(initial_path),
        "particle_sha256": digest(final_path),
        "rng_sha256": digest(rng_path),
    }


def run(model: str) -> None:
    """Build every required search/cap variant and write one component manifest"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument("--search", choices=("both", "kdtree", "morton"), default="both")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = project_root/"qav"/"logs"/"swarm"/"rocm"
    if scope != "all":
        scope_root = scope_root/"groups"/scope
    out_dir = scope_root/model
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    searches = ("morton", "kdtree") if args.search == "both" else (args.search,)
    event_caps = (1, 32) if model == "test_colchain_2d" else (32,)
    fragmentation = model == "test_colchain_frag_2d"
    executable = model_executable(project_root, model, "rocm", "swarm")
    variants = {}
    for search in searches:
        for event_cap in event_caps:
            tag = f"{search}_cap{event_cap}"
            variant_dir = out_dir/tag
            make_base = [
                "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=rocm",
                f"GPU_TARGET={args.target}", f"COLLISION_SEARCH={search}",
                f"CHAIN_CAP={event_cap}", f"QAV_SCOPE={scope}", f"OUT_TAG={tag}",
            ]
            subprocess.run([*make_base, "clean"], check=True)
            subprocess.run(make_base, check=True)
            if args.build_only:
                continue

            first = run_once(
                executable, project_root, variant_dir, search, event_cap, fragmentation
            )
            second = run_once(
                executable, project_root, variant_dir, search, event_cap, fragmentation
            )
            path_deterministic = first["particle_sha256"] == second["particle_sha256"] \
                and first["rng_sha256"] == second["rng_sha256"]
            controller_consistent = controller_equivalent(
                first["controller"]["comparison"], second["controller"]["comparison"]
            )
            passed = first["finite"] and first["positive_species"] \
                and first["changed_particles"] > 0 \
                and first["mass_relative_error"] <= 2.0e-12 \
                and first["provenance"] and first["controller"]["passed"] \
                and (not fragmentation or first["fragmentation_exercised"]) \
                and path_deterministic and controller_consistent
            variants[tag] = {
                "search": search,
                "event_cap": event_cap,
                "runs": [first, second],
                "path_repeat_deterministic": path_deterministic,
                "controller_repeat_consistent": controller_consistent,
                "passed": passed,
            }
            print(
                f"{model}/{tag}: {'PASS' if passed else 'FAIL'} "
                f"changed={first['changed_particles']} shrunk={first['shrunk_particles']} "
                f"baths={first['controller']['bath_count']} "
                f"launches={first['controller']['continuation_launches']}",
                flush=True,
            )

    if args.build_only:
        return

    initial_equal = len({
        result["runs"][0]["initial_sha256"] for result in variants.values()
    }) == 1
    cap_pathwise_equal = True
    if len(event_caps) > 1:
        for search in searches:
            small = variants[f"{search}_cap1"]["runs"][0]
            large = variants[f"{search}_cap32"]["runs"][0]
            cap_pathwise_equal = cap_pathwise_equal \
                and small["particle_sha256"] == large["particle_sha256"] \
                and small["rng_sha256"] == large["rng_sha256"]
    passed = all(result["passed"] for result in variants.values()) \
        and initial_equal and cap_pathwise_equal
    manifest = {
        "schema": 3,
        "model": model,
        "backend": "rocm",
        "gpu_target": args.target,
        "tier": "publication",
        "integrator": "frozen_bath",
        "variants": variants,
        "initial_byte_equal": initial_equal,
        "cap_pathwise_equal": cap_pathwise_equal,
        "mass_tolerance": 2.0e-12,
        "passed": passed,
        "finished_utc": utc_now(),
    }
    manifest_path = out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"{model}: {'PASS' if passed else 'FAIL'} variants={len(variants)} "
        f"initial_equal={initial_equal} cap_equal={cap_pathwise_equal} "
        f"manifest={manifest_path}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)
