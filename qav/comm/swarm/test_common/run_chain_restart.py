#!/usr/bin/env python3

"""Validate production collision-chain restart behavior at an output boundary"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

from run_chain import (
    NUMBER_INDEX,
    PARTICLE_FIELDS,
    SIZE_INDEX,
    controller_equivalent,
    digest,
    read_particle,
    represented_mass,
    validate_controller,
)


EXPECTED_PARTICLES = 2048
VELOCITY_TOLERANCE = 2.0e-14
MASS_TOLERANCE = 2.0e-12


def utc_now() -> str:
    """Return the manifest timestamp in UTC"""

    return datetime.now(timezone.utc).isoformat()


def require_artifacts(paths: tuple[Path, ...]) -> None:
    """Reject an incomplete production checkpoint before comparing it"""

    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing collision-chain restart artifact: {path}")


def validate_geometry(path: Path, backend: str, expected: dict[str, int]) -> dict:
    """Validate cumulative hierarchy-reuse counters written by the production runtime"""

    data = json.loads(path.read_text())
    observed = {
        key: data.get(key)
        for key in (
            "search_calls",
            "geometry_builds",
            "geometry_reuses",
            "geometry_invalidations",
        )
    }
    passed = data.get("schema") == 1 and data.get("backend") == backend \
        and all(observed[key] == value for key, value in expected.items()) \
        and observed["search_calls"] == observed["geometry_builds"] + observed["geometry_reuses"]
    return {
        "path": str(path),
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def compare_particle_state(reference, resumed) -> dict:
    """Separate exact physical-state fields from the reversible velocity-file transform"""

    if len(reference) != len(resumed):
        raise RuntimeError("continuous and resumed particle checkpoints have different lengths")
    if len(reference) != EXPECTED_PARTICLES*PARTICLE_FIELDS:
        raise RuntimeError(
            f"particle checkpoint contains {len(reference) // PARTICLE_FIELDS} records; "
            f"expected {EXPECTED_PARTICLES}"
        )

    exact_indices = (0, 1, 2, SIZE_INDEX, NUMBER_INDEX)
    velocity_indices = (3, 4, 5)
    exact_equal = all(
        struct.pack("=d", reference[base + field]) == struct.pack("=d", resumed[base + field])
        for base in range(0, len(reference), PARTICLE_FIELDS)
        for field in exact_indices
    )
    velocity_error = [
        resumed[base + field] - reference[base + field]
        for base in range(0, len(reference), PARTICLE_FIELDS)
        for field in velocity_indices
    ]
    velocity_linf = max((abs(value) for value in velocity_error), default=0.0)
    velocity_l2 = math.sqrt(
        math.fsum(value*value for value in velocity_error)
        / max(1, len(velocity_error))
    )
    finite = all(math.isfinite(value) for value in reference) \
        and all(math.isfinite(value) for value in resumed)
    positive = all(
        resumed[base + SIZE_INDEX] > 0.0 and resumed[base + NUMBER_INDEX] > 0.0
        for base in range(0, len(resumed), PARTICLE_FIELDS)
    )
    return {
        "raw_byte_equal": reference.tobytes() == resumed.tobytes(),
        "position_species_byte_equal": exact_equal,
        "velocity_l2": velocity_l2,
        "velocity_linf": velocity_linf,
        "finite": finite,
        "positive_species": positive,
        "passed": finite and positive and exact_equal and velocity_linf <= VELOCITY_TOLERANCE,
    }


def run_variant(
    executable: Path,
    project_root: Path,
    variant_dir: Path,
    search: str,
    backend: str,
) -> dict:
    """Compare a continuous two-frame run with a new process resumed from frame one"""

    variant_dir.mkdir(parents=True, exist_ok=True)
    for path in variant_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    subprocess.run([str(executable)], cwd=project_root, check=True)

    initial_path = variant_dir/"particle_00000.dat"
    checkpoint_path = variant_dir/"particle_00001.dat"
    checkpoint_rng_path = variant_dir/"rngstate_00001.dat"
    final_path = variant_dir/"particle_00002.dat"
    final_rng_path = variant_dir/"rngstate_00002.dat"
    controller_path = variant_dir/"collision_chain_00002.json"
    geometry_one_path = variant_dir/"collision_geometry_00001.json"
    geometry_two_path = variant_dir/"collision_geometry_00002.json"
    variable_path = variant_dir/"variables.txt"
    require_artifacts((
        initial_path,
        checkpoint_path,
        checkpoint_rng_path,
        final_path,
        final_rng_path,
        controller_path,
        geometry_one_path,
        geometry_two_path,
        variable_path,
    ))

    initial = read_particle(initial_path)
    continuous = read_particle(final_path)
    continuous_controller = validate_controller(controller_path, False)
    continuous_geometry_one = validate_geometry(
        geometry_one_path,
        backend,
        {
            "search_calls": 1,
            "geometry_builds": 1,
            "geometry_reuses": 0,
            "geometry_invalidations": 0,
        },
    )
    continuous_geometry_two = validate_geometry(
        geometry_two_path,
        backend,
        {
            "search_calls": 2,
            "geometry_builds": 1,
            "geometry_reuses": 1,
            "geometry_invalidations": 0,
        },
    )
    continuous_particle_hash = digest(final_path)
    continuous_rng_hash = digest(final_rng_path)
    checkpoint_particle_hash = digest(checkpoint_path)
    checkpoint_rng_hash = digest(checkpoint_rng_path)

    archive_dir = variant_dir/"continuous"
    archive_dir.mkdir()
    for path in (final_path, final_rng_path, controller_path, geometry_two_path):
        shutil.copy2(path, archive_dir/path.name)

    # launch a new process so device allocations, search indices, and controller memory are rebuilt
    subprocess.run([str(executable), "1"], cwd=project_root, check=True)
    require_artifacts((
        checkpoint_path,
        checkpoint_rng_path,
        final_path,
        final_rng_path,
        controller_path,
        geometry_two_path,
    ))

    resumed = read_particle(final_path)
    resumed_controller = validate_controller(controller_path, False)
    resumed_geometry = validate_geometry(
        geometry_two_path,
        backend,
        {
            "search_calls": 1,
            "geometry_builds": 1,
            "geometry_reuses": 0,
            "geometry_invalidations": 0,
        },
    )
    state = compare_particle_state(continuous, resumed)
    controller_match = controller_equivalent(
        continuous_controller["comparison"], resumed_controller["comparison"]
    )
    controller_byte_equal = digest(archive_dir/controller_path.name) == digest(controller_path)
    rng_equal = continuous_rng_hash == digest(final_rng_path)
    checkpoint_unchanged = checkpoint_particle_hash == digest(checkpoint_path) \
        and checkpoint_rng_hash == digest(checkpoint_rng_path)

    changed = sum(
        continuous[base + SIZE_INDEX] != initial[base + SIZE_INDEX]
        for base in range(0, len(continuous), PARTICLE_FIELDS)
    )
    mass_initial = represented_mass(initial)
    mass_continuous = represented_mass(continuous)
    mass_resumed = represented_mass(resumed)
    mass_error_continuous = abs(mass_continuous - mass_initial) / mass_initial
    mass_error_resumed = abs(mass_resumed - mass_initial) / mass_initial
    variables = variable_path.read_text()
    provenance = "COLLISION_INTEGRATOR = chain" in variables \
        and f"COLLISION_SEARCH = {search}" in variables \
        and "COL_EVENT_CAP  = 32" in variables \
        and "COL_CONTROLLER_AUDIT = path_integrated" in variables

    passed = state["passed"] and rng_equal and checkpoint_unchanged \
        and controller_match and continuous_controller["passed"] \
        and resumed_controller["passed"] and changed > 0 and provenance \
        and continuous_geometry_one["passed"] and continuous_geometry_two["passed"] \
        and resumed_geometry["passed"] \
        and mass_error_continuous <= MASS_TOLERANCE \
        and mass_error_resumed <= MASS_TOLERANCE
    return {
        "search": search,
        "event_cap": 32,
        "changed_particles": changed,
        "state": state,
        "rng_byte_equal": rng_equal,
        "input_checkpoint_unchanged": checkpoint_unchanged,
        "controller_equivalent": controller_match,
        "controller_byte_equal": controller_byte_equal,
        "continuous_controller": continuous_controller,
        "resumed_controller": resumed_controller,
        "geometry": {
            "continuous_frame_one": continuous_geometry_one,
            "continuous_frame_two": continuous_geometry_two,
            "resumed_frame_two": resumed_geometry,
        },
        "mass_initial": mass_initial,
        "mass_continuous": mass_continuous,
        "mass_resumed": mass_resumed,
        "continuous_mass_relative_error": mass_error_continuous,
        "resumed_mass_relative_error": mass_error_resumed,
        "provenance": provenance,
        "continuous_particle_sha256": continuous_particle_hash,
        "resumed_particle_sha256": digest(final_path),
        "continuous_rng_sha256": continuous_rng_hash,
        "resumed_rng_sha256": digest(final_rng_path),
        "passed": passed,
    }


def run(model: str, backend: str) -> None:
    """Build both search variants and write one restart qualification manifest"""

    if backend not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported GPU backend: {backend}")
    default_target = os.environ.get("CUDA_ARCH", "sm_80") if backend == "cuda" \
        else os.environ.get("AMDGPU_TARGET", "gfx942")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=default_target)
    parser.add_argument("--search", choices=("both", "kdtree", "morton"), default="both")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    model_dir = project_root/"qav"/backend/"swarm"/model
    scope = os.environ.get("QAV_SCOPE", "qualification")
    scope_root = project_root/"qav"/"logs"/"swarm"/backend
    if scope != "all":
        scope_root = scope_root/"groups"/scope
    out_dir = scope_root/model
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    executable = model_dir/"gamedev"
    searches = ("morton", "kdtree") if args.search == "both" else (args.search,)
    variants = {}
    for search in searches:
        tag = f"{search}_cap32"
        variant_dir = out_dir/tag
        make_base = [
            "make", "-C", str(project_root), f"MODEL={model}", f"GPU_BACKEND={backend}",
            f"GPU_TARGET={args.target}", f"COLLISION_SEARCH={search}",
            "CHAIN_CAP=32", f"QAV_SCOPE={scope}", f"OUT_TAG={tag}",
        ]
        subprocess.run([*make_base, "clean"], check=True)
        subprocess.run(make_base, check=True)
        if args.build_only:
            continue

        result = run_variant(executable, project_root, variant_dir, search, backend)
        variants[tag] = result
        print(
            f"{model}/{tag}: {'PASS' if result['passed'] else 'FAIL'} "
            f"changed={result['changed_particles']} "
            f"state={result['state']['position_species_byte_equal']} "
            f"rng={result['rng_byte_equal']} controller={result['controller_equivalent']} "
            f"geometry={all(item['passed'] for item in result['geometry'].values())}",
            flush=True,
        )

    if args.build_only:
        return

    passed = all(result["passed"] for result in variants.values())
    manifest = {
        "schema": 1,
        "model": model,
        "backend": backend,
        "gpu_target": args.target,
        "tier": "qualification",
        "integrator": "chain",
        "restart_frame": 1,
        "final_frame": 2,
        "variants": variants,
        "velocity_tolerance": VELOCITY_TOLERANCE,
        "mass_tolerance": MASS_TOLERANCE,
        "passed": passed,
        "finished_utc": utc_now(),
    }
    manifest_path = out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"{model}: {'PASS' if passed else 'FAIL'} variants={len(variants)} "
        f"manifest={manifest_path}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)
