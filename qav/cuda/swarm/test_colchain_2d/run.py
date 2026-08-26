#!/usr/bin/env python3

"""Compile and exercise the guarded CUDA collision chain through the production runtime"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


MODEL = Path(__file__).resolve().parent.name
EXPECTED_PARTICLES = 2048
PARTICLE_FIELDS = 8
SIZE_INDEX = 6
NUMBER_INDEX = 7


def utc_now() -> str:
    """Return the manifest timestamp in UTC"""

    return datetime.now(timezone.utc).isoformat()


def read_particle(path: Path) -> array:
    """Read the production swarm checkpoint as native double values"""

    values = array("d")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if len(values) % PARTICLE_FIELDS != 0:
        raise RuntimeError(f"invalid particle checkpoint length: {path}")
    return values


def represented_mass(values: array) -> float:
    """Sum the size-cubed mass proxy whose constant grain-density factor cancels"""

    return math.fsum(
        values[index + NUMBER_INDEX]*values[index + SIZE_INDEX]**3
        for index in range(0, len(values), PARTICLE_FIELDS)
    )


def digest(path: Path) -> str:
    """Hash one exact checkpoint for repeat-determinism evidence"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_once(executable: Path, project_root: Path, out_dir: Path) -> dict:
    """Run one fresh realization and return physical and byte-level diagnostics"""

    subprocess.run([str(executable)], cwd=project_root, check=True)
    initial_path = out_dir/"particle_00000.dat"
    final_path = out_dir/"particle_00001.dat"
    rng_path = out_dir/"rngstate_00001.dat"
    variable_path = out_dir/"variables.txt"
    for path in (initial_path, final_path, rng_path, variable_path):
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
    mass_initial = represented_mass(initial)
    mass_final = represented_mass(final)
    mass_relative = abs(mass_final - mass_initial) / mass_initial
    variables = variable_path.read_text()
    provenance = (
        "COLLISION_INTEGRATOR = chain" in variables
        and "COLLISION_SEARCH = morton" in variables
    )
    return {
        "particle_count": len(final) // PARTICLE_FIELDS,
        "finite": finite,
        "positive_species": positive,
        "changed_particles": changed,
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_relative_error": mass_relative,
        "provenance": provenance,
        "particle_sha256": digest(final_path),
        "rng_sha256": digest(rng_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=os.environ.get("CUDA_ARCH", "sm_80"))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    scope = os.environ.get("QAV_SCOPE", "qualification")
    scope_root = project_root/"qav"/"logs"/"swarm"/"cuda"
    if scope != "all":
        scope_root = scope_root/"groups"/scope
    out_dir = scope_root/MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file():
            path.unlink()

    make_base = [
        "make", "-C", str(project_root), f"MODEL={MODEL}", "GPU_BACKEND=cuda",
        f"GPU_TARGET={args.target}", "COLLISION_SEARCH=morton", f"QAV_SCOPE={scope}",
    ]
    subprocess.run([*make_base, "clean"], check=True)
    subprocess.run(make_base, check=True)
    if args.build_only:
        return

    executable = Path(__file__).resolve().parent/"gamedev"
    first = run_once(executable, project_root, out_dir)
    second = run_once(executable, project_root, out_dir)
    deterministic = (
        first["particle_sha256"] == second["particle_sha256"]
        and first["rng_sha256"] == second["rng_sha256"]
    )
    passed = (
        first["finite"]
        and first["positive_species"]
        and first["changed_particles"] > 0
        and first["mass_relative_error"] <= 2.0e-12
        and first["provenance"]
        and deterministic
    )
    manifest = {
        "schema": 1,
        "model": MODEL,
        "backend": "cuda",
        "gpu_target": args.target,
        "tier": "qualification",
        "integrator": "chain",
        "search": "morton",
        "runs": [first, second],
        "repeat_deterministic": deterministic,
        "mass_tolerance": 2.0e-12,
        "passed": passed,
        "finished_utc": utc_now(),
    }
    manifest_path = out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"{MODEL}: {'PASS' if passed else 'FAIL'} "
        f"changed={first['changed_particles']} "
        f"mass_relative={first['mass_relative_error']:.3e} "
        f"repeat_equal={deterministic} manifest={manifest_path}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
