#!/usr/bin/env python3

"""Compare collision-search reuse with a forced-rebuild production baseline"""

from __future__ import annotations

from array import array
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


PARTICLE_FIELDS = 8
EXPECTED_PARTICLES = 2048


def utc_now() -> str:
    """Return the manifest timestamp in UTC"""

    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    """Hash one binary artifact for exact baseline comparison"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_particle(path: Path) -> array:
    """Read one native-double particle checkpoint"""

    values = array("d")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if len(values) != EXPECTED_PARTICLES*PARTICLE_FIELDS:
        raise RuntimeError(f"unexpected particle checkpoint length: {path}")
    return values


def validate_geometry(path: Path, backend: str, fresh: bool) -> dict:
    """Check hierarchy accounting for the reused or forced-rebuild execution"""

    data = json.loads(path.read_text())
    calls = data.get("search_calls")
    builds = data.get("geometry_builds")
    reuses = data.get("geometry_reuses")
    invalidations = data.get("geometry_invalidations")
    common = data.get("schema") == 1 and data.get("backend") == backend \
        and isinstance(calls, int) and isinstance(builds, int) \
        and isinstance(reuses, int) and isinstance(invalidations, int) \
        and calls >= 4 and invalidations >= 2 \
        and calls == builds + reuses
    if fresh:
        accounting = builds == calls and reuses == 0 and invalidations*2 == calls
    else:
        accounting = builds == invalidations + 1 \
            and reuses == invalidations - 1 \
            and calls == 2*invalidations
    return {
        "search_calls": calls,
        "geometry_builds": builds,
        "geometry_reuses": reuses,
        "geometry_invalidations": invalidations,
        "passed": common and accounting,
    }


def run_once(executable: Path, project_root: Path, out_dir: Path, backend: str, fresh: bool) -> dict:
    """Run one production variant and collect exact state plus search accounting"""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    subprocess.run([str(executable)], cwd=project_root, check=True)

    particle_path = out_dir/"particle_00001.dat"
    rng_path = out_dir/"rngstate_00001.dat"
    geometry_path = out_dir/"collision_geometry_00001.json"
    variable_path = out_dir/"variables.txt"
    for path in (particle_path, rng_path, geometry_path, variable_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing collision-geometry artifact: {path}")

    particle = read_particle(particle_path)
    geometry = validate_geometry(geometry_path, backend, fresh)
    return {
        "particle": particle,
        "particle_sha256": digest(particle_path),
        "rng_sha256": digest(rng_path),
        "geometry": geometry,
        "variables": variable_path.read_text(),
        "finite": all(value == value and abs(value) != float("inf") for value in particle),
    }


def run(model: str, backend: str) -> None:
    """Build all integrator/search baselines and archive one qualification manifest"""

    if backend not in {"cuda", "rocm"}:
        raise ValueError(f"unsupported GPU backend: {backend}")
    default_target = os.environ.get("CUDA_ARCH", "sm_80") if backend == "cuda" \
        else os.environ.get("AMDGPU_TARGET", "gfx942")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=default_target)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    model_dir = project_root/"qav"/backend/"swarm"/model
    scope = os.environ.get("QAV_SCOPE", "chain")
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
    backend_flag = "CUDA_FLAGS" if backend == "cuda" else "ROCM_FLAGS"
    records = {}
    integrators = (
        ("bernoulli_direct", ["-DBERNOULLI"], "bernoulli_direct", "direct"),
        ("bernoulli_cache", ["-DBERNOULLI", "-DKNN_CACHE"], "bernoulli_cache", "cached"),
        ("frozen_bath", [], "frozen_bath", "cached"),
    )
    for integrator, integrator_defines, provenance_integrator, provenance_neighbors in integrators:
        for search in ("kdtree", "morton"):
            pair = {}
            for mode in ("reuse", "fresh"):
                tag = f"{integrator}_{search}_{mode}"
                defines = list(integrator_defines)
                if mode == "fresh":
                    defines.append("-DKNN_FRESH")
                make_base = [
                    "make", "-C", str(project_root), f"MODEL={model}",
                    f"GPU_BACKEND={backend}", f"GPU_TARGET={args.target}",
                    f"COLLISION_SEARCH={search}", f"QAV_SCOPE={scope}", f"OUT_TAG={tag}",
                    f"{backend_flag}={' '.join(defines)}",
                ]
                subprocess.run([*make_base, "clean"], check=True)
                subprocess.run(make_base, check=True)
                if args.build_only:
                    continue
                pair[mode] = run_once(
                    executable, project_root, out_dir/tag, backend, mode == "fresh"
                )

            if args.build_only:
                continue
            reuse = pair["reuse"]
            fresh = pair["fresh"]
            expected_integrator = f"COLLISION_INTEGRATOR = {provenance_integrator}"
            provenance = expected_integrator in reuse["variables"] \
                and f"COLLISION_NEIGHBORS = {provenance_neighbors}" in reuse["variables"] \
                and f"COLLISION_SEARCH = {search}" in reuse["variables"]
            exact = reuse["particle_sha256"] == fresh["particle_sha256"] \
                and reuse["rng_sha256"] == fresh["rng_sha256"]
            passed = reuse["finite"] and fresh["finite"] and exact and provenance \
                and reuse["geometry"]["passed"] and fresh["geometry"]["passed"]
            records[f"{integrator}_{search}"] = {
                "reuse": {key: value for key, value in reuse.items() if key not in {"particle", "variables"}},
                "fresh": {key: value for key, value in fresh.items() if key not in {"particle", "variables"}},
                "exact_particle_and_rng": exact,
                "provenance": provenance,
                "passed": passed,
            }
            print(
                f"{model}/{integrator}/{search}: {'PASS' if passed else 'FAIL'} "
                f"builds={reuse['geometry']['geometry_builds']}/{fresh['geometry']['geometry_builds']} "
                f"reuses={reuse['geometry']['geometry_reuses']} exact={exact}",
                flush=True,
            )

    if args.build_only:
        return

    bernoulli_equivalence = {}
    for search in ("kdtree", "morton"):
        modes = {}
        for mode in ("reuse", "fresh"):
            direct = records[f"bernoulli_direct_{search}"][mode]
            cached = records[f"bernoulli_cache_{search}"][mode]
            particle_equal = direct["particle_sha256"] == cached["particle_sha256"]
            rng_equal = direct["rng_sha256"] == cached["rng_sha256"]
            modes[mode] = {
                "particle_equal": particle_equal,
                "rng_equal": rng_equal,
                "passed": particle_equal and rng_equal,
            }
        comparison_passed = all(value["passed"] for value in modes.values())
        bernoulli_equivalence[search] = {
            "modes": modes,
            "passed": comparison_passed,
        }
        print(
            f"{model}/bernoulli_direct_vs_cache/{search}: "
            f"{'PASS' if comparison_passed else 'FAIL'} "
            f"reuse={modes['reuse']['passed']} fresh={modes['fresh']['passed']}",
            flush=True,
        )

    passed = all(record["passed"] for record in records.values()) \
        and all(record["passed"] for record in bernoulli_equivalence.values())
    manifest = {
        "schema": 1,
        "model": model,
        "backend": backend,
        "gpu_target": args.target,
        "tier": "qualification",
        "records": records,
        "bernoulli_direct_cache_equivalence": bernoulli_equivalence,
        "passed": passed,
        "finished_utc": utc_now(),
    }
    manifest_path = out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"{model}: {'PASS' if passed else 'FAIL'} variants={len(records)} "
        f"manifest={manifest_path}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)
