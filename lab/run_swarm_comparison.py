#!/usr/bin/env python3

"""Build and compare the copied swarm runtime with both exact KNN backends

Each backend receives the same model constants, deterministic initialization
seed, collision physics, and CUDA compiler flags.  Backend-specific object and
output directories prevent one build from reusing the other's objects.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "par_size",
    "par_numr",
)
PARTICLE_DTYPE = np.dtype([(name, "<f8") for name in FIELDS])
KNN_PATTERN = re.compile(
    r"\[KNN\](?: backend=(?P<backend>\w+) build_ms=(?P<build>[0-9.eE+-]+)"
    r" persistent_bytes=(?P<memory>\d+)| rate_ms=(?P<rate>[0-9.eE+-]+)"
    r" event_ms=(?P<event>[0-9.eE+-]+) max_rate=(?P<max_rate>[0-9.eE+-]+))"
)


def json_default(value: object) -> object:
    """Convert NumPy scalar diagnostics without masking unsupported result types"""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported result type: {type(value).__name__}")


def run_live(command: list[str], cwd: Path, log_path: Path, timeout: float) -> tuple[int, str, float]:
    """Stream one executable to the terminal while preserving the same text in its log"""

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("failed to capture simulation output")

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        """Read in a separate thread so the timeout still works when the executable is silent"""

        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output: list[str] = []
    deadline = started + timeout

    with log_path.open("w") as log_file:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                process.kill()
                process.wait()
                raise RuntimeError(f"simulation exceeded {timeout:g} seconds; see {log_path}")

            try:
                line = output_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue

            if line is None:
                break

            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            output.append(line)

    return_code = process.wait()
    reader.join()
    return return_code, "".join(output), time.perf_counter() - started


def run_backend(
    root: Path,
    model: str,
    backend: str,
    arch: str,
    timeout: float,
    skip_build: bool,
) -> dict[str, object]:
    """Compile and execute one backend, retaining its complete terminal log"""

    output_dir = root/"out"/"swarm"/model/backend
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_build:
        clean = subprocess.run(
            [
                "make",
                "-C",
                str(root),
                "swarm-clean",
                f"SWARM_MODEL={model}",
                f"COLLISION_SEARCH={backend}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        build = subprocess.run(
            [
                "make",
                "-C",
                str(root),
                "swarm",
                f"SWARM_MODEL={model}",
                f"COLLISION_SEARCH={backend}",
                f"ARCH={arch}",
                "PTXAS=1",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        build_output = clean.stdout + build.stdout
        (output_dir/"build.txt").write_text(build_output)
        if clean.returncode != 0 or build.returncode != 0:
            print(build_output, end="", flush=True)
            raise RuntimeError(
                f"{model}/{backend} did not compile; see {output_dir/'build.txt'}"
            )

    executable = root/"bin"/f"gamedev_{model}_{backend}"

    return_code, run_output, wall_seconds = run_live(
        [str(executable)], root.parent, output_dir/"run.txt", timeout
    )

    if return_code != 0:
        raise RuntimeError(
            f"{model}/{backend} exited with status {return_code}; "
            f"see {output_dir/'run.txt'}"
        )

    timing: dict[str, object] = {
        "wall_seconds": wall_seconds,
        "build_ms": 0.0,
        "rate_ms": 0.0,
        "event_ms": 0.0,
        "persistent_bytes": None,
        "maximum_rate": 0.0,
    }
    for match in KNN_PATTERN.finditer(run_output):
        if match.group("build") is not None:
            timing["build_ms"] += float(match.group("build"))
            timing["persistent_bytes"] = int(match.group("memory"))
        else:
            timing["rate_ms"] += float(match.group("rate"))
            timing["event_ms"] += float(match.group("event"))
            timing["maximum_rate"] = max(
                float(timing["maximum_rate"]), float(match.group("max_rate"))
            )
    return timing


def read_particles(path: Path) -> np.ndarray:
    """Load the fixed laboratory multisize particle record"""

    if not path.exists():
        raise FileNotFoundError(f"missing particle checkpoint: {path}")
    data = np.fromfile(path, dtype=PARTICLE_DTYPE)
    if data.size == 0:
        raise ValueError(f"empty particle checkpoint: {path}")
    return data


def field_error(reference: np.ndarray, candidate: np.ndarray, name: str) -> dict[str, object]:
    """Report byte equality, maximum error, and a scale-aware relative L2 norm"""

    ref = reference[name]
    test = candidate[name]
    delta = test - ref
    ref_norm = np.linalg.norm(ref)
    delta_norm = np.linalg.norm(delta)
    relative_l2 = delta_norm/ref_norm if ref_norm > 0.0 else delta_norm
    return {
        "byte_equal": bool(ref.tobytes() == test.tobytes()),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(relative_l2),
    }


def array_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    l2_tolerance: float,
    max_tolerance: float,
) -> dict[str, object]:
    """Compare deterministic floating arrays with explicit absolute and relative diagnostics"""

    if reference.shape != candidate.shape:
        raise ValueError("probe array sizes differ")
    delta = candidate - reference
    ref_norm = np.linalg.norm(reference)
    relative_l2 = np.linalg.norm(delta)/ref_norm if ref_norm > 0.0 else np.linalg.norm(delta)
    max_abs = float(np.max(np.abs(delta)))
    max_scale = max(1.0, float(np.max(np.abs(reference))))
    max_relative = max_abs/max_scale
    zero_mask_equal = bool(np.array_equal(reference == 0.0, candidate == 0.0))
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    passed = bool(
        finite and zero_mask_equal
        and relative_l2 <= l2_tolerance and max_relative <= max_tolerance
    )
    return {
        "passed": passed,
        "byte_equal": bool(reference.tobytes() == candidate.tobytes()),
        "finite": finite,
        "zero_mask_equal": zero_mask_equal,
        "max_abs": max_abs,
        "max_relative": max_relative,
        "relative_l2": float(relative_l2),
    }


def exact_array_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Compare integer topology probes exactly and report where disagreement starts"""

    if reference.shape != candidate.shape:
        raise ValueError("probe array sizes differ")
    mismatch = reference != candidate
    mismatch_count = int(np.count_nonzero(mismatch))
    first_mismatch = int(np.flatnonzero(mismatch)[0]) if mismatch_count else None
    result = {
        "passed": mismatch_count == 0,
        "byte_equal": bool(reference.tobytes() == candidate.tobytes()),
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count/reference.size,
        "first_mismatch": first_mismatch,
    }
    if first_mismatch is not None:
        result["reference_value"] = int(reference[first_mismatch])
        result["candidate_value"] = int(candidate[first_mismatch])
    return result


def evaluate_constant(expression: str) -> float:
    """Evaluate the small arithmetic expressions used by model constants without eval"""

    binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left*right,
        ast.Div: lambda left, right: left/right,
    }
    unary = {
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "M_PI":
            return math.pi
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported constant expression: {expression}")

    return visit(ast.parse(expression, mode="eval").body)


def read_model_constant(root: Path, model: str, name: str) -> float:
    """Read one numeric constant from the selected laboratory model header"""

    header = root/"mod"/model/"const_defs.cuh"
    pattern = re.compile(
        rf"^\s*const\s+(?:int|real)\s+{re.escape(name)}\s*=\s*([^;]+);",
        re.MULTILINE,
    )
    match = pattern.search(header.read_text())
    if match is None:
        raise ValueError(f"missing {name} in {header}")
    return evaluate_constant(match.group(1).strip())


def mixed_neighbor_hash(indices: np.ndarray) -> int:
    """Reproduce the commutative 64-bit GPU neighbor fingerprint"""

    mask = (1 << 64) - 1
    total = 0
    for index in indices:
        value = (int(index) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30))*0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27))*0x94D049BB133111EB) & mask
        value ^= value >> 31
        total = (total + value) & mask
    return total


def periodic_neighbor_forensic(
    root: Path,
    model: str,
    particles: np.ndarray,
    query: int,
    neighbor_count: int,
    kdtree_hash: int,
    morton_hash: int,
    tie_tolerance: float,
) -> dict[str, object]:
    """Adjudicate one topology disagreement in physical and GPU search coordinates"""

    n_x = int(read_model_constant(root, model, "N_X"))
    x_min = read_model_constant(root, model, "X_MIN")
    x_max = read_model_constant(root, model, "X_MAX")
    width = x_max - x_min

    x = particles["position_x"]
    y = particles["position_y"]
    z = particles["position_z"]
    radius = y*np.sin(z)
    height = y*np.cos(z)
    delta_x = x - x[query]
    if n_x > 1:
        delta_x -= width*np.floor(delta_x/width + 0.5)

    physical_distance_sq = (
        radius*radius + radius[query]*radius[query]
        - 2.0*radius*radius[query]*np.cos(delta_x)
        + (height - height[query])**2
    )
    physical_distance_sq = np.maximum(physical_distance_sq, 0.0)
    physical_distance_sq[query] = np.inf

    # Both GPU paths first round spherical positions to float and then evaluate
    # the Cartesian map in float.  Reconstruct that common input and accumulate
    # its distances in double so the oracle tests the search rather than the
    # shared coordinate quantization.
    x_float = x.astype(np.float32)
    y_float = y.astype(np.float32)
    z_float = z.astype(np.float32)
    radius_float = y_float*np.sin(z_float)
    cartesian = np.empty((particles.size, 3), dtype=np.float32)
    cartesian[:, 0] = radius_float*np.cos(x_float)
    cartesian[:, 1] = radius_float*np.sin(x_float)
    cartesian[:, 2] = y_float*np.cos(z_float)
    query_images = [cartesian[query]]
    if n_x > 1 and width < 2.0*math.pi - 1.0e-12:
        width_float = np.float32(width)
        sin_width = np.sin(width_float)
        cos_width = np.cos(width_float)
        query_point = cartesian[query]
        for sign in (-1.0, 1.0):
            sin_angle = np.float32(sign)*sin_width
            query_images.append(np.array((
                cos_width*query_point[0] - sin_angle*query_point[1],
                sin_angle*query_point[0] + cos_width*query_point[1],
                query_point[2],
            ), dtype=np.float32))

    cartesian_double = cartesian.astype(np.float64)
    search_distance_sq = np.full(particles.size, np.inf)
    for query_image in query_images:
        delta = cartesian_double - query_image.astype(np.float64)
        search_distance_sq = np.minimum(
            search_distance_sq, np.einsum("ij,ij->i", delta, delta)
        )
    search_distance_sq[query] = np.inf

    def adjudicate(distance_sq: np.ndarray) -> dict[str, object]:
        """Return the exact top-K fingerprint and boundary gap for one metric"""

        # Only K+1 entries are needed, so keep the million-particle forensic linear
        candidates = np.argpartition(distance_sq, neighbor_count)[:neighbor_count + 1]
        order = candidates[np.lexsort((candidates, distance_sq[candidates]))]
        reference_hash = mixed_neighbor_hash(order[:neighbor_count])
        if reference_hash == kdtree_hash == morton_hash:
            reference_backend = "both"
        elif reference_hash == kdtree_hash:
            reference_backend = "kdtree"
        elif reference_hash == morton_hash:
            reference_backend = "morton"
        else:
            reference_backend = "neither"
        kth_distance = math.sqrt(float(distance_sq[order[neighbor_count - 1]]))
        next_distance = math.sqrt(float(distance_sq[order[neighbor_count]]))
        distance_gap = next_distance - kth_distance
        relative_distance_gap = distance_gap/max(kth_distance, np.finfo(np.float64).tiny)
        return {
            "reference_backend": reference_backend,
            "reference_hash": reference_hash,
            "kth_neighbor": int(order[neighbor_count - 1]),
            "next_neighbor": int(order[neighbor_count]),
            "kth_distance": kth_distance,
            "next_distance": next_distance,
            "distance_gap": distance_gap,
            "relative_distance_gap": relative_distance_gap,
        }

    physical = adjudicate(physical_distance_sq)
    search = adjudicate(search_distance_sq)
    return {
        "query": query,
        "neighbor_count": neighbor_count,
        "kdtree_hash": kdtree_hash,
        "morton_hash": morton_hash,
        "search_reference": search,
        "physical_reference": physical,
        "near_float_tie": bool(search["relative_distance_gap"] <= tie_tolerance),
        "query_position": {
            "x": float(x[query]),
            "y": float(y[query]),
            "z": float(z[query]),
        },
        "angular_boundary_distance": float(min(x[query] - x_min, x_max - x[query])),
    }


def ks_distance_sorted(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the empirical two-sample KS distance between sorted arrays"""

    samples = np.concatenate((reference, candidate))
    ref_cdf = np.searchsorted(reference, samples, side="right")/reference.size
    test_cdf = np.searchsorted(candidate, samples, side="right")/candidate.size
    return float(np.max(np.abs(ref_cdf - test_cdf)))


def distribution_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    family_alpha: float,
    comparisons: int,
) -> dict[str, object]:
    """Compare evolved samples using a Bonferroni-controlled two-sample KS test"""

    alpha = family_alpha/comparisons
    critical = np.sqrt(
        -0.5*np.log(alpha/2.0)
        *(reference.size + candidate.size)/(reference.size*candidate.size)
    )
    ref_sorted = np.sort(reference)
    test_sorted = np.sort(candidate)
    distance = ks_distance_sorted(ref_sorted, test_sorted)
    sorted_norm = np.linalg.norm(ref_sorted)
    sorted_relative_l2 = np.linalg.norm(test_sorted - ref_sorted)/sorted_norm \
        if sorted_norm > 0.0 else np.linalg.norm(test_sorted - ref_sorted)
    return {
        "passed": bool(distance <= critical),
        "ks_distance": distance,
        "ks_critical": float(critical),
        "field_alpha": alpha,
        "sorted_relative_l2": float(sorted_relative_l2),
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "reference_std": float(np.std(reference)),
        "candidate_std": float(np.std(candidate)),
    }


def represented_mass(data: np.ndarray) -> float:
    """Return mass up to the common compact-grain factor pi*rho/6"""

    return float(np.sum(data["par_numr"]*data["par_size"]**3, dtype=np.float64))


def compare_model(
    root: Path,
    model: str,
    tolerance: float,
    probe_l2_tolerance: float,
    probe_max_tolerance: float,
    mass_tolerance: float,
    ks_alpha: float,
    knn_tie_tolerance: float,
) -> dict[str, object]:
    """Compare deterministic search outputs and statistical evolved distributions"""

    base = root/"out"/"swarm"/model
    kd_initial_path = base/"kdtree"/"particle_00000.dat"
    mo_initial_path = base/"morton"/"particle_00000.dat"
    kd_final_path = base/"kdtree"/"particle_00001.dat"
    mo_final_path = base/"morton"/"particle_00001.dat"

    kd_initial = read_particles(kd_initial_path)
    mo_initial = read_particles(mo_initial_path)
    kd_final = read_particles(kd_final_path)
    mo_final = read_particles(mo_final_path)
    if not (kd_initial.size == mo_initial.size == kd_final.size == mo_final.size):
        raise ValueError("backend particle counts differ")

    fields = {name: field_error(kd_final, mo_final, name) for name in FIELDS}
    distributions = {
        name: distribution_error(kd_final[name], mo_final[name], ks_alpha, len(FIELDS))
        for name in FIELDS
    }
    finite = bool(
        all(np.isfinite(kd_final[name]).all() and np.isfinite(mo_final[name]).all() for name in FIELDS)
    )
    initial_byte_equal = kd_initial_path.read_bytes() == mo_initial_path.read_bytes()

    kd_rng = base/"kdtree"/"rngstate_00001.dat"
    mo_rng = base/"morton"/"rngstate_00001.dat"
    rng_byte_equal = kd_rng.exists() and mo_rng.exists() and kd_rng.read_bytes() == mo_rng.read_bytes()

    kd_rate = np.fromfile(base/"kdtree"/"col_rate_probe.dat", dtype="<f8")
    mo_rate = np.fromfile(base/"morton"/"col_rate_probe.dat", dtype="<f8")
    kd_dist = np.fromfile(base/"kdtree"/"col_dist_probe.dat", dtype="<f8")
    mo_dist = np.fromfile(base/"morton"/"col_dist_probe.dat", dtype="<f8")
    kd_hash = np.fromfile(base/"kdtree"/"col_hash_probe.dat", dtype="<u8")
    mo_hash = np.fromfile(base/"morton"/"col_hash_probe.dat", dtype="<u8")
    kd_count = np.fromfile(base/"kdtree"/"col_count_probe.dat", dtype="<u4")
    mo_count = np.fromfile(base/"morton"/"col_count_probe.dat", dtype="<u4")
    if not (
        kd_rate.size == mo_rate.size == kd_dist.size == mo_dist.size
        == kd_hash.size == mo_hash.size == kd_count.size == mo_count.size
        == kd_final.size
    ):
        raise ValueError("collision probe counts differ from the particle count")
    hash_probe = exact_array_error(kd_hash, mo_hash)
    count_probe = exact_array_error(kd_count, mo_count)
    common_topology = kd_hash == mo_hash
    probes = {
        "neighbor_hash": hash_probe,
        "neighbor_count": count_probe,
        "collision_rate": array_error(
            kd_rate, mo_rate, probe_l2_tolerance, probe_max_tolerance
        ),
        "knn_radius": array_error(
            kd_dist, mo_dist, probe_l2_tolerance, probe_max_tolerance
        ),
    }
    topology_forensics = []
    if hash_probe["mismatch_count"] and bool(count_probe["passed"]):
        for mismatch_query in np.flatnonzero(~common_topology):
            topology_forensics.append(periodic_neighbor_forensic(
                root,
                model,
                kd_initial,
                int(mismatch_query),
                int(kd_count[mismatch_query]),
                int(kd_hash[mismatch_query]),
                int(mo_hash[mismatch_query]),
                knn_tie_tolerance,
            ))

    near_tie_equivalent = bool(topology_forensics) and all(
        item["near_float_tie"]
        and item["search_reference"]["reference_backend"] in ("both", "kdtree", "morton")
        for item in topology_forensics
    )
    hash_probe["exact_passed"] = bool(hash_probe["passed"])
    hash_probe["near_tie_equivalent"] = near_tie_equivalent
    hash_probe["passed"] = bool(hash_probe["passed"] or near_tie_equivalent)

    # A boundary tie can legitimately exchange one almost equidistant grain,
    # whose physical properties need not be equal.  Preserve the all-query
    # diagnostic, but judge arithmetic agreement only where both backends chose
    # the same neighbor identities.
    if np.any(common_topology):
        common_rate = array_error(
            kd_rate[common_topology], mo_rate[common_topology],
            probe_l2_tolerance, probe_max_tolerance
        )
    else:
        common_rate = dict(probes["collision_rate"])
        common_rate["passed"] = False
    probes["collision_rate"]["all_query_passed"] = bool(probes["collision_rate"]["passed"])
    probes["collision_rate"]["common_topology"] = common_rate
    probes["collision_rate"]["topology_exempt_count"] = int(np.count_nonzero(~common_topology))
    probes["collision_rate"]["passed"] = bool(common_rate["passed"] and hash_probe["passed"])

    kd_mass_initial = represented_mass(kd_initial)
    mo_mass_initial = represented_mass(mo_initial)
    kd_mass_final = represented_mass(kd_final)
    mo_mass_final = represented_mass(mo_final)
    kd_mass_drift = (kd_mass_final - kd_mass_initial)/kd_mass_initial
    mo_mass_drift = (mo_mass_final - mo_mass_initial)/mo_mass_initial
    final_mass_mismatch = (mo_mass_final - kd_mass_final)/kd_mass_final

    trajectory_close = all(float(item["relative_l2"]) <= tolerance for item in fields.values())
    probe_pass = all(bool(item["passed"]) for item in probes.values())
    distribution_pass = all(bool(item["passed"]) for item in distributions.values())
    mass_pass = max(
        abs(kd_mass_drift), abs(mo_mass_drift), abs(final_mass_mismatch)
    ) <= mass_tolerance
    passed = initial_byte_equal and finite and probe_pass and distribution_pass and mass_pass

    return {
        "model": model,
        "passed": passed,
        "particle_count": int(kd_final.size),
        "initial_byte_equal": initial_byte_equal,
        "rng_byte_equal": rng_byte_equal,
        "finite": finite,
        "trajectory_close": trajectory_close,
        "trajectory_tolerance": tolerance,
        "probe_l2_tolerance": probe_l2_tolerance,
        "probe_max_tolerance": probe_max_tolerance,
        "knn_tie_tolerance": knn_tie_tolerance,
        "mass_tolerance": mass_tolerance,
        "ks_family_alpha": ks_alpha,
        "probes": probes,
        "topology_forensic": topology_forensics[0] if topology_forensics else None,
        "topology_forensics": topology_forensics,
        "fields": fields,
        "distributions": distributions,
        "kdtree_mass_drift": kd_mass_drift,
        "morton_mass_drift": mo_mass_drift,
        "final_mass_mismatch": final_mass_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["collision_disk_2d", "collision_wedge_2d", "collision_disk_3d"],
    )
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--probe-l2-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--probe-max-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--knn-tie-tolerance",
        type=float,
        default=8.0*np.finfo(np.float32).eps,
        help="relative K-boundary gap treated as a single-precision topology tie",
    )
    parser.add_argument("--mass-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--ks-alpha", type=float, default=1.0e-2)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="reanalyze existing backend outputs without rebuilding or rerunning",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    all_passed = True

    for model in args.models:
        result_path = root/"out"/"swarm"/f"comparison_{model}.json"
        if args.analyze_only:
            print(f"\n=== {model}: analyze existing outputs ===", flush=True)
            prior = json.loads(result_path.read_text()) if result_path.exists() else {}
            prior_timing = prior.get("timing", {})
            empty_timing = {
                "wall_seconds": 0.0,
                "build_ms": 0.0,
                "rate_ms": 0.0,
                "event_ms": 0.0,
                "persistent_bytes": None,
                "maximum_rate": 0.0,
            }
            kd_timing = prior_timing.get("kdtree", empty_timing)
            mo_timing = prior_timing.get("morton", empty_timing)
        else:
            print(f"\n=== {model}: kdtree ===", flush=True)
            kd_timing = run_backend(
                root, model, "kdtree", args.arch, args.timeout, args.skip_build
            )
            print(f"=== {model}: morton ===", flush=True)
            mo_timing = run_backend(
                root, model, "morton", args.arch, args.timeout, args.skip_build
            )

        result = compare_model(
            root,
            model,
            args.tolerance,
            args.probe_l2_tolerance,
            args.probe_max_tolerance,
            args.mass_tolerance,
            args.ks_alpha,
            args.knn_tie_tolerance,
        )
        result["timing"] = {"kdtree": kd_timing, "morton": mo_timing}
        if float(mo_timing["wall_seconds"]) > 0.0:
            result["wall_speed_ratio"] = (
                float(kd_timing["wall_seconds"])/float(mo_timing["wall_seconds"])
            )
        kd_memory = kd_timing["persistent_bytes"]
        mo_memory = mo_timing["persistent_bytes"]
        if kd_memory is not None and mo_memory is not None and int(kd_memory) > 0:
            result["persistent_memory_ratio"] = int(mo_memory)/int(kd_memory)

        print(
            f"{model}: passed={result['passed']} "
            f"initial_equal={result['initial_byte_equal']} "
            f"probes={all(item['passed'] for item in result['probes'].values())} "
            f"distributions={all(item['passed'] for item in result['distributions'].values())} "
            f"rng_equal={result['rng_byte_equal']} "
            f"mass_mismatch={result['final_mass_mismatch']:.3e}"
        )
        for name, probe in result["probes"].items():
            if "mismatch_count" in probe:
                print(
                    f"  probe {name}: passed={probe['passed']} "
                    f"mismatches={probe['mismatch_count']}"
                )
            else:
                print(
                    f"  probe {name}: passed={probe['passed']} "
                    f"rel_L2={probe['relative_l2']:.3e} "
                    f"max_rel={probe['max_relative']:.3e}"
                )
                if name == "collision_rate" and "common_topology" in probe:
                    common = probe["common_topology"]
                    print(
                        f"    common topology: passed={common['passed']} "
                        f"rel_L2={common['relative_l2']:.3e} "
                        f"max_rel={common['max_relative']:.3e} "
                        f"excluded_ties={probe['topology_exempt_count']}"
                    )
        forensic = result["topology_forensic"]
        if forensic is not None:
            print(
                f"  topology mismatches={len(result['topology_forensics'])} "
                f"near_tie_equivalent={result['probes']['neighbor_hash']['near_tie_equivalent']}"
            )
            for item in result["topology_forensics"]:
                print(
                    f"    query={item['query']} "
                    f"search_reference={item['search_reference']['reference_backend']} "
                    f"search_K_gap={item['search_reference']['relative_distance_gap']:.3e} "
                    f"near_float_tie={item['near_float_tie']} "
                    f"physical_reference={item['physical_reference']['reference_backend']} "
                    f"physical_K_gap={item['physical_reference']['relative_distance_gap']:.3e} "
                    f"boundary_angle={item['angular_boundary_distance']:.3e}"
                )
        for name in ("par_size", "par_numr"):
            distribution = result["distributions"][name]
            print(
                f"  distribution {name}: passed={distribution['passed']} "
                f"KS={distribution['ks_distance']:.3e}/"
                f"{distribution['ks_critical']:.3e}"
            )

        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
        print(f"result: {result_path}")
        all_passed = all_passed and bool(result["passed"])

    if not all_passed:
        raise SystemExit("one or more swarm backend comparisons failed")


if __name__ == "__main__":
    main()
