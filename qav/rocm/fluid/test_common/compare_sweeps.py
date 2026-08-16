#!/usr/bin/env python3

"""Compare matched thread-sweep and block-sweep fluid outputs"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("dustdens", "dustvelx", "dustvely", "dustvelz")
OPTIONAL_FIELDS = ("optdepth",)


def read_field(directory: Path, field: str, frame: int, expected: int) -> np.ndarray:
    """Read one header-free double-precision field and verify its cell count"""

    path = directory / f"{field}_{frame:05d}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"missing output field: {path}")

    values = np.fromfile(path, dtype=np.float64)
    if values.size != expected:
        raise ValueError(f"{path}: found {values.size} values, expected {expected}")
    return values


def cell_volumes(nx: int, ny: int, nz: int, z_min: float, z_max: float) -> np.ndarray:
    """Construct the finite-volume weights used by the production spherical mesh"""

    x_min, x_max = 0.0, 2.0*np.pi
    y_min, y_max = 0.5, 2.5
    dx = (x_max - x_min) / nx
    dy = (y_max/y_min)**(1.0/ny)
    y_faces = y_min*dy**np.arange(ny + 1)

    if nz == 1:
        vol_y = (y_faces[1:]**2 - y_faces[:-1]**2) / 2.0
        vol_z = np.ones(1)
    else:
        vol_y = (y_faces[1:]**3 - y_faces[:-1]**3) / 3.0
        z_faces = np.linspace(z_min, z_max, nz + 1)
        vol_z = np.cos(z_faces[:-1]) - np.cos(z_faces[1:])

    # Binary fields use ix + iy*nx + iz*nx*ny, corresponding to this array shape
    return dx*vol_z[:, None, None]*vol_y[None, :, None]*np.ones((1, 1, nx))


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the unweighted relative Euclidean difference"""

    difference_norm = float(np.linalg.norm(candidate - reference))
    reference_norm = float(np.linalg.norm(reference))
    return difference_norm/reference_norm if reference_norm > 0.0 else difference_norm


def accepted_steps(path: Path) -> int:
    """Read the accepted-step count from a structured run record"""

    if not path.is_file():
        raise FileNotFoundError(f"missing run record: {path}")
    record = json.loads(path.read_text())
    count = record.get("accepted_steps")
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"invalid accepted_steps in {path}")
    return count


def compare_pair(
    reference_dir: Path,
    candidate_dir: Path,
    frame: int,
    nx: int,
    ny: int,
    nz: int,
    z_min: float,
    z_max: float,
) -> dict[str, object]:
    """Compare one matched pair, print its metrics, and raise on a failed equivalence check"""

    expected = nx*ny*nz
    fields = list(REQUIRED_FIELDS)
    for field in OPTIONAL_FIELDS:
        ref_exists = (reference_dir/f"{field}_{frame:05d}.dat").is_file()
        opt_exists = (candidate_dir/f"{field}_{frame:05d}.dat").is_file()
        if ref_exists != opt_exists:
            raise FileNotFoundError(f"{field}: present in only one output directory")
        if ref_exists:
            fields.append(field)

    print(f"comparing frame {frame:05d}")
    print(f"thread: {reference_dir}")
    print(f"block:  {candidate_dir}")
    print()
    print(f"{'field':<12} {'count':>12} {'byte equal':>12} {'max abs':>14} {'relative L2':>14}")

    metrics: dict[str, dict[str, float | bool | int]] = {}
    for field in fields:
        reference = read_field(reference_dir, field, frame, expected)
        candidate = read_field(candidate_dir, field, frame, expected)
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
            raise ValueError(f"{field}: nonfinite value found")

        max_abs = float(np.max(np.abs(candidate - reference)))
        rel_l2 = relative_l2(reference, candidate)
        byte_equal = (
            reference_dir/f"{field}_{frame:05d}.dat"
        ).read_bytes() == (
            candidate_dir/f"{field}_{frame:05d}.dat"
        ).read_bytes()
        metrics[field] = {
            "count": expected,
            "byte_equal": byte_equal,
            "max_abs": max_abs,
            "relative_l2": rel_l2,
        }
        print(f"{field:<12} {expected:12d} {str(byte_equal):>12} {max_abs:14.6e} {rel_l2:14.6e}")

    # Identical initialization proves that both executables received the same physical state
    for field in fields:
        ref_path = reference_dir/f"{field}_00000.dat"
        opt_path = candidate_dir/f"{field}_00000.dat"
        if ref_path.read_bytes() != opt_path.read_bytes():
            raise AssertionError(f"{field}: initial thread and block fields are not byte-identical")

    volumes = cell_volumes(nx, ny, nz, z_min, z_max)
    ref_dens0 = read_field(reference_dir, "dustdens", 0, expected)
    opt_dens0 = read_field(candidate_dir, "dustdens", 0, expected)
    ref_dens1 = read_field(reference_dir, "dustdens", frame, expected)
    opt_dens1 = read_field(candidate_dir, "dustdens", frame, expected)
    if np.min(ref_dens1) < 0.0 or np.min(opt_dens1) < 0.0:
        raise ValueError("negative final density found")

    def mass(values: np.ndarray) -> float:
        return float(np.sum(values.reshape(nz, ny, nx)*volumes))

    ref_mass0, opt_mass0 = mass(ref_dens0), mass(opt_dens0)
    ref_mass1, opt_mass1 = mass(ref_dens1), mass(opt_dens1)
    ref_drift = (ref_mass1 - ref_mass0)/ref_mass0
    opt_drift = (opt_mass1 - opt_mass0)/opt_mass0
    mass_mismatch = (opt_mass1 - ref_mass1)/ref_mass1
    ref_steps = accepted_steps(reference_dir/"run.json")
    opt_steps = accepted_steps(candidate_dir/"run.json")

    print()
    print(f"minimum density: thread={np.min(ref_dens1):.8e} block={np.min(opt_dens1):.8e}")
    print(f"thread mass drift: {ref_drift:.8e}")
    print(f"block mass drift:  {opt_drift:.8e}")
    print(f"final mass mismatch: {mass_mismatch:.8e}")
    print(f"accepted steps: thread={ref_steps} block={opt_steps}")

    # These tolerances detect implementation mistakes while allowing harmless floating-point reordering
    if metrics["dustdens"]["relative_l2"] > 1.0e-7:
        raise AssertionError("thread/block density mismatch exceeds 1e-7")
    for field in ("dustvelx", "dustvely", "dustvelz"):
        if metrics[field]["relative_l2"] > 1.0e-5:
            raise AssertionError(f"{field}: thread/block mismatch exceeds 1e-5")
    if abs(mass_mismatch) > 1.0e-10:
        raise AssertionError("thread/block mass mismatch exceeds 1e-10")
    result: dict[str, object] = {
        "frame": frame,
        "shape": [nz, ny, nx],
        "fields": metrics,
        "thread_mass_drift": ref_drift,
        "block_mass_drift": opt_drift,
        "final_mass_mismatch": mass_mismatch,
        "thread_steps": ref_steps,
        "block_steps": opt_steps,
        "passed": True,
    }
    print("sweep comparison: PASS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="compare matched thread and block sweep outputs")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--z-min", type=float, default=0.5*np.pi)
    parser.add_argument("--z-max", type=float, default=0.5*np.pi)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = compare_pair(
        args.reference, args.candidate, args.frame,
        args.nx, args.ny, args.nz, args.z_min, args.z_max,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
