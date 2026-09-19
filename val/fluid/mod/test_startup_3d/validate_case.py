#!/usr/bin/env python3

"""Measure convergence of the initialized polar advection-diffusion residual"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np


def read_values(out_dir: Path, name: str, resolution: int, count: int) -> np.ndarray:
    """Load one binary field and reject stale output with an incompatible shape"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{name} contains {values.size} values; expected {count}")
    return values


def weighted_norms(
    residual: np.ndarray, scale: np.ndarray, volume: np.ndarray,
) -> dict[str, float]:
    """Normalize the uncancelled tendency by the two resolved component tendencies"""

    volume_sum = float(np.sum(volume))
    residual_l1 = float(np.sum(np.abs(residual)*volume)/volume_sum)
    scale_l1 = float(np.sum(scale*volume)/volume_sum)
    residual_l2 = float(np.sqrt(np.sum(residual*residual*volume)/volume_sum))
    scale_l2 = float(np.sqrt(np.sum(scale*scale*volume)/volume_sum))
    return {
        "l1": residual_l1/scale_l1,
        "l2": residual_l2/scale_l2,
        "linf": float(np.max(np.abs(residual))/np.max(scale)),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare the discrete polar advection and diffusion tendencies"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    count = nx*ny*nz
    dt = float(meta["probe_dt"])

    initial = read_values(out_dir, "dustdens_initial", resolution, count).reshape(nz, ny, nx)
    advected = read_values(out_dir, "dustdens_advection", resolution, count).reshape(nz, ny, nx)
    diffused = read_values(out_dir, "dustdens_diffusion", resolution, count).reshape(nz, ny, nx)

    adv_rate = (advected - initial)/dt
    diff_rate = (diffused - initial)/dt
    residual = adv_rate + diff_rate
    scale = np.abs(adv_rate) + np.abs(diff_rate)

    ratio = (float(meta["y_max"])/float(meta["y_min"]))**(1.0/ny)
    y_face = float(meta["y_min"])*ratio**np.arange(ny + 1)
    vol_y = (y_face[1:]**3 - y_face[:-1]**3)/3.0
    z_face = np.linspace(float(meta["z_min"]), float(meta["z_max"]), nz + 1)
    vol_z = np.cos(z_face[:-1]) - np.cos(z_face[1:])
    volume = vol_z[:, None, None]*vol_y[None, :, None]*np.ones((1, 1, nx))

    cancellation = weighted_norms(residual, scale, volume)
    initial_mass = float(np.sum(initial*volume))
    residual_mass_rate = float(np.sum(residual*volume))
    component_mass_rate = float(np.sum(scale*volume))
    mass_balance = abs(residual_mass_rate)/component_mass_rate

    finite = bool(
        np.all(np.isfinite(initial)) and np.all(np.isfinite(advected))
        and np.all(np.isfinite(diffused)) and np.all(np.isfinite(residual))
    )
    nonnegative = bool(np.min(initial) >= 0.0 and np.min(advected) >= 0.0 and np.min(diffused) >= 0.0)
    passed = finite and nonnegative and initial_mass > 0.0 and mass_balance < 1.0e-6

    return {
        "case": "startup_3d",
        "resolution": resolution,
        "primary_field": "polar_residual",
        "reference_class": "continuum-initial-balance",
        "terminal_mode": "convergence",
        "probe_dt": dt,
        "initial_mass": initial_mass,
        "residual_mass_balance": mass_balance,
        "convergence_field": "polar_residual",
        "minimum_order": 1.5,
        "errors": {"polar_residual": cancellation},
        "finite": finite,
        "nonnegative": nonnegative,
        "passed": passed,
    }
