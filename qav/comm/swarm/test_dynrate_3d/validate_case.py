#!/usr/bin/env python3

"""Reconstruct every enabled production dynamics-rate candidate"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def norm_set(error: np.ndarray) -> dict[str, float]:
    """Return common absolute norms"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare the device maximum with an independent host reconstruction"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir/f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    actual = np.fromfile(out_dir/f"dyn_rate_N{resolution}.dat", dtype=np.float64)
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    z_min, z_max = float(meta["z_min"]), float(meta["z_max"])
    cfl = float(meta["cfl_dyn"])
    dx = (x_max - x_min)/nx
    dy = (y_max/y_min)**(1.0/ny)
    dz = (z_max - z_min)/nz
    gms = float(meta["gms"])
    nu = float(meta["nu"])

    names = (
        "orbital", "orbital_phase", "cross_x", "cross_y", "cross_z",
        "accel_y", "accel_z", "visc_y", "visc_z",
        "diff_y", "drift_y", "diff_x", "drift_x", "diff_z", "drift_z",
    )
    expected = np.zeros(count)
    bound_violation = np.zeros(count)
    step_fraction = np.zeros(count)
    dominant: dict[str, int] = {}
    for idx in range(count - 1):
        _, y, z, lx, vy, lz = state[:, idx]
        R = y*math.sin(z)
        omega = math.sqrt(gms/R**3)
        iy = int(math.log(y/y_min)/math.log(dy))
        iy = min(iy, ny - 1)
        dr = y_min*dy**iy*(dy - 1.0)

        grav_y = -gms/(y*y)
        cent_y = lx*lx/(R*R*y) + lz*lz/(y**3)
        accel_y = abs(grav_y + cent_y)
        torq_z = lx*lx/(R*R)*math.cos(z)/math.sin(z)
        accel_z = abs(torq_z)/y

        h_g = float(meta["aspr_0"])*R**(0.5*(float(meta["idx_q"]) + 1.0))
        cyl_frac = R/y
        strat = (cyl_frac - 1.0)/(h_g*h_g)
        grad_strat_R = cyl_frac*(1.0 - cyl_frac*cyl_frac)/(h_g*h_g) \
            - (float(meta["idx_q"]) + 1.0)*strat
        grad_strat_Z = -cyl_frac*(1.0 - cyl_frac*cyl_frac)/(h_g*h_g)
        grad_rhog_R = float(meta["idx_p"]) - 0.5*(float(meta["idx_q"]) + 3.0) + grad_strat_R
        grad_rhog_Z = grad_strat_Z
        term_R = 3.0*nu*(grad_rhog_R + 0.5)
        term_Z = float(meta["idx_q"])*nu*(1.0 + grad_rhog_Z)
        visc_R = -(term_R - term_Z)/R

        diff_x = nu/float(meta["schmidt_x"])
        diff_R = nu/float(meta["schmidt_R"])
        diff_Z = nu/float(meta["schmidt_Z"])
        drift_R = diff_R/R
        sin_z, cos_z = math.sin(z), math.cos(z)
        diff_y = diff_R*sin_z*sin_z + diff_Z*cos_z*cos_z
        drift_y = drift_R*sin_z
        diff_z = diff_R*cos_z*cos_z + diff_Z*sin_z*sin_z
        drift_z = drift_R*cos_z
        cell_x = R*dx
        cell_z = y*dz

        candidates = np.array([
            omega/cfl,
            omega/(dx*cfl),
            abs(lx)/(R*R*dx*cfl),
            abs(vy)/(dr*cfl),
            abs(lz)/(y*y*dz*cfl),
            math.sqrt(accel_y/(2.0*cfl*dr)),
            math.sqrt(accel_z/(2.0*cfl*y*dz)),
            abs(visc_R*sin_z)/(dr*cfl),
            abs(visc_R*cos_z)/(y*dz*cfl),
            2.0*diff_y/(cfl*cfl*dr*dr),
            abs(drift_y)/(cfl*dr),
            2.0*diff_x/(cfl*cfl*cell_x*cell_x),
            0.0,
            2.0*diff_z/(cfl*cfl*cell_z*cell_z),
            abs(drift_z)/(cfl*cell_z),
        ])
        expected[idx] = float(np.max(candidates))
        dominant_name = names[int(np.argmax(candidates))]
        dominant[dominant_name] = dominant.get(dominant_name, 0) + 1
        bound_violation[idx] = max(0.0, float(np.max(candidates - actual[idx])))
        step_fraction[idx] = float(np.max(candidates/actual[idx])) if actual[idx] > 0.0 else math.inf

    error = actual - expected
    scale = np.maximum(np.abs(expected), 1.0)
    relative = error/scale
    activation = bool(meta["viscous_flow"] and meta["density_diffusion"] and len(dominant) >= 3)
    passed = bool(
        activation
        and np.all(np.isfinite(actual))
        and actual[-1] == 0.0
        and np.max(np.abs(relative[:-1])) < 3.0e-11
        and np.max(bound_violation[:-1]) < 3.0e-11*np.max(scale[:-1])
        and np.max(step_fraction[:-1]) <= 1.0 + 3.0e-11
    )
    return {
        "case": "dynrate_3d",
        "resolution": resolution,
        "reference_class": "independent-formula-reconstruction",
        "activation": activation,
        "candidate_families": list(names),
        "dominant_candidate_counts": dominant,
        "inactive_rate": float(actual[-1]),
        "maximum_candidate_step_fraction": float(np.max(step_fraction[:-1])),
        "errors": {
            "rate_absolute": norm_set(error),
            "rate_relative": norm_set(relative),
            "candidate_bound": norm_set(bound_violation),
        },
        "passed": passed,
    }
