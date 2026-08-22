#!/usr/bin/env python3

"""Validate exact radial crossing, inactive sentinels, and downstream exclusion"""

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
    """Compare crossing records with exact times and verify every archived inactive diagnostic"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir/f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    absorption_time = np.fromfile(out_dir/f"absorption_time_N{resolution}.dat", dtype=np.float64)
    dyn_rate = np.fromfile(out_dir/f"dyn_rate_N{resolution}.dat", dtype=np.float64)
    dustdens = np.fromfile(out_dir/f"dustdens_N{resolution}.dat", dtype=np.float64)
    optdepth = np.fromfile(out_dir/f"optdepth_N{resolution}.dat", dtype=np.float64)
    col_active = np.fromfile(out_dir/f"col_active_N{resolution}.dat", dtype=np.float64)
    col_rate = np.fromfile(out_dir/f"col_rate_N{resolution}.dat", dtype=np.float64)
    col_dist = np.fromfile(out_dir/f"col_dist_N{resolution}.dat", dtype=np.float64)
    dt = float(meta["dt"])
    exact_hit = np.array([float(meta["outer_hit"]), float(meta["inner_hit"])])
    delay = absorption_time[:2] - exact_hit

    absorbed_state = np.concatenate((
        state[1, :2],
        state[2, :2] - 0.5*math.pi,
        state[3:, :2].ravel(),
    ))
    survivor_exact_y = float(meta["survivor_y_initial"]) + float(meta["survivor_vy"])*float(meta["time"])
    survivor2_exact_y = float(meta["survivor2_y_initial"]) + float(meta["survivor2_vy"])*float(meta["time"])
    survivor_error = np.concatenate((
        np.array([state[0, 2], state[1, 2] - survivor_exact_y, state[2, 2] - 0.5*math.pi,
                  state[3, 2], state[4, 2] - float(meta["survivor_vy"]), state[5, 2]]),
        np.array([state[0, 3], state[1, 3] - survivor2_exact_y, state[2, 3] - 0.5*math.pi,
                  state[3, 3], state[4, 3] - float(meta["survivor2_vy"]), state[5, 3]]),
    ))
    rate_error = dyn_rate[:2]
    ny = int(meta["ny"])
    ratio = (1.5/0.5)**(1.0/ny)
    radial_faces = 0.5*ratio**np.arange(ny + 1)
    cell_measure = 2.0*math.pi*0.5*(radial_faces[1:]**2 - radial_faces[:-1]**2)
    grid_mass = float(np.sum(dustdens*cell_measure))
    archived_mass = float(meta["deposited_mass"])
    mass_error = max(abs(grid_mass - 2.0), abs(archived_mass - grid_mass))
    activation = bool(
        meta["constant_radial_specialization"]
        and meta["production_transport_boundary"]
        and meta["production_downstream_exclusion"]
        and meta["production_optdepth_exclusion"]
        and meta["production_collision_mask"]
        and meta["production_collision_rate"]
        and int(meta["morton_overflow_max"]) == 0
    )
    passed = bool(
        activation
        and np.all(np.isfinite(state))
        and np.all(delay >= -2.0e-14)
        and np.all(delay <= dt*(1.0 + 2.0e-12))
        and np.all(absorption_time[2:] < 0.0)
        and np.max(np.abs(absorbed_state)) < 2.0e-14
        and np.max(np.abs(survivor_error)) < 2.0e-12
        and np.max(np.abs(rate_error)) < 2.0e-14
        and np.all(np.isfinite(dyn_rate[2:])) and np.all(dyn_rate[2:] > 0.0)
        and np.all(np.isfinite(dustdens)) and np.min(dustdens) >= 0.0
        and np.all(np.isfinite(optdepth)) and np.min(optdepth) >= 0.0 and np.max(optdepth) > 0.0
        and np.array_equal(col_active, np.array([0.0, 0.0, 1.0, 1.0]))
        and np.all(col_rate[:2] == 0.0) and np.all(col_dist[:2] == 0.0)
        and np.all(np.isfinite(col_rate[2:])) and np.all(col_rate[2:] > 0.0)
        and np.all(np.isfinite(col_dist[2:])) and np.all(col_dist[2:] >= 0.0)
        and mass_error < 2.0e-12
    )
    return {
        "case": "absorb_path_1d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized-exact-crossing",
        "activation": activation,
        "crossing_time_error_bound": dt,
        "deposited_active_mass": grid_mass,
        "errors": {
            "absorption_delay": norm_set(delay),
            "absorbed_sentinel": norm_set(absorbed_state),
            "survivor_state": norm_set(survivor_error),
            "inactive_dyn_rate": norm_set(rate_error),
            "active_mass": {"l1": mass_error, "l2": mass_error, "linf": mass_error},
            "optical_depth_finite": norm_set(np.minimum(optdepth, 0.0)),
            "collision_active_mask": norm_set(col_active - np.array([0.0, 0.0, 1.0, 1.0])),
            "inactive_collision_rate": norm_set(col_rate[:2]),
            "inactive_collision_radius": norm_set(col_dist[:2]),
        },
        "passed": passed,
    }
