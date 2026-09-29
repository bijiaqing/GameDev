#!/usr/bin/env python3

"""validate multi-step constant-coefficient drag velocity and displacement"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def norm_set(error: np.ndarray) -> dict[str, float]:
    """return common absolute norms for one particle ensemble"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
        "linf": float(np.max(np.abs(error))),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """evaluate exact constant-drag velocity and its time-integrated radial path"""

    meta = json.loads((out_dir / f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir / f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    stopping = np.fromfile(out_dir / f"stopping_time_N{resolution}.dat", dtype=np.float64)
    time = float(meta["time"])
    y0 = float(meta["y_initial"])
    velocity0 = float(meta["vy_initial"])
    gas_velocity = float(meta["gas_velocity"])
    force = float(meta["force"])
    # solve dv/dt=-(v-v_g)/t_s+F and integrate the same solution once to obtain y(t)
    equilibrium = gas_velocity + force * stopping
    decay = np.exp(-time / stopping)
    exact_velocity = equilibrium + (velocity0 - equilibrium) * decay
    exact_position = y0 + equilibrium * time + stopping * (velocity0 - equilibrium) * (1.0 - decay)
    position_error = state[1] - exact_position
    velocity_error = state[4] - exact_velocity
    # a radial-only test also verifies that the inactive azimuthal and polar components stay fixed
    inactive_error = np.concatenate((state[0], state[2] - 0.5 * math.pi, state[3], state[5]))
    activation = bool(meta["constant_drag_specialization"])
    return {
        "case": "drag_path_1d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized",
        "convergence_field": "position",
        "minimum_order": 1.8,
        "activation": activation,
        "stopping_times": stopping.tolist(),
        "errors": {
            "position": norm_set(position_error),
            "velocity": norm_set(velocity_error),
            "inactive": norm_set(inactive_error),
        },
        "passed": bool(
            activation
            and np.all(np.isfinite(state))
            and np.max(np.abs(position_error)) < 1.0e-2
            and np.max(np.abs(velocity_error)) < 2.0e-12
            and np.max(np.abs(inactive_error)) < 2.0e-14
        ),
    }
