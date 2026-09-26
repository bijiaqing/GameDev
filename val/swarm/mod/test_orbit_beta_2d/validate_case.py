#!/usr/bin/env python3

"""validate radiation-pressure trajectories against an exact reduced-gravity orbit"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def norm_set(error: np.ndarray) -> dict[str, float]:
    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
        "linf": float(np.max(np.abs(error))),
    }


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    anomaly = mean_anomaly.copy()
    for _ in range(30):
        anomaly -= (anomaly - eccentricity * np.sin(anomaly) - mean_anomaly) / (
            1.0 - eccentricity * np.cos(anomaly)
        )
    return anomaly


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = json.loads((out_dir / f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir / f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    mean_initial = np.fromfile(out_dir / f"mean_anomaly_N{resolution}.dat", dtype=np.float64)
    size = np.fromfile(out_dir / f"size_N{resolution}.dat", dtype=np.float64)
    a, e, beta = float(meta["semimajor"]), float(meta["eccentricity"]), float(meta["beta"])
    beta_values = beta / (size / float(meta["size_ref"]))
    mu = 1.0 - beta_values
    anomaly = solve_kepler(mean_initial + np.sqrt(mu / a**3) * float(meta["time"]), e)
    exact_x = np.arctan2(np.sqrt(1.0 - e**2) * np.sin(anomaly), np.cos(anomaly) - e)
    exact_y = a * (1.0 - e * np.cos(anomaly))
    exact_lx = np.sqrt(mu * a * (1.0 - e**2))
    exact_vy = np.sqrt(mu / a) * e * np.sin(anomaly) / (1.0 - e * np.cos(anomaly))
    wrap_x = (state[0] - exact_x + math.pi) % (2.0 * math.pi) - math.pi
    state_error = np.concatenate(
        (
            wrap_x,
            state[1] - exact_y,
            state[2] - 0.5 * math.pi,
            state[3] - exact_lx,
            state[4] - exact_vy,
            state[5],
        )
    )
    energy_error = (
        0.5 * ((state[3] / state[1]) ** 2 + state[4] ** 2) - mu / state[1] + mu / (2.0 * a)
    )
    activation = bool(
        meta["zero_drag_specialization"]
        and meta["zero_optical_depth"]
        and meta["unit_radiation_taper"]
        and np.all((beta_values > 0.0) & (beta_values < 1.0))
        and np.unique(beta_values).size == 2
    )
    return {
        "case": "orbit_beta_2d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized",
        "convergence_field": "state",
        "minimum_order": 1.8,
        "activation": activation,
        "errors": {
            "state": norm_set(state_error),
            "energy": norm_set(energy_error),
            "angular_momentum": norm_set(state[3] - exact_lx),
        },
        "passed": bool(
            activation and np.all(np.isfinite(state)) and np.max(np.abs(state_error)) < 2.0e-2
        ),
    }
