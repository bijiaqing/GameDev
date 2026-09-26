#!/usr/bin/env python3

"""validate non-circular drag-free swarm trajectories against Kepler's equation"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def norm_set(error: np.ndarray) -> dict[str, float]:
    """return common absolute norms for one flattened phase-space error"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
        "linf": float(np.max(np.abs(error))),
    }


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    """solve Kepler's equation independently of the GPU trajectory integrator"""

    anomaly = mean_anomaly.copy()
    for _ in range(30):
        anomaly -= (anomaly - eccentricity * np.sin(anomaly) - mean_anomaly) / (
            1.0 - eccentricity * np.cos(anomaly)
        )
    return anomaly


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """measure azimuthal error across the periodic seam on the principal interval"""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def analyze(out_dir: Path, resolution: int) -> dict:
    """compare the numerical phase-space state and invariants with an exact eccentric orbit"""

    meta = json.loads((out_dir / f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir / f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    mean_initial = np.fromfile(out_dir / f"mean_anomaly_N{resolution}.dat", dtype=np.float64)
    semimajor = float(meta["semimajor"])
    eccentricity = float(meta["eccentricity"])
    time = float(meta["time"])
    # advance mean anomaly analytically, then convert to the production spherical-coordinate state
    mean_motion = semimajor**-1.5
    mean_final = mean_initial + mean_motion * time
    anomaly = solve_kepler(mean_final, eccentricity)
    exact_x = np.arctan2(
        np.sqrt(1.0 - eccentricity**2) * np.sin(anomaly), np.cos(anomaly) - eccentricity
    )
    exact_y = semimajor * (1.0 - eccentricity * np.cos(anomaly))
    exact_lx = math.sqrt(semimajor * (1.0 - eccentricity**2)) * np.ones(count)
    exact_vy = (
        semimajor**-0.5 * eccentricity * np.sin(anomaly) / (1.0 - eccentricity * np.cos(anomaly))
    )

    state_error = np.concatenate(
        (
            wrap_angle(state[0] - exact_x),
            state[1] - exact_y,
            state[2] - 0.5 * math.pi,
            state[3] - exact_lx,
            state[4] - exact_vy,
            state[5],
        )
    )
    # energy and angular momentum expose compensating position-velocity errors a state norm hides
    energy = 0.5 * ((state[3] / state[1]) ** 2 + state[4] ** 2) - 1.0 / state[1]
    exact_energy = -0.5 / semimajor
    energy_error = energy - exact_energy
    angular_error = state[3] - exact_lx
    activation = bool(meta["zero_drag_specialization"])
    return {
        "case": "orbit_ecc_2d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized",
        "convergence_field": "state",
        "minimum_order": 1.8,
        "activation": activation,
        "errors": {
            "state": norm_set(state_error),
            "energy": norm_set(energy_error),
            "angular_momentum": norm_set(angular_error),
        },
        "passed": bool(
            activation and np.all(np.isfinite(state)) and np.max(np.abs(state_error)) < 2.0e-2
        ),
    }
