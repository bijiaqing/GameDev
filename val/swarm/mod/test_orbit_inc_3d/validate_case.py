#!/usr/bin/env python3

"""validate full spherical swarm transport against an exact inclined Kepler orbit"""

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


def rotate(xp: np.ndarray, yp: np.ndarray, node: float, inc: float, peri: float) -> np.ndarray:
    x1 = np.cos(peri) * xp - np.sin(peri) * yp
    y1 = np.sin(peri) * xp + np.cos(peri) * yp
    return np.array(
        (
            np.cos(node) * x1 - np.sin(node) * np.cos(inc) * y1,
            np.sin(node) * x1 + np.cos(node) * np.cos(inc) * y1,
            np.sin(inc) * y1,
        )
    )


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = json.loads((out_dir / f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    state = np.fromfile(out_dir / f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)
    mean_initial = np.fromfile(out_dir / f"mean_anomaly_N{resolution}.dat", dtype=np.float64)
    a, e = float(meta["semimajor"]), float(meta["eccentricity"])
    inc, node, peri = float(meta["inclination"]), float(meta["node"]), float(meta["periapsis"])
    E = solve_kepler(mean_initial + a**-1.5 * float(meta["time"]), e)
    denom = 1.0 - e * np.cos(E)
    pos = rotate(a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E), node, inc, peri)
    vel = rotate(
        -a * np.sin(E) * a**-1.5 / denom,
        a * np.sqrt(1.0 - e * e) * np.cos(E) * a**-1.5 / denom,
        node,
        inc,
        peri,
    )
    y = np.sqrt(np.sum(pos * pos, axis=0))
    x = np.arctan2(pos[1], pos[0])
    z = np.arccos(pos[2] / y)
    sinz, cosz, sinx, cosx = np.sin(z), np.cos(z), np.sin(x), np.cos(x)
    vy = vel[0] * sinz * cosx + vel[1] * sinz * sinx + vel[2] * cosz
    lx = y * sinz * (-vel[0] * sinx + vel[1] * cosx)
    lz = y * (vel[0] * cosz * cosx + vel[1] * cosz * sinx - vel[2] * sinz)
    wrap_x = (state[0] - x + math.pi) % (2.0 * math.pi) - math.pi
    state_error = np.concatenate(
        (wrap_x, state[1] - y, state[2] - z, state[3] - lx, state[4] - vy, state[5] - lz)
    )
    R = state[1] * np.sin(state[2])
    vphi = state[3] / R
    vtheta = state[5] / state[1]
    energy_error = 0.5 * (vphi * vphi + state[4] ** 2 + vtheta * vtheta) - 1.0 / state[1] + 0.5 / a
    activation = bool(
        meta["zero_drag_specialization"]
        and meta["zero_diffusivity_specialization"]
        and meta["transport_only_schedule"]
    )
    return {
        "case": "orbit_inc_3d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized",
        "convergence_field": "state",
        "minimum_order": 1.8,
        "activation": activation,
        "errors": {"state": norm_set(state_error), "energy": norm_set(energy_error)},
        "passed": bool(
            activation and np.all(np.isfinite(state)) and np.max(np.abs(state_error)) < 3.0e-2
        ),
    }
