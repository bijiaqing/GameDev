#!/usr/bin/env python3

"""Validate linear settling plus production vertical diffusion against its exact finite-step law"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def normal_cdf(values: np.ndarray, mean: float, deviation: float) -> np.ndarray:
    """Evaluate a Gaussian CDF without requiring SciPy on compute nodes"""

    scale = math.sqrt(2.0)*deviation
    return np.fromiter(
        (0.5*math.erfc(-(float(value) - mean)/scale) for value in values),
        dtype=np.float64,
        count=values.size,
    )


def ks_distance(samples: np.ndarray, mean: float, deviation: float) -> float:
    """Return the two-sided one-sample Kolmogorov-Smirnov distance"""

    ordered = np.sort(samples)
    cdf = normal_cdf(ordered, mean, deviation)
    count = ordered.size
    upper = np.arange(1, count + 1, dtype=np.float64)/count - cdf
    lower = cdf - np.arange(count, dtype=np.float64)/count
    return float(max(np.max(upper), np.max(lower)))


def norm_set(error: np.ndarray) -> dict[str, float]:
    """Return common absolute norms for one diagnostic vector"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Measure periodic azimuthal displacement on the principal interval"""

    return (angle + math.pi)%(2.0*math.pi) - math.pi


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare particle moments and CDF with the discrete Ornstein-Uhlenbeck recurrence"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    count = int(meta["np"])
    steps = int(meta["steps"])
    dt = float(meta["dt"])
    gamma = float(meta["gamma"])
    diffusivity = float(meta["diffusivity"])
    height_initial = float(meta["height_initial"])
    radius_initial = float(meta["radius_initial"])
    state = np.fromfile(out_dir/f"state_N{resolution}.dat", dtype=np.float64).reshape(6, count)

    radius = state[1]*np.sin(state[2])
    height = state[1]*np.cos(state[2])
    vx = state[3]/radius
    vz = state[5]/state[1]
    velocity_R = state[4]*np.sin(state[2]) + vz*np.cos(state[2])
    velocity_Z = state[4]*np.cos(state[2]) - vz*np.sin(state[2])
    x_initial = -math.pi + (np.arange(count) + 0.5)*2.0*math.pi/count

    factor = 1.0 - gamma*dt
    mean_exact = factor**steps*height_initial
    variance_exact = 2.0*diffusivity*dt*(1.0 - factor**(2*steps))/(1.0 - factor*factor)
    deviation_exact = math.sqrt(variance_exact)
    mean_error = float(np.mean(height) - mean_exact)
    variance_error = float(np.var(height) - variance_exact)

    # use conservative six-standard-error moment gates and allocate one third of the familywise alpha to the CDF gate
    alpha_each = 0.01/3.0
    mean_limit = 6.0*deviation_exact/math.sqrt(count)
    variance_limit = 6.0*variance_exact*math.sqrt(2.0/(count - 1))
    ks_value = ks_distance(height, mean_exact, deviation_exact)
    ks_limit = math.sqrt(math.log(2.0/alpha_each)/(2.0*count))

    velocity_error = np.concatenate((
        velocity_R - float(meta["velocity_R"]),
        vx - float(meta["velocity_x"]),
        velocity_Z - float(meta["velocity_Z"]),
    ))
    inactive_error = np.concatenate((
        radius - radius_initial,
        wrap_angle(state[0] - x_initial),
    ))
    activation = bool(
        meta["local_linear_settling"]
        and meta["production_vertical_diffusion"]
        and meta["radial_diffusion_suppressed"]
        and meta["azimuthal_diffusion_suppressed"]
    )
    passed = bool(
        activation
        and np.all(np.isfinite(state))
        and abs(mean_error) <= mean_limit
        and abs(variance_error) <= variance_limit
        and ks_value <= ks_limit
        and np.max(np.abs(velocity_error)) < 2.0e-11
        and np.max(np.abs(inactive_error)) < 2.0e-12
    )
    return {
        "case": "settle_diffuse_3d",
        "resolution": resolution,
        "reference_class": "coefficient-specialized-statistical",
        "activation": activation,
        "ensemble_size": count,
        "familywise_alpha": 0.01,
        "statistics": {
            "height": {
                "mean_error": mean_error,
                "mean_limit": mean_limit,
                "variance_error": variance_error,
                "variance_limit": variance_limit,
                "ks_distance": ks_value,
                "ks_limit": ks_limit,
                "mean_exact_discrete": mean_exact,
                "variance_exact_discrete": variance_exact,
            },
        },
        "errors": {
            "velocity_invariance": norm_set(velocity_error),
            "inactive_coordinates": norm_set(inactive_error),
        },
        "passed": passed,
    }
