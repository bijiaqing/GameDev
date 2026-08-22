#!/usr/bin/env python3

"""Validate positivity-controlled cyclic Crank-Nicolson diffusion"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def norm_set(error: np.ndarray) -> dict[str, float]:
    """Return unweighted norms for the uniform periodic line"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def read_values(out_dir: Path, name: str, resolution: int, count: int) -> np.ndarray:
    """Load one binary line and reject stale output with the wrong length"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{name} contains {values.size} values; expected {count}")
    return values


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare automatic subcycling with discrete CN and an equivalent manual sequence"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx = int(meta["nx"])
    dt = float(meta["dt"])
    radius = float(meta["radius"])
    diffusivity = float(meta["diffusivity"])
    mode = int(meta["mode"])
    sub_count = int(meta["sub_count"])
    dx = 2.0*math.pi/nx
    x0 = np.arange(nx)*dx
    x1 = x0 + dx
    mode_average = (np.sin(mode*x1) - np.sin(mode*x0))/(mode*dx)

    initial = read_values(out_dir, "density_initial", resolution, nx)
    automatic = read_values(out_dir, "density_auto", resolution, nx)
    manual = read_values(out_dir, "density_manual", resolution, nx)
    minima = read_values(out_dir, "substep_minimum", resolution, sub_count)

    # use the exact eigenvalue of the second-difference operator, not the continuum m**2 eigenvalue
    decay_rate = 4.0*diffusivity*math.sin(0.5*mode*dx)**2/(radius*radius*dx*dx)
    dt_sub = dt/sub_count
    amplification = ((1.0 - 0.5*decay_rate*dt_sub)/(1.0 + 0.5*decay_rate*dt_sub))**sub_count
    discrete = 1.0 + 0.9*amplification*mode_average
    # the continuum mode is retained separately to measure refinement order of the full spatial discretization
    continuum = 1.0 + 0.9*math.exp(-diffusivity*mode*mode*dt/(radius*radius))*mode_average

    discrete_error = automatic - discrete
    manual_error = automatic - manual
    continuum_error = automatic - continuum
    activation = bool(meta["constant_diffusivity"] and meta["automatic_subcycling"] and sub_count > 1)
    passed = bool(
        activation
        and np.all(np.isfinite(automatic))
        and np.max(np.abs(discrete_error)) < 5.0e-11
        and np.max(np.abs(manual_error)) < 5.0e-11
        and np.min(minima) >= -5.0e-14
        and np.max(np.abs(initial - (1.0 + 0.9*mode_average))) < 5.0e-15
    )
    return {
        "case": "diffusion_poslimit",
        "resolution": resolution,
        "primary_field": "continuum_density",
        "convergence_field": "continuum_density",
        "minimum_order": 1.8,
        "reference_class": "discrete-algorithm",
        "sub_count": sub_count,
        "minimum_substep_density": float(np.min(minima)),
        "activation": activation,
        "errors": {
            "continuum_density": norm_set(continuum_error),
            "discrete_density": norm_set(discrete_error),
            "automatic_manual": norm_set(manual_error),
        },
        "passed": passed,
    }
