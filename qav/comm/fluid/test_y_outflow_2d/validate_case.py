#!/usr/bin/env python3

"""Validate radial outflow against the exact geometric characteristic solution"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

Y_MIN, Y_MAX = 0.5, 2.5
OUTFLOW_SPEED = 0.2
SUPPORT_MIN, SUPPORT_MAX = 1.7, 2.5


def compact_bump(value: np.ndarray) -> np.ndarray:
    """Evaluate the smooth initial density profile"""

    center = 0.5*(SUPPORT_MIN + SUPPORT_MAX)
    half_width = 0.5*(SUPPORT_MAX - SUPPORT_MIN)
    coordinate = (value - center)/half_width
    result = np.zeros_like(value)
    active = np.abs(coordinate) < 1.0
    result[active] = np.exp(1.0 - 1.0/(1.0 - coordinate[active]**2))
    return result


def gauss_integral(function, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Integrate independently over every radial cell"""

    nodes, weights = np.polynomial.legendre.leggauss(32)
    midpoint = 0.5*(lower + upper)
    radius = 0.5*(upper - lower)
    points = midpoint[:, None] + radius[:, None]*nodes
    return radius*np.sum(function(points)*weights, axis=1)


def read_field(out_dir: Path, name: str, resolution: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Load one x-fastest production field"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    if values.size != math.prod(shape):
        raise ValueError(f"{name} contains {values.size} values; expected {math.prod(shape)}")
    return values.reshape(shape)


def norm_set(error: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    """Return finite-volume weighted absolute norms"""

    weights = np.broadcast_to(volume, error.shape)
    total = float(np.sum(weights))
    return {
        "l1": float(np.sum(weights*np.abs(error))/total),
        "l2": float(np.sqrt(np.sum(weights*error*error)/total)),
        "linf": float(np.max(np.abs(error))),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """Construct exact cell averages and compare density, momentum, and escaped mass"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)
    dimension = 2 if nz == 1 else 3
    time = float(meta["time"])

    ratio = (Y_MAX/Y_MIN)**(1.0/ny)
    faces = Y_MIN*ratio**np.arange(ny + 1)
    lower, upper = faces[:-1], faces[1:]
    radial_volume = (upper**dimension - lower**dimension)/dimension
    if nz == 1:
        polar_volume = np.ones(1)
    else:
        polar_faces = np.linspace(0.35, math.pi - 0.35, nz + 1)
        polar_volume = np.cos(polar_faces[:-1]) - np.cos(polar_faces[1:])
    volume = polar_volume[:, None, None]*radial_volume[None, :, None]*np.ones((1, 1, nx))

    shift = OUTFLOW_SPEED*time
    exact_integral = gauss_integral(
        lambda radius: compact_bump(radius - shift)*(radius - shift)**(dimension - 1),
        lower,
        upper,
    )
    exact_line = exact_integral/radial_volume
    exact_density = np.broadcast_to(exact_line[None, :, None], shape)

    density = read_field(out_dir, "dustdens_final", resolution, shape)
    density_initial = read_field(out_dir, "dustdens_initial", resolution, shape)
    mx = read_field(out_dir, "dustmomx_final", resolution, shape)
    my = read_field(out_dir, "dustmomy_final", resolution, shape)
    mz = read_field(out_dir, "dustmomz_final", resolution, shape)

    initial_mass = float(np.sum(volume*density_initial))
    numerical_mass = float(np.sum(volume*density))
    analytical_mass = float(np.sum(volume*exact_density))
    mass_error = abs(numerical_mass - analytical_mass)/initial_mass
    escaped_exact = (initial_mass - analytical_mass)/initial_mass
    escaped_numerical = (initial_mass - numerical_mass)/initial_mass
    momentum_error = my - OUTFLOW_SPEED*density
    transverse_error = np.concatenate((mx.ravel(), mz.ravel()))
    activation = bool(
        meta["case"] == f"y_outflow_{dimension}d"
        and abs(time - 1.0) < 1.0e-13
        and escaped_exact > 0.05
    )
    return {
        "case": f"y_outflow_{dimension}d",
        "resolution": resolution,
        "reference_class": "operator-specialized-exact-characteristic",
        "primary_field": "density",
        "convergence_field": "density",
        "minimum_order": 0.75,
        "activation": activation,
        "initial_mass": initial_mass,
        "analytical_remaining_mass": analytical_mass,
        "numerical_remaining_mass": numerical_mass,
        "analytical_escaped_fraction": escaped_exact,
        "numerical_escaped_fraction": escaped_numerical,
        "mass_relative_change": abs(escaped_numerical),
        "errors": {
            "density": norm_set(density - exact_density, volume),
            "radial_momentum_closure": norm_set(momentum_error, volume),
            "transverse_momentum": norm_set(transverse_error, np.ones_like(transverse_error)),
            "remaining_mass_relative": {
                "l1": mass_error,
                "l2": mass_error,
                "linf": mass_error,
            },
        },
        "passed": bool(
            activation
            and np.all(np.isfinite(density))
            and np.min(density) >= -2.0e-13
            and mass_error < 3.0e-2
            and np.max(np.abs(momentum_error)) < 2.0e-11
            and np.max(np.abs(transverse_error)) < 2.0e-13
            and np.max(np.abs(density - exact_density)) < 1.5e-1
        ),
    }
