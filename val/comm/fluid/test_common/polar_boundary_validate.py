#!/usr/bin/env python3

"""Validate polar outflow and reflection-symmetric HALF_DISK transport"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUTFLOW_SUPPORT = (2.45, 2.75)
REFLECT_SUPPORT = (0.70, math.pi - 0.70)
MINIMUM_SEQUENCE_ORDER = 0.75


def compact_bump(value: np.ndarray, support: tuple[float, float]) -> np.ndarray:
    """Evaluate the smooth compact profile used by the test driver"""

    lower, upper = support
    center = 0.5*(lower + upper)
    half_width = 0.5*(upper - lower)
    coordinate = (value - center)/half_width
    result = np.zeros_like(value)
    active = np.abs(coordinate) < 1.0
    result[active] = np.exp(1.0 - 1.0/(1.0 - coordinate[active]**2))
    return result


def gauss_integral(function, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Integrate one function independently over every polar cell"""

    nodes, weights = np.polynomial.legendre.leggauss(32)
    midpoint = 0.5*(lower + upper)
    radius = 0.5*(upper - lower)
    points = midpoint[:, None] + radius[:, None]*nodes
    return radius*np.sum(function(points)*weights, axis=1)


def read_field(out_dir: Path, name: str, resolution: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Load one x-fastest field and reject stale output with the wrong size"""

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
    """Construct exact cell averages and assess the selected polar boundary"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    case = str(meta["case"])
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)
    time = float(meta["time"])

    y_ratio = (float(meta["y_max"])/float(meta["y_min"]))**(1.0/ny)
    y_face = float(meta["y_min"])*y_ratio**np.arange(ny + 1)
    y_center = np.sqrt(y_face[:-1]*y_face[1:])
    vol_y = (y_face[1:]**3 - y_face[:-1]**3)/3.0
    area_z = 0.5*(y_face[1:]**2 - y_face[:-1]**2)
    geom_z = area_z/vol_y

    z_face = np.linspace(float(meta["z_min"]), float(meta["z_max"]), nz + 1)
    z_i, z_o = z_face[:-1], z_face[1:]
    vol_z = np.cos(z_i) - np.cos(z_o)
    volume = vol_z[:, None, None]*vol_y[None, :, None]*np.ones((1, 1, nx))

    density = read_field(out_dir, "dustdens_final", resolution, shape)
    density_initial = read_field(out_dir, "dustdens_initial", resolution, shape)
    mx = read_field(out_dir, "dustmomx_final", resolution, shape)
    my = read_field(out_dir, "dustmomy_final", resolution, shape)
    mz = read_field(out_dir, "dustmomz_final", resolution, shape)

    exact_density = np.empty(shape)
    exact_mz = np.empty(shape)
    if case == "z_outflow_3d":
        rate = float(meta["outflow_rate"])
        shift = rate*time
        q_int = gauss_integral(
            lambda z: compact_bump(z - shift, OUTFLOW_SUPPORT), z_i, z_o,
        )
        shell_lz = y_center*rate/geom_z
        exact_density[:] = (q_int/vol_z)[:, None, None]
        exact_mz[:] = exact_density*shell_lz[None, :, None]
        activation_detail = {
            "support_initial_upper": OUTFLOW_SUPPORT[1],
            "support_final_upper": OUTFLOW_SUPPORT[1] + shift,
            "outer_boundary": float(meta["z_max"]),
        }
    elif case == "z_reflect_3d":
        rate = float(meta["reflect_rate"])
        compression = 1.0 - rate*time
        if compression <= 0.0:
            raise ValueError("reflection reference requires 1-a*t > 0")

        def initial_coordinate(z: np.ndarray) -> np.ndarray:
            return 0.5*math.pi - (0.5*math.pi - z)/compression

        q_int = gauss_integral(
            lambda z: compact_bump(initial_coordinate(z), REFLECT_SUPPORT)/compression,
            z_i, z_o,
        )
        exact_density[:] = (q_int/vol_z)[:, None, None]
        for iy in range(ny):
            lz_scale = y_center[iy]*rate/(geom_z[iy]*compression)
            mz_int = gauss_integral(
                lambda z: compact_bump(initial_coordinate(z), REFLECT_SUPPORT)
                / compression*lz_scale*(0.5*math.pi - z),
                z_i, z_o,
            )
            exact_mz[:, iy, :] = (mz_int/vol_z)[:, None]
        activation_detail = {
            "compression_factor": compression,
            "midplane_initial_q": float(compact_bump(np.array([0.5*math.pi]), REFLECT_SUPPORT)[0]),
            "analytical_midplane_speed": 0.0,
        }
    else:
        raise ValueError(f"unknown polar-boundary case: {case}")

    initial_mass = float(np.sum(volume*density_initial))
    numerical_mass = float(np.sum(volume*density))
    analytical_mass = float(np.sum(volume*exact_density))
    mass_error = abs(numerical_mass - analytical_mass)/initial_mass
    transverse = np.sqrt(mx*mx + my*my)
    finite = bool(
        np.all(np.isfinite(density)) and np.all(np.isfinite(mx))
        and np.all(np.isfinite(my)) and np.all(np.isfinite(mz))
    )
    nonnegative = bool(np.min(density) >= -2.0e-13)

    if case == "z_outflow_3d":
        escaped_exact = (initial_mass - analytical_mass)/initial_mass
        escaped_numerical = (initial_mass - numerical_mass)/initial_mass
        activated = bool(
            OUTFLOW_SUPPORT[1] < float(meta["z_max"])
            and OUTFLOW_SUPPORT[1] + float(meta["outflow_rate"])*time > float(meta["z_max"])
            and escaped_exact > 0.05
        )
        mass_diagnostic = abs(escaped_numerical)
        mass_label = "escaped frac"
        mass_limit = 5.0e-3
    else:
        escaped_exact = 0.0
        escaped_numerical = (initial_mass - numerical_mass)/initial_mass
        activated = bool(
            abs(float(meta["z_max"]) - 0.5*math.pi) < 1.0e-13
            and activation_detail["compression_factor"] < 0.9
            and activation_detail["midplane_initial_q"] > 0.1
        )
        mass_diagnostic = abs(escaped_numerical)
        mass_label = "mass rel"
        mass_limit = 2.0e-11

    density_limits = (4.0e-3, 5.0e-2)
    momentum_limit = 4.0e-3 if case == "z_outflow_3d" else 1.0e-2
    return {
        "case": case,
        "resolution": resolution,
        "reference_class": "operator-specialized-exact-characteristic",
        "primary_field": "density",
        "convergence_field": "density",
        "minimum_order": MINIMUM_SEQUENCE_ORDER,
        "sequence_requirements": [
            {
                "name": "density_l1_accuracy",
                "field": "density",
                "norm": "l1",
                "maximum_final_error": density_limits[0],
                "minimum_final_order": MINIMUM_SEQUENCE_ORDER,
                "require_monotonic": True,
            },
            {
                "name": "density_linf_accuracy",
                "field": "density",
                "norm": "linf",
                "maximum_final_error": density_limits[1],
                "require_monotonic": True,
            },
            {
                "name": "polar_momentum_l1_accuracy",
                "field": "polar_momentum",
                "norm": "l1",
                "maximum_final_error": momentum_limit,
                "minimum_final_order": MINIMUM_SEQUENCE_ORDER,
                "require_monotonic": True,
            },
            {
                "name": "boundary_mass_balance",
                "field": "remaining_mass_relative",
                "norm": "linf",
                "maximum_final_error": mass_limit,
                "require_monotonic": case == "z_outflow_3d",
            },
        ],
        "activation": activated,
        "activation_detail": activation_detail,
        "state_finite": finite,
        "minimum_density": float(np.min(density)),
        "initial_mass": initial_mass,
        "analytical_remaining_mass": analytical_mass,
        "numerical_remaining_mass": numerical_mass,
        "analytical_escaped_fraction": escaped_exact,
        "numerical_escaped_fraction": escaped_numerical,
        "mass_relative_change": mass_diagnostic,
        "mass_diagnostic_label": mass_label,
        "errors": {
            "density": norm_set(density - exact_density, volume),
            "polar_momentum": norm_set(mz - exact_mz, volume),
            "transverse_momentum": norm_set(transverse, volume),
            "remaining_mass_relative": {
                "l1": mass_error,
                "l2": mass_error,
                "linf": mass_error,
            },
        },
        "passed": bool(
            activated and finite and nonnegative
            and np.max(transverse) < 2.0e-13
        ),
    }
