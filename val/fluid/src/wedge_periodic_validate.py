#!/usr/bin/env python3

"""Validate periodic azimuthal operators on a non-2pi wedge"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
from pathlib import Path

import numpy as np

BUMP_SUPPORT = (0.45, 0.75)
MINIMUM_SEQUENCE_ORDER = 0.75


def wrap_x(value: np.ndarray, lower: float, period: float) -> np.ndarray:
    """Map coordinates into the configured half-open wedge"""

    return lower + np.mod(value - lower, period)


def periodic_bump(value: np.ndarray, lower: float, period: float) -> np.ndarray:
    """Evaluate the compact pulse after periodic wrapping"""

    coordinate = wrap_x(value, lower, period)
    center = 0.5*sum(BUMP_SUPPORT)
    half_width = 0.5*(BUMP_SUPPORT[1] - BUMP_SUPPORT[0])
    normalized = (coordinate - center)/half_width
    result = np.zeros_like(coordinate)
    active = np.abs(normalized) < 1.0
    result[active] = np.exp(1.0 - 1.0/(1.0 - normalized[active]**2))
    return result


def gauss_average(function, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Calculate independent 32-point cell averages"""

    nodes, weights = np.polynomial.legendre.leggauss(32)
    midpoint = 0.5*(lower + upper)
    radius = 0.5*(upper - lower)
    points = midpoint[:, None] + radius[:, None]*nodes
    return np.sum(function(points)*weights, axis=1)*0.5


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
    """Construct the exact wedge-periodic solution and assess one completed run"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    case = str(meta["case"])
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)
    time = float(meta["time"])
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    period = x_max - x_min

    x_face = np.linspace(x_min, x_max, nx + 1)
    x_i, x_o = x_face[:-1], x_face[1:]
    y_ratio = (float(meta["y_max"])/float(meta["y_min"]))**(1.0/ny)
    y_face = float(meta["y_min"])*y_ratio**np.arange(ny + 1)
    y_center = np.sqrt(y_face[:-1]*y_face[1:])
    vol_y = 0.5*(y_face[1:]**2 - y_face[:-1]**2)
    volume = vol_y[None, :, None]*np.ones((1, 1, nx))*period/nx

    density_initial = read_field(out_dir, "dustdens_initial", resolution, shape)
    density = read_field(out_dir, "dustdens_final", resolution, shape)
    mx = read_field(out_dir, "dustmomx_final", resolution, shape)
    my = read_field(out_dir, "dustmomy_final", resolution, shape)
    mz = read_field(out_dir, "dustmomz_final", resolution, shape)

    exact_density = np.empty(shape)
    if case == "x_wedge_transport_2d":
        rate = float(meta["transport_rate"])
        shift = rate*time
        profile = gauss_average(
            lambda x: periodic_bump(x - shift, x_min, period), x_i, x_o,
        )
        exact_density[:] = 1.0 + 0.5*profile[None, None, :]
        exact_mx = exact_density*y_center[None, :, None]**2*rate
        exact_my = 0.10*exact_density
        exact_mz = -0.05*exact_density
        wrapped_band = x_face[:-1] < x_min + BUMP_SUPPORT[1] + shift - x_max
        wrapped_signal = float(np.max(exact_density[:, :, wrapped_band] - 1.0)) \
            if np.any(wrapped_band) else 0.0
        activation_detail = {
            "period": period,
            "initial_support_upper": BUMP_SUPPORT[1],
            "unwrapped_final_support_upper": BUMP_SUPPORT[1] + shift,
            "outer_seam": x_max,
            "wrapped_signal": wrapped_signal,
        }
        activated = bool(
            BUMP_SUPPORT[1] < x_max
            and BUMP_SUPPORT[1] + shift > x_max
            and wrapped_signal > 0.05
        )
        density_limits = (5.0e-3, 6.0e-2)
        momentum_limit = 5.0e-3
    elif case == "x_wedge_diffusion_2d":
        diffusivity = float(meta["diffusivity"])
        phase = float(meta["mode_phase"])
        wave_number = 2.0*math.pi/period
        mode_avg = (np.sin(wave_number*(x_o - phase))
            - np.sin(wave_number*(x_i - phase))) / (wave_number*(x_o - x_i))
        decay = np.exp(-diffusivity*wave_number**2*time/y_center**2)
        exact_density[:] = 1.0 + 0.1*decay[None, :, None]*mode_avg[None, None, :]
        exact_mx = 0.70*exact_density
        exact_my = -0.15*exact_density
        exact_mz = 0.11*exact_density
        seam_gradient = abs(-0.1*wave_number*math.sin(wave_number*(x_min - phase)))
        activation_detail = {
            "period": period,
            "wave_number": wave_number,
            "seam_gradient": seam_gradient,
            "inner_decay": float(decay[0]),
            "outer_decay": float(decay[-1]),
        }
        activated = bool(
            abs(period - 2.0*math.pi) > 1.0
            and seam_gradient > 0.05
            and float(np.max(decay) - np.min(decay)) > 0.1
        )
        density_limits = (2.0e-3, 2.0e-2)
        momentum_limit = 2.0e-3
    else:
        raise ValueError(f"unknown wedge-periodic case: {case}")

    initial_mass = float(np.sum(volume*density_initial))
    final_mass = float(np.sum(volume*density))
    mass_relative = abs(final_mass - initial_mass)/initial_mass
    momentum_error = np.sqrt((mx - exact_mx)**2 + (my - exact_my)**2 + (mz - exact_mz)**2)
    finite = bool(
        np.all(np.isfinite(density)) and np.all(np.isfinite(mx))
        and np.all(np.isfinite(my)) and np.all(np.isfinite(mz))
    )
    nonnegative = bool(np.min(density) >= -2.0e-13)

    return {
        "case": case,
        "resolution": resolution,
        "reference_class": "operator-specialized-exact-periodic",
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
                "name": "momentum_vector_l1_accuracy",
                "field": "momentum_vector",
                "norm": "l1",
                "maximum_final_error": momentum_limit,
                "minimum_final_order": MINIMUM_SEQUENCE_ORDER,
                "require_monotonic": True,
            },
            {
                "name": "periodic_mass_conservation",
                "field": "mass_relative",
                "norm": "linf",
                "maximum_final_error": 2.0e-11,
            },
        ],
        "activation": activated,
        "activation_detail": activation_detail,
        "state_finite": finite,
        "minimum_density": float(np.min(density)),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "mass_relative_change": mass_relative,
        "errors": {
            "density": norm_set(density - exact_density, volume),
            "momentum_vector": norm_set(momentum_error, volume),
            "mass_relative": {
                "l1": mass_relative,
                "l2": mass_relative,
                "linf": mass_relative,
            },
        },
        "passed": bool(activated and finite and nonnegative and mass_relative <= 2.0e-11),
    }
