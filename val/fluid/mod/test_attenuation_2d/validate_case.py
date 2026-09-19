#!/usr/bin/env python3

"""Validate 2D optical-depth reconstruction and frozen attenuated forcing"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
from pathlib import Path

import numpy as np

Y_MIN, Y_MAX = 0.5, 2.5
ASPR_0 = 0.5
STOKES_0 = 0.1
IDX_P = 2.0
BETA_0 = 1.0
KAPPA_0 = 1.0


def norm_set(error: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Return finite-volume weighted norms on the logarithmic radial grid"""

    weights = np.broadcast_to(weight, error.shape)
    weight_sum = float(np.sum(weights))
    return {
        "l1": float(np.sum(weights*np.abs(error))/weight_sum),
        "l2": float(np.sqrt(np.sum(weights*error*error)/weight_sum)),
        "linf": float(np.max(np.abs(error))),
    }


def read_values(out_dir: Path, name: str, resolution: int, shape: tuple[int, int]) -> np.ndarray:
    """Load one binary field and reject stale output with the wrong shape"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    if values.size != math.prod(shape):
        raise ValueError(f"{name} contains {values.size} values; expected {math.prod(shape)}")
    return values.reshape(shape)


def analyze(out_dir: Path, resolution: int) -> dict:
    """Reconstruct the discrete optical-depth scan and the one-step attenuated source response"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny = int(meta["nx"]), int(meta["ny"])
    dt = float(meta["dt"])
    power = float(meta["power"])
    shape = (ny, nx)
    ratio = (Y_MAX/Y_MIN)**(1.0/ny)
    faces = Y_MIN*ratio**np.arange(ny + 1)
    radius = np.sqrt(faces[:-1]*faces[1:])
    dr = np.diff(faces)
    rho_ext = radius**power
    sigma_d = math.sqrt(2.0*math.pi)*ASPR_0*radius*rho_ext
    increment = KAPPA_0*rho_ext*dr
    # match the production inclusive radial scan, then reconstruct the cell-center value used by source_update
    outer_tau = np.cumsum(increment)
    inner_tau = np.concatenate(([0.0], outer_tau[:-1]))
    center_fraction = 1.0/(math.sqrt(ratio) + 1.0)
    center_tau = inner_tau + center_fraction*(outer_tau - inner_tau)

    # use the continuum integral only for convergence order; exact same-grid checks use outer_tau above
    if abs(power + 1.0) < 1.0e-14:
        continuum_tau = np.log(faces[1:]/Y_MIN)
    else:
        continuum_tau = (faces[1:]**(power + 1.0) - Y_MIN**(power + 1.0))/(power + 1.0)

    density = read_values(out_dir, "density", resolution, shape)
    optical = read_values(out_dir, "optdepth", resolution, shape)
    lx = read_values(out_dir, "lx", resolution, shape)
    vy = read_values(out_dir, "vy", resolution, shape)
    lz = read_values(out_dir, "lz", resolution, shape)

    # independently reproduce the closed drag-force coefficients for the frozen state configured by the test
    omega = radius**-1.5
    stokes = STOKES_0/radius**IDX_P
    stopping = stokes/omega
    tau_drag = dt/stopping
    relax = -np.expm1(-tau_drag)
    decay = 1.0 - relax
    weight_new = stopping*(tau_drag - relax)/tau_drag
    weight_old = stopping*relax - weight_new
    lx_gas = omega*radius*radius
    lx_exact = relax*lx_gas
    beta = BETA_0*np.exp(-center_tau)
    grav_old = -(1.0 - beta)/radius**2
    grav_new = grav_old
    cent_new = lx_exact*lx_exact/radius**3
    vy_exact = weight_old*grav_old + weight_new*(grav_new + cent_new)

    expected_density = np.broadcast_to(sigma_d[:, None], shape)
    expected_optical = np.broadcast_to(outer_tau[:, None], shape)
    expected_lx = np.broadcast_to(lx_exact[:, None], shape)
    expected_vy = np.broadcast_to(vy_exact[:, None], shape)
    expected_lz = np.zeros(shape)
    continuum_outer = np.broadcast_to(continuum_tau[:, None], shape)
    discrete_error = optical - expected_optical
    source_components = np.stack((
        lx - expected_lx, vy - expected_vy, lz - expected_lz,
    ))
    source_error = source_components
    cell_weight = np.broadcast_to(
        (0.5*(faces[1:]**2 - faces[:-1]**2))[:, None], shape,
    )
    source_weight = np.broadcast_to(cell_weight, source_components.shape)
    activation = bool(meta["well_mixed_2d"] and float(meta["radiation_taper"]) == 1.0)
    passed = bool(
        activation
        and np.all(np.isfinite(source_error))
        and np.max(np.abs(density - expected_density)) < 5.0e-14
        and np.max(np.abs(discrete_error)) < 5.0e-12
        and np.max(np.abs(source_error)) < 2.0e-11
    )
    return {
        "case": "attenuation_2d",
        "resolution": resolution,
        "power": power,
        "primary_field": "continuum_optdepth",
        "convergence_field": "continuum_optdepth",
        "minimum_order": 1.8,
        "reference_class": "operator-specialized",
        "activation": activation,
        "maximum_center_optdepth": float(np.max(center_tau)),
        "errors": {
            "continuum_optdepth": norm_set(optical - continuum_outer, cell_weight),
            "discrete_optdepth": norm_set(discrete_error, cell_weight),
            "density": norm_set(density - expected_density, cell_weight),
            "source_response": norm_set(source_error, source_weight),
        },
        "passed": passed,
    }
