#!/usr/bin/env python3

"""Validate the exact radial metric in fixed-polar-resolution advection"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SUPPORT = (0.70, 1.00)


def compact_bump(value: np.ndarray) -> np.ndarray:
    """Evaluate the smooth compact profile transported in q=rho*sin(z)"""

    center = 0.5*sum(SUPPORT)
    half_width = 0.5*(SUPPORT[1] - SUPPORT[0])
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


def scalar_norm(value: float) -> dict[str, float]:
    """Expose one scalar diagnostic through the common error-record interface"""

    return {"l1": value, "l2": value, "linf": value}


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare every radial shell with its exact metric-dependent translation"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)

    y_ratio = (float(meta["y_max"])/float(meta["y_min"]))**(1.0/ny)
    y_face = float(meta["y_min"])*y_ratio**np.arange(ny + 1)
    y_center = np.sqrt(y_face[:-1]*y_face[1:])
    vol_y = (y_face[1:]**3 - y_face[:-1]**3)/3.0
    area_z = 0.5*(y_face[1:]**2 - y_face[:-1]**2)
    radial_metric = y_center*area_z/vol_y
    angular_rate = float(meta["polar_rate"])*radial_metric

    z_face = np.linspace(float(meta["z_min"]), float(meta["z_max"]), nz + 1)
    z_i, z_o = z_face[:-1], z_face[1:]
    vol_z = np.cos(z_i) - np.cos(z_o)
    dx = (float(meta["x_max"]) - float(meta["x_min"]))/nx
    volume = dx*vol_z[:, None, None]*vol_y[None, :, None]*np.ones((1, 1, nx))

    density = read_field(out_dir, "dustdens_final", resolution, shape)
    density_initial = read_field(out_dir, "dustdens_initial", resolution, shape)
    mx = read_field(out_dir, "dustmomx_final", resolution, shape)
    my = read_field(out_dir, "dustmomy_final", resolution, shape)
    mz = read_field(out_dir, "dustmomz_final", resolution, shape)

    exact_density = np.empty(shape)
    exact_mx = np.empty(shape)
    exact_my = np.empty(shape)
    exact_mz = np.empty(shape)
    for iy in range(ny):
        shift = angular_rate[iy]*float(meta["time"])
        q_int = gauss_integral(lambda z: compact_bump(z - shift), z_i, z_o)
        shell_density = q_int/vol_z
        shell_lz = float(meta["polar_rate"])*y_center[iy]**2
        exact_density[:, iy, :] = shell_density[:, None]
        exact_mx[:, iy, :] = 0.70*shell_density[:, None]
        exact_my[:, iy, :] = 0.05*shell_density[:, None]
        exact_mz[:, iy, :] = shell_lz*shell_density[:, None]

    # a cell-center approximation would replace the exact radial metric by one; retain that deliberately wrong reference
    # so the coarsest radial grid must demonstrate that this test can distinguish the production factor
    center_q_int = gauss_integral(
        lambda z: compact_bump(z - float(meta["polar_rate"])*float(meta["time"])), z_i, z_o,
    )
    center_density = np.broadcast_to((center_q_int/vol_z)[:, None, None], shape)

    momentum_error = np.sqrt(
        (mx - exact_mx)**2 + (my - exact_my)**2 + (mz - exact_mz)**2
    )
    shell_spread = float(np.max(np.abs(density - density[:, :1, :])))
    initial_mass = float(np.sum(volume*density_initial))
    final_mass = float(np.sum(volume*density))
    mass_relative = abs(final_mass - initial_mass)/initial_mass
    metric_spread = float(np.max(radial_metric) - np.min(radial_metric))
    center_metric_gap = abs(float(meta["polar_rate"])*(1.0 - radial_metric[0])*float(meta["time"]))
    finite = bool(
        np.all(np.isfinite(density)) and np.all(np.isfinite(mx))
        and np.all(np.isfinite(my)) and np.all(np.isfinite(mz))
    )

    density_error = norm_set(density - exact_density, volume)
    center_density_error = norm_set(density - center_density, volume)
    momentum_norm = norm_set(momentum_error, volume)
    metric_discrimination = center_density_error["l1"]/max(density_error["l1"], np.finfo(float).tiny)
    return {
        "case": "z_metric_3d",
        "resolution": resolution,
        "reference_class": "operator-specialized-exact-characteristic",
        "primary_field": "density",
        "convergence_field": None,
        "terminal_mode": "fixed-N_Z metric sequence",
        "sequence_requirements": [
            {
                "name": "density_accuracy",
                "field": "density",
                "norm": "l1",
                "maximum_final_error": 5.0e-3,
            },
            {
                "name": "momentum_accuracy",
                "field": "momentum_vector",
                "norm": "l1",
                "maximum_final_error": 1.0e-2,
            },
            {
                "name": "full_3d_mass_conservation",
                "field": "mass_relative",
                "norm": "linf",
                "maximum_final_error": 2.0e-11,
            },
            {
                "name": "radial_shell_reproducibility",
                "field": "shell_density_spread",
                "norm": "linf",
                "maximum_final_error": 1.0e-6,
            },
        ],
        "activation": bool(
            nz == 128 and ny == resolution//8 and center_metric_gap > 1.0e-4
            and (resolution != 32 or metric_discrimination >= 2.0)
        ),
        "activation_detail": {
            "fixed_nz": nz,
            "varied_ny": ny,
            "radial_metric": float(radial_metric[0]),
            "radial_metric_spread": metric_spread,
            "center_metric_phase_gap": center_metric_gap,
            "center_metric_density_l1": center_density_error["l1"],
            "exact_metric_density_l1": density_error["l1"],
            "coarse_metric_discrimination": metric_discrimination,
            "minimum_exact_support": SUPPORT[0] + float(np.min(angular_rate))*float(meta["time"]),
            "maximum_exact_support": SUPPORT[1] + float(np.max(angular_rate))*float(meta["time"]),
        },
        "state_finite": finite,
        "minimum_density": float(np.min(density)),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "mass_relative_change": mass_relative,
        "mass_diagnostic_label": "mass rel",
        "errors": {
            "density": density_error,
            "center_metric_density": center_density_error,
            "momentum_vector": momentum_norm,
            "shell_density_spread": scalar_norm(shell_spread),
            "mass_relative": scalar_norm(mass_relative),
        },
        "passed": bool(
            finite and np.min(density) >= -2.0e-13
            and density_error["l1"] <= 5.0e-3
            and density_error["linf"] <= 7.0e-2
            and momentum_norm["l1"] <= 1.0e-2
            and shell_spread <= 1.0e-6
            and mass_relative <= 2.0e-11
            and metric_spread <= 2.0e-13
        ),
    }
