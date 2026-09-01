#!/usr/bin/env python3

"""Validate imported-gas sampling, interpolation, temporal blending, and Stokes calibration"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


QUERY_COUNT = 4
X_MIN = -math.pi
X_MAX = math.pi
Y_MIN = 0.5
Y_MAX = 1.5
Z_MIN = 0.35
Z_MAX = math.pi - 0.35
SIGMA_0 = 1.0
R_0 = 1.0
S_0 = 1.0
ASPR_0 = 0.05
IDX_Q = -1.0
STOKES_0 = 0.2


def norm_set(error: np.ndarray) -> dict[str, float]:
    """Return the common absolute error norms"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def axis_stencil(location: float, count: int, reference: float, kind: str, ratio: float) -> tuple[int, int, float]:
    """Reconstruct one production cell-centred interpolation stencil independently"""

    base = int(math.floor(location))
    decimal = location - math.floor(location)
    if kind == "x":
        bounded = location < reference or location > count + reference - 1.0
        if decimal == reference:
            return base, 0, 0.0
        if not bounded:
            return (base, 1, decimal - reference) if decimal > reference \
                else (base, -1, reference - decimal)
        return (base, 1 - count, decimal - reference) if decimal >= reference \
            else (base, count - 1, reference - decimal)

    bounded = location <= reference or location >= count + reference - 1.0
    if bounded:
        return base, 0, 0.0
    if kind == "z":
        return (base, 1, decimal - reference) if decimal >= reference \
            else (base, -1, reference - decimal)
    if decimal >= reference:
        fraction = (ratio**(decimal - reference) - 1.0) / (ratio - 1.0)
        return base, 1, fraction
    fraction = (ratio**(decimal - reference) - 1.0) / (1.0/ratio - 1.0)
    return base, -1, fraction


def interpolate(field: np.ndarray, location: tuple[float, float, float], ratio: float, reference_y: float) -> float:
    """Evaluate the eight-cell scalar interpolation using independently reconstructed weights"""

    nz, ny, nx = field.shape
    ix, step_x, frac_x = axis_stencil(location[0], nx, 0.5, "x", ratio)
    iy, step_y, frac_y = axis_stencil(location[1], ny, reference_y, "y", ratio)
    iz, step_z, frac_z = axis_stencil(location[2], nz, 0.5, "z", ratio)
    value = 0.0
    for dz, weight_z in ((0, 1.0 - frac_z), (step_z, frac_z)):
        for dy, weight_y in ((0, 1.0 - frac_y), (step_y, frac_y)):
            for dx, weight_x in ((0, 1.0 - frac_x), (step_x, frac_x)):
                value += field[iz + dz, iy + dy, ix + dx]*weight_x*weight_y*weight_z
    return float(value)


def source_fields(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct the two imported frames from the test prescription rather than archived device values"""

    base = np.empty((4, nz, ny, nx), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                base[0, iz, iy, ix] = 2.0 + 0.03*ix + 0.04*iy + 0.02*iz
                base[1, iz, iy, ix] = -0.2 + 0.01*ix - 0.02*iy + 0.03*iz
                base[2, iz, iy, ix] = 0.4 - 0.04*ix + 0.015*iy + 0.025*iz
                base[3, iz, iy, ix] = -0.1 + 0.02*ix + 0.03*iy - 0.01*iz
    following = np.empty_like(base)
    following[0] = 1.3*base[0] + 0.2
    following[1] = 0.7*base[1] - 0.05
    following[2] = 1.2*base[2] + 0.08
    following[3] = 0.8*base[3] - 0.03
    return base, following


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare the GPU and host records with analytical and independently reconstructed references"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    cell_count = nx*ny*nz
    base, following = source_fields(nx, ny, nz)
    ratio = (Y_MAX/Y_MIN)**(1.0/ny)
    mesh_dim = 3.0
    reference_y = math.log(
        (mesh_dim/(mesh_dim + 1.0))*(ratio**(mesh_dim + 1.0) - 1.0)
        / (ratio**mesh_dim - 1.0)
    ) / math.log(ratio)
    query = np.array([
        [1.5, 2.0 + reference_y, 1.5],
        [1.75, 2.35 + reference_y, 1.7],
        [3.75, 3.8 + reference_y, 2.2],
        [0.25, 0.1, 0.2],
    ])

    spatial = np.fromfile(out_dir/f"spatial_N{resolution}.dat", dtype=np.float64).reshape(5, QUERY_COUNT)
    expected_spatial = np.empty_like(spatial)
    for idx, location in enumerate(query):
        for field in range(4):
            expected_spatial[field, idx] = interpolate(base[field], tuple(location), ratio, reference_y)
        y = Y_MIN*ratio**location[1]
        z = Z_MIN + location[2]*(Z_MAX - Z_MIN)/nz
        cyl_radius = y*math.sin(z)
        h_g = ASPR_0*(cyl_radius/R_0)**(0.5*(IDX_Q + 1.0))
        H_g = h_g*cyl_radius
        H_g0 = ASPR_0*R_0
        rhog_0 = SIGMA_0/(math.sqrt(2.0*math.pi)*H_g0)
        expected_spatial[4, idx] = STOKES_0*(S_0/S_0)*rhog_0*H_g0 \
            / (expected_spatial[0, idx]*H_g)

    temporal_25 = np.fromfile(
        out_dir/f"temporal_25_N{resolution}.dat", dtype=np.float64
    ).reshape(4, cell_count).reshape(base.shape)
    temporal_70 = np.fromfile(
        out_dir/f"temporal_70_N{resolution}.dat", dtype=np.float64
    ).reshape(4, cell_count).reshape(base.shape)
    expected_25 = 0.75*base + 0.25*following
    expected_70 = 0.30*base + 0.70*following

    anchor = np.fromfile(out_dir/f"anchor_N{resolution}.dat", dtype=np.float64)
    H_g0 = ASPR_0*R_0
    rhog_0 = SIGMA_0/(math.sqrt(2.0*math.pi)*H_g0)
    expected_anchor = np.array([rhog_0, STOKES_0, 0.1*rhog_0, 10.0*STOKES_0])
    checks = np.fromfile(out_dir/f"profile_checks_N{resolution}.dat", dtype=np.float64)

    spatial_error = spatial - expected_spatial
    temporal_25_error = temporal_25 - expected_25
    temporal_70_error = temporal_70 - expected_70
    anchor_error = anchor - expected_anchor
    check_error = checks - 1.0
    errors = {
        "spatial_interpolation": norm_set(spatial_error[:4]),
        "local_stokes": norm_set(spatial_error[4]),
        "temporal_interpolation_25": norm_set(temporal_25_error),
        "temporal_interpolation_70": norm_set(temporal_70_error),
        "stokes_anchor_gap": norm_set(anchor_error),
        "profile_validation": norm_set(check_error),
    }
    passed = bool(
        meta["case"] == "import_3d"
        and meta["resolution"] == resolution
        and (nx, ny, nz) == (4, 8, 4)
        and all(np.isfinite(values).all() for values in (
            spatial, temporal_25, temporal_70, anchor, checks,
        ))
        and max(values["linf"] for values in errors.values()) < 2.0e-12
    )
    return {
        "case": "import_3d",
        "resolution": resolution,
        "reference_class": "independent-grid-interpolation-and-analytical-stokes-anchor",
        "profile_checks": {
            "sparse_support": bool(checks[0] == 1.0),
            "zero_total_rejected": bool(checks[1] == 1.0),
            "negative_density_rejected": bool(checks[2] == 1.0),
            "negative_ratio_rejected": bool(checks[3] == 1.0),
            "nan_density_rejected": bool(checks[4] == 1.0),
            "infinite_ratio_rejected": bool(checks[5] == 1.0),
            "overflow_mass_rejected": bool(checks[6] == 1.0),
        },
        "errors": errors,
        "passed": passed,
    }
