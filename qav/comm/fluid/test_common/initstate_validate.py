#!/usr/bin/env python3

"""Independently reconstruct the complete production fluid initialization state"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


SIGMA_0 = 1.3
METAL_Z = 0.017
ASPR_0 = 0.12
IDX_P = -0.8
IDX_Q = -0.4
STOKES_0 = 0.03
ALPHA = 0.004
SCHMIDT_Z = 2.0
R_0 = 1.0


def norms(error: np.ndarray) -> dict[str, float]:
    """Return unweighted diagnostics for one initialization field"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def read_values(out_dir: Path, name: str, resolution: int, count: int) -> np.ndarray:
    """Load one binary field and reject stale output with the wrong size"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{name} contains {values.size} values; expected {count}")
    return values


def convolved_profile(
    radial_min: float, domain_y_min: float, radial_max: float, count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the host convolution from its documented quadrature rather than GPU output"""

    radius = np.linspace(radial_min, radial_max, count + 1)
    dr = (radial_max - radial_min)/count
    smooth = 0.05*R_0
    source_min = domain_y_min + 2.0*smooth
    source_max = radial_max - 2.0*smooth
    width = 0.5*smooth
    normalization = 1.0/(math.sqrt(2.0*math.pi)*width)
    profile = np.zeros(count + 1)
    for source_radius in radius:
        if source_radius < source_min or source_radius > source_max:
            continue
        sigma_d = METAL_Z*SIGMA_0*(source_radius/R_0)**IDX_P
        offset = radius - source_radius
        profile += sigma_d*normalization*np.exp(-0.5*(offset/width)**2)*dr
    return radius, profile


def interpolate_profile(radius: np.ndarray, axis: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Apply the production piecewise-linear interpolation, including its endpoint convention"""

    result = np.zeros_like(radius)
    active = (radius >= axis[0]) & (radius <= axis[-1])
    location = (radius[active] - axis[0])/(axis[1] - axis[0])
    lower = np.floor(location).astype(int)
    lower = np.minimum(lower, profile.size - 2)
    fraction = location - lower
    result[active] = (1.0 - fraction)*profile[lower] + fraction*profile[lower + 1]
    return result


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare convolved density, gas targets, and all initialized velocity components"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    shape = (nz, ny, nx)
    count = nx*ny*nz

    ratio = (y_max/y_min)**(1.0/ny)
    y = y_min*ratio**(np.arange(ny) + 0.5)
    if nz == 1:
        z = np.array([0.5*math.pi])
        radial_min = y_min
    else:
        z = z_min + (np.arange(nz) + 0.5)*(z_max - z_min)/nz
        radial_min = y_min*min(math.sin(z_min), math.sin(z_max))
    Y = y[None, :, None]
    ZETA = z[:, None, None]
    R = np.broadcast_to(Y*np.sin(ZETA), shape)
    Z = np.broadcast_to(Y*np.cos(ZETA), shape)

    profile_axis, expected_profile = convolved_profile(radial_min, y_min, y_max, ny)
    recorded_profile = read_values(out_dir, "initdens", resolution, ny + 1)
    profile_error = recorded_profile - expected_profile
    sigma_d = interpolate_profile(R, profile_axis, expected_profile)

    h_g = ASPR_0*(R/R_0)**(0.5*(IDX_Q + 1.0))
    omega = R**-1.5
    v_k = R*omega
    spherical_radius = np.sqrt(R*R + Z*Z)
    stratification = np.exp((R/spherical_radius - 1.0)/(h_g*h_g))
    sigma_g = SIGMA_0*(R/R_0)**IDX_P
    rhog = sigma_g/(math.sqrt(2.0*math.pi)*h_g*R)*stratification
    stokes = STOKES_0/(R/R_0)**IDX_P/stratification
    eta = -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g
                + IDX_Q*(1.0 - R/spherical_radius))
    vx_g = v_k*np.sqrt(np.maximum(1.0 - 2.0*eta, 0.0))
    nu = ALPHA*h_g*h_g*R*R*omega
    if nz == 1:
        vR_g = -3.0*nu*((IDX_Q + 1.5) + IDX_P + 0.5)/R
        base_density = sigma_d
    else:
        cyl_fraction = R/spherical_radius
        strat_log = (cyl_fraction - 1.0)/(h_g*h_g)
        grad_strat_R = cyl_fraction*(1.0 - cyl_fraction*cyl_fraction)/(h_g*h_g) \
            - (IDX_Q + 1.0)*strat_log
        grad_strat_Z = -cyl_fraction*(1.0 - cyl_fraction*cyl_fraction)/(h_g*h_g)
        grad_rhog_R = IDX_P - 0.5*(IDX_Q + 3.0) + grad_strat_R
        grad_rhog_Z = grad_strat_Z
        term_R = 3.0*nu*((IDX_Q + 1.5) + grad_rhog_R + 0.5)
        term_Z = IDX_Q*nu*(1.0 + grad_rhog_Z)
        vR_g = -(term_R - term_Z)/R
        H_d = h_g*R*np.sqrt((ALPHA/SCHMIDT_Z)/(STOKES_0/(R/R_0)**IDX_P))
        base_density = sigma_d*np.exp(-0.5*Z*Z/(H_d*H_d))/(math.sqrt(2.0*math.pi)*H_d)

    density = read_values(out_dir, "dustdens", resolution, count).reshape(shape)
    # production intentionally applies one deterministic random multiplier per azimuth; infer that nuisance factor once and
    # then require the complete radial-polar field to follow the independently reconstructed physical profile
    multiplier = np.zeros(nx)
    noise_spread = np.zeros(nx)
    for ix in range(nx):
        active = base_density[:, :, ix] > 1.0e-250
        ratios = density[:, :, ix][active]/base_density[:, :, ix][active]
        multiplier[ix] = float(np.median(ratios))
        noise_spread[ix] = float(np.max(np.abs(ratios - multiplier[ix])))
    expected_density = base_density*multiplier[None, None, :]

    diagnostic = read_values(out_dir, "diagnostic", resolution, 7*count).reshape(7, *shape)
    expected_diagnostic = np.stack((h_g, stratification, rhog, stokes, eta, vR_g, vx_g))
    diagnostic_error = diagnostic - expected_diagnostic

    vR = (vR_g + 2.0*stokes*(vx_g - v_k))/(1.0 + stokes*stokes)
    vx = vx_g - 0.5*stokes*vR
    vz_diff = np.zeros(shape)
    if nz > 1:
        dz = (z_max - z_min)/nz
        y_grid = np.broadcast_to(Y, shape)
        gradient = np.empty(shape)
        gradient[0] = (expected_density[1] - expected_density[0])/dz
        gradient[-1] = (expected_density[-1] - expected_density[-2])/dz
        if nz > 2:
            gradient[1:-1] = (expected_density[2:] - expected_density[:-2])/(2.0*dz)
        active = expected_density > 0.0
        vz_diff[active] = (
            (nu[active]/SCHMIDT_Z)*gradient[active]
            /(y_grid[active]*expected_density[active])
        )
    expected_vx = R*vx
    expected_vy = vR*np.sin(ZETA)
    expected_vz = np.zeros(shape) if nz == 1 else Y*(vR*np.cos(ZETA) + vz_diff)
    velocity = np.stack((
        read_values(out_dir, "dustvelx", resolution, count).reshape(shape),
        read_values(out_dir, "dustvely", resolution, count).reshape(shape),
        read_values(out_dir, "dustvelz", resolution, count).reshape(shape),
    ))
    velocity_error = velocity - np.stack((expected_vx, expected_vy, expected_vz))
    density_error = density - expected_density

    finite = bool(
        np.all(np.isfinite(density)) and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(diagnostic))
    )
    passed = bool(
        finite
        and np.max(np.abs(profile_error)) < 5.0e-13
        and np.max(noise_spread) < 5.0e-12
        and np.max(np.abs(density_error)) < 5.0e-12
        and np.max(np.abs(diagnostic_error)) < 5.0e-12
        and np.max(np.abs(velocity_error)) < 2.0e-10
    )
    return {
        "case": meta["case"],
        "resolution": resolution,
        "primary_field": "velocity",
        "reference_class": "production-initialization",
        "terminal_mode": "verification",
        "azimuthal_noise_multipliers": multiplier.tolist(),
        "finite": finite,
        "errors": {
            "convolved_profile": norms(profile_error),
            "azimuthal_noise_consistency": norms(noise_spread),
            "density": norms(density_error),
            "gas_targets": norms(diagnostic_error),
            "velocity": norms(velocity_error),
        },
        "passed": passed,
    }
