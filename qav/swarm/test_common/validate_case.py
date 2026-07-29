#!/usr/bin/env python3

"""Validate raw swarm CUDA outputs against independent analytical results"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


X_MIN, X_MAX = -math.pi, math.pi
Y_MIN, Y_MAX = 0.5, 1.5
Z_MIN, Z_MAX = 0.35, math.pi - 0.35
NU = 2.0e-2
STOKES_0 = 0.2
BETA_0 = 0.2
C_LIGHT = 25.0
KAPPA_0 = 0.7
ASPR_0 = 0.05


def read_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_array(out_dir: Path, name: str, resolution: int, count: int) -> np.ndarray:
    path = out_dir / f"{name}_N{resolution}.dat"
    values = np.fromfile(path, dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{path} contains {values.size} values; expected {count}")
    return values


def norms(values: np.ndarray) -> dict[str, float]:
    return {
        "l1": float(np.mean(np.abs(values))),
        "l2": float(np.sqrt(np.mean(values * values))),
        "linf": float(np.max(np.abs(values))),
    }


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def load_state(out_dir: Path, resolution: int, nparticle: int) -> np.ndarray:
    return read_array(out_dir, "state", resolution, 6*nparticle).reshape(6, nparticle)


def radial_measures(ny: int, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = (Y_MAX / Y_MIN) ** (1.0 / ny)
    edges = Y_MIN * ratio ** np.arange(ny + 1)
    volume = (edges[1:]**dimension - edges[:-1]**dimension) / dimension
    return edges, volume, edges[:-1]*(ratio - 1.0)


def analyze_grid(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)
    density = read_array(out_dir, "dustdens", resolution, nx*ny*nz).reshape(shape)
    optical = read_array(out_dir, "optdepth", resolution, nx*ny*nz).reshape(shape)
    dimension = 2 if nz == 1 else 3
    edges, vol_y, dr = radial_measures(ny, dimension)
    vol_x = 2.0*math.pi / nx
    if nz == 1:
        vol_z = np.ones(1)
    else:
        zedges = np.linspace(Z_MIN, Z_MAX, nz + 1)
        vol_z = np.cos(zedges[:-1]) - np.cos(zedges[1:])

    cell_volume = vol_z[:, None, None]*vol_y[None, :, None]*vol_x
    mass = 1.0 / (nx*ny*nz)
    expected_density = np.broadcast_to(mass/cell_volume, shape)

    extinction = mass/cell_volume
    if nz == 1:
        ratio = (Y_MAX / Y_MIN) ** (1.0 / ny)
        dim = 2.0
        offset = math.log((dim/(dim + 1.0))*(ratio**(dim + 1.0) - 1.0)
                          / (ratio**dim - 1.0), ratio)
        radius = Y_MIN*ratio**(np.arange(ny) + offset)
        extinction = extinction/(math.sqrt(2.0*math.pi)*ASPR_0*radius[None, :, None])
    increment = KAPPA_0*extinction*dr[None, :, None]
    expected_optical = np.cumsum(np.broadcast_to(increment, shape), axis=1)
    mass_relative_change = float(np.sum(density*cell_volume) - 1.0)

    return {
        "case": meta["case"],
        "resolution": resolution,
        "errors": {
            "density": norms(density - expected_density),
            "optdepth": norms(optical - expected_optical),
        },
        "mass_relative_change": mass_relative_change,
        "passed": bool(
            np.max(np.abs(density - expected_density)) < 5.0e-12
            and np.max(np.abs(optical - expected_optical)) < 5.0e-11
            and abs(mass_relative_change) < 5.0e-12
        ),
    }


def analyze_orbit(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    state = load_state(out_dir, resolution, nparticle)
    initial_x = X_MIN + (np.arange(nparticle) + 0.5)*(X_MAX - X_MIN)/nparticle
    phase_error = wrap_angle(state[0] - initial_x)
    error = np.concatenate((phase_error, state[1] - 1.0, state[3] - 1.0, state[4]))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"state": norms(error)},
        "passed": bool(np.all(np.isfinite(state)) and np.max(np.abs(error)) < 0.5),
    }


def analyze_relaxation(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    state = load_state(out_dir, resolution, nparticle)
    species = read_array(out_dir, "species", resolution, 2*nparticle).reshape(2, nparticle)
    size = species[0]
    inv_ts = 1.0/(STOKES_0*size)
    lx_initial = 1.2
    lx_gas = 1.0
    beta = np.zeros_like(size)
    if meta["case"] in {"radiation_2d", "prdrag_2d"}:
        beta = BETA_0/size

    if meta["case"] != "prdrag_2d":
        expected = lx_gas + (lx_initial - lx_gas)*np.exp(-inv_ts*dt)
        lx_half = lx_gas + (lx_initial - lx_gas)*np.exp(-0.5*inv_ts*dt)
        force_y = -(1.0 - beta) + lx_half*lx_half
        expected_vy = force_y/inv_ts*(-np.expm1(-inv_ts*dt))
    else:
        rate_x = inv_ts + beta/C_LIGHT
        rate_y = inv_ts + 2.0*beta/C_LIGHT
        expected = np.exp(-rate_x*dt)*lx_initial + (-np.expm1(-rate_x*dt))*inv_ts*lx_gas/rate_x
        lx_half = (
            np.exp(-0.5*rate_x*dt)*lx_initial
            + (-np.expm1(-0.5*rate_x*dt))*inv_ts*lx_gas/rate_x
        )
        force_y = -(1.0 - beta) + lx_half*lx_half
        expected_vy = (-np.expm1(-rate_y*dt))*force_y/rate_y

    expected_y = 1.0 + 0.5*expected_vy*dt
    error = np.concatenate((state[3] - expected, state[4] - expected_vy, state[1] - expected_y))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"response": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze_diffusion(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    state = load_state(out_dir, resolution, nparticle)
    expected_variance = 2.0*NU*dt

    if meta["case"] == "diffusion_2d":
        samples = wrap_angle(state[0])
        mean_expected = 0.0
        velocity_x = state[4]*np.cos(state[0]) - (state[3]/state[1])*np.sin(state[0])
        velocity_y = state[4]*np.sin(state[0]) + (state[3]/state[1])*np.cos(state[0])
        velocity_error = np.concatenate((velocity_x - 0.2, velocity_y - 0.7))
        components = {"x": samples}
    else:
        radius = state[1]*np.sin(state[2])
        height = state[1]*np.cos(state[2])
        components = {"R": radius - 1.0, "Z": height}
        mean_expected = NU*dt

        vx = state[3]/radius
        vz = state[5]/state[1]
        vR = state[4]*np.sin(state[2]) + vz*np.cos(state[2])
        vZ = state[4]*np.cos(state[2]) - vz*np.sin(state[2])
        velocity_error = np.concatenate((vR - 0.2, vx - 0.7, vZ))

    statistics = {}
    passed = np.max(np.abs(velocity_error)) < 2.0e-12
    for name, sample in components.items():
        target_mean = mean_expected if name == "R" else 0.0
        mean_error = float(np.mean(sample) - target_mean)
        variance_error = float(np.var(sample) - expected_variance)
        mean_limit = 6.0*math.sqrt(expected_variance/nparticle)
        variance_limit = 6.0*expected_variance*math.sqrt(2.0/(nparticle - 1))
        statistics[name] = {
            "mean_error": mean_error,
            "mean_limit": mean_limit,
            "variance_error": variance_error,
            "variance_limit": variance_limit,
        }
        passed = passed and abs(mean_error) <= mean_limit and abs(variance_error) <= variance_limit

    return {
        "case": meta["case"], "resolution": resolution,
        "statistics": statistics,
        "errors": {"velocity": norms(velocity_error)},
        "passed": bool(passed),
    }


def ball_measure(dimension: int, radius: float, distance: float) -> float:
    if dimension == 2:
        full = math.pi*radius*radius
        cap = radius*radius*math.acos(distance/radius) - distance*math.sqrt(radius*radius - distance*distance)
    else:
        full = 4.0*math.pi*radius**3/3.0
        cap = math.pi*(radius - distance)**2*(2.0*radius + distance)/3.0
    return full*(1.0 - cap/full)


def analyze_collision(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    values = read_array(out_dir, "collision", resolution, 6)
    dimension = 2 if meta["case"] == "collision_2d" else 3
    radius = 0.2
    mass_i = math.pi/6.0
    mass_j = math.pi*8.0/6.0
    interior_measure = math.pi*radius**2 if dimension == 2 else 4.0*math.pi*radius**3/3.0
    polar_measure = interior_measure if dimension == 2 else ball_measure(dimension, radius, 0.25*radius)
    expected = np.array([
        interior_measure,
        ball_measure(dimension, radius, 0.25*radius),
        polar_measure,
        0.3*7.0,
        0.3*7.0*(mass_i + mass_j),
        0.3*7.0*mass_i*mass_j,
    ])
    error = values - expected
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"collision": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = read_meta(out_dir / f"meta_N{resolution}.txt")
    case = meta["case"]
    if case.startswith("grid_"):
        result = analyze_grid(out_dir, resolution, meta)
    elif case == "orbit_2d":
        result = analyze_orbit(out_dir, resolution, meta)
    elif case in {"drag_2d", "radiation_2d", "prdrag_2d"}:
        result = analyze_relaxation(out_dir, resolution, meta)
    elif case.startswith("diffusion_"):
        result = analyze_diffusion(out_dir, resolution, meta)
    elif case.startswith("collision_"):
        result = analyze_collision(out_dir, resolution, meta)
    else:
        raise ValueError(f"unknown swarm verification case: {case}")
    if not result["passed"]:
        raise AssertionError(f"{case} failed at resolution {resolution}: {result}")
    return result
