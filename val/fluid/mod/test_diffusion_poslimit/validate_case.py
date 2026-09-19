#!/usr/bin/env python3

"""Validate positivity-controlled cyclic Crank-Nicolson diffusion"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

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


def front_geometry(meta: dict) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return cell measures and the lower/upper unit-time CN coefficients"""

    nx, ny, nz = (int(meta[key]) for key in ("nx", "ny", "nz"))
    direction = str(meta["direction"])
    diffusivity = float(meta["diffusivity"])
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    z_min, z_max = float(meta["z_min"]), float(meta["z_max"])
    dy = (y_max/y_min)**(1.0/ny)
    y_face = y_min*dy**np.arange(ny + 1)
    y_cent = y_min*dy**(np.arange(ny) + 0.5)
    dz = (z_max - z_min)/nz
    z_face = z_min + dz*np.arange(nz + 1)
    z_cent = z_min + dz*(np.arange(nz) + 0.5)

    measures = np.empty(nx*ny*nz)
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    if direction == "x":
        dx = (float(meta["x_max"]) - float(meta["x_min"]))/nx
        for iz in range(nz):
            for iy in range(ny):
                dx_len = y_cent[iy]*math.sin(z_cent[iz])*dx
                measures[iy*nx + iz*nx*ny:(iy + 1)*nx + iz*nx*ny] = dx_len
                coefficient = 0.5*diffusivity/(dx_len*dx_len)
                lowers.append(np.full(nx, coefficient))
                uppers.append(np.full(nx, coefficient))
    elif direction == "y":
        mesh_dim = 2.0 + float(nz > 1)
        vol_y = (y_face[:-1]**mesh_dim)*(dy**mesh_dim - 1.0)/mesh_dim
        area_y = y_face**(mesh_dim - 1.0)
        dr_i = y_cent*(dy - 1.0)/dy
        dr_o = y_cent*(dy - 1.0)
        lower = 0.5*area_y[:-1]*diffusivity/(dr_i*vol_y)
        upper = 0.5*area_y[1:]*diffusivity/(dr_o*vol_y)
        lower[0] = 0.0
        upper[-1] = 0.0
        for iz in range(nz):
            for ix in range(nx):
                measures[ix + np.arange(ny)*nx + iz*nx*ny] = vol_y
                lowers.append(lower.copy())
                uppers.append(upper.copy())
    else:
        vol_z = np.cos(z_face[:-1]) - np.cos(z_face[1:])
        for iy in range(ny):
            measure = y_cent[iy]*vol_z
            lower = 0.5*np.sin(z_face[:-1])*diffusivity/(y_cent[iy]**2*dz*vol_z)
            upper = 0.5*np.sin(z_face[1:])*diffusivity/(y_cent[iy]**2*dz*vol_z)
            lower[0] = 0.0
            upper[-1] = 0.0
            for ix in range(nx):
                measures[ix + iy*nx + np.arange(nz)*nx*ny] = measure
                lowers.append(lower.copy())
                uppers.append(upper.copy())
    return measures, lowers, uppers


def line_indices(meta: dict) -> list[np.ndarray]:
    """Return flattened indices for every independent directional line"""

    nx, ny, nz = (int(meta[key]) for key in ("nx", "ny", "nz"))
    if meta["direction"] == "x":
        return [np.arange(nx) + iy*nx + iz*nx*ny for iz in range(nz) for iy in range(ny)]
    if meta["direction"] == "y":
        return [ix + np.arange(ny)*nx + iz*nx*ny for iz in range(nz) for ix in range(nx)]
    return [ix + iy*nx + np.arange(nz)*nx*ny for iy in range(ny) for ix in range(nx)]


def unlimited_trial(
    density: np.ndarray, meta: dict, lowers: list[np.ndarray], uppers: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Reproduce the unlimited CN density trial and its largest donor-outflow fraction"""

    dt = float(meta["dt"])
    diffusivity = float(meta["diffusivity"])
    direction = str(meta["direction"])
    trial = np.empty_like(density)
    maximum_outflow = 0.0
    measures, _, _ = front_geometry(meta)
    for indices, lower_unit, upper_unit in zip(line_indices(meta), lowers, uppers):
        old = density[indices]
        lower = dt*lower_unit
        upper = dt*upper_unit
        count = old.size
        matrix = np.zeros((count, count))
        rhs = np.empty(count)
        if direction == "x":
            coefficient = upper[0]
            for index in range(count):
                before = (index - 1) % count
                after = (index + 1) % count
                matrix[index, index] = 1.0 + 2.0*coefficient
                matrix[index, before] = -coefficient
                matrix[index, after] = -coefficient
                rhs[index] = (
                    coefficient*old[before] + (1.0 - 2.0*coefficient)*old[index]
                    + coefficient*old[after]
                )
        else:
            for index in range(count):
                matrix[index, index] = 1.0 + lower[index] + upper[index]
                if index > 0:
                    matrix[index, index - 1] = -lower[index]
                if index < count - 1:
                    matrix[index, index + 1] = -upper[index]
                previous = old[index - 1] if index > 0 else old[index]
                following = old[index + 1] if index < count - 1 else old[index]
                rhs[index] = (
                    lower[index]*previous + (1.0 - lower[index] - upper[index])*old[index]
                    + upper[index]*following
                )
        new = np.linalg.solve(matrix, rhs)
        trial[indices] = new

        if direction == "x":
            dx_len = measures[indices][0]
            flux = -0.5*diffusivity*((np.roll(old, -1) - old) + (np.roll(new, -1) - new))/dx_len
            incoming_face = np.roll(flux, 1)
        else:
            flux = np.zeros(count)
            flux[:-1] = -(upper[:-1]*measures[indices][:-1]/dt)*(
                (old[1:] - old[:-1]) + (new[1:] - new[:-1])
            )
            incoming_face = np.concatenate(([0.0], flux[:-1]))
        out_rate = np.maximum(flux, 0.0) + np.maximum(-incoming_face, 0.0)
        maximum_outflow = max(
            maximum_outflow,
            float(np.max(dt*out_rate/(old*measures[indices]))),
        )
    return trial, maximum_outflow


def analyze_front(out_dir: Path, resolution: int, meta: dict) -> dict:
    """Check conservative and convex properties of the activated donor limiter"""

    count = int(meta["nx"])*int(meta["ny"])*int(meta["nz"])
    density_initial = read_values(out_dir, "density_initial", resolution, count)
    density_final = read_values(out_dir, "density_final", resolution, count)
    momentum_initial = [
        read_values(out_dir, f"momentum_{axis}_initial", resolution, count)
        for axis in "xyz"
    ]
    momentum_final = [
        read_values(out_dir, f"momentum_{axis}_final", resolution, count)
        for axis in "xyz"
    ]
    measures, lowers, uppers = front_geometry(meta)
    unlimited, maximum_outflow = unlimited_trial(density_initial, meta, lowers, uppers)

    conserved = [density_initial, *momentum_initial]
    advanced = [density_final, *momentum_final]
    conservation = np.array([
        abs(np.sum(after*measures) - np.sum(before*measures))
        / max(abs(np.sum(before*measures)), np.sum(np.abs(before)*measures), 1.0)
        for before, after in zip(conserved, advanced)
    ])

    range_excess = []
    energy_growth = []
    for before, after in zip(momentum_initial, momentum_final):
        q_initial = before/density_initial
        q_final = after/density_final
        range_excess.append(max(
            float(np.min(q_initial) - np.min(q_final)),
            float(np.max(q_final) - np.max(q_initial)),
            0.0,
        ))
        energy_initial = 0.5*np.sum(density_initial*q_initial*q_initial*measures)
        energy_final = 0.5*np.sum(density_final*q_final*q_final*measures)
        energy_growth.append(max(float((energy_final - energy_initial)/energy_initial), 0.0))

    limiter_change = float(np.max(np.abs(density_final - unlimited)))
    activation = maximum_outflow > 0.9 and limiter_change > 1.0e-10
    finite = all(np.all(np.isfinite(values)) for values in advanced)
    passed = bool(
        finite
        and np.min(density_final) >= -5.0e-13
        and np.max(conservation) < 5.0e-12
        and max(range_excess) < 5.0e-12
        and max(energy_growth) < 5.0e-12
        and activation
    )
    return {
        "case": "diffusion_poslimit",
        "variant": meta["direction"],
        "resolution": resolution,
        "primary_field": "conservation",
        "terminal_mode": "checks",
        "reference_class": "invariant-domain",
        "activation": activation,
        "maximum_raw_outflow_fraction": maximum_outflow,
        "limiter_density_change": limiter_change,
        "minimum_density": float(np.min(density_final)),
        "maximum_primitive_range_excess": max(range_excess),
        "maximum_quadratic_growth": max(energy_growth),
        "errors": {
            "conservation": norm_set(conservation),
            "primitive_range": norm_set(np.asarray(range_excess)),
            "quadratic_growth": norm_set(np.asarray(energy_growth)),
        },
        "passed": passed,
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare automatic subcycling with discrete CN and an equivalent manual sequence"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    if meta.get("mode") == "donor_front":
        return analyze_front(out_dir, resolution, meta)
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
