#!/usr/bin/env python3

"""Validate repeated periodic FARGO-advection and diffusion compositions"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def read_field(out_dir: Path, name: str, resolution: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Load one x-fastest field and reject stale output with an incompatible grid"""

    values = np.fromfile(out_dir/f"{name}_N{resolution}.dat", dtype=np.float64)
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{name} contains {values.size} values; expected {expected}")
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
    """Expose one scalar conservation diagnostic through the common error interface"""

    return {"l1": value, "l2": value, "linf": value}


def analyze(out_dir: Path, resolution: int) -> dict:
    """Compare the long ring with its exact translated-and-damped Fourier mode"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    shape = (nz, ny, nx)

    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    dx = (x_max - x_min)/nx
    x_i = x_min + np.arange(nx)*dx
    x_o = x_i + dx

    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    dy = (y_max/y_min)**(1.0/ny)
    y_face = y_min*dy**np.arange(ny + 1)
    y = np.sqrt(y_face[:-1]*y_face[1:])
    radial_volume = 0.5*(y_face[1:]**2 - y_face[:-1]**2)
    volume = dx*radial_volume[None, :, None]*np.ones((1, 1, nx))

    fields = {
        name: read_field(out_dir, name, resolution, shape)
        for name in (
            "dustdens_initial", "dustmomx_initial", "dustmomy_initial", "dustmomz_initial",
            "dustdens_final", "dustmomx_final", "dustmomy_final", "dustmomz_final",
        )
    }

    time = float(meta["time"])
    mode = int(meta["mode"])
    phase = float(meta["mode_phase"])
    omega = float(meta["angular_rate"])
    diffusivity = float(meta["diffusivity"])
    mode_average = (
        np.sin(mode*(x_o - omega*time - phase))
        - np.sin(mode*(x_i - omega*time - phase))
    )/(mode*dx)
    decay = np.exp(-diffusivity*mode*mode*time/(y*y))
    exact_density = y[None, :, None]**2*(
        1.0 + 0.1*decay[None, :, None]*mode_average[None, None, :]
    )
    exact_mx = exact_density*y[None, :, None]**2*omega
    exact_my = exact_density*float(meta["radial_speed"])
    exact_mz = exact_density*float(meta["polar_moment"])

    final_density = fields["dustdens_final"]
    final_mx = fields["dustmomx_final"]
    final_my = fields["dustmomy_final"]
    final_mz = fields["dustmomz_final"]
    momentum_vector_error = np.sqrt(
        (final_mx - exact_mx)**2 + (final_my - exact_my)**2 + (final_mz - exact_mz)**2
    )

    conservation: dict[str, float] = {}
    for label, initial_name, final_name in (
        ("mass", "dustdens_initial", "dustdens_final"),
        ("mx", "dustmomx_initial", "dustmomx_final"),
        ("my", "dustmomy_initial", "dustmomy_final"),
        ("mz", "dustmomz_initial", "dustmomz_final"),
    ):
        initial_integral = float(np.sum(volume*fields[initial_name]))
        final_integral = float(np.sum(volume*fields[final_name]))
        conservation[label] = abs(final_integral - initial_integral)/abs(initial_integral)

    finite = all(np.all(np.isfinite(field)) for field in fields.values())
    revolutions = omega*time/(2.0*math.pi)
    inner_decay = float(decay[0])
    outer_decay = float(decay[-1])
    closure_linf = max(
        float(np.max(np.abs(final_mx - final_density*y[None, :, None]**2*omega))),
        float(np.max(np.abs(final_my - final_density*float(meta["radial_speed"])))),
        float(np.max(np.abs(final_mz - final_density*float(meta["polar_moment"])))),
    )
    return {
        "case": "ring_long_2d",
        "resolution": resolution,
        "reference_class": "operator-composed-exact-Fourier",
        "primary_field": "density",
        "convergence_field": "density",
        "minimum_order": 1.0,
        "sequence_requirements": [
            {
                "name": "long_duration_density_accuracy",
                "field": "density",
                "norm": "l1",
                "maximum_final_error": 1.0e-2,
            },
            {
                "name": "long_duration_momentum_accuracy",
                "field": "momentum_vector",
                "norm": "l1",
                "maximum_final_error": 2.0e-2,
            },
            *(
                {
                    "name": f"periodic_{label}_conservation",
                    "field": f"{label}_relative_change",
                    "norm": "linf",
                    "maximum_final_error": 2.0e-11,
                }
                for label in ("mass", "mx", "my", "mz")
            ),
        ],
        "activation": bool(
            meta["case"] == "ring_long_2d"
            and nx == resolution and ny == 8 and nz == 1
            and abs(revolutions - 4.0) <= 2.0e-13
            and abs(float(meta["shift_cells"]) - 3.25) <= 1.0e-13
            and int(meta["steps"]) >= resolution
            and inner_decay < 0.2 and outer_decay > 0.8
        ),
        "activation_detail": {
            "revolutions": revolutions,
            "steps": int(meta["steps"]),
            "shift_cells": float(meta["shift_cells"]),
            "inner_mode_decay": inner_decay,
            "outer_mode_decay": outer_decay,
        },
        "state_finite": bool(finite),
        "minimum_density": float(np.min(final_density)),
        "momentum_closure_linf": closure_linf,
        "conserved_integrals_relative_change": conservation,
        "mass_relative_change": conservation["mass"],
        "mass_diagnostic_label": "mass rel",
        "errors": {
            "density": norm_set(final_density - exact_density, volume),
            "momentum_vector": norm_set(momentum_vector_error, volume),
            **{
                f"{label}_relative_change": scalar_norm(value)
                for label, value in conservation.items()
            },
        },
        "passed": bool(
            finite
            and np.min(final_density) >= -2.0e-13
            and closure_linf <= 2.0e-10
            and max(conservation.values()) <= 2.0e-11
        ),
    }
