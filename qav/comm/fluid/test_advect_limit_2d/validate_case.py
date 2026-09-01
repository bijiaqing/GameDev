#!/usr/bin/env python3

"""Validate conservative near-vacuum advection and exact invariant scaling"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def norm_set(error: np.ndarray) -> dict[str, float]:
    """Return common absolute norms"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def restrict(available: float, change: float, scale: float) -> float:
    """Reconstruct one production invariant constraint independently"""

    if change >= 0.0:
        return scale
    return min(scale, (1.0 - 1.0e-12)*available/(-change)) if available > 0.0 else 0.0


def analyze(out_dir: Path, resolution: int) -> dict:
    """Check one sharp production update and the controlled limiter activation"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    count = int(meta["nx"])
    initial = np.fromfile(out_dir/f"state_initial_N{resolution}.dat", dtype=np.float64).reshape(4, count)
    final = np.fromfile(out_dir/f"state_final_N{resolution}.dat", dtype=np.float64).reshape(4, count)
    probe = np.fromfile(out_dir/f"limit_probe_N{resolution}.dat", dtype=np.float64)

    rhod, mx, my, mz = 1.0, 0.2, -0.1, 0.05
    corr_rhod, corr_mx, corr_my, corr_mz = -1.2, 0.7, 0.0, 0.0
    bounds = (-0.5, 0.5, -0.4, 0.4, -0.2, 0.2)
    expected_scale = 1.0
    expected_scale = restrict(rhod, corr_rhod, expected_scale)
    expected_scale = restrict(mx - bounds[0]*rhod, corr_mx - bounds[0]*corr_rhod, expected_scale)
    expected_scale = restrict(bounds[1]*rhod - mx, bounds[1]*corr_rhod - corr_mx, expected_scale)
    expected_scale = restrict(my - bounds[2]*rhod, corr_my - bounds[2]*corr_rhod, expected_scale)
    expected_scale = restrict(bounds[3]*rhod - my, bounds[3]*corr_rhod - corr_my, expected_scale)
    expected_scale = restrict(mz - bounds[4]*rhod, corr_mz - bounds[4]*corr_rhod, expected_scale)
    expected_scale = restrict(bounds[5]*rhod - mz, bounds[5]*corr_rhod - corr_mz, expected_scale)
    expected_probe = np.array([
        expected_scale,
        rhod + expected_scale*corr_rhod,
        mx + expected_scale*corr_mx,
        my + expected_scale*corr_my,
        mz + expected_scale*corr_mz,
    ])

    conserved_error = np.sum(final, axis=1) - np.sum(initial, axis=1)
    final_density = final[0]
    active = final_density > float(meta["rho_vac"])
    primitive = final[1:, active]/final_density[active]
    initial_primitive = initial[1:]/initial[0]
    primitive_min = np.min(initial_primitive, axis=1)
    primitive_max = np.max(initial_primitive, axis=1)
    bound_violation = np.maximum(
        np.max(primitive_min[:, None] - primitive, axis=1),
        np.max(primitive - primitive_max[:, None], axis=1),
    )
    # assess the invariant in conserved form so division by a near-vacuum density cannot amplify harmless roundoff
    invariant_violation = np.maximum.reduce([
        primitive_min[:, None]*final_density[None, :] - final[1:],
        final[1:] - primitive_max[:, None]*final_density[None, :],
        np.zeros_like(final[1:]),
    ])
    scale_error = probe - expected_probe
    activation = bool(
        meta["production_advection"]
        and meta["controlled_limiter_probe"]
        and 0.0 < expected_scale < 1.0
        and np.min(initial[0]) < 20.0*float(meta["rho_vac"])
    )
    passed = bool(
        activation
        and np.all(np.isfinite(final))
        and np.all(np.isfinite(probe))
        and np.min(final_density) > 0.0
        and np.max(np.abs(conserved_error)) < 5.0e-13
        and np.max(invariant_violation) < 5.0e-15
        and np.max(bound_violation) < 1.0e-4
        and np.max(np.abs(scale_error)) < 5.0e-15
    )
    return {
        "case": "advect_limit_2d",
        "resolution": resolution,
        "reference_class": "conservation-and-discrete-limiter",
        "activation": activation,
        "minimum_density": float(np.min(final_density)),
        "limiter_scale": float(probe[0]),
        "expected_limiter_scale": expected_scale,
        "errors": {
            "conserved_state": norm_set(conserved_error),
            "primitive_bounds": norm_set(np.maximum(bound_violation, 0.0)),
            "conserved_invariant": norm_set(invariant_violation),
            "limiter_probe": norm_set(scale_error),
        },
        "passed": passed,
    }
