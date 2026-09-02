#!/usr/bin/env python3

"""Validate the production physical collision kernel against independent scalar formulas."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ALPHA = 1.0e-4
ASPR_0 = 0.05
M_MOL = 2.3*1.66054e-24
REYNOLDS_0 = 1.0e8
RHO_0 = 1.0
SIGMA_0 = 1.0
STOKES_0 = 0.2
X_SEC = 2.0e-15


def norms(error: np.ndarray) -> dict[str, float]:
    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def turbulent_velocity(stokes_large: float, stokes_small: float, re_inv: float) -> float:
    """Evaluate the six published Ormel-Cuzzi branches without calling GameDev code."""

    sound_speed = ASPR_0
    gas_velocity_sq = 1.5*ALPHA*sound_speed*sound_speed
    epsilon = stokes_small/stokes_large
    y_a = 1.6
    y_s = (1.6015125 - 0.63119577*stokes_large
           + 0.32938936*stokes_large**2 - 0.29847604*stokes_large**3)

    if stokes_large < 0.2*re_inv:
        value = gas_velocity_sq*(stokes_large - stokes_small)**2/re_inv
    elif stokes_large < re_inv/y_a:
        value = gas_velocity_sq*(stokes_large - stokes_small)/(stokes_large + stokes_small)
        value *= (stokes_large/(1.0 + re_inv/stokes_large)
                  - stokes_small/(1.0 + re_inv/stokes_small))
    elif stokes_large < 5.0*re_inv:
        coefficient = (stokes_large - stokes_small)/(stokes_large + stokes_small)
        coefficient *= (stokes_large/(1.0 + y_a)
                        - stokes_small**2/(stokes_small + y_a*stokes_large))
        coefficient += 2.0*(y_a*stokes_large - re_inv) + stokes_large/(1.0 + y_a)
        coefficient -= stokes_large**2/(stokes_large + re_inv)
        coefficient += stokes_small**2/(y_a*stokes_large + stokes_small)
        coefficient -= stokes_small**2/(stokes_small + re_inv)
        value = gas_velocity_sq*coefficient
    elif stokes_large < 0.2:
        value = gas_velocity_sq*stokes_large
        value *= (2.0*y_a - (1.0 + epsilon)
                  + 2.0/(1.0 + epsilon)*(1.0/(1.0 + y_a)
                  + epsilon**3/(y_a + epsilon)))
    elif stokes_large < 1.0:
        value = gas_velocity_sq*stokes_large
        value *= (2.0*y_s - (1.0 + epsilon)
                  + 2.0/(1.0 + epsilon)*(1.0/(1.0 + y_s)
                  + epsilon**3/(y_s + epsilon)))
    else:
        value = gas_velocity_sq*(1.0/(1.0 + stokes_large)
                                 + 1.0/(1.0 + stokes_small))
    return math.sqrt(value)


def brownian_velocity(size_i: float, size_j: float) -> float:
    mass_i = math.pi*RHO_0*size_i**3/6.0
    mass_j = math.pi*RHO_0*size_j**3/6.0
    value = math.sqrt(
        8.0*ASPR_0**2*M_MOL*(mass_i + mass_j)/(math.pi*mass_i*mass_j)
    )
    return min(value, ASPR_0)


def cartesian_velocity(x: float, angular_momentum: float, radial_velocity: float) -> np.ndarray:
    azimuthal_velocity = angular_momentum
    return np.array([
        radial_velocity*math.cos(x) - azimuthal_velocity*math.sin(x),
        radial_velocity*math.sin(x) + azimuthal_velocity*math.cos(x),
        0.0,
    ])


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    values = np.fromfile(out_dir/f"collision_physics_N{resolution}.dat", dtype=np.float64)
    if values.size != int(meta["result_count"]):
        raise ValueError(f"expected {meta['result_count']} collision values, found {values.size}")

    code_unit = bool(meta["code_unit"])
    re_inv = 1.0/math.sqrt(REYNOLDS_0 if code_unit
                           else 0.5*ALPHA*SIGMA_0*X_SEC/M_MOL)
    boundaries = [0.2*re_inv, re_inv/1.6, 5.0*re_inv, 0.2, 1.0]
    stokes_large = [
        0.02*re_inv,
        0.40*re_inv,
        2.00*re_inv,
        math.sqrt(5.0*re_inv*0.2),
        0.5,
        2.0,
        *(value*factor for value in boundaries for factor in (1.0 - 1.0e-6, 1.0 + 1.0e-6)),
    ]
    turbulence = [turbulent_velocity(value, 0.3*value, re_inv) for value in stokes_large]

    size = np.asarray(meta["size"], dtype=np.float64)
    number = np.asarray(meta["number"], dtype=np.float64)
    brownian = [0.0, 0.0] if code_unit else [
        brownian_velocity(float(size[0]), float(size[1])),
        brownian_velocity(1.0e-12, 1.0e-12),
    ]
    velocity_i = cartesian_velocity(
        float(meta["position_x"][0]), float(meta["velocity_lx"][0]), float(meta["velocity_y"][0])
    )
    velocity_j = cartesian_velocity(
        float(meta["position_x"][1]), float(meta["velocity_lx"][1]), float(meta["velocity_y"][1])
    )
    resolved = float(np.linalg.norm(velocity_i - velocity_j))
    stokes_i, stokes_j = STOKES_0*size
    turbulent_pair = turbulent_velocity(max(stokes_i, stokes_j), min(stokes_i, stokes_j), re_inv)
    brownian_pair = 0.0 if code_unit else brownian_velocity(float(size[0]), float(size[1]))
    pair_velocity = math.sqrt(resolved**2 + turbulent_pair**2 + brownian_pair**2)
    cross_section = math.pi*(size[0] + size[1])**2/4.0
    overlap_depth = math.sqrt(2.0*math.pi*(ASPR_0**2 + ASPR_0**2))
    custom_rate = number[1]*pair_velocity*cross_section/overlap_depth

    expected = np.asarray([
        re_inv,
        *turbulence,
        *brownian,
        pair_velocity,
        custom_rate,
        resolved,
    ])
    error = values - expected
    relative = np.abs(error)/np.maximum(np.abs(expected), 1.0e-300)

    # the six interior points and five lower/upper pairs must exercise all production branch choices
    branch_ids = []
    for value in stokes_large:
        if value < 0.2*re_inv:
            branch_ids.append(1)
        elif value < re_inv/1.6:
            branch_ids.append(2)
        elif value < 5.0*re_inv:
            branch_ids.append(3)
        elif value < 0.2:
            branch_ids.append(4)
        elif value < 1.0:
            branch_ids.append(5)
        else:
            branch_ids.append(6)
    boundary_coverage = all(
        branch_ids[6 + 2*idx] == idx + 1 and branch_ids[7 + 2*idx] == idx + 2
        for idx in range(5)
    )
    activation = (
        branch_ids[:6] == [1, 2, 3, 4, 5, 6]
        and boundary_coverage
        and (code_unit or math.isclose(brownian[1], ASPR_0, rel_tol=0.0, abs_tol=0.0))
    )
    passed = bool(
        activation and np.all(np.isfinite(values))
        and float(np.max(relative)) < 2.0e-11
    )
    return {
        "case": str(meta["case"]),
        "resolution": resolution,
        "reference_class": "analytical-fixed-pair",
        "activation": {
            "all_six_turbulent_regimes": branch_ids[:6] == [1, 2, 3, 4, 5, 6],
            "all_five_boundaries_straddled": boundary_coverage,
            "brownian_branch": not code_unit,
            "brownian_sound_speed_cap": None if code_unit else brownian[1] == ASPR_0,
            "custom_kernel": True,
            "vertically_integrated_overlap": True,
        },
        "maximum_relative_error": float(np.max(relative)),
        "errors": {"physical_collision": norms(error)},
        "passed": passed,
    }
