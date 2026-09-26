#!/usr/bin/env python3

"""validate the production physical collision kernel against independent scalar formulas"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

ALPHA = 1.0e-4
ASPR_0 = 0.05
M_MOL = 2.3 * 1.66054e-24
REYNOLDS_0 = 1.0e8
RHO_0 = 1.0
SIGMA_0 = 1.0
STOKES_0 = 0.2
V_FRAG = 1.0
X_SEC = 2.0e-15


def norms(error: np.ndarray) -> dict[str, float]:
    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
        "linf": float(np.max(np.abs(error))),
    }


def turbulent_velocity(
    stokes_large: float,
    stokes_small: float,
    re_inv: float,
    radius: float = 1.0,
) -> float:
    """evaluate the six published Ormel-Cuzzi branches without calling GameDev code"""

    h_g = ASPR_0 * radius ** (0.5 * (-1.0 + 1.0))
    sound_speed = h_g / math.sqrt(radius)
    gas_velocity_sq = 1.5 * ALPHA * sound_speed * sound_speed
    epsilon = stokes_small / stokes_large
    y_a = 1.6
    y_s = (
        1.6015125
        - 0.63119577 * stokes_large
        + 0.32938936 * stokes_large**2
        - 0.29847604 * stokes_large**3
    )

    if stokes_large < 0.2 * re_inv:
        value = gas_velocity_sq * (stokes_large - stokes_small) ** 2 / re_inv
    elif stokes_large < re_inv / y_a:
        value = gas_velocity_sq * (stokes_large - stokes_small) / (stokes_large + stokes_small)
        value *= stokes_large / (1.0 + re_inv / stokes_large) - stokes_small / (
            1.0 + re_inv / stokes_small
        )
    elif stokes_large < 5.0 * re_inv:
        coefficient = (stokes_large - stokes_small) / (stokes_large + stokes_small)
        coefficient *= stokes_large / (1.0 + y_a) - stokes_small**2 / (
            stokes_small + y_a * stokes_large
        )
        coefficient += 2.0 * (y_a * stokes_large - re_inv) + stokes_large / (1.0 + y_a)
        coefficient -= stokes_large**2 / (stokes_large + re_inv)
        coefficient += stokes_small**2 / (y_a * stokes_large + stokes_small)
        coefficient -= stokes_small**2 / (stokes_small + re_inv)
        value = gas_velocity_sq * coefficient
    elif stokes_large < 0.2:
        value = gas_velocity_sq * stokes_large
        value *= (
            2.0 * y_a
            - (1.0 + epsilon)
            + 2.0 / (1.0 + epsilon) * (1.0 / (1.0 + y_a) + epsilon**3 / (y_a + epsilon))
        )
    elif stokes_large < 1.0:
        value = gas_velocity_sq * stokes_large
        value *= (
            2.0 * y_s
            - (1.0 + epsilon)
            + 2.0 / (1.0 + epsilon) * (1.0 / (1.0 + y_s) + epsilon**3 / (y_s + epsilon))
        )
    else:
        value = gas_velocity_sq * (1.0 / (1.0 + stokes_large) + 1.0 / (1.0 + stokes_small))
    return math.sqrt(value)


def brownian_velocity(size_i: float, size_j: float, radius: float = 1.0) -> float:
    mass_i = math.pi * RHO_0 * size_i**3 / 6.0
    mass_j = math.pi * RHO_0 * size_j**3 / 6.0
    value = math.sqrt(
        8.0
        * (ASPR_0 / math.sqrt(radius)) ** 2
        * M_MOL
        * (mass_i + mass_j)
        / (math.pi * mass_i * mass_j)
    )
    return min(value, ASPR_0 / math.sqrt(radius))


def local_pair_velocity(
    velocity_i: np.ndarray,
    velocity_j: np.ndarray,
    size_i: float,
    size_j: float,
    radius: float,
    height: float,
    code_unit: bool,
) -> tuple[float, float]:
    """reference the query-local drift, settling, turbulence and Brownian prescription"""

    h_g = ASPR_0
    sigma_g = SIGMA_0 * radius**2
    gas_strat = math.exp(
        (radius / math.sqrt(radius * radius + height * height) - 1.0) / (h_g * h_g)
    )
    stokes_i = STOKES_0 * size_i / (radius**2 * gas_strat)
    stokes_j = STOKES_0 * size_j / (radius**2 * gas_strat)
    re_inv = 1.0 / math.sqrt(
        REYNOLDS_0 * sigma_g * gas_strat / SIGMA_0
        if code_unit
        else 0.5 * ALPHA * sigma_g * gas_strat * X_SEC / M_MOL
    )
    resolved = float(np.linalg.norm(velocity_i - velocity_j))
    turbulent = turbulent_velocity(max(stokes_i, stokes_j), min(stokes_i, stokes_j), re_inv, radius)
    brownian = 0.0 if code_unit else brownian_velocity(size_i, size_j, radius)
    # Stored particle velocities are separately checked by the Cartesian diagnostic.
    # Collision rates use equilibrium differential drift at the query location.
    omega = radius ** (-1.5)
    eta = -0.5 * ((2.0 - 0.5 - 1.5) * h_g * h_g - (1.0 - radius / math.hypot(radius, height)))
    vn = -eta * radius * omega
    fi, fj = 1.0 / (1.0 + stokes_i**2), 1.0 / (1.0 + stokes_j**2)
    radial = 2.0 * vn * (stokes_i * fi - stokes_j * fj)
    azimuthal = vn * (fi - fj)
    vertical = height * omega * (min(stokes_i, 0.5) - min(stokes_j, 0.5))
    return resolved, math.sqrt(radial**2 + azimuthal**2 + vertical**2 + turbulent**2 + brownian**2)


def cartesian_velocity(
    x: float,
    lx: float,
    vy: float,
    z: float = 0.5 * math.pi,
    lz: float = 0.0,
    y: float = 1.0,
) -> np.ndarray:
    sinz = math.sin(z)
    cosz = math.cos(z)
    vx = lx / (y * sinz)
    vz = lz / y
    vR = vy * sinz + vz * cosz
    vZ = vy * cosz - vz * sinz
    return np.array(
        [
            vR * math.cos(x) - vx * math.sin(x),
            vR * math.sin(x) + vx * math.cos(x),
            vZ,
        ]
    )


def cache_radius_reference(meta: dict) -> float:
    """reconstruct the seam distance from the float coordinates searched on the GPU"""

    z = float(meta["seam_z"])
    radius = np.float32(math.sin(z))
    x_owner = float(meta["x_min"]) + float(meta["seam_offset"])
    x_partner = float(meta["x_max"]) - float(meta["seam_offset"])
    owner = np.asarray(
        [
            np.float32(radius * np.float32(math.cos(x_owner))),
            np.float32(radius * np.float32(math.sin(x_owner))),
        ],
        dtype=np.float32,
    )
    partner = np.asarray(
        [
            np.float32(radius * np.float32(math.cos(x_partner))),
            np.float32(radius * np.float32(math.sin(x_partner))),
        ],
        dtype=np.float32,
    )
    width = np.float32(float(meta["x_max"]) - float(meta["x_min"]))
    sin_width = np.float32(math.sin(float(width)))
    cos_width = np.float32(math.cos(float(width)))
    image = np.asarray(
        [
            np.float32(cos_width * partner[0] + sin_width * partner[1]),
            np.float32(-sin_width * partner[0] + cos_width * partner[1]),
        ],
        dtype=np.float32,
    )
    delta = owner - image
    distance_sq = np.float32(delta[0] * delta[0] + delta[1] * delta[1])
    return math.sqrt(float(distance_sq))


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = json.loads((out_dir / f"meta_N{resolution}.json").read_text())
    values = np.fromfile(out_dir / f"collision_physics_N{resolution}.dat", dtype=np.float64)
    if values.size != int(meta["result_count"]):
        raise ValueError(f"expected {meta['result_count']} collision values, found {values.size}")

    code_unit = bool(meta["code_unit"])
    re_inv = 1.0 / math.sqrt(REYNOLDS_0 if code_unit else 0.5 * ALPHA * SIGMA_0 * X_SEC / M_MOL)
    boundaries = [0.2 * re_inv, re_inv / 1.6, 5.0 * re_inv, 0.2, 1.0]
    stokes_large = [
        0.02 * re_inv,
        0.40 * re_inv,
        2.00 * re_inv,
        math.sqrt(5.0 * re_inv * 0.2),
        0.5,
        2.0,
        *(value * factor for value in boundaries for factor in (1.0 - 1.0e-6, 1.0 + 1.0e-6)),
    ]
    turbulence = [turbulent_velocity(value, 0.3 * value, re_inv) for value in stokes_large]

    size = np.asarray(meta["size"], dtype=np.float64)
    number = np.asarray(meta["number"], dtype=np.float64)
    brownian = (
        [0.0, 0.0]
        if code_unit
        else [
            brownian_velocity(float(size[0]), float(size[1])),
            brownian_velocity(1.0e-12, 1.0e-12),
        ]
    )
    dimension = int(meta["dimension"])
    velocity_i = cartesian_velocity(
        float(meta["position_x"][0]), float(meta["velocity_lx"][0]), float(meta["velocity_y"][0])
    )
    velocity_j = cartesian_velocity(
        float(meta["position_x"][1]), float(meta["velocity_lx"][1]), float(meta["velocity_y"][1])
    )
    resolved, pair_velocity = local_pair_velocity(
        velocity_i, velocity_j, float(size[0]), float(size[1]), 1.0, 0.0, code_unit
    )
    cross_section = math.pi * (size[0] + size[1]) ** 2 / 4.0
    overlap_depth = math.sqrt(2.0 * math.pi * (ASPR_0**2 + ASPR_0**2))
    rate_scale = 1.0 / overlap_depth if dimension == 2 else 1.0
    custom_rate = number[1] * pair_velocity * cross_section * rate_scale

    seam_offset = float(meta["seam_offset"])
    width = float(meta["x_max"]) - float(meta["x_min"])
    seam_z = float(meta["seam_z"])
    seam_velocity = [float(value) for value in meta["seam_velocity"]]
    seam_owner = cartesian_velocity(
        float(meta["x_min"]) + seam_offset, *seam_velocity[:2], seam_z, seam_velocity[2]
    )
    seam_partner = cartesian_velocity(
        float(meta["x_max"]) - seam_offset - width,
        *seam_velocity[:2],
        seam_z,
        seam_velocity[2],
    )
    seam_partner_wrong = cartesian_velocity(
        float(meta["x_max"]) - seam_offset,
        *seam_velocity[:2],
        seam_z,
        seam_velocity[2],
    )
    interior_owner = cartesian_velocity(+seam_offset, *seam_velocity[:2], seam_z, seam_velocity[2])
    interior_partner = cartesian_velocity(
        -seam_offset, *seam_velocity[:2], seam_z, seam_velocity[2]
    )
    seam_radius = math.sin(seam_z)
    seam_height = math.cos(seam_z)
    _, image_velocity = local_pair_velocity(
        seam_owner,
        seam_partner,
        float(size[0]),
        float(size[1]),
        seam_radius,
        seam_height,
        code_unit,
    )
    _, image_velocity_wrong = local_pair_velocity(
        seam_owner,
        seam_partner_wrong,
        float(size[0]),
        float(size[1]),
        seam_radius,
        seam_height,
        code_unit,
    )
    _, interior_velocity = local_pair_velocity(
        interior_owner,
        interior_partner,
        float(size[0]),
        float(size[1]),
        seam_radius,
        seam_height,
        code_unit,
    )
    image_rate = number[1] * image_velocity * cross_section * rate_scale
    interior_rate = number[1] * interior_velocity * cross_section * rate_scale
    neighbor = 3 * 1 + 1

    cache_radius = cache_radius_reference(meta)
    cache_measure = (
        math.pi * cache_radius**2 if dimension == 2 else 4.0 * math.pi * cache_radius**3 / 3.0
    )
    _, self_velocity_0 = local_pair_velocity(
        seam_owner,
        seam_owner,
        float(size[0]),
        float(size[0]),
        seam_radius,
        seam_height,
        code_unit,
    )
    _, self_velocity_1 = local_pair_velocity(
        seam_partner_wrong,
        seam_partner_wrong,
        float(size[1]),
        float(size[1]),
        seam_radius,
        seam_height,
        code_unit,
    )
    self_section_0 = math.pi * size[0] ** 2
    self_section_1 = math.pi * size[1] ** 2
    cache_numerator_0 = number[0] * self_velocity_0 * self_section_0 * rate_scale + image_rate
    reverse_image_rate = number[0] * image_velocity * cross_section * rate_scale
    cache_numerator_1 = (
        number[1] * self_velocity_1 * self_section_1 * rate_scale + reverse_image_rate
    )

    # use the measured float-geometry volume in the rate oracle, then check that
    # volume separately against the exact double-precision seam geometry
    measured_measure = values[36:38]
    cache_rate_0 = cache_numerator_0 / measured_measure[0]
    cache_rate_1 = cache_numerator_1 / measured_measure[1]

    expected = np.asarray(
        [
            re_inv,
            *turbulence,
            *brownian,
            pair_velocity,
            custom_rate,
            resolved,
            image_velocity,
            interior_velocity,
            image_velocity_wrong,
            image_rate,
            interior_rate,
            neighbor,
            1,
            1,
            cache_rate_0,
            cache_rate_1,
        ]
    )
    error = values[:32] - expected
    relative = np.abs(error) / np.maximum(np.abs(expected), 1.0e-300)
    expected_measure = np.asarray([cache_measure, cache_measure])
    measure_error = measured_measure - expected_measure
    measure_relative = np.abs(measure_error) / expected_measure
    cache_codes = [
        {int(value) for value in values[32:34]},
        {int(value) for value in values[34:36]},
    ]
    expected_codes = [{0, 4}, {2, 3}]
    cache_images = cache_codes == expected_codes

    # the six interior points and five lower/upper pairs must exercise all production branch choices
    branch_ids = []
    for value in stokes_large:
        if value < 0.2 * re_inv:
            branch_ids.append(1)
        elif value < re_inv / 1.6:
            branch_ids.append(2)
        elif value < 5.0 * re_inv:
            branch_ids.append(3)
        elif value < 0.2:
            branch_ids.append(4)
        elif value < 1.0:
            branch_ids.append(5)
        else:
            branch_ids.append(6)
    boundary_coverage = all(
        branch_ids[6 + 2 * idx] == idx + 1 and branch_ids[7 + 2 * idx] == idx + 2
        for idx in range(5)
    )
    activation = (
        branch_ids[:6] == [1, 2, 3, 4, 5, 6]
        and boundary_coverage
        and (code_unit or math.isclose(brownian[1], ASPR_0, rel_tol=0.0, abs_tol=0.0))
        and math.isclose(values[24], values[22], rel_tol=2.0e-11)
        and math.isclose(values[23], values[22], rel_tol=2.0e-11)
        and np.array_equal(values[27:30], expected[27:30])
        and cache_images
        and np.all(measured_measure > 0.0)
    )
    passed = bool(
        activation
        and np.all(np.isfinite(values))
        and float(np.max(relative)) < 2.0e-11
        and float(np.max(measure_relative)) < 2.0e-4
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
            "vertically_integrated_overlap": dimension == 2,
            "query_local_image_invariance": bool(
                math.isclose(values[24], values[22], rel_tol=2.0e-11)
            ),
            "query_local_interior_equivalence": bool(
                math.isclose(values[23], values[22], rel_tol=2.0e-11)
            ),
            "periodic_image_code": bool(np.array_equal(values[27:30], expected[27:30])),
            "search_cache_images": bool(cache_images),
            "cached_periodic_image_rate": bool(relative[30] < 2.0e-11 and relative[31] < 2.0e-11),
        },
        "maximum_relative_error": float(np.max(relative)),
        "maximum_measure_relative_error": float(np.max(measure_relative)),
        "errors": {
            "physical_collision": norms(error),
            "knn_measure": norms(measure_error),
        },
        "passed": passed,
    }
