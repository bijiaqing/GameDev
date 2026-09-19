#!/usr/bin/env python3

"""Validate raw swarm CUDA outputs against independent analytical results"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
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
ALPHA = 1.0e-4
IDX_P = 2.0
INIT_SMIN = 0.05
INIT_SMAX = 6.4


def read_meta(path: Path) -> dict[str, object]:
    """Read and validate the structured metadata emitted by the CUDA driver"""

    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return values


def read_array(out_dir: Path, name: str, resolution: int, count: int) -> np.ndarray:
    """Load one raw float64 artifact and reject truncated or stale files"""

    path = out_dir / f"{name}_N{resolution}.dat"
    values = np.fromfile(path, dtype=np.float64)
    if values.size != count:
        raise ValueError(f"{path} contains {values.size} values; expected {count}")
    return values


def norms(values: np.ndarray) -> dict[str, float]:
    """Return the common absolute L1, L2, and Linf summaries"""

    return {
        "l1": float(np.mean(np.abs(values))),
        "l2": float(np.sqrt(np.mean(values * values))),
        "linf": float(np.max(np.abs(values))),
    }


def wrap_angle(values: np.ndarray) -> np.ndarray:
    """Map angular errors to the principal interval from -pi through pi"""

    return (values + math.pi) % (2.0 * math.pi) - math.pi


def load_state(out_dir: Path, resolution: int, nparticle: int) -> np.ndarray:
    """Load the six position and velocity components as one component-first array"""

    return read_array(out_dir, "state", resolution, 6*nparticle).reshape(6, nparticle)


def analyze_relaxation(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    """Evaluate the exact constant-coefficient P-R drag response"""

    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    state = load_state(out_dir, resolution, nparticle)
    species = read_array(out_dir, "species", resolution, 2*nparticle).reshape(2, nparticle)
    size = species[0]
    inv_ts = 1.0/(STOKES_0*size)
    lx_initial = 1.2
    lx_gas = 1.0
    beta = BETA_0/size
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
    """Check stochastic moments and preservation of Cartesian velocity after displacement"""

    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    state = load_state(out_dir, resolution, nparticle)
    diffusivity = NU/(1.0 + 0.2**2)
    expected_variance = 2.0*diffusivity*dt

    if meta["case"] == "diffusion_1d":
        samples = state[1] - 1.0
        mean_expected = diffusivity*(1.0 + 4.0*0.2**2/(1.0+0.2**2))*dt
        velocity_error = np.concatenate((
            state[3]/state[1] - 0.7,
            state[4] - 0.2,
            state[0],
            state[2] - 0.5*math.pi,
            state[5],
        ))
        components = {"R": samples}
    elif meta["case"] == "diffusion_2d":
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
        mean_expected = diffusivity*(1.0 + 4.0*0.2**2/(1.0+0.2**2))*dt

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


def analyze_wedge_diffusion(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    """Check seam crossing and velocity reprojection at the unique unwrapped endpoint"""

    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    initial = read_array(out_dir, "state_initial", resolution, 6*nparticle).reshape(6, nparticle)
    final = load_state(out_dir, resolution, nparticle)
    width = float(meta["x_max"]) - float(meta["x_min"])
    delta_x = (final[0] - initial[0] + 0.5*width) % width - 0.5*width
    x_unwrapped = initial[0] + delta_x

    radius_initial = initial[1]*np.sin(initial[2])
    vx_initial = initial[3]/radius_initial
    vz_initial = initial[5]/initial[1]
    vR_initial = initial[4]*np.sin(initial[2]) + vz_initial*np.cos(initial[2])
    vZ_initial = initial[4]*np.cos(initial[2]) - vz_initial*np.sin(initial[2])
    vx_cart = vR_initial*np.cos(initial[0]) - vx_initial*np.sin(initial[0])
    vy_cart = vR_initial*np.sin(initial[0]) + vx_initial*np.cos(initial[0])

    radius_final = final[1]*np.sin(final[2])
    vR_final = vx_cart*np.cos(x_unwrapped) + vy_cart*np.sin(x_unwrapped)
    vx_final = vy_cart*np.cos(x_unwrapped) - vx_cart*np.sin(x_unwrapped)
    expected_lx = radius_final*vx_final
    expected_vy = vR_final*np.sin(final[2]) + vZ_initial*np.cos(final[2])
    expected_lz = (vR_final*np.cos(final[2]) - vZ_initial*np.sin(final[2]))*final[1]
    velocity_error = np.concatenate((
        final[3] - expected_lx,
        final[4] - expected_vy,
        final[5] - expected_lz,
    ))

    lower = initial[0] < 0.0
    upper = ~lower
    crossed_lower = int(np.count_nonzero(lower & (final[0] > 0.0)))
    crossed_upper = int(np.count_nonzero(upper & (final[0] < 0.0)))
    unique_unwrap = bool(np.max(np.abs(delta_x)) < 0.5*width)
    stokes = 0.2/radius_initial**2/np.exp((radius_initial/initial[1]-1.0)/0.05**2)
    expected_variance = float(np.mean(2.0*NU*dt/((1.0+stokes**2)*radius_initial**2)))
    mean_error = float(np.mean(delta_x))
    variance_error = float(np.var(delta_x) - expected_variance)
    mean_limit = 6.0*math.sqrt(expected_variance/nparticle)
    variance_limit = 6.0*expected_variance*math.sqrt(2.0/(nparticle - 1))
    passed = bool(
        unique_unwrap and crossed_lower > 0 and crossed_upper > 0
        and np.max(np.abs(velocity_error)) < 2.0e-12
        and abs(mean_error) <= mean_limit and abs(variance_error) <= variance_limit
    )
    return {
        "case": meta["case"], "resolution": resolution,
        "crossed_lower": crossed_lower, "crossed_upper": crossed_upper,
        "unique_unwrap": unique_unwrap,
        "statistics": {"x": {
            "mean_error": mean_error, "mean_limit": mean_limit,
            "variance_error": variance_error, "variance_limit": variance_limit,
        }},
        "errors": {"velocity": norms(velocity_error)},
        "passed": passed,
    }


def initialization_spans(radius: float, meta: dict[str, str]) -> list[tuple[float, float]]:
    """Intersect a cylindrical line with the independently reconstructed spherical domain"""

    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    if radius < 0.0 or radius > y_max:
        return []

    polar_lo = -math.inf if z_max >= math.pi else radius*math.cos(z_max)/math.sin(z_max)
    polar_hi = +math.inf if z_min <= 0.0 else radius*math.cos(z_min)/math.sin(z_min)
    outer = math.sqrt(max(y_max*y_max - radius*radius, 0.0))
    height_lo = max(polar_lo, -outer)
    height_hi = min(polar_hi, +outer)
    if height_hi <= height_lo:
        return []
    if radius >= y_min:
        return [(height_lo, height_hi)]

    inner = math.sqrt(max(y_min*y_min - radius*radius, 0.0))
    spans = []
    if min(height_hi, -inner) > height_lo:
        spans.append((height_lo, min(height_hi, -inner)))
    if height_hi > max(height_lo, +inner):
        spans.append((max(height_lo, +inner), height_hi))
    return spans


def normal_interval_mass(height_lo: float, height_hi: float, scale_height: float) -> float:
    """Integrate a normalized Gaussian without subtracting nearly equal unit CDFs"""

    inv_width = 1.0/(math.sqrt(2.0)*scale_height)
    if height_lo >= 0.0:
        return 0.5*(math.erfc(height_lo*inv_width) - math.erfc(height_hi*inv_width))
    if height_hi <= 0.0:
        return 0.5*(math.erfc(-height_hi*inv_width) - math.erfc(-height_lo*inv_width))
    return 0.5*(math.erf(height_hi*inv_width) - math.erf(height_lo*inv_width))


def initialization_scale_height(radius: float, size: float) -> float:
    """Evaluate the test model's settled scale height independently of production helpers"""

    gas_aspect = ASPR_0
    stokes_mid = STOKES_0*size/radius**IDX_P
    return gas_aspect*radius*math.sqrt(ALPHA/stokes_mid)


def initialization_profile(meta: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the Gaussian-convolved surface profile used by the test"""

    ny = int(meta["ny"])
    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])
    z_min = float(meta["z_min"])
    z_max = float(meta["z_max"])
    radial_min = y_min*min(math.sin(z_min), math.sin(z_max))
    profile_dr = (y_max - radial_min)/ny
    radius = radial_min + profile_dr*np.arange(ny + 1)
    smooth = 0.05
    source_min = y_min + 2.0*smooth
    source_max = y_max - 2.0*smooth
    kernel_std = 0.5*smooth
    kernel_norm = 1.0/(math.sqrt(2.0*math.pi)*kernel_std)
    surface = np.zeros(ny + 1)
    for source_radius in radius:
        if source_radius < source_min or source_radius > source_max:
            continue
        source_surface = 1.0e-2*source_radius**IDX_P
        offset = radius - source_radius
        surface += source_surface*kernel_norm*np.exp(-0.5*(offset/kernel_std)**2)*profile_dr
    return radius, surface


def initialization_radial_cdf(
    size: float, meta: dict[str, str], profile_radius: np.ndarray, profile_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct an independent radial marginal from analytic vertical containment"""

    radial_bin_count = max(2048, 4*int(meta["ny"]))
    radial_axis = np.linspace(profile_radius[0], float(meta["y_max"]), radial_bin_count + 1)
    surface = np.interp(radial_axis, profile_radius, profile_surface)
    radial_density = np.zeros(radial_bin_count + 1)
    for idx, radius in enumerate(radial_axis):
        if radius <= 0.0:
            continue
        scale_height = initialization_scale_height(radius, size)
        containment = sum(
            normal_interval_mass(height_lo, height_hi, scale_height)
            for height_lo, height_hi in initialization_spans(radius, meta)
        )
        radial_density[idx] = radius*surface[idx]*containment

    interval_mass = 0.5*(radial_density[:-1] + radial_density[1:])*np.diff(radial_axis)
    cdf = np.concatenate(([0.0], np.cumsum(interval_mass)))
    domain_mass = 2.0*math.pi*cdf[-1]
    cdf /= cdf[-1]
    return radial_axis, cdf, domain_mass


def uniform_ks(samples: np.ndarray) -> float:
    """Return the two-sided one-sample KS distance from a uniform distribution"""

    values = np.sort(samples)
    count = values.size
    upper = np.arange(1, count + 1)/count - values
    lower = values - np.arange(count)/count
    return float(max(np.max(upper), np.max(lower)))


def analyze_initialization(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    """Validate continuous finite-domain initialization and representative-mass normalization"""

    nparticle = int(meta["np"])
    initial = read_array(out_dir, "initial", resolution, 4*nparticle).reshape(4, nparticle)
    mass_bank = read_array(out_dir, "mass_bank", resolution, 128)
    mass_summary = read_array(out_dir, "mass_summary", resolution, 3)
    x, y, z, size = initial
    radius = y*np.sin(z)
    height = y*np.cos(z)

    profile_radius, profile_surface = initialization_profile(meta)
    radial_references = {}
    expected_mass = np.empty(128)
    size_axis = np.exp(np.linspace(math.log(INIT_SMIN), math.log(INIT_SMAX), 128))
    size_mid = math.sqrt(INIT_SMIN*INIT_SMAX)
    dlog_size = (math.log(INIT_SMAX) - math.log(INIT_SMIN))/(size_axis.size - 1)
    mid_location = (math.log(size_mid) - math.log(INIT_SMIN))/dlog_size
    mid_index = min(int(mid_location), size_axis.size - 2)
    mid_fraction = mid_location - mid_index
    reference_indices = {0, mid_index, mid_index + 1, size_axis.size - 1}
    for idx_size, test_size in enumerate(size_axis):
        radial_axis, radial_cdf, domain_mass = initialization_radial_cdf(
            test_size, meta, profile_radius, profile_surface
        )
        expected_mass[idx_size] = domain_mass
        if idx_size in reference_indices:
            radial_references[idx_size] = (radial_axis, radial_cdf)

    radial_axis = radial_references[0][0]
    middle_cdf = (
        (1.0 - mid_fraction)*radial_references[mid_index][1]
        + mid_fraction*radial_references[mid_index + 1][1]
    )
    middle_mass = math.exp(
        (1.0 - mid_fraction)*math.log(expected_mass[mid_index])
        + mid_fraction*math.log(expected_mass[mid_index + 1])
    )
    _, middle_cdf_exact, middle_mass_exact = initialization_radial_cdf(
        size_mid, meta, profile_radius, profile_surface
    )

    population_ranges = (
        (0, nparticle//3, INIT_SMIN, radial_references[0][1], expected_mass[0]),
        (nparticle//3, 2*nparticle//3, size_mid, middle_cdf, middle_mass),
        (2*nparticle//3, nparticle, INIT_SMAX, radial_references[127][1], expected_mass[127]),
    )
    grain_sizes_match = all(
        np.all(size[idx_lo:idx_hi] == test_size)
        for idx_lo, idx_hi, test_size, _, _ in population_ranges
    )

    relative_mass_error = (mass_bank - expected_mass)/expected_mass
    radial_pit = np.empty(nparticle)
    vertical_pit = np.empty(nparticle)
    for idx_lo, idx_hi, test_size, radial_cdf, _ in population_ranges:
        for idx in range(idx_lo, idx_hi):
            radial_pit[idx] = np.interp(radius[idx], radial_axis, radial_cdf)

            spans = initialization_spans(radius[idx], meta)
            scale_height = initialization_scale_height(radius[idx], test_size)
            span_masses = [normal_interval_mass(lo, hi, scale_height) for lo, hi in spans]
            total_mass = sum(span_masses)
            mass_before = 0.0
            for (height_lo, height_hi), span_mass in zip(spans, span_masses):
                if height[idx] >= height_hi:
                    mass_before += span_mass
                    continue
                if height[idx] > height_lo:
                    mass_before += normal_interval_mass(height_lo, height[idx], scale_height)
                break
            vertical_pit[idx] = mass_before/total_mass

    population_sizes = [idx_hi - idx_lo for idx_lo, idx_hi, *_ in population_ranges]
    radial_ks_values = [uniform_ks(radial_pit[idx_lo:idx_hi]) for idx_lo, idx_hi, *_ in population_ranges]
    vertical_ks_values = [uniform_ks(vertical_pit[idx_lo:idx_hi]) for idx_lo, idx_hi, *_ in population_ranges]
    ks_limits = [6.0/math.sqrt(count) for count in population_sizes]
    radial_ks = max(radial_ks_values)
    vertical_ks = max(vertical_ks_values)

    expected_weight_sum = sum(
        (idx_hi - idx_lo)*domain_mass
        for idx_lo, idx_hi, _, _, domain_mass in population_ranges
    )
    expected_mass_norm = mass_summary[0]*nparticle/expected_weight_sum
    mass_norm_error = (mass_summary[2] - expected_mass_norm)/expected_mass_norm
    represented_error = (mass_summary[1] - mass_summary[0])/mass_summary[0]
    inside = (
        (x >= float(meta["x_min"])) & (x <= float(meta["x_max"]))
        & (y >= float(meta["y_min"])) & (y <= float(meta["y_max"]))
        & (z >= float(meta["z_min"])) & (z <= float(meta["z_max"]))
    )

    return {
        "case": meta["case"],
        "resolution": resolution,
        "errors": {
            "domain_mass_relative": norms(relative_mass_error),
            "mass_norm_relative": norms(np.array([mass_norm_error])),
            "represented_mass_relative": norms(np.array([represented_error])),
            "radial_cdf_ks": norms(np.array([radial_ks])),
            "vertical_cdf_ks": norms(np.array([vertical_ks])),
            "middle_cdf_interpolation": norms(middle_cdf - middle_cdf_exact),
            "middle_mass_interpolation_relative": norms(
                np.array([(middle_mass - middle_mass_exact)/middle_mass_exact])
            ),
        },
        "population_sizes": population_sizes,
        "population_grain_sizes": [INIT_SMIN, size_mid, INIT_SMAX],
        "radial_cdf_ks_by_population": radial_ks_values,
        "vertical_cdf_ks_by_population": vertical_ks_values,
        "ks_limits": ks_limits,
        "population_grain_sizes_match": bool(grain_sizes_match),
        "inside_fraction": float(np.mean(inside)),
        "mass_norm": float(mass_summary[2]),
        "passed": bool(
            np.all(np.isfinite(initial))
            and np.all(np.isfinite(mass_bank))
            and grain_sizes_match
            and np.max(np.abs(relative_mass_error)) < 2.0e-12
            and abs(mass_norm_error) < 5.0e-12
            and abs(represented_error) < 5.0e-12
            and all(value < limit for value, limit in zip(radial_ks_values, ks_limits))
            and all(value < limit for value, limit in zip(vertical_ks_values, ks_limits))
            and np.all(inside)
        ),
    }




def analyze(out_dir: Path, resolution: int) -> dict:
    """Dispatch one retained publication case and reject failed criteria"""

    meta = read_meta(out_dir / f"meta_N{resolution}.json")
    if int(meta["resolution"]) != resolution:
        raise ValueError(
            f"metadata resolution {meta['resolution']} does not match requested N={resolution}"
        )
    case = meta["case"]
    if case == "prdrag_2d":
        result = analyze_relaxation(out_dir, resolution, meta)
    elif isinstance(case, str) and case.startswith("diffusion_wedge_"):
        result = analyze_wedge_diffusion(out_dir, resolution, meta)
    elif isinstance(case, str) and case.startswith("diffusion_"):
        result = analyze_diffusion(out_dir, resolution, meta)
    elif case == "initial_3d":
        result = analyze_initialization(out_dir, resolution, meta)
    else:
        raise ValueError(f"unknown retained swarm publication case: {case}")
    if not result["passed"]:
        raise AssertionError(f"{case} failed at resolution {resolution}: {result}")
    return result
