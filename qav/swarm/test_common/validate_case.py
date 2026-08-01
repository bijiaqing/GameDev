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
ALPHA = 1.0e-4
IDX_P = 2.0
INIT_SMIN = 0.05
INIT_SMAX = 6.4


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
    state = load_state(out_dir, resolution, int(meta["np"]))
    inactive_error = np.concatenate((
        state[0] if nx == 1 else np.zeros(0),
        state[2] - 0.5*math.pi if nz == 1 else np.zeros(0),
        state[5] if nz == 1 else np.zeros(0),
    ))
    inactive_max = float(np.max(np.abs(inactive_error))) if inactive_error.size else 0.0

    return {
        "case": meta["case"],
        "resolution": resolution,
        "errors": {
            "density": norms(density - expected_density),
            "optdepth": norms(optical - expected_optical),
            "inactive": norms(inactive_error) if inactive_error.size else norms(np.zeros(1)),
        },
        "mass_relative_change": mass_relative_change,
        "passed": bool(
            np.max(np.abs(density - expected_density)) < 5.0e-12
            and np.max(np.abs(optical - expected_optical)) < 5.0e-11
            and abs(mass_relative_change) < 5.0e-12
            and inactive_max == 0.0
        ),
    }


def analyze_orbit(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    state = load_state(out_dir, resolution, nparticle)
    radial = meta["case"] == "orbit_1d"
    initial_x = np.zeros(nparticle) if radial else X_MIN + (np.arange(nparticle) + 0.5)*(X_MAX - X_MIN)/nparticle
    phase_error = state[0] - initial_x if radial else wrap_angle(state[0] - initial_x)
    error = np.concatenate((phase_error, state[1] - 1.0, state[2] - 0.5*math.pi,
                            state[3] - 1.0, state[4], state[5]))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"state": norms(error)},
        "passed": bool(
            np.all(np.isfinite(state))
            and np.max(np.abs(error)) < (5.0e-13 if radial else 0.5)
        ),
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
    if meta["case"] in {"radiation_1d", "radiation_2d", "prdrag_1d", "prdrag_2d"}:
        beta = BETA_0/size

    if meta["case"] not in {"prdrag_1d", "prdrag_2d"}:
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
    if meta["case"].endswith("_1d"):
        error = np.concatenate((error, state[0], state[2] - 0.5*math.pi, state[5]))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"response": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze_viscflow(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    state = load_state(out_dir, resolution, nparticle)
    radius = 0.7 + 0.6*np.arange(nparticle)/(nparticle - 1.0)
    stokes = STOKES_0/radius**2
    velocity_k = radius**-0.5
    velocity_gas_r = -3.0*NU*(2.0 + 0.5)/radius
    velocity_r = velocity_gas_r/(1.0 + stokes**2)
    velocity_x = velocity_k - 0.5*stokes*velocity_r
    error = np.concatenate((
        state[0],
        state[1] - radius,
        state[2] - 0.5*math.pi,
        state[3] - radius*velocity_x,
        state[4] - velocity_r,
        state[5],
    ))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"viscous_flow": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze_diffusion(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    nparticle = int(meta["np"])
    dt = float(meta["dt"])
    state = load_state(out_dir, resolution, nparticle)
    expected_variance = 2.0*NU*dt

    if meta["case"] == "diffusion_1d":
        samples = state[1] - 1.0
        mean_expected = NU*dt
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


def ball_measure(dimension: int, radius: float, distance: float) -> float:
    if dimension == 2:
        full = math.pi*radius*radius
        cap = radius*radius*math.acos(distance/radius) - distance*math.sqrt(radius*radius - distance*distance)
    else:
        full = 4.0*math.pi*radius**3/3.0
        cap = math.pi*(radius - distance)**2*(2.0*radius + distance)/3.0
    return full*(1.0 - cap/full)


def analyze_collision(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    values = read_array(out_dir, "collision", resolution, 7)
    radial = meta["case"] == "collision_1d"
    dimension = 2 if meta["case"] in {"collision_1d", "collision_2d"} else 3
    radius = 0.2
    mass_i = math.pi/6.0
    mass_j = math.pi*8.0/6.0
    if radial:
        annulus = lambda center, width: math.pi*(min(Y_MAX, center + width)**2
                                                  - max(Y_MIN, center - width)**2)
        interior_measure = annulus(1.0, radius)
        inner_measure = annulus(Y_MIN + 0.25*radius, radius)
        polar_measure = annulus(Y_MAX - 0.25*radius, radius)
        both_measure = annulus(1.0, 0.6)
    else:
        interior_measure = math.pi*radius**2 if dimension == 2 else 4.0*math.pi*radius**3/3.0
        inner_measure = ball_measure(dimension, radius, 0.25*radius)
        polar_measure = interior_measure if dimension == 2 else ball_measure(dimension, radius, 0.25*radius)
        both_measure = 0.0
    expected = np.array([
        interior_measure,
        inner_measure,
        polar_measure,
        0.3*7.0,
        0.3*7.0*(mass_i + mass_j),
        0.3*7.0*mass_i*mass_j,
        both_measure,
    ])
    error = values - expected
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"collision": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze_import(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    values = read_array(out_dir, "import", resolution, 2*int(meta["np"])).reshape(2, -1)
    iy = 2*np.arange(values.shape[1])
    sigma_g = 1.0 + 0.1*iy
    expected_stokes = STOKES_0/sigma_g
    expected_re = 1.0/np.sqrt(1.0e8*sigma_g)
    error = np.concatenate((values[0] - expected_stokes, values[1] - expected_re))
    return {
        "case": meta["case"], "resolution": resolution,
        "errors": {"import": norms(error)},
        "passed": bool(np.max(np.abs(error)) < 2.0e-13),
    }


def analyze_boundary(out_dir: Path, resolution: int, meta: dict[str, str]) -> dict:
    values = read_array(out_dir, "boundary", resolution, 42).reshape(7, 6)
    nx, nz = int(meta["nx"]), int(meta["nz"])
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    z_min, z_max = float(meta["z_min"]), float(meta["z_max"])
    x_width = x_max - x_min
    y_width = y_max - y_min
    z_width = z_max - z_min if nz > 1 else 0.0
    z_mid = 0.5*(z_min + z_max) if nz > 1 else 0.5*math.pi
    halfdisk = meta["case"] == "boundary_half"

    def wrap_x(x: float) -> float:
        if nx == 1:
            return 0.5*(x_min + x_max)
        return x_min + (x - x_min) % x_width

    def reflect(value: float, lower: float, upper: float) -> float:
        while value < lower or value > upper:
            if value < lower:
                value = 2.0*lower - value
            if value > upper:
                value = 2.0*upper - value
        if value >= upper:
            value = upper - 1.0e-12*(upper - lower)
        return value

    expected = np.zeros((7, 6))
    expected[0] = (
        wrap_x(x_max + 0.25*x_width),
        reflect(y_min - 0.125*y_width, y_min, y_max),
        reflect(z_min - 0.125*z_width, z_min, z_max) if nz > 1 else 0.5*math.pi,
        1.0, -2.0, 3.0,
    )
    expected[1] = (
        wrap_x(x_min - 0.25*x_width),
        reflect(y_max + 0.125*y_width, y_min, y_max),
        reflect(z_max + 0.125*z_width, z_min, z_max) if nz > 1 else 0.5*math.pi,
        1.0, -2.0, 3.0,
    )
    expected[2] = (
        wrap_x(0.5*(x_min + x_max)),
        reflect(y_min - 2.25*y_width, y_min, y_max),
        z_mid,
        1.0, -2.0, 3.0,
    )

    # Both radial transport exits are absorbed after periodic azimuthal wrapping.
    expected[3] = (wrap_x(x_max + 0.25*x_width), 0.0, 0.5*math.pi, 0.0, 0.0, 0.0)
    expected[4] = (wrap_x(x_min - 0.25*x_width), 0.0, 0.5*math.pi, 0.0, 0.0, 0.0)

    if nz == 1:
        expected[5] = (wrap_x(x_max + 0.25*x_width), 1.0, 0.5*math.pi, 1.0, -2.0, 0.0)
        expected[6] = (wrap_x(x_min - 0.25*x_width), 1.0, 0.5*math.pi, 1.0, 2.0, 0.0)
    else:
        expected[5] = (wrap_x(x_max + 0.25*x_width), 0.0, 0.5*math.pi, 0.0, 0.0, 0.0)
        if halfdisk:
            expected[6] = (
                wrap_x(x_min - 0.25*x_width), 1.0,
                math.pi - (z_max + 0.125*z_width), 1.0, 2.0, -3.0,
            )
        else:
            expected[6] = (wrap_x(x_min - 0.25*x_width), 0.0, 0.5*math.pi, 0.0, 0.0, 0.0)

    error = values - expected
    return {
        "case": meta["case"],
        "resolution": resolution,
        "errors": {"boundary": norms(error)},
        "passed": bool(np.all(np.isfinite(values)) and np.max(np.abs(error)) < 2.0e-13),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = read_meta(out_dir / f"meta_N{resolution}.txt")
    if int(meta["resolution"]) != resolution:
        raise ValueError(
            f"metadata resolution {meta['resolution']} does not match requested N={resolution}"
        )
    case = meta["case"]
    if case.startswith("grid_"):
        result = analyze_grid(out_dir, resolution, meta)
    elif case in {"orbit_1d", "orbit_2d"}:
        result = analyze_orbit(out_dir, resolution, meta)
    elif case in {"drag_1d", "drag_2d", "radiation_1d", "radiation_2d", "prdrag_1d", "prdrag_2d"}:
        result = analyze_relaxation(out_dir, resolution, meta)
    elif case == "viscflow_1d":
        result = analyze_viscflow(out_dir, resolution, meta)
    elif case.startswith("diffusion_"):
        result = analyze_diffusion(out_dir, resolution, meta)
    elif case == "initial_3d":
        result = analyze_initialization(out_dir, resolution, meta)
    elif case.startswith("collision_"):
        result = analyze_collision(out_dir, resolution, meta)
    elif case == "import_1d":
        result = analyze_import(out_dir, resolution, meta)
    elif case.startswith("boundary_"):
        result = analyze_boundary(out_dir, resolution, meta)
    else:
        raise ValueError(f"unknown swarm verification case: {case}")
    if not result["passed"]:
        raise AssertionError(f"{case} failed at resolution {resolution}: {result}")
    return result
