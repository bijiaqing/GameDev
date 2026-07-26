#!/usr/bin/env python3
"""Independent post-processing for the CUDA verification models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


Y_MIN, Y_MAX = 0.5, 2.5
X_MIN, X_MAX = 0.0, 2.0 * math.pi
Q0, EPS, MODE = 1.0, 0.1, 2
DIFFUSIVITY = 5.0e-2
K2 = 1.694299217770420
K3 = 1.874562003084784


def read_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_field(out_dir: Path, name: str, resolution: int, shape: tuple[int, int, int]) -> np.ndarray:
    path = out_dir / f"{name}_N{resolution}.dat"
    data = np.fromfile(path, dtype=np.float64)
    expected = math.prod(shape)
    if data.size != expected:
        raise ValueError(f"{path} contains {data.size} values; expected {expected}")
    return data.reshape(shape)


def gauss_average(function, lower: np.ndarray, upper: np.ndarray, measure=None) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(16)
    midpoint = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    points = midpoint[..., None] + radius[..., None] * nodes
    values = function(points)
    if measure is not None:
        values = values * measure(points)
    return radius * np.sum(values * weights, axis=-1)


def compact_bump(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    u = (value - center) / half_width
    result = np.zeros_like(value)
    active = np.abs(u) < 1.0
    result[active] = np.exp(1.0 - 1.0 / (1.0 - u[active] ** 2))
    return result


def radial_mode_table(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent RK4 integration of Q''+(d-1)Q'/y+k^2Q=0."""
    k = K2 if dimension == 2 else K3
    count = 200_001
    y = np.linspace(Y_MIN, Y_MAX, count)
    step = y[1] - y[0]
    q = np.empty(count)
    derivative = np.empty(count)
    q[0], derivative[0] = 1.0, 0.0

    def rhs(radius: float, value: float, slope: float) -> tuple[float, float]:
        return slope, -(dimension - 1.0) * slope / radius - k * k * value

    for i in range(count - 1):
        r0, q0, p0 = y[i], q[i], derivative[i]
        k1q, k1p = rhs(r0, q0, p0)
        k2q, k2p = rhs(r0 + 0.5 * step, q0 + 0.5 * step * k1q, p0 + 0.5 * step * k1p)
        k3q, k3p = rhs(r0 + 0.5 * step, q0 + 0.5 * step * k2q, p0 + 0.5 * step * k2p)
        k4q, k4p = rhs(r0 + step, q0 + step * k3q, p0 + step * k3p)
        q[i + 1] = q0 + step * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
        derivative[i + 1] = p0 + step * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    return y, q


def norm_set(error: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    total_volume = np.sum(volume)
    return {
        "l1": float(np.sum(volume * np.abs(error)) / total_volume),
        "l2": float(np.sqrt(np.sum(volume * error * error) / total_volume)),
        "linf": float(np.max(np.abs(error))),
    }


def gas_density(radius: np.ndarray, beta: float) -> np.ndarray:
    aspect = 0.5
    power = 2.0 - 4.0 * beta
    sigma = radius**power
    return sigma / (np.sqrt(2.0 * np.pi) * aspect * radius)


def analyze(out_dir: Path, resolution: int) -> dict:
    meta = read_meta(out_dir / f"meta_N{resolution}.txt")
    case = meta["case"]
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    time = float(meta["time"])
    shape = (nz, ny, nx)

    dx = (X_MAX - X_MIN) / nx
    x0 = X_MIN + np.arange(nx) * dx
    x1 = x0 + dx
    xc = 0.5 * (x0 + x1)

    ratio = (Y_MAX / Y_MIN) ** (1.0 / ny)
    yf = Y_MIN * ratio ** np.arange(ny + 1)
    y0, y1 = yf[:-1], yf[1:]
    yc = np.sqrt(y0 * y1)

    if nz == 1:
        zf = np.array([0.5 * np.pi, 0.5 * np.pi])
        zc = np.array([0.5 * np.pi])
        polar_volume = np.ones(1)
        dimension = 2
    else:
        if case == "z_diffusion":
            zf = np.linspace(0.0, 0.5 * np.pi, nz + 1)
        else:
            zf = np.linspace(0.35, np.pi - 0.35, nz + 1)
        zc = 0.5 * (zf[:-1] + zf[1:])
        polar_volume = np.cos(zf[:-1]) - np.cos(zf[1:])
        dimension = 3

    radial_volume = (y1**dimension - y0**dimension) / dimension
    volume = polar_volume[:, None, None] * radial_volume[None, :, None] * np.ones((1, 1, nx))

    results: dict = {
        "case": case,
        "resolution": resolution,
        "time": time,
        "steps": int(meta["steps"]),
        "cfl": float(meta["cfl"]),
        "errors": {},
    }

    if case == "optdepth":
        tau = read_field(out_dir, "optdepth_final", resolution, shape)
        power = float(meta["power"])
        if abs(power + 1.0) < 1.0e-14:
            exact_line = np.log(yf[1:] / Y_MIN)
        else:
            exact_line = (yf[1:] ** (power + 1.0) - Y_MIN ** (power + 1.0)) / (power + 1.0)
        exact = np.broadcast_to(exact_line[None, :, None], shape)
        results["errors"]["optdepth"] = norm_set(tau - exact, volume)
        return results

    dens = read_field(out_dir, "dustdens_final", resolution, shape)
    momx = read_field(out_dir, "dustmomx_final", resolution, shape)
    momy = read_field(out_dir, "dustmomy_final", resolution, shape)
    momz = read_field(out_dir, "dustmomz_final", resolution, shape)
    velx = read_field(out_dir, "dustvelx_final", resolution, shape)
    vely = read_field(out_dir, "dustvely_final", resolution, shape)
    velz = read_field(out_dir, "dustvelz_final", resolution, shape)
    dens_initial = read_field(out_dir, "dustdens_initial", resolution, shape)

    exact_dens = np.empty(shape)
    exact_momx = np.empty(shape)
    exact_momy = np.empty(shape)
    exact_momz = np.empty(shape)

    if case == "x_transport":
        mode_avg = (np.sin(MODE * (x1 - time)) - np.sin(MODE * (x0 - time))) / (MODE * dx)
        exact_dens[:] = Q0 + EPS * mode_avg[None, None, :]
        radius = yc[None, :, None]
        exact_momx[:] = exact_dens * radius * radius
        exact_momy.fill(0.0)
        exact_momz.fill(0.0)
    elif case == "x_diffusion":
        mode_avg = (np.sin(MODE * x1) - np.sin(MODE * x0)) / (MODE * dx)
        radius = yc[None, :, None]
        decay = np.exp(-DIFFUSIVITY * MODE * MODE * time / (radius * radius))
        exact_dens[:] = Q0 + EPS * decay * mode_avg[None, None, :]
        exact_momx[:] = 0.7 * exact_dens
        exact_momy[:] = -0.15 * exact_dens
        exact_momz[:] = 0.11 * exact_dens
    elif case.startswith("y_transport"):
        lam = 1.0 + 0.2 * time
        measure = lambda y: y ** (dimension - 1)
        rho_int = gauss_average(
            lambda y: lam ** (-dimension) * compact_bump(y / lam, 1.0, 1.8), y0, y1, measure
        )
        momy_int = gauss_average(
            lambda y: lam ** (-dimension) * compact_bump(y / lam, 1.0, 1.8) * 0.2 * y / lam,
            y0, y1, measure,
        )
        line_dens = rho_int / radial_volume
        line_momy = momy_int / radial_volume
        exact_dens[:] = line_dens[None, :, None]
        exact_momx[:] = 0.7 * exact_dens
        exact_momy[:] = line_momy[None, :, None]
        exact_momz[:] = 0.11 * exact_dens
    elif case == "z_transport":
        for j, radius in enumerate(yc):
            shift = 0.15 * time / (radius * radius)
            rho_int = gauss_average(lambda z: compact_bump(z - shift, 0.80, 1.30), zf[:-1], zf[1:])
            exact_dens[:, j, :] = (rho_int / polar_volume)[:, None]
        exact_momx[:] = 0.7 * exact_dens
        exact_momy[:] = 0.05 * exact_dens
        exact_momz[:] = 0.15 * exact_dens
    elif case.startswith("y_diffusion"):
        table_y, table_q = radial_mode_table(dimension)
        mode_int = gauss_average(
            lambda y: np.interp(y, table_y, table_q), y0, y1,
            lambda y: y ** (dimension - 1),
        )
        mode_avg = mode_int / radial_volume
        k = K2 if dimension == 2 else K3
        line = Q0 + EPS * mode_avg * np.exp(-DIFFUSIVITY * k * k * time)
        exact_dens[:] = line[None, :, None]
        exact_momx[:] = 0.7 * exact_dens
        exact_momy[:] = -0.15 * exact_dens
        exact_momz[:] = 0.11 * exact_dens
    elif case == "z_diffusion":
        mode_int = gauss_average(
            lambda z: 0.5 * (3.0 * np.cos(z) ** 2 - 1.0), zf[:-1], zf[1:], np.sin
        )
        mode_avg = mode_int / polar_volume
        for j, radius in enumerate(yc):
            line = Q0 + EPS * mode_avg * np.exp(-DIFFUSIVITY * 6.0 * time / (radius * radius))
            exact_dens[:, j, :] = line[:, None]
        exact_momx[:] = 0.7 * exact_dens
        exact_momy[:] = -0.15 * exact_dens
        exact_momz[:] = 0.11 * exact_dens
    elif case == "source_drag":
        stiffness = np.array([1.0e-6, 1.0e-3, 0.1, 1.0, 10.0, 1.0e2, 1.0e4, 1.0e6])
        ts = time / stiffness
        decay = np.exp(-stiffness)

        def source_exact(initial, gas, force0, force1):
            slope = (force1 - force0) / time
            return (gas + (initial - gas) * decay + force0 * ts * (1.0 - decay)
                    + slope * (ts * time - ts * ts * (1.0 - decay)))

        vx = source_exact(1.3, 0.4, -0.3, 0.2)
        vy = source_exact(-0.8, -0.2, 0.7, -0.1)
        vz = source_exact(0.6, 0.1, -0.5, 0.9)
        exact_dens.fill(1.0)
        exact_momx[:] = vx[None, None, :]
        exact_momy[:] = vy[None, None, :]
        exact_momz[:] = vz[None, None, :]
    elif case.startswith("ring_"):
        beta = 0.2 if case in {"ring_radiation_2d", "ring_all_2d"} else 0.0
        has_diffusion = case in {"ring_diffusion_2d", "ring_all_2d"}
        mode_values = np.empty((ny, nx))
        ell = np.empty(ny)
        rho_g = gas_density(yc, beta)
        for j, radius in enumerate(yc):
            omega = math.sqrt((1.0 - beta) / radius**3)
            decay = math.exp(-DIFFUSIVITY * MODE * MODE * time / radius**2) if has_diffusion else 1.0
            average = (np.sin(MODE * (x1 - omega * time))
                     - np.sin(MODE * (x0 - omega * time))) / (MODE * dx)
            mode_values[j] = Q0 + EPS * decay * average
            ell[j] = math.sqrt((1.0 - beta) * radius)
        exact_dens[:] = (rho_g[:, None] * mode_values)[None, :, :]
        exact_momx[:] = exact_dens * ell[None, :, None]
        exact_momy.fill(0.0)
        exact_momz.fill(0.0)
    else:
        raise ValueError(f"Unknown verification case {case}")

    for name, numerical, exact in (
        ("density", dens, exact_dens),
        ("momx", momx, exact_momx),
        ("momy", momy, exact_momy),
        ("momz", momz, exact_momz),
    ):
        results["errors"][name] = norm_set(numerical - exact, volume)

    active = exact_dens > 1.0e-12
    active_volume = np.where(active, volume, 0.0)
    radius = yc[None, :, None] * np.sin(zc)[:, None, None]
    sphere_radius = yc[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        exact_velx = np.where(active, exact_momx / exact_dens / radius, 0.0)
        exact_vely = np.where(active, exact_momy / exact_dens, 0.0)
        exact_velz = np.where(active, exact_momz / exact_dens / sphere_radius, 0.0)
    for name, numerical, exact in (
        ("velx", velx, exact_velx),
        ("vely", vely, exact_vely),
        ("velz", velz, exact_velz),
    ):
        results["errors"][name] = norm_set(np.where(active, numerical - exact, 0.0), active_volume)

    if "diffusion" in case or case.startswith("ring_"):
        if case.startswith("ring_"):
            beta = 0.2 if case in {"ring_radiation_2d", "ring_all_2d"} else 0.0
            rho_g = gas_density(yc, beta)[None, :, None]
        else:
            rho_g = 1.0
        results["errors"]["ratio"] = norm_set(
            dens/rho_g - exact_dens/rho_g, volume
        )

    if case in {"ring_radiation_2d", "ring_all_2d"}:
        tau = read_field(out_dir, "optdepth_final", resolution, shape)
        results["errors"]["optdepth"] = norm_set(tau, volume)

    mass_initial = float(np.sum(volume * dens_initial))
    mass_final = float(np.sum(volume * dens))
    results["mass_relative_change"] = abs(mass_final - mass_initial) / max(abs(mass_initial), 1.0e-300)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("resolution", type=int)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    result = analyze(args.out_dir, args.resolution)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write:
        args.write.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
