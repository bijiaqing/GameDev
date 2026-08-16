#!/usr/bin/env python3

"""Compare verification output with analytical finite-volume reference fields

The CUDA driver writes raw float64 arrays plus a metadata text file.  This
module reconstructs the same grid, evaluates analytical *cell averages* rather
than point samples, computes geometry-weighted error norms, and returns a plain
dictionary suitable for JSON serialization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# These constants mirror test_common/const_defs.cuh.  Keeping the analytical
# parameters here makes validation independent of the numerical output itself;
# an incorrect simulation cannot silently redefine its expected answer.
Y_MIN, Y_MAX = 0.5, 2.5
X_MIN, X_MAX = 0.0, 2.0 * math.pi
Q0, EPS, MODE = 1.0, 0.1, 2
DIFFUSIVITY = 5.0e-2
K2 = 1.694299217770420
K3 = 1.874562003084784


def read_meta(path: Path) -> dict[str, object]:
    """Read and validate the CUDA driver's structured metadata record"""

    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return values


def read_field(out_dir: Path, name: str, resolution: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Load one raw float64 field and restore its (z, y, x) grid shape"""

    path = out_dir / f"{name}_N{resolution}.dat"
    data = np.fromfile(path, dtype=np.float64)
    expected = math.prod(shape)
    if data.size != expected:
        raise ValueError(f"{path} contains {data.size} values; expected {expected}")
    # CUDA uses idx = ix + iy*N_X + iz*N_X*N_Y, so x is the fastest-varying
    # index.  C-order reshape into (nz, ny, nx) reproduces exactly that layout.
    return data.reshape(shape)


def gauss_average(function, lower: np.ndarray, upper: np.ndarray, measure=None) -> np.ndarray:
    """Integrate a function over many cells with 16-point Gauss-Legendre rules"""

    # lower and upper may be arrays.  Appending a quadrature axis with [..., None]
    # evaluates all cells at once through NumPy broadcasting.  Despite the
    # historical name, this function returns integrals; callers divide by the
    # appropriate geometric cell volume when they require cell averages.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    midpoint = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    points = midpoint[..., None] + radius[..., None] * nodes
    values = function(points)
    if measure is not None:
        values = values * measure(points)
    return radius * np.sum(values * weights, axis=-1)


def compact_bump(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Evaluate a smooth compact-support profile that vanishes at both edges"""

    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    u = (value - center) / half_width
    result = np.zeros_like(value)
    # Evaluate the exponential only inside its support to avoid division by zero
    # at |u|=1 and meaningless overflow outside the bump.
    active = np.abs(u) < 1.0
    result[active] = np.exp(1.0 - 1.0 / (1.0 - u[active] ** 2))
    return result


def radial_mode_table(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the radial Neumann eigenmode used by the diffusion exact solution"""

    # The mode satisfies q'' + (d-1)q'/r + k**2 q = 0.  K2 and K3 are selected
    # so q'=0 at both radial boundaries, while q(Y_MIN)=1 fixes normalization.
    # A dense RK4 table avoids depending on SciPy Bessel functions and is later
    # interpolated only for high-order cell integration.
    k = K2 if dimension == 2 else K3
    count = 200_001
    y = np.linspace(Y_MIN, Y_MAX, count)
    step = y[1] - y[0]
    q = np.empty(count)
    derivative = np.empty(count)
    q[0], derivative[0] = 1.0, 0.0

    def rhs(radius: float, value: float, slope: float) -> tuple[float, float]:
        # Convert the second-order eigenvalue equation into two first-order ODEs.
        return slope, -(dimension - 1.0) * slope / radius - k * k * value

    # Classical fourth-order Runge-Kutta advances q and q' together on the dense
    # reference grid.  k1 through k4 are slopes sampled across one radial step.
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
    """Compute volume-weighted L1, L2, and unweighted maximum error norms"""

    total_volume = np.sum(volume)
    return {
        "l1": float(np.sum(volume * np.abs(error)) / total_volume),
        "l2": float(np.sqrt(np.sum(volume * error * error) / total_volume)),
        "linf": float(np.max(np.abs(error))),
    }


def gas_density(radius: np.ndarray, beta: float) -> np.ndarray:
    """Return the 2D gas surface-density profile configured for the ring tests"""

    power = 2.0 - 4.0 * beta
    return radius**power


def analyze(out_dir: Path, resolution: int) -> dict:
    """Analyze one completed case and return all scalar metrics"""

    # Metadata tells the validator which compile-time branch produced the files
    # and provides the realized grid and final integration time.
    meta = read_meta(out_dir / f"meta_N{resolution}.json")
    case = meta["case"]
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    time = float(meta["time"])
    shape = (nz, ny, nx)

    # Reconstruct azimuthal faces and centers on the uniform periodic mesh.
    dx = (X_MAX - X_MIN) / nx
    x0 = X_MIN + np.arange(nx) * dx
    x1 = x0 + dx
    xc = 0.5 * (x0 + x1)

    # Radial cells are logarithmically spaced, so their centers are geometric
    # rather than arithmetic means of the two faces.
    ratio = (Y_MAX / Y_MIN) ** (1.0 / ny)
    yf = Y_MIN * ratio ** np.arange(ny + 1)
    y0, y1 = yf[:-1], yf[1:]
    yc = np.sqrt(y0 * y1)

    if nz == 1:
        # A single polar cell represents the 2D cylindrical midplane model.
        zf = np.array([0.5 * np.pi, 0.5 * np.pi])
        zc = np.array([0.5 * np.pi])
        polar_volume = np.ones(1)
        dimension = 2
    else:
        # Polar diffusion uses a hemisphere so the Legendre mode has natural
        # zero-flux boundaries.  Other 3D tests avoid the coordinate poles.
        if case == "z_diffusion":
            zf = np.linspace(0.0, 0.5 * np.pi, nz + 1)
        else:
            zf = np.linspace(0.35, np.pi - 0.35, nz + 1)
        zc = 0.5 * (zf[:-1] + zf[1:])
        polar_volume = np.cos(zf[:-1]) - np.cos(zf[1:])
        dimension = 3

    # Only relative weights enter the normalized norms and relative mass change,
    # so the common azimuthal width dx can be omitted from every cell volume.
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
        # For rho=r**p and unit opacity, tau(r_face)=integral rho dr.  p=-1 has
        # the logarithmic antiderivative and all other powers use r**(p+1)/(p+1).
        tau = read_field(out_dir, "optdepth_final", resolution, shape)
        power = float(meta["power"])
        if abs(power + 1.0) < 1.0e-14:
            exact_line = np.log(yf[1:] / Y_MIN)
        else:
            exact_line = (yf[1:] ** (power + 1.0) - Y_MIN ** (power + 1.0)) / (power + 1.0)
        exact = np.broadcast_to(exact_line[None, :, None], shape)
        results["errors"]["optdepth"] = norm_set(tau - exact, volume)
        return results

    # Dynamical tests save both conserved fields and physical output velocities.
    # The initial density is retained solely for the mass-conservation metric.
    rhod = read_field(out_dir, "dustdens_final", resolution, shape)
    mx = read_field(out_dir, "dustmomx_final", resolution, shape)
    my = read_field(out_dir, "dustmomy_final", resolution, shape)
    mz = read_field(out_dir, "dustmomz_final", resolution, shape)
    velx = read_field(out_dir, "dustvelx_final", resolution, shape)
    vely = read_field(out_dir, "dustvely_final", resolution, shape)
    velz = read_field(out_dir, "dustvelz_final", resolution, shape)
    rhod_initial = read_field(out_dir, "dustdens_initial", resolution, shape)

    # Allocate full analytical fields once; the case branch below fills every
    # entry using broadcasting or explicit radial loops.
    exact_rhod = np.empty(shape)
    exact_mx = np.empty(shape)
    exact_my = np.empty(shape)
    exact_mz = np.empty(shape)

    if case == "x_transport":
        # A sinusoidal cell average translates by angular speed one.  The x
        # momentum follows density times the prescribed specific angular momentum R**2.
        mode_avg = (np.sin(MODE * (x1 - time)) - np.sin(MODE * (x0 - time))) / (MODE * dx)
        exact_rhod[:] = Q0 + EPS * mode_avg[None, None, :]
        radius = yc[None, :, None]
        exact_mx[:] = exact_rhod * radius * radius
        exact_my.fill(0.0)
        exact_mz.fill(0.0)
    elif case == "x_diffusion":
        # Each azimuthal Fourier mode decays as exp(-D*m**2*t/R**2), with the
        # spherical metric introducing the radius-dependent R**-2 factor.
        mode_avg = (np.sin(MODE * x1) - np.sin(MODE * x0)) / (MODE * dx)
        radius = yc[None, :, None]
        decay = np.exp(-DIFFUSIVITY * MODE * MODE * time / (radius * radius))
        exact_rhod[:] = Q0 + EPS * decay * mode_avg[None, None, :]
        exact_mx[:] = 0.7 * exact_rhod
        exact_my[:] = -0.15 * exact_rhod
        exact_mz[:] = 0.11 * exact_rhod
    elif case.startswith("y_transport"):
        # The prescribed homologous flow v_y=a*y expands coordinates by
        # lambda=1+a*t and dilutes density by lambda**dimension.  Quadrature
        # produces exact finite-volume density and radial momentum averages.
        lam = 1.0 + 0.2 * time
        measure = lambda y: y ** (dimension - 1)
        rho_int = gauss_average(
            lambda y: lam ** (-dimension) * compact_bump(y / lam, 1.0, 1.8), y0, y1, measure
        )
        my_int = gauss_average(
            lambda y: lam ** (-dimension) * compact_bump(y / lam, 1.0, 1.8) * 0.2 * y / lam,
            y0, y1, measure,
        )
        line_rhod = rho_int / radial_volume
        line_my = my_int / radial_volume
        exact_rhod[:] = line_rhod[None, :, None]
        exact_mx[:] = 0.7 * exact_rhod
        exact_my[:] = line_my[None, :, None]
        exact_mz[:] = 0.11 * exact_rhod
    elif case == "z_transport":
        # Choose one shell representative polar velocity so the exact integrated
        # polar face-area-to-volume factor gives angular rate 0.15 at every y.
        area_z = 0.5 * (y1**2 - y0**2)
        geom_z = area_z / radial_volume
        shell_lz = 0.15 * yc / geom_z
        shift = 0.15 * time
        for j in range(ny):
            rho_int = gauss_average(lambda z: compact_bump(z - shift, 0.80, 1.30), zf[:-1], zf[1:])
            exact_rhod[:, j, :] = (rho_int / polar_volume)[:, None]
        exact_mx[:] = 0.7 * exact_rhod
        exact_my[:] = 0.05 * exact_rhod
        exact_mz[:] = shell_lz[None, :, None] * exact_rhod
    elif case.startswith("y_diffusion"):
        # A radial Laplacian eigenmode preserves its shape and decays globally as
        # exp(-D*k**2*t).  The 2D and 3D cases use their corresponding eigenvalue
        # and radial volume measure.
        table_y, table_q = radial_mode_table(dimension)
        mode_int = gauss_average(
            lambda y: np.interp(y, table_y, table_q), y0, y1,
            lambda y: y ** (dimension - 1),
        )
        mode_avg = mode_int / radial_volume
        k = K2 if dimension == 2 else K3
        line = Q0 + EPS * mode_avg * np.exp(-DIFFUSIVITY * k * k * time)
        exact_rhod[:] = line[None, :, None]
        exact_mx[:] = 0.7 * exact_rhod
        exact_my[:] = -0.15 * exact_rhod
        exact_mz[:] = 0.11 * exact_rhod
    elif case == "z_diffusion":
        # P2(cos z) is a spherical angular Laplacian eigenfunction with eigenvalue
        # -l(l+1)=-6.  Physical angular diffusion therefore decays as
        # exp(-6*D*t/R**2) independently on each radial shell.
        mode_int = gauss_average(
            lambda z: 0.5 * (3.0 * np.cos(z) ** 2 - 1.0), zf[:-1], zf[1:], np.sin
        )
        mode_avg = mode_int / polar_volume
        for j, radius in enumerate(yc):
            line = Q0 + EPS * mode_avg * np.exp(-DIFFUSIVITY * 6.0 * time / (radius * radius))
            exact_rhod[:, j, :] = line[:, None]
        exact_mx[:] = 0.7 * exact_rhod
        exact_my[:] = -0.15 * exact_rhod
        exact_mz[:] = 0.11 * exact_rhod
    elif case == "source_drag":
        # The eight x cells represent eight values of dt/ts spanning twelve
        # orders of magnitude; x is being used as a parameter index, not space.
        stiffness = np.array([1.0e-6, 1.0e-3, 0.1, 1.0, 10.0, 1.0e2, 1.0e4, 1.0e6])
        ts = time / stiffness
        decay = np.exp(-stiffness)
        drag_relax = -np.expm1(-stiffness)

        def source_exact(initial, gas, force0, force1):
            # Closed-form solution of dv/dt=-(v-v_g)/ts+F(t) for a force that
            # varies linearly from force0 to force1 during the single step.  The
            # two endpoint weights mirror the CUDA branch: direct subtraction is
            # safe for ordinary h=dt/ts, while a series avoids cancellation when
            # h is small.
            weight_old = np.empty_like(stiffness)
            weight_new = np.empty_like(stiffness)
            small = stiffness < 1.0e-4

            h = stiffness[small]
            h2 = h*h
            h3 = h2*h
            weight_old[small] = time*(0.5 - h/3.0 + h2/8.0  - h3/30.0)
            weight_new[small] = time*(0.5 - h/6.0 + h2/24.0 - h3/120.0)

            regular = ~small
            weight_new[regular] = (
                ts[regular]*(stiffness[regular] - drag_relax[regular]) / stiffness[regular]
            )
            weight_old[regular] = ts[regular]*drag_relax[regular] - weight_new[regular]

            return (gas + (initial - gas)*decay
                    + force0*weight_old + force1*weight_new)

        vx = source_exact(1.3, 0.4, -0.3, 0.2)
        vy = source_exact(-0.8, -0.2, 0.7, -0.1)
        vz = source_exact(0.6, 0.1, -0.5, 0.9)
        exact_rhod.fill(1.0)
        exact_mx[:] = vx[None, None, :]
        exact_my[:] = vy[None, None, :]
        exact_mz[:] = vz[None, None, :]
    elif case.startswith("ring_"):
        # Combined ring tests rotate each radial shell at its radiation-modified
        # Keplerian omega.  Optional azimuthal density diffusion damps the
        # Fourier mode on top of the prescribed radial density profile.
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
        exact_rhod[:] = (rho_g[:, None] * mode_values)[None, :, :]
        exact_mx[:] = exact_rhod * ell[None, :, None]
        exact_my.fill(0.0)
        exact_mz.fill(0.0)
    else:
        raise ValueError(f"Unknown verification case {case}")

    # Compare conserved fields directly before converting the analytical angular
    # momenta into the physical velocities written by save_state.
    for name, numerical, exact in (
        ("density", rhod, exact_rhod),
        ("momx", mx, exact_mx),
        ("momy", my, exact_my),
        ("momz", mz, exact_mz),
    ):
        results["errors"][name] = norm_set(numerical - exact, volume)

    # Velocity errors are meaningful only where both solutions resolve dust
    # density.  Requiring numerical density excludes cells where the production
    # recovery deliberately writes a vacuum fallback velocity.  Scale the floor
    # to the analytical solution so the mask remains dimensionally meaningful.
    density_floor = 1.0e-12*float(np.max(np.abs(exact_rhod)))
    active = (exact_rhod > density_floor) & (rhod > density_floor)
    active_volume = np.where(active, volume, 0.0)
    radius = yc[None, :, None] * np.sin(zc)[:, None, None]
    sphere_radius = yc[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        exact_velx = np.where(active, exact_mx / exact_rhod / radius, 0.0)
        exact_vely = np.where(active, exact_my / exact_rhod, 0.0)
        exact_velz = np.where(active, exact_mz / exact_rhod / sphere_radius, 0.0)
    for name, numerical, exact in (
        ("velx", velx, exact_velx),
        ("vely", vely, exact_vely),
        ("velz", velz, exact_velz),
    ):
        results["errors"][name] = norm_set(np.where(active, numerical - exact, 0.0), active_volume)

    if case in {"ring_radiation_2d", "ring_all_2d"}:
        # Ring radiation tests set KAPPA_0=0 to isolate radiation acceleration
        # without attenuation, making zero optical depth the analytical answer.
        tau = read_field(out_dir, "optdepth_final", resolution, shape)
        results["errors"]["optdepth"] = norm_set(tau, volume)

    # The common omitted azimuthal factor cancels from this relative change.
    mass_initial = float(np.sum(volume * rhod_initial))
    mass_final = float(np.sum(volume * rhod))
    results["mass_relative_change"] = abs(mass_final - mass_initial) / max(abs(mass_initial), 1.0e-300)
    return results


def main() -> None:
    # This standalone interface is useful for reanalyzing files after copying
    # them from a cluster.  --write optionally stores the same JSON that is shown.
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
