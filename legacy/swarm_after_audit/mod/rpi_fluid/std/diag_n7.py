#!/usr/bin/env python3
"""Mock the conservative invariant-domain correction proposed for N7."""

import numpy as np

import review_mock_tests as R
from diag_y import A, TEND, Y_MAX, Y_MIN, compact_bump


def theta_state(dens, mom, ddens, dmom, vel_min, vel_max):
    theta = 1.0

    def restrict(available, change):
        nonlocal theta
        if change < 0.0:
            if available <= 0.0:
                theta = 0.0
            else:
                theta = min(theta, (1.0 - 1.0e-12)*available/(-change))

    restrict(dens, ddens)
    restrict(mom - vel_min*dens, dmom - vel_min*ddens)
    restrict(vel_max*dens - mom, vel_max*ddens - dmom)
    return max(0.0, min(1.0, theta))


def velocity_bounds(vel, idx):
    lo = max(0, idx - 1)
    hi = min(len(vel), idx + 2)
    vel_min = np.min(vel[lo:hi])
    vel_max = np.max(vel[lo:hi])
    pad = 1.0e-12*max(1.0, abs(vel_min), abs(vel_max))
    return vel_min - pad, vel_max + pad


def euler_fct(dens, mom, dt, yf, dimension, weight):
    count = len(dens)
    volume = (yf[1:]**dimension - yf[:-1]**dimension)/dimension
    area = yf**(dimension - 1.0)
    vel = np.where(dens >= R.RHO_VAC, mom/np.maximum(dens, 1.0e-300), 0.0)

    edge_dens = R.ppm_edges_nonuniform(dens, weight)
    edge_vel = R.ppm_edges_nonuniform(vel, weight)
    flux_dens = np.zeros(count + 1)
    flux_mom = np.zeros(count + 1)
    corr_dens = np.zeros(count + 1)
    corr_mom = np.zeros(count + 1)

    speed = vel[0]
    if speed < 0.0:
        flux_dens[0] = speed*max(dens[0], 0.0)
        flux_mom[0] = flux_dens[0]*vel[0]

    for face in range(1, count):
        left = face - 1
        right = face

        dens_l = max(R.ppm_face_value(edge_dens, dens, left, face, True, 0.0), 0.0)
        dens_r = max(R.ppm_face_value(edge_dens, dens, right, face + 1, False, 0.0), 0.0)
        vel_l = R.ppm_face_value(edge_vel, vel, left, face, True, 0.0)
        vel_r = R.ppm_face_value(edge_vel, vel, right, face + 1, False, 0.0)
        high_dens, high_mom = R.hll_2(
            vel_l, vel_r, dens_l, dens_l*vel_l, dens_r, dens_r*vel_r
        )
        low_dens, low_mom = R.hll_2(
            vel[left], vel[right], dens[left], mom[left], dens[right], mom[right]
        )

        flux_dens[face] = low_dens
        flux_mom[face] = low_mom
        corr_dens[face] = high_dens - low_dens
        corr_mom[face] = high_mom - low_mom

    speed = vel[-1]
    if speed > 0.0:
        flux_dens[-1] = speed*max(dens[-1], 0.0)
        flux_mom[-1] = flux_dens[-1]*vel[-1]

    dens_new = dens - dt*(area[1:]*flux_dens[1:] - area[:-1]*flux_dens[:-1])/volume
    mom_new = mom - dt*(area[1:]*flux_mom[1:] - area[:-1]*flux_mom[:-1])/volume

    for face in range(1, count):
        left = face - 1
        right = face
        ddens_l = -dt*area[face]*corr_dens[face]/volume[left]
        dmom_l = -dt*area[face]*corr_mom[face]/volume[left]
        ddens_r = dt*area[face]*corr_dens[face]/volume[right]
        dmom_r = dt*area[face]*corr_mom[face]/volume[right]

        vel_min_l, vel_max_l = velocity_bounds(vel, left)
        vel_min_r, vel_max_r = velocity_bounds(vel, right)
        theta = min(
            theta_state(dens_new[left], mom_new[left], ddens_l, dmom_l, vel_min_l, vel_max_l),
            theta_state(dens_new[right], mom_new[right], ddens_r, dmom_r, vel_min_r, vel_max_r),
        )

        dens_new[left] += theta*ddens_l
        mom_new[left] += theta*dmom_l
        dens_new[right] += theta*ddens_r
        mom_new[right] += theta*dmom_r

    return dens_new, mom_new


def advect_rk3(dens, mom, dt, yf, dimension, weight):
    dens_0 = dens.copy()
    mom_0 = mom.copy()
    dens, mom = euler_fct(dens, mom, dt, yf, dimension, weight)
    dens, mom = euler_fct(dens, mom, dt, yf, dimension, weight)
    dens = 0.75*dens_0 + 0.25*dens
    mom = 0.75*mom_0 + 0.25*mom
    dens, mom = euler_fct(dens, mom, dt, yf, dimension, weight)
    return dens_0/3.0 + 2.0*dens/3.0, mom_0/3.0 + 2.0*mom/3.0


def run(resolution, dimension):
    ratio = (Y_MAX/Y_MIN)**(1.0/resolution)
    yf = Y_MIN*ratio**np.arange(resolution + 1)
    volume = (yf[1:]**dimension - yf[:-1]**dimension)/dimension
    area = yf**(dimension - 1.0)
    weight = R.ppm_nonuniform_weights(yf**dimension/dimension)
    dens = np.empty(resolution)
    mom = np.empty(resolution)
    for idx in range(resolution):
        y = np.linspace(yf[idx], yf[idx + 1], 256)
        profile = compact_bump(y, 1.0, 1.8)
        dens[idx] = np.trapezoid(profile*y**(dimension - 1.0), y)/volume[idx]
        mom[idx] = np.trapezoid(profile*A*y*y**(dimension - 1.0), y)/volume[idx]

    mass_0 = np.sum(dens*volume)
    time = 0.0
    steps = 0
    vel_min = np.inf
    vel_max = -np.inf
    while time < TEND - 1.0e-14:
        vel = np.where(dens >= R.RHO_VAC, mom/np.maximum(dens, 1.0e-300), 0.0)
        rate = np.max(np.abs(vel)*area[1:]/volume)
        dt = min(0.5/max(rate, 1.0e-300), TEND - time)
        dens, mom = advect_rk3(dens, mom, dt, yf, dimension, weight)
        active = dens >= R.RHO_VAC
        vel = np.where(active, mom/np.maximum(dens, 1.0e-300), 0.0)
        vel_min = min(vel_min, np.min(vel[active]))
        vel_max = max(vel_max, np.max(vel[active]))
        time += dt
        steps += 1

    factor = 1.0 + A*TEND
    exact = np.empty(resolution)
    for idx in range(resolution):
        y = np.linspace(yf[idx], yf[idx + 1], 512)
        exact[idx] = factor**(-dimension)*np.trapezoid(
            compact_bump(y/factor, 1.0, 1.8)*y**(dimension - 1.0), y
        )/volume[idx]
    error = np.sum(volume*np.abs(dens - exact))/np.sum(volume)
    mass_error = abs(np.sum(dens*volume) - mass_0)/mass_0
    return error, mass_error, steps, vel_min, vel_max


if __name__ == "__main__":
    for dimension in (2, 3):
        errors = []
        for resolution in (32, 64, 128, 256):
            result = run(resolution, dimension)
            errors.append(result[0])
            print(f"d={dimension} N={resolution}: L1={result[0]:.8e} mass={result[1]:.2e} "
                  f"steps={result[2]} velocity=[{result[3]:.6e},{result[4]:.6e}]")
        orders = np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.0)
        print(f"d={dimension} orders={orders}")
