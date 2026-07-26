#!/usr/bin/env python3
"""Decisive spatial-truncation test for f_advection_y: measure the error of the
semi-discrete flux divergence for a smooth manufactured state, isolating space
from time. Also compares X-sweep order on the same Gaussian bump."""
import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, ppm_edges_nonuniform,
                               ppm_face_value, hll_2, advect_x, advect_y, cell_average,
                               Y_MIN, Y_MAX, RHO_VAC)

# --- spatial defect of one Y step -----------------------------------------
def spatial_defect(d, n):
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    rho0 = lambda y: np.exp(-((y - 1.6)/0.12)**2)
    dens = cell_average(rho0, yf, d)
    v0   = cell_average(lambda y: np.ones_like(y), yf, d)
    momy = dens*1.0
    dt = 1e-9
    d2, m2 = advect_y(dens, momy, dt, yf, d, w)
    upd = (d2 - dens)/dt                       # measured discrete divergence
    # exact: -(1/y^{d-1}) d(y^{d-1} rho v)/dy, cell-averaged = -(A_o F_o - A_i F_i)/vol
    # with F = rho*v at faces (exact point values, smooth)
    Ff = rho0(yf)*1.0
    exact = -(area[1:]*Ff[1:] - area[:-1]*Ff[:-1])/vol
    return upd, exact, vol

for d in (2,):
    errs = []
    for n in (64, 128, 256, 512):
        upd, exact, vol = spatial_defect(d, n)
        errs.append(np.sum(vol*np.abs(upd - exact))/np.sum(vol))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    print(f"Y spatial defect d={d}: L1={np.array2string(np.array(errs), precision=3)} orders={np.round(orders,3)}")

# --- X sweep order on the same Gaussian bump (periodic, uniform grid) -----
def x_bump_order():
    errs = []
    for n in (64, 128, 256, 512):
        dx = 2.0*np.pi/n
        x = (np.arange(n) + 0.5)*dx
        dens = np.exp(-((np.minimum(np.abs(x - np.pi), 2*np.pi - np.abs(x - np.pi)))/0.5)**2)
        exact = dens.copy()
        momx = dens.copy()
        dt = 3.25*dx
        nsteps = int(round(2.0*np.pi/dt)); dt = 2.0*np.pi/nsteps
        for _ in range(nsteps):
            dens, momx = advect_x(dens, momx, dt)
        errs.append(np.sum(np.abs(dens - exact))*dx/(2*np.pi))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    print(f"X bump translation: L1={np.array2string(np.array(errs), precision=3)} orders={np.round(orders,3)}")

x_bump_order()
