#!/usr/bin/env python3
"""Locate the dominant error in the T7 homologous/linear Y-sweep cases."""
import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, advect_y, cell_average,
                               Y_MIN, Y_MAX, RHO_VAC)

a_diag = 0.2

def run_case(d, n, cfl_num, mode, t_end, sig=0.12, y0=1.6):
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    rho0 = lambda y: np.exp(-((y - y0)/sig)**2)
    if mode == "homologous":
        dens = cell_average(rho0, yf, d)
        momy = cell_average(lambda y: rho0(y)*a_diag*y, yf, d)
    else:
        dens = cell_average(rho0, yf, d)
        momy = cell_average(lambda y: rho0(y)*1.0, yf, d)
    t = 0.0
    while t < t_end - 1e-14:
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), t_end - t)
        dens, momy = advect_y(dens, momy, dt, yf, d, w)
        t += dt
    if mode == "homologous":
        lam = 1.0 + a_diag*t_end
        exact = cell_average(lambda y: lam**(-d)*rho0(y/lam), yf, d)
    else:
        v = 1.0
        exact = cell_average(lambda y: ((y - v*t_end)/y)**(d-1)*rho0(y - v*t_end), yf, d)
    err = dens - exact
    yc = np.sqrt(yf[:-1]*yf[1:])
    i_max = np.argmax(np.abs(err)*vol)
    L1 = np.sum(vol*np.abs(err))/np.sum(vol)
    Linf = np.max(np.abs(err))
    print(f"  {mode:10s} d={d} N={n:4d} t={t_end}: L1={L1:.3e} Linf={Linf:.3e} "
          f"max-err cell y={yc[i_max]:.4f} (bump centre ~{y0 + (a_diag*y0*t_end if mode=='homologous' else t_end):.3f})")
    return L1

print("== shorter time, bump kept far from outer boundary ==")
for mode in ("linear", "homologous"):
    for n in (64, 128, 256, 512):
        run_case(2, n, 0.5, mode, t_end=0.25)

print("== wider bump (better resolved at fixed N) ==")
for mode in ("linear", "homologous"):
    for n in (64, 128, 256, 512):
        run_case(2, n, 0.5, mode, t_end=0.25, sig=0.24)

print("== error-location check, homologous N=512 ==")
run_case(2, 512, 0.5, "homologous", t_end=0.5)
run_case(2, 512, 0.5, "linear", t_end=0.5)
