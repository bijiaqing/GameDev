#!/usr/bin/env python3



import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, advect_y, cell_average,
                               Y_MIN, Y_MAX, RHO_VAC)

a_diag = 0.2

def run_case(d, n, cfl_num, mode, t_end=0.5):
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    rho0 = lambda y: np.exp(-((y - 1.6)/0.12)**2)
    if mode == "homologous":
        dens = cell_average(rho0, yf, d)
        momy = cell_average(lambda y: rho0(y)*a_diag*y, yf, d)
    else:
        dens = cell_average(rho0, yf, d)
        momy = cell_average(lambda y: rho0(y)*1.0, yf, d)
    mass0 = np.sum(dens*vol)
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
    L1 = np.sum(vol*np.abs(dens - exact))/np.sum(vol)
    merr = abs(np.sum(dens*vol) - mass0)/mass0
    return L1, merr

for mode in ("linear", "homologous"):
    for d in (2,):
        errs = []
        for n in (64, 128, 256, 512):
            L1, merr = run_case(d, n, 0.5, mode)
            errs.append(L1)
        orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
        print(f"{mode:10s} d={d} CFL=0.5  L1={np.array2string(np.array(errs), precision=3)}  "
              f"orders={np.round(orders,3)}")


d = 2
for mode in ("linear", "homologous"):
    errs = []
    for cfl in (0.4, 0.2, 0.1, 0.05):
        L1, _ = run_case(d, 256, cfl, mode)
        errs.append(L1)
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    print(f"{mode:10s} d={d} N=256, CFL refinement L1={np.array2string(np.array(errs), precision=3)} "
          f"temporal orders={np.round(orders,3)}")
