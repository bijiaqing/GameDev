#!/usr/bin/env python3
"""Uniform-density homologous control: separates geometric dilution error from
bump-shape/limiter error. Also reports velocity error."""
import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, advect_y, cell_average,
                               Y_MIN, Y_MAX, RHO_VAC)

a_diag = 0.2

def run_uniform(d, n, cfl_num, t_end):
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    dens = np.ones(n)          # uniform unit density
    momy = cell_average(lambda y: 1.0*a_diag*y, yf, d)   # v = a y
    t = 0.0
    while t < t_end - 1e-14:
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), t_end - t)
        dens, momy = advect_y(dens, momy, dt, yf, d, w)
        t += dt
    lam = 1.0 + a_diag*t_end
    yc = np.sqrt(yf[:-1]*yf[1:])
    vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    err_rho = np.max(np.abs(dens - lam**(-d)))
    err_v = np.max(np.abs(vely - a_diag*yc/lam))
    return err_rho, err_v

print("uniform-density homologous: exact rho = lam^-d (constant), v = a y/lam")
for d in (2, 3):
    errs_r, errs_v = [], []
    for n in (64, 128, 256, 512):
        er, ev = run_uniform(d, n, 0.5, 0.5)
        errs_r.append(er); errs_v.append(ev)
    orr = np.log(np.array(errs_r[:-1])/np.array(errs_r[1:]))/np.log(2.0)
    orv = np.log(np.array(errs_v[:-1])/np.array(errs_v[1:]))/np.log(2.0)
    print(f"d={d}: rho err={np.array2string(np.array(errs_r), precision=3)} orders={np.round(orr,2)}")
    print(f"d={d}: vel err={np.array2string(np.array(errs_v), precision=3)} orders={np.round(orv,2)}")
