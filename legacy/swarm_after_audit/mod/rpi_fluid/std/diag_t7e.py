#!/usr/bin/env python3
"""Structure of the homologous error: profile, peak-only order, off-peak order."""
import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, advect_y, cell_average,
                               Y_MIN, Y_MAX, RHO_VAC)

a_diag = 0.2

def run(d, n, t_end=0.5, cfl=0.5):
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    rho0 = lambda y: np.exp(-((y - 1.6)/0.12)**2)
    dens = cell_average(rho0, yf, d)
    momy = cell_average(lambda y: rho0(y)*a_diag*y, yf, d)
    t = 0.0
    while t < t_end - 1e-14:
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl/max(rate, 1e-300), t_end - t)
        dens, momy = advect_y(dens, momy, dt, yf, d, w)
        t += dt
    lam = 1.0 + a_diag*t_end
    exact = cell_average(lambda y: lam**(-d)*rho0(y/lam), yf, d)
    yc = np.sqrt(yf[:-1]*yf[1:])
    return yc, vol, dens, exact

d = 2
for n in (256, 512):
    yc, vol, dens, exact = run(d, n)
    err = dens - exact
    i_pk = np.argmax(exact)
    print(f"N={n}: peak cell y={yc[i_pk]:.4f}, exact={exact[i_pk]:.5f}, err={err[i_pk]:+.3e}")
    sel = slice(i_pk-6, i_pk+7)
    print("  y        exact      err        rel")
    for i in range(i_pk-6, i_pk+7):
        print(f"  {yc[i]:.4f}  {exact[i]:.5f}  {err[i]:+.3e}  {err[i]/max(exact[i],1e-30):+.3e}")
    off = np.ones(len(err), bool); off[i_pk-2:i_pk+3] = False
    L1_off = np.sum(vol[off]*np.abs(err[off]))/np.sum(vol[off])
    print(f"  off-peak L1 = {L1_off:.3e}")

print()
errs_pk, errs_off = [], []
for n in (64, 128, 256, 512):
    yc, vol, dens, exact = run(d, n)
    err = dens - exact
    i_pk = np.argmax(exact)
    errs_pk.append(abs(err[i_pk]))
    off = np.ones(len(err), bool); off[i_pk-2:i_pk+3] = False
    errs_off.append(np.sum(vol[off]*np.abs(err[off]))/np.sum(vol[off]))
print("peak err:", np.array2string(np.array(errs_pk), precision=3),
      "orders:", np.round(np.log(np.array(errs_pk[:-1])/np.array(errs_pk[1:]))/np.log(2), 3))
print("off-peak L1:", np.array2string(np.array(errs_off), precision=3),
      "orders:", np.round(np.log(np.array(errs_off[:-1])/np.array(errs_off[1:]))/np.log(2), 3))
