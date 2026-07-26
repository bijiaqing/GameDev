#!/usr/bin/env python3
"""Reproduce the low-density CFL defect: run the exact y_transport_cyl case with
the SSPRK(3,3) mock, count steps, and locate anomalous-velocity cells."""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND
from diag_y6 import advect_y_rk

def run(N, d=2, rk=3, cfl_num=0.5, tend=TEND, verbose=False):
    r = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    yc = np.sqrt(yf[:-1]*yf[1:])
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = R.ppm_nonuniform_weights(yf**d/d)
    dens = np.empty(N); momy = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 256)
        dens[i] = np.trapezoid(compact_bump(yy, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
        momy[i] = np.trapezoid(compact_bump(yy, 1.0, 1.8)*A*yy*yy**(d-1), yy)/vol[i]
    t = 0.0; steps = 0
    dts = []
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dts.append(dt)
        dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, rk)
        t += dt; steps += 1
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    anom = np.where((np.abs(vely) > 0.6) & (dens < 1e-6))[0]
    print(f"N={N} rk={rk}: steps={steps} (dt range [{min(dts):.2e},{max(dts):.2e}])")
    if verbose and len(anom):
        for i in anom[:12]:
            print(f"   cell y={yc[i]:.4f}: dens={dens[i]:.3e} momy={momy[i]:.3e} vely={vely[i]:+.3f}")
    # also report the rate-limiting cell
    rate_cells = np.abs(vely)*area[1:]/vol
    imax = int(np.argmax(rate_cells))
    print(f"   rate-limiting cell: y={yc[imax]:.4f} dens={dens[imax]:.3e} vely={vely[imax]:+.3f} rate={rate_cells[imax]:.3f}")
    return steps

print("SSPRK(3,3) step counts:")
run(64, verbose=True)
run(128, verbose=True)
run(256, verbose=True)
