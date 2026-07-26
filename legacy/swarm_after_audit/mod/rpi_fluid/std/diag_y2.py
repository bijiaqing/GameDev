#!/usr/bin/env python3
"""Diagnose the irregular Y-transport orders after the SSPRK2 fix.
Reproduces the CUDA y_transport_cyl numbers exactly (L1: 1.4364e-3, 2.7296e-4,
8.7991e-5 at N=64,128,256). Dissect: error location, interior norms, CFL series."""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, advect_y_ssprk2, Y_MIN, Y_MAX, A, TEND, LAM

def run(N, d=2, cfl_num=0.5, tend=TEND):
    r = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = R.ppm_nonuniform_weights(yf**d/d)
    dens = np.empty(N); momy = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 256)
        dens[i] = np.trapezoid(compact_bump(yy, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
        momy[i] = np.trapezoid(compact_bump(yy, 1.0, 1.8)*A*yy*yy**(d-1), yy)/vol[i]
    t = 0.0
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dens, momy = advect_y_ssprk2(dens, momy, dt, yf, d, w)
        t += dt
    exact = np.empty(N)
    lam = 1.0 + A*tend
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = np.trapezoid(lam**(-d)*compact_bump(yy/lam, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
    return yf, vol, dens, exact

def orders(errs):
    return np.round(np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2), 3)

print("== error profile at N=256 (top-8 error cells) ==")
yf, vol, dens, exact = run(256)
err = dens - exact
yc = np.sqrt(yf[:-1]*yf[1:])
idx = np.argsort(-np.abs(err))[:8]
for i in sorted(idx):
    print(f"  y={yc[i]:.4f}  dens={dens[i]:.5f}  exact={exact[i]:.5f}  err={err[i]:+.3e}")

print("\n== global L1 orders, cfl series ==")
for cfl in (0.5, 0.25, 0.1, 0.05):
    errs = []
    for N in (64, 128, 256):
        yf, vol, dens, exact = run(N, cfl_num=cfl)
        errs.append(np.sum(vol*np.abs(dens-exact))/np.sum(vol))
    print(f"  cfl={cfl:.2f}: L1={np.array2string(np.array(errs), precision=4)} orders={orders(errs)}")

print("\n== interior norm (cells with yc in [1.2, 2.0], away from bump edges) ==")
for cfl in (0.5, 0.05):
    errs = []
    for N in (64, 128, 256):
        yf, vol, dens, exact = run(N, cfl_num=cfl)
        yc = np.sqrt(yf[:-1]*yf[1:])
        m = (yc >= 1.2) & (yc <= 2.0)
        errs.append(np.sum(vol[m]*np.abs((dens-exact)[m]))/np.sum(vol[m]))
    print(f"  cfl={cfl:.2f}: interior L1={np.array2string(np.array(errs), precision=4)} orders={orders(errs)}")

print("\n== norm excluding cells with exact < 1e-6 (vacuum-interface excluded) ==")
for cfl in (0.5, 0.05):
    errs = []
    for N in (64, 128, 256):
        yf, vol, dens, exact = run(N, cfl_num=cfl)
        m = exact > 1e-6
        errs.append(np.sum(vol[m]*np.abs((dens-exact)[m]))/np.sum(vol[m]))
    print(f"  cfl={cfl:.2f}: live-cell L1={np.array2string(np.array(errs), precision=4)} orders={orders(errs)}")

print("\n== norm excluding the peak cell +- 3 ==")
for cfl in (0.5,):
    errs = []
    for N in (64, 128, 256):
        yf, vol, dens, exact = run(N, cfl_num=cfl)
        ipk = np.argmax(exact)
        m = np.ones(N, bool); m[max(0,ipk-3):ipk+4] = False
        errs.append(np.sum(vol[m]*np.abs((dens-exact)[m]))/np.sum(vol[m]))
    print(f"  cfl={cfl:.2f}: off-peak L1={np.array2string(np.array(errs), precision=4)} orders={orders(errs)}")
