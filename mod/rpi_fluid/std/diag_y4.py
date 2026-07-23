#!/usr/bin/env python3
"""Characterize the residual cfl-dependent error: scaling with cfl, location,
positivity-scaling role, and comparison with the former traced scheme."""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, advect_y_ssprk2, Y_MIN, Y_MAX, A, TEND, LAM
from diag_t7f import advect_y2   # former characteristic-traced scheme

def run(N, d=2, cfl_num=0.5, scheme="ssprk2", positivity=True, tend=TEND):
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
        if scheme == "ssprk2":
            dens, momy = advect_y_ssprk2(dens, momy, dt, yf, d, w)
        else:
            dens, momy = advect_y2(dens, momy, dt, yf, d, w, True)
        t += dt
    lam = 1.0 + A*tend
    exact = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = np.trapezoid(lam**(-d)*compact_bump(yy/lam, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
    return yf, vol, dens, exact

def L1(vol, a, b):
    return np.sum(vol*np.abs(a-b))/np.sum(vol)

print("== E(cfl) at N=256 ==")
for cfl in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    yf, vol, dens, exact = run(256, cfl_num=cfl)
    print(f"  cfl={cfl:.2f}: L1={L1(vol,dens,exact):.4e}")

print("\n== error profile at leading edge, cfl 0.5 vs 0.05 (N=256) ==")
for cfl in (0.5, 0.05):
    yf, vol, dens, exact = run(256, cfl_num=cfl)
    yc = np.sqrt(yf[:-1]*yf[1:])
    err = dens - exact
    idx = np.argsort(-np.abs(err))[:6]
    s = ", ".join(f"y={yc[i]:.4f}:{err[i]:+.2e}" for i in sorted(idx))
    print(f"  cfl={cfl:.2f}: {s}")

print("\n== L1 restricted to leading-edge region y in [1.75, 1.95] vs rest ==")
for cfl in (0.5, 0.25, 0.05):
    yf, vol, dens, exact = run(256, cfl_num=cfl)
    yc = np.sqrt(yf[:-1]*yf[1:])
    m = (yc >= 1.75) & (yc <= 1.95)
    e_edge = np.sum(vol[m]*np.abs((dens-exact)[m]))/np.sum(vol[m])
    e_rest = np.sum(vol[~m]*np.abs((dens-exact)[~m]))/np.sum(vol[~m])
    print(f"  cfl={cfl:.2f}: edge L1={e_edge:.4e}   rest L1={e_rest:.4e}")

print("\n== old traced scheme (diag_t7f.advect_y2) for comparison ==")
for cfl in (0.5, 0.05):
    errs = []
    for N in (64, 128, 256):
        yf, vol, dens, exact = run(N, cfl_num=cfl, scheme="traced")
        errs.append(L1(vol, dens, exact))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2)
    print(f"  cfl={cfl:.2f}: L1={np.array2string(np.array(errs), precision=4)} orders={np.round(orders,3)}")
