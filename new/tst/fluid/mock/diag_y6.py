#!/usr/bin/env python3



import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, advect_y_ssprk2, Y_MIN, Y_MAX, A, TEND

def advect_y_rk(dens, momy, dt, yf, d, w, order=2):

    N = len(dens)
    powy = float(d)
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    area = yf**(powy-1.0)
    def L(dens, momy):
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        ed = R.ppm_edges_nonuniform(dens, w)
        ev = R.ppm_edges_nonuniform(vely, w)
        F = np.zeros(N+1); M = np.zeros(N+1)
        for iy in range(N-1):
            dL = max(R.ppm_face_value(ed, dens, iy, iy+1, True, 0.0), 0.0)
            dR = max(R.ppm_face_value(ed, dens, iy+1, iy+2, False, 0.0), 0.0)
            vL = R.ppm_face_value(ev, vely, iy, iy+1, True, 0.0)
            vR = R.ppm_face_value(ev, vely, iy+1, iy+2, False, 0.0)
            fd, fm = R.hll_2(vL, vR, dL, dL*vL, dR, dR*vR)
            F[iy+1], M[iy+1] = fd, fm
        so = vely[N-1]
        F[N] = so*max(dens[N-1],0.0) if so > 0.0 else 0.0; M[N] = F[N]*vely[N-1]
        si = vely[0]
        F[0] = si*max(dens[0],0.0) if si < 0.0 else 0.0;    M[0] = F[0]*vely[0]
        scale = np.ones(N)
        for i in range(N):
            leave = dt*(area[i+1]*max(F[i+1],0.0) + area[i]*max(-F[i],0.0))
            allow = (1.0-1e-12)*max(dens[i],0.0)*vol[i]
            if leave > allow and leave > 0.0: scale[i] = allow/leave
        Fs, Ms = F.copy(), M.copy()
        Fs[0] *= scale[0]; Ms[0] *= scale[0]
        for i in range(1, N):
            iup = i-1 if F[i] >= 0.0 else i
            Fs[i] *= scale[iup]; Ms[i] *= scale[iup]
        Fs[N] *= scale[N-1]; Ms[N] *= scale[N-1]
        dd = -dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
        dm = -dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
        return dd, dm
    if order == 2:
        d1, m1 = L(dens, momy)
        dens1, momy1 = dens + d1, momy + m1
        dens1 = np.maximum(dens1, 0.0)
        d2, m2 = L(dens1, momy1)
        return 0.5*(dens + dens1 + d2), 0.5*(momy + momy1 + m2)

    d1, m1 = L(dens, momy)
    dens1, momy1 = dens + d1, momy + m1
    dens1 = np.maximum(dens1, 0.0)
    d2, m2 = L(dens1, momy1)
    dens2 = 0.75*dens + 0.25*(dens1 + d2)
    momy2 = 0.75*momy + 0.25*(momy1 + m2)
    dens2 = np.maximum(dens2, 0.0)
    d3, m3 = L(dens2, momy2)
    return (dens/3 + 2*(dens2 + d3)/3), (momy/3 + 2*(momy2 + m3)/3)

def run(N, d=2, cfl_num=0.5, rk=2, hw=0.4, tend=TEND):
    r = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = R.ppm_nonuniform_weights(yf**d/d)
    lo, hi = 1.4 - hw, 1.4 + hw
    dens = np.empty(N); momy = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 256)
        dens[i] = np.trapezoid(compact_bump(yy, lo, hi)*yy**(d-1), yy)/vol[i]
        momy[i] = np.trapezoid(compact_bump(yy, lo, hi)*A*yy*yy**(d-1), yy)/vol[i]
    t = 0.0
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        if rk == 2:
            dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, 2)
        else:
            dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, 3)
        t += dt
    lam = 1.0 + A*tend
    exact = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = lam**(-d)*np.trapezoid(compact_bump(yy/lam, lo, hi)*yy**(d-1), yy)/vol[i]
    L1 = np.sum(vol*np.abs(dens-exact))/np.sum(vol)
    yc = np.sqrt(yf[:-1]*yf[1:])
    edge = (yc >= hi*lam - 0.15) & (yc <= hi*lam + 0.05)
    e_edge = np.sum(vol[edge]*np.abs((dens-exact)[edge]))/np.sum(vol[edge])
    return L1, e_edge

print("(a) bump half-width x2 (0.8, hw=0.8) at cfl=0.5, SSPRK2:")
errs = [run(N, hw=0.8)[0] for N in (64,128,256)]
print("   L1:", np.array2string(np.array(errs), precision=4),
      "orders:", np.round(np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2), 3))

print("(b) SSPRK(3,3) vs SSPRK2 at cfl=0.5, production bump:")
for rk in (2, 3):
    errs = [run(N, rk=rk)[0] for N in (64,128,256)]
    print(f"   rk{rk} L1:", np.array2string(np.array(errs), precision=4),
          "orders:", np.round(np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2), 3))

print("(c) edge-only error scaling at cfl=0.5 (SSPRK2):")
for N in (64, 128, 256):
    L1, ee = run(N)
    print(f"   N={N}: edge L1={ee:.4e}  (global L1={L1:.4e})")
