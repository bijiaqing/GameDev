#!/usr/bin/env python3


import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND, LAM

ACTIV = {"scale": 0, "neg": 0}

def advect_y_instr(dens, momy, dt, yf, d, w):
    N = len(dens)
    powy = float(d)
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    area = yf**(powy-1.0)
    dens0, momy0 = dens.copy(), momy.copy()
    for stage in range(2):
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
            if leave > allow and leave > 0.0:
                scale[i] = allow/leave
                ACTIV["scale"] += 1
        Fs, Ms = F.copy(), M.copy()
        Fs[0] *= scale[0]; Ms[0] *= scale[0]
        for i in range(1, N):
            iup = i-1 if F[i] >= 0.0 else i
            Fs[i] *= scale[iup]; Ms[i] *= scale[iup]
        Fs[N] *= scale[N-1]; Ms[N] *= scale[N-1]
        dens = dens - dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
        momy = momy - dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
        neg = dens < 0.0
        ACTIV["neg"] += int(neg.sum())
        dens[neg] = 0.0; momy[neg] = 0.0
    return 0.5*(dens0 + dens), 0.5*(momy0 + momy)

def run(N, d=2, cfl_num=0.5, floor=0.0, tend=TEND):
    r = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = R.ppm_nonuniform_weights(yf**d/d)
    dens = np.empty(N); momy = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 256)
        dens[i] = floor + np.trapezoid(compact_bump(yy, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
        momy[i] = np.trapezoid((floor + compact_bump(yy, 1.0, 1.8))*A*yy*yy**(d-1), yy)/vol[i]
    t = 0.0
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dens, momy = advect_y_instr(dens, momy, dt, yf, d, w)
        t += dt
    lam = 1.0 + A*tend
    exact = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = floor*lam**(-d) + np.trapezoid(lam**(-d)*compact_bump(yy/lam, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
    L1 = np.sum(vol*np.abs(dens-exact))/np.sum(vol)
    return L1

print("floored test (no vacuum): dens = bump + 0.01")
for cfl in (0.5, 0.25, 0.05):
    ACTIV["scale"] = 0; ACTIV["neg"] = 0
    errs = [run(N, cfl_num=cfl, floor=0.01) for N in (64, 128, 256)]
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2)
    print(f"  cfl={cfl:.2f}: L1={np.array2string(np.array(errs), precision=4)} orders={np.round(orders,3)}"
          f"  (scale activations={ACTIV['scale']}, neg clamps={ACTIV['neg']})")

print("original test (vacuum present):")
for cfl in (0.5, 0.25, 0.05):
    ACTIV["scale"] = 0; ACTIV["neg"] = 0
    errs = [run(N, cfl_num=cfl, floor=0.0) for N in (64, 128, 256)]
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2)
    print(f"  cfl={cfl:.2f}: L1={np.array2string(np.array(errs), precision=4)} orders={np.round(orders,3)}"
          f"  (scale activations={ACTIV['scale']}, neg clamps={ACTIV['neg']})")
