#!/usr/bin/env python3

import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND

N = 128; d = 2
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

def L_instrumented(dens, momy, dt, stage):
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

    j = int(np.argmin(np.abs(yc - 0.9922)))
    print(f"  stage {stage}: cell j={j} y={yc[j]:.4f} dens={dens[j]:.3e} momy={momy[j]:.3e}")
    print(f"    F[j]={F[j]:+.4e} F[j+1]={F[j+1]:+.4e} M[j]={M[j]:+.4e} M[j+1]={M[j+1]:+.4e}")
    print(f"    scale[j]={scale[j]:.4f} scale[j-1]={scale[j-1]:.4f} scale[j+1]={scale[j+1]:.4f}")
    Fs, Ms = F.copy(), M.copy()
    Fs[0] *= scale[0]; Ms[0] *= scale[0]
    for i in range(1, N):
        iup = i-1 if F[i] >= 0.0 else i
        Fs[i] *= scale[iup]; Ms[i] *= scale[iup]
    Fs[N] *= scale[N-1]; Ms[N] *= scale[N-1]
    print(f"    after scaling: Fs[j]={Fs[j]:+.4e} Fs[j+1]={Fs[j+1]:+.4e} Ms[j]={Ms[j]:+.4e} Ms[j+1]={Ms[j+1]:+.4e}")
    dd = -dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
    dm = -dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
    print(f"    dd[j]={dd[j]:+.4e} dm[j]={dm[j]:+.4e}  -> dens+dd={dens[j]+dd[j]:.3e}")
    return dd, dm

t = 0.0
for step in range(6):
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    rate = np.max(np.abs(vely)*area[1:]/vol)
    dt = min(0.5/max(rate, 1e-300), TEND - t)
    if step == 5:
        print(f"step 6 (dt={dt:.3e}):")
        d1, m1 = L_instrumented(dens, momy, dt, 0)
        dens1, momy1 = dens + d1, momy + m1
        dens1[dens1 < 0] = 0.0; momy1[dens1 < 0] = 0.0
        d2, m2 = L_instrumented(dens1, momy1, dt, 1)
        dens2 = 0.75*dens + 0.25*(dens1 + d2)
        momy2 = 0.75*momy + 0.25*(momy1 + m2)
        dens2[dens2 < 0] = 0.0; momy2[dens2 < 0] = 0.0
        d3, m3 = L_instrumented(dens2, momy2, dt, 2)
        dens = dens/3 + 2*(dens2 + d3)/3
        momy = momy/3 + 2*(momy2 + m3)/3
    else:
        from diag_y6 import advect_y_rk
        dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, 3)
    t += dt
vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
j = int(np.argmin(np.abs(yc - 0.9922)))
print(f"after step 6: y={yc[j]:.4f} dens={dens[j]:.3e} momy={momy[j]:.3e} vely={vely[j]:+.4f}")
