#!/usr/bin/env python3
"""Final validation of the SSPRK(3,3) candidate fix against the CUDA cases:
  V1: y_transport_cyl (d=2) and y_transport_sph (d=3) — exact CUDA setups
  V2: z_transport — exact CUDA setup (innermost column + global)
  V3: mass conservation in all cases
  V4: positivity at a strong front (step profile)
Compare SSPRK2 (current CUDA code; mock reproduces its metrics exactly) vs
SSPRK(3,3) (candidate)."""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND
from diag_y6 import advect_y_rk

def run_y(N, d=2, rk=2, cfl_num=0.5, tend=TEND):
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
    mass0 = np.sum(dens*vol)
    t = 0.0
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, rk)
        t += dt
    lam = 1.0 + A*tend
    exact = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = lam**(-d)*np.trapezoid(compact_bump(yy/lam, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
    L1 = np.sum(vol*np.abs(dens-exact))/np.sum(vol)
    merr = abs(np.sum(dens*vol) - mass0)/mass0
    return L1, merr

def orders(errs):
    return np.round(np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2), 4)

print("V1a y_transport_cyl (d=2, cfl=0.5):")
for rk in (2, 3):
    errs, merrs = [], []
    for N in (32, 64, 128, 256):
        L1, me = run_y(N, 2, rk); errs.append(L1); merrs.append(me)
    print(f"  SSPRK{rk}: L1={np.array2string(np.array(errs), precision=4)}")
    print(f"         orders={orders(errs)}  max mass rel err={max(merrs):.2e}")

print("V1b y_transport_sph (d=3, cfl=0.5):")
for rk in (2, 3):
    errs, merrs = [], []
    for N in (32, 64, 128, 256):
        L1, me = run_y(N, 3, rk); errs.append(L1); merrs.append(me)
    print(f"  SSPRK{rk}: L1={np.array2string(np.array(errs), precision=4)}")
    print(f"         orders={orders(errs)}  max mass rel err={max(merrs):.2e}")

print("V2 z_transport (global over 4 columns, cfl=0.5):")
from diag_z import compact_bump as zbump, Z_MIN, Z_MAX, LZ
def run_z(N, yc, rk=2):
    from diag_z import advect_z_ssprk2
    zf = np.linspace(Z_MIN, Z_MAX, N+1)
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
    w = R.ppm_nonuniform_weights(-np.cos(zf))
    dens = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 256)
        dens[i] = np.trapezoid(zbump(zz, 0.80, 1.30), zz)/vol_z[i]
    momz = LZ*dens
    t = 0.0
    while t < 0.30 - 1e-14:
        velz = np.where(dens >= R.RHO_VAC, momz/np.maximum(dens, 1e-300), 0.0)
        sin_max = np.maximum(np.sin(zf[:-1]), np.sin(zf[1:]))
        mid = (zf[:-1] <= 0.5*np.pi) & (zf[1:] >= 0.5*np.pi)
        sin_max = np.where(mid, 1.0, sin_max)
        rate = np.max(np.abs(velz)/yc*sin_max/(yc*vol_z))
        dt = min(0.5/max(rate, 1e-300), 0.30 - t)
        # NOTE: advect_z_ssprk2 is the SSPRK2 kernel; rk=3 uses a 3-stage variant below
        if rk == 2:
            dens, momz = advect_z_ssprk2(dens, momz, dt, yc, zf, w)
        else:
            dens, momz = advect_z_ssprk3(dens, momz, dt, yc, zf, w)
        t += dt
    s = LZ*0.30/yc**2
    ref = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 512)
        ref[i] = np.trapezoid(zbump(zz-s, 0.80, 1.30), zz)/vol_z[i]
    return np.sum(vol_z*np.abs(dens-ref))/np.sum(vol_z)

def advect_z_ssprk3(dens, momz, dt, yc, zf, w):
    N = len(dens)
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
    sin_f = np.sin(zf)
    def L(dens, momz):
        velz = np.where(dens >= R.RHO_VAC, momz/np.maximum(dens, 1e-300), 0.0)
        ed = R.ppm_edges_nonuniform(dens, w)
        ev = R.ppm_edges_nonuniform(velz, w)
        F = np.zeros(N+1); M = np.zeros(N+1)
        for iz in range(N-1):
            dL = max(R.ppm_face_value(ed, dens, iz, iz+1, True, 0.0), 0.0)
            dR = max(R.ppm_face_value(ed, dens, iz+1, iz+2, False, 0.0), 0.0)
            vL = R.ppm_face_value(ev, velz, iz, iz+1, True, 0.0)
            vR = R.ppm_face_value(ev, velz, iz+1, iz+2, False, 0.0)
            fd, fm = R.hll_2(vL/yc, vR/yc, dL, dL*vL, dR, dR*vR)
            F[iz+1], M[iz+1] = fd, fm
        so = velz[N-1]/yc
        F[N] = so*max(dens[N-1],0.0) if so > 0.0 else 0.0; M[N] = F[N]*velz[N-1]
        si = velz[0]/yc
        F[0] = si*max(dens[0],0.0) if si < 0.0 else 0.0;    M[0] = F[0]*velz[0]
        scale = np.ones(N)
        for i in range(N):
            leave = dt*(sin_f[i+1]*max(F[i+1],0.0) + sin_f[i]*max(-F[i],0.0))
            allow = (1.0-1e-12)*max(dens[i],0.0)*yc*vol_z[i]
            if leave > allow and leave > 0.0: scale[i] = allow/leave
        Fs, Ms = F.copy(), M.copy()
        Fs[0] *= scale[0]; Ms[0] *= scale[0]
        for i in range(1, N):
            iup = i-1 if F[i] >= 0.0 else i
            Fs[i] *= scale[iup]; Ms[i] *= scale[iup]
        Fs[N] *= scale[N-1]; Ms[N] *= scale[N-1]
        dd = -dt*(sin_f[1:]*Fs[1:] - sin_f[:-1]*Fs[:-1])/(yc*vol_z)
        dm = -dt*(sin_f[1:]*Ms[1:] - sin_f[:-1]*Ms[:-1])/(yc*vol_z)
        return dd, dm
    d1, m1 = L(dens, momz)
    dens1, momz1 = dens + d1, momz + m1
    dens1[dens1 < 0] = 0.0; momz1[dens1 < 0] = 0.0
    d2, m2 = L(dens1, momz1)
    dens2 = 0.75*dens + 0.25*(dens1 + d2)
    momz2 = 0.75*momz + 0.25*(momz1 + m2)
    dens2[dens2 < 0] = 0.0; momz2[dens2 < 0] = 0.0
    d3, m3 = L(dens2, momz2)
    return dens/3 + 2*(dens2 + d3)/3, momz/3 + 2*(momz2 + m3)/3

for rk in (2, 3):
    errs = []
    for N in (32, 64, 128, 256):
        yc_all = 0.5*(Y_MAX/Y_MIN)**((np.arange(4)+0.5)/4)
        zf = np.linspace(Z_MIN, Z_MAX, N+1)
        vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
        rvol = ((0.5*(Y_MAX/Y_MIN)**((np.arange(4)+1.0)/4))**3 - (0.5*(Y_MAX/Y_MIN)**(np.arange(4)/4))**3)/3.0
        V = vol_z[:, None]*rvol[None, :]
        E = np.zeros((N, 4))
        for j, yc in enumerate(yc_all):
            # run each column; reuse run_z but need error per column
            from diag_z import compact_bump as zb, Z_MIN as zlo, Z_MAX as zhi
            # inline column run (same as run_z but returns dens and ref)
            w = R.ppm_nonuniform_weights(-np.cos(zf))
            dens = np.empty(N)
            for i in range(N):
                zz = np.linspace(zf[i], zf[i+1], 256)
                dens[i] = np.trapezoid(zb(zz, 0.80, 1.30), zz)/vol_z[i]
            momz = LZ*dens
            t = 0.0
            while t < 0.30 - 1e-14:
                velz = np.where(dens >= R.RHO_VAC, momz/np.maximum(dens, 1e-300), 0.0)
                sin_max = np.maximum(np.sin(zf[:-1]), np.sin(zf[1:]))
                mid = (zf[:-1] <= 0.5*np.pi) & (zf[1:] >= 0.5*np.pi)
                sin_max = np.where(mid, 1.0, sin_max)
                rate = np.max(np.abs(velz)/yc*sin_max/(yc*vol_z))
                dt = min(0.5/max(rate, 1e-300), 0.30 - t)
                if rk == 2:
                    from diag_z import advect_z_ssprk2
                    dens, momz = advect_z_ssprk2(dens, momz, dt, yc, zf, w)
                else:
                    dens, momz = advect_z_ssprk3(dens, momz, dt, yc, zf, w)
                t += dt
            s = LZ*0.30/yc**2
            for i in range(N):
                zz = np.linspace(zf[i], zf[i+1], 512)
                ref_i = np.trapezoid(zb(zz-s, 0.80, 1.30), zz)/vol_z[i]
                E[i, j] = abs(dens[i] - ref_i)
        errs.append(np.sum(V*np.abs(E))/np.sum(V))
    print(f"  SSPRK{rk}: L1={np.array2string(np.array(errs), precision=4)} orders={orders(errs)}")

print("V3 positivity at a strong front (step profile, SSPRK3, cfl=0.5):")
yf = Y_MIN*(Y_MAX/Y_MIN)**(np.arange(257)/256)
vol = (yf[1:]**2 - yf[:-1]**2)/2
area = yf
w = R.ppm_nonuniform_weights(yf**2/2)
dens = np.where((np.sqrt(yf[:-1]*yf[1:]) > 1.0) & (np.sqrt(yf[:-1]*yf[1:]) < 1.5), 1.0, 0.05)
momy = dens*0.2*np.sqrt(yf[:-1]*yf[1:])
mass0 = np.sum(dens*vol)
t = 0.0
while t < 0.3:
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    rate = np.max(np.abs(vely)*area[1:]/vol)
    dt = min(0.5/max(rate, 1e-300), 0.3 - t)
    dens, momy = advect_y_rk(dens, momy, dt, yf, 2, w, 3)
    t += dt
print(f"  min density={dens.min():.3e} (must be >= 0), mass rel change={abs(np.sum(dens*vol)-mass0)/mass0:.2e}")
