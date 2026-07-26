#!/usr/bin/env python3
"""Transcribe the CURRENT (SSPRK2) f_advection_z and run the z_transport case,
comparing against both candidate exact references."""
import numpy as np
import review_mock_tests as R

Y_MIN, Y_MAX = 0.5, 2.5
Z_MIN, Z_MAX = 0.35, np.pi - 0.35
LZ, TEND = 0.15, 0.30

def compact_bump(z, lo, hi):
    c = 0.5*(lo+hi); hw = 0.5*(hi-lo)
    u = (z-c)/hw
    out = np.zeros_like(z)
    m = np.abs(u) < 1.0
    out[m] = np.exp(1.0 - 1.0/(1.0-u[m]**2))
    return out

def advect_z_ssprk2(dens, momz, dt, yc, zf, w):
    N = len(dens)
    dz = zf[1] - zf[0]
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
    sin_f = np.sin(zf)
    dens0, momz0 = dens.copy(), momz.copy()
    for stage in range(2):
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
        dens = dens - dt*(sin_f[1:]*Fs[1:] - sin_f[:-1]*Fs[:-1])/(yc*vol_z)
        momz = momz - dt*(sin_f[1:]*Ms[1:] - sin_f[:-1]*Ms[:-1])/(yc*vol_z)
        neg = dens < 0.0; dens[neg] = 0.0; momz[neg] = 0.0
    return 0.5*(dens0 + dens), 0.5*(momz0 + momz)

def run(N, cfl_num=0.5):
    yc = 0.61142227   # innermost radial column
    zf = np.linspace(Z_MIN, Z_MAX, N+1)
    zc = 0.5*(zf[:-1]+zf[1:])
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
    w = R.ppm_nonuniform_weights(-np.cos(zf))
    # initial FV averages of rho0 = B(z)/sin z (i.e. rho*sin z = B), as in the CUDA test
    dens = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 256)
        dens[i] = np.trapezoid(compact_bump(zz, 0.80, 1.30), zz)/vol_z[i]
    momz = LZ*dens
    shift = 0.0
    t = 0.0
    while t < TEND - 1e-14:
        velz = np.where(dens >= R.RHO_VAC, momz/np.maximum(dens, 1e-300), 0.0)
        sin_max = np.maximum(np.sin(zf[:-1]), np.sin(zf[1:]))
        mid = (zf[:-1] <= 0.5*np.pi) & (zf[1:] >= 0.5*np.pi)
        sin_max = np.where(mid, 1.0, sin_max)
        rate = np.max(np.abs(velz)/yc*sin_max/(yc*vol_z))
        dt = min(cfl_num/max(rate, 1e-300), TEND - t)
        dens, momz = advect_z_ssprk2(dens, momz, dt, yc, zf, w)
        t += dt
    # exact reference: rho = B(z-s)/sin z -> sin-weighted cell average = int B(z-s) dz / vol_z
    s = LZ*TEND/yc**2
    ref_b = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 512)
        ub = compact_bump(zz - s, 0.80, 1.30)
        ref_b[i] = np.trapezoid(ub, zz)/vol_z[i]
    L1b = np.sum(vol_z*np.abs(dens - ref_b))/np.sum(vol_z)
    Linfb = np.max(np.abs(dens - ref_b))
    print(f"N={N}: vs exact(translation of rho sin z) L1={L1b:.4e} Linf={Linfb:.4e}")
    return L1b

for N in (64, 128, 256):
    run(N)
