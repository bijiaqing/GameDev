#!/usr/bin/env python3



import numpy as np
from diag_z import advect_z_ssprk2, compact_bump, Y_MIN, Y_MAX, Z_MIN, Z_MAX, LZ, TEND
import review_mock_tests as R

def run_column(N, yc, cfl_num=0.5):
    zf = np.linspace(Z_MIN, Z_MAX, N+1)
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])
    w = R.ppm_nonuniform_weights(-np.cos(zf))
    dens = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 256)
        dens[i] = np.trapezoid(compact_bump(zz, 0.80, 1.30), zz)/(zf[i+1]-zf[i])
    momz = LZ*dens
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
    return dens

def refs(N, yc, zf, vol_z):
    s = LZ*TEND/yc**2
    ra = np.empty(N); rb = np.empty(N)
    for i in range(N):
        zz = np.linspace(zf[i], zf[i+1], 512)
        ra[i] = np.trapezoid(compact_bump(zz-s, 0.80, 1.30)*np.sin(zz-s), zz)/vol_z[i]
        rb[i] = np.trapezoid(compact_bump(zz-s, 0.80, 1.30), zz)/vol_z[i]
    return ra, rb

for N in (64, 256):
    yc_all = 0.5*(Y_MAX/Y_MIN)**((np.arange(4)+0.5)/4)
    zf = np.linspace(Z_MIN, Z_MAX, N+1)
    vol_z = np.cos(zf[:-1]) - np.cos(zf[1:])

    rvol = ( (0.5*(Y_MAX/Y_MIN)**((np.arange(4)+1.0)/4))**3 - (0.5*(Y_MAX/Y_MIN)**(np.arange(4)/4))**3 )/3.0
    errs_a, errs_b = [], []
    for j, yc in enumerate(yc_all):
        dens = run_column(N, yc)
        ra, rb = refs(N, yc, zf, vol_z)
        errs_a.append((dens - ra)*1.0); errs_b.append((dens - rb)*1.0)
    V = vol_z[:, None]*rvol[None, :]
    Ea = np.abs(np.concatenate(errs_a, axis=0).reshape(N, 4))
    Eb = np.abs(np.concatenate(errs_b, axis=0).reshape(N, 4))
    L1a = np.sum(V*Ea)/np.sum(V); L1b = np.sum(V*Eb)/np.sum(V)
    Linfa = Ea.max(); Linfb = Eb.max()
    print(f"N={N}: vs plan      L1={L1a:.4e} Linf={Linfa:.4e}")
    print(f"N={N}: vs transl.   L1={L1b:.4e} Linf={Linfb:.4e}")
