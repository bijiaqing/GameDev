#!/usr/bin/env python3
"""Trace the first appearance of anomalous negative-velocity cells step by step."""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND
from diag_y6 import advect_y_rk

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

t = 0.0
for step in range(12):
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    rate = np.max(np.abs(vely)*area[1:]/vol)
    dt = min(0.5/max(rate, 1e-300), TEND - t)
    dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, 3)
    t += dt
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    bad = np.where((vely < -0.05) & (dens > 0))[0]
    if len(bad):
        print(f"step {step+1} (dt={dt:.2e}): {len(bad)} anomalous cells")
        for i in bad[:8]:
            print(f"   y={yc[i]:.4f} dens={dens[i]:.3e} momy={momy[i]:.3e} vely={vely[i]:+.4f}")
    else:
        print(f"step {step+1} (dt={dt:.2e}): no anomalous cells yet; "
              f"min dens in [0.6,1.1] = {dens[(yc>0.6)&(yc<1.1)].min():.2e}")
