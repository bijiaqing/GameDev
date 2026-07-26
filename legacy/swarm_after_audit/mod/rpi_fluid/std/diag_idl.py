#!/usr/bin/env python3
"""Mock implementation + validation of the proposed conservative invariant-domain
flux limiter for the Y/Z SSPRK(3,3) sweeps.

Design (per RK stage, per face):
  F = F^L + theta_f (F^H - F^L),   0 <= theta_f <= 1,
with F^L the first-order cell-centred upwind flux (cell-average states) and F^H
the PPM/HLL flux. Per cell the limited update is constrained to the convex
invariant set
  rho >= 0,  v_{k,min} rho <= m_k <= v_{k,max} rho   (k = x,y,z),
with v_{k,min/max} from the min/max of the stage velocities over the cell and its
two neighbours (plus a small margin). theta_f is the per-face minimum of the two
adjacent cells' positivity ratios over the 1 + 2*3 linear constraints (per-cell
DMP form of the bound-preserving limiter). One shared theta_f per face keeps the
single equal-and-opposite conservative face flux; the convex set is preserved by
the Shu-Osher combinations.
"""
import numpy as np
import review_mock_tests as R
from diag_y import compact_bump, Y_MIN, Y_MAX, A, TEND

def advect_y_rk_idl(dens, momx, momy, momz, dt, yf, d, w, order=3, margin=1.0e-6):
    N = len(dens)
    powy = float(d)
    vol = (yf[1:]**d - yf[:-1]**d)/powy
    area = yf**(powy - 1.0)

    def L(dens, momx, momy, momz):
        # recover primitives (production vacuum convention)
        velx = np.where(dens >= R.RHO_VAC, momx/np.maximum(dens, 1e-300), np.sqrt(1.0*np.sqrt(yf[:-1]*yf[1:])))
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        velz = np.where(dens >= R.RHO_VAC, momz/np.maximum(dens, 1e-300), 0.0)

        # ---- high-order fluxes (PPM/HLL, production) ----
        eds = [R.ppm_edges_nonuniform(q, w) for q in (dens, velx, vely, velz)]
        FH = np.zeros((4, N+1))
        for iy in range(N-1):
            states = []
            for f, e, q in zip(range(4), eds, (dens, velx, vely, velz)):
                Ls = R.ppm_face_value(e, q, iy, iy+1, True, 0.0)
                Rs = R.ppm_face_value(e, q, iy+1, iy+2, False, 0.0)
                states.append((max(Ls, 0.0) if f == 0 else Ls,
                               max(Rs, 0.0) if f == 0 else Rs))
            (dL, dR), (xL, xR), (yL, yR), (zL, zR) = states
            fd, fx, fy, fz = 0.0, 0.0, 0.0, 0.0
            fd, fx, fy, fz = R.hll_2(yL, yR, dL, dL*xL, dR, dR*xR)[:2] if False else (None,)*4
            # full 4-component HLL (as in helpers.cuh)
            mLs = (dL*xL, dL*yL, dL*zL); mRs = (dR*xR, dR*yR, dR*zR)
            if yL >= 0.0 and yR >= 0.0:
                FH[0, iy+1] = yL*dL; FH[1, iy+1] = yL*mLs[0]; FH[2, iy+1] = yL*mLs[1]; FH[3, iy+1] = yL*mLs[2]
            elif yL <= 0.0 and yR <= 0.0:
                FH[0, iy+1] = yR*dR; FH[1, iy+1] = yR*mRs[0]; FH[2, iy+1] = yR*mRs[1]; FH[3, iy+1] = yR*mRs[2]
            else:
                wL = min(yL, yR); wR = max(yL, yR); inv = 1.0/(wR - wL)
                FH[0, iy+1] = (wR*yL*dL - wL*yR*dR + wL*wR*(dR - dL))*inv
                for k in range(3):
                    FH[k+1, iy+1] = (wR*yL*mLs[k] - wL*yR*mRs[k] + wL*wR*(mRs[k] - mLs[k]))*inv
        # boundaries (outflow only)
        so = vely[N-1]
        FH[0, N] = so*max(dens[N-1], 0.0) if so > 0.0 else 0.0
        for k, v in enumerate((velx[N-1], vely[N-1], velz[N-1])):
            FH[k+1, N] = FH[0, N]*v
        si = vely[0]
        FH[0, 0] = si*max(dens[0], 0.0) if si < 0.0 else 0.0
        for k, v in enumerate((velx[0], vely[0], velz[0])):
            FH[k+1, 0] = FH[0, 0]*v

        # ---- low-order fluxes (first-order upwind, cell-average states) ----
        FL = np.zeros((4, N+1))
        for iy in range(N-1):
            vface = 0.5*(vely[iy] + vely[iy+1])
            up = iy if vface >= 0.0 else iy+1
            FL[0, iy+1] = vface*dens[up]
            FL[1, iy+1] = vface*dens[up]*velx[up]
            FL[2, iy+1] = vface*dens[up]*vely[up]
            FL[3, iy+1] = vface*dens[up]*velz[up]
        FL[0, N] = FH[0, N]; FL[:, N] = FH[:, N]
        FL[0, 0] = FH[0, 0]; FL[:, 0] = FH[:, 0]

        # ---- velocity bounds from neighbour stage velocities ----
        vb_min = np.empty((3, N)); vb_max = np.empty((3, N))
        vels = (velx, vely, velz)
        for k in range(3):
            v0 = vels[k]
            vb_min[k] = np.minimum(v0, np.minimum(np.roll(v0, 1), np.roll(v0, -1))) - margin
            vb_max[k] = np.maximum(v0, np.maximum(np.roll(v0, 1), np.roll(v0, -1))) + margin
        vb_min[:, 0] = np.minimum(vels[0][0], vels[0][1]) - margin if False else vb_min[:, 0]
        vb_max[:, -1] = vb_max[:, -1]

        # ---- per-cell positivity ratios over the 7 linear constraints ----
        AD = FH - FL                      # antidiffusive fluxes, shape (4, N+1)
        UL = np.stack((dens, momx, momy, momz))   # (4, N)
        # low-order update
        UL_new = UL.copy()
        for c in range(4):
            UL_new[c] = UL[c] - dt*(area[1:]*FL[c, 1:] - area[:-1]*FL[c, :-1])/vol

        theta_cell = np.ones(N)
        # constraint C(U) = a*dens + b*mom_k >= 0 with antidiffusive delta
        cons = [(1.0, 0, 1.0)]          # rho >= 0  (a=1, b unused)
        for k in range(3):
            cons.append((-vb_min[k], k+1, 1.0))   # m_k - v_min*rho >= 0
            cons.append(( vb_max[k], k+1, -1.0))  # v_max*rho - m_k >= 0
        for a, cidx, sgn in cons:
            if cidx == 0:
                C_UL = UL_new[0]
                dC = -dt*(area[1:]*AD[0, 1:] - area[:-1]*AD[0, :-1])/vol
            else:
                C_UL = sgn*UL_new[cidx] + a*UL_new[0]
                dC = sgn*(-dt*(area[1:]*AD[cidx, 1:] - area[:-1]*AD[cidx, :-1])/vol) \
                     + a*(-dt*(area[1:]*AD[0, 1:] - area[:-1]*AD[0, :-1])/vol)
            viol = dC < 0.0
            if np.any(viol):
                ratio = np.ones(N)
                ratio[viol] = np.clip(C_UL[viol]/(-dC[viol]), 0.0, 1.0)
                theta_cell = np.minimum(theta_cell, ratio)

        # ---- limit per face (shared coefficient, upwind-independent) ----
        theta_face = np.minimum(theta_cell[:-1], theta_cell[1:])   # interior faces 1..N-1
        theta = np.ones(N+1)
        theta[1:N] = theta_face
        # boundary faces keep their raw flux (already first-order outflow)
        Flim = FL + theta[None, :]*AD

        dd = np.empty((4, N))
        for c in range(4):
            dd[c] = -dt*(area[1:]*Flim[c, 1:] - area[:-1]*Flim[c, :-1])/vol
        return dd

    d1 = L(dens, momx, momy, momz)
    s1 = np.stack((dens, momx, momy, momz)) + d1
    s1[0][s1[0] < 0] = 0.0
    d2 = L(*s1)
    s2 = 0.75*np.stack((dens, momx, momy, momz)) + 0.25*(s1 + d2)
    s2[0][s2[0] < 0] = 0.0
    d3 = L(*s2)
    out = np.stack((dens, momx, momy, momz))/3.0 + 2.0*(s2 + d3)/3.0
    out[0][out[0] < 0] = 0.0
    return out

def run(N, d=2, rk=3, cfl_num=0.5, tend=TEND, limiter=True, verbose=False):
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
    momx = 0.7*dens; momz = 0.11*dens
    mass0 = np.sum(dens*vol)
    t = 0.0; steps = 0; dtmin = 1e300
    while t < tend - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dtmin = min(dtmin, dt)
        out = advect_y_rk_idl(dens, momx, momy, momz, dt, yf, d, w, rk)
        dens, momx, momy, momz = out
        t += dt; steps += 1
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
    lam = 1.0 + A*tend
    exact = np.empty(N)
    for i in range(N):
        yy = np.linspace(yf[i], yf[i+1], 512)
        exact[i] = lam**(-d)*np.trapezoid(compact_bump(yy/lam, 1.0, 1.8)*yy**(d-1), yy)/vol[i]
    L1 = np.sum(vol*np.abs(dens-exact))/np.sum(vol)
    merr = abs(np.sum(dens*vol) - mass0)/mass0
    vmax = np.max(np.abs(vely[dens > 1e-12])) if np.any(dens > 1e-12) else 0.0
    print(f"  N={N} rk={rk} limiter={limiter}: steps={steps} (dtmin={dtmin:.2e}) "
          f"L1={L1:.4e} mass={merr:.1e} max|vely|={vmax:.3f}")
    return L1, steps

if __name__ == "__main__":
    print("invariant-domain limiter, y_transport_cyl, cfl=0.5, SSPRK(3,3):")
    for N in (64, 128, 256):
        run(N, verbose=True)
