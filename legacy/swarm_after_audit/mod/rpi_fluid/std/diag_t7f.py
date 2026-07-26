#!/usr/bin/env python3
"""A/B tests for the Y sweep: limiter on/off, log/uniform grid."""
import numpy as np
import review_mock_tests as R

a_diag = 0.2

# --- advect_y variant with switchable limiter and grid --------------------
def advect_y2(dens, momy, dt, yf, d, w, use_limiter):
    N = len(dens)
    powy = float(d)
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    area = yf**(powy - 1.0)
    vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)

    def edges(cv):
        e = np.zeros(N+1)
        e[0] = cv[0]
        for f in range(1, N):
            if 2 <= f <= N-2:
                e[f] = w[f] @ cv[f-2:f+2]
            else:
                e[f] = w[f,0]*cv[f-1] + w[f,1]*cv[f]
            lo, hi = min(cv[f-1], cv[f]), max(cv[f-1], cv[f])
            e[f] = min(max(e[f], lo), hi)
        e[N] = cv[N-1]
        return e

    ed, ev = edges(dens), edges(vely)

    def face(edge, cv, iL, iR, up_left, cfl):
        if use_limiter:
            ql, qr, dq, q6 = R.ppm_limit(cv[iL], edge[iL], edge[iR])
        else:
            q0 = cv[iL]; ql, qr = edge[iL], edge[iR]
            dq = qr - ql; q6 = 6.0*q0 - 3.0*(ql+qr)
        if up_left:
            return qr - 0.5*cfl*(dq - (1.0-2.0*cfl/3.0)*q6)
        return ql + 0.5*cfl*(dq + (1.0-2.0*cfl/3.0)*q6)

    F = np.zeros(N+1); Md = np.zeros(N+1)
    for iy in range(N-1):
        y_face = yf[iy+1]
        s_face = y_face**powy/powy
        s_inL = yf[iy]**powy/powy
        s_outR = yf[iy+2]**powy/powy
        cflL = (s_face - (y_face-abs(vely[iy])*dt)**powy/powy)/(s_face - s_inL)
        cflR = ((y_face+abs(vely[iy+1])*dt)**powy/powy - s_face)/(s_outR - s_face)
        dL = max(face(ed, dens, iy, iy+1, True, cflL), 0.0)
        dR = max(face(ed, dens, iy+1, iy+2, False, cflR), 0.0)
        vL = face(ev, vely, iy, iy+1, True, cflL)
        vR = face(ev, vely, iy+1, iy+2, False, cflR)
        fd, fm = R.hll_2(vL, vR, dL, dL*vL, dR, dR*vR)
        F[iy+1], Md[iy+1] = fd, fm
    so = vely[N-1]
    F[N] = so*max(dens[N-1],0.0) if so > 0 else 0.0; Md[N] = F[N]*vely[N-1]
    si = vely[0]
    F[0] = si*max(dens[0],0.0) if si < 0 else 0.0;    Md[0] = F[0]*vely[0]
    scale = np.ones(N)
    for i in range(N):
        leave = dt*(area[i+1]*max(F[i+1],0.0) + area[i]*max(-F[i],0.0))
        allow = (1.0-1e-12)*max(dens[i],0.0)*vol[i]
        if leave > allow and leave > 0.0: scale[i] = allow/leave
    Fs, Ms = F.copy(), Md.copy()
    Fs[0] *= scale[0]; Ms[0] *= scale[0]
    for i in range(1, N):
        iup = i-1 if F[i] >= 0 else i
        Fs[i] *= scale[iup]; Ms[i] *= scale[iup]
    Fs[N] *= scale[N-1]; Ms[N] *= scale[N-1]
    dn = dens - dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
    mn = momy - dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
    neg = dn < 0; dn[neg] = 0; mn[neg] = 0
    return dn, mn

def run(mode, d, n, use_limiter, uniform_grid, t_end=0.5):
    if uniform_grid:
        yf = np.linspace(R.Y_MIN, R.Y_MAX, n+1)
    else:
        yf = R.Y_MIN*(R.Y_MAX/R.Y_MIN)**(np.arange(n+1)/n)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = R.ppm_nonuniform_weights(yf**d/d)
    rho0 = lambda y: np.exp(-((y - 1.6)/0.12)**2)
    dens = R.cell_average(rho0, yf, d)
    momy = R.cell_average(lambda y: rho0(y)*(a_diag*y if mode=="homologous" else 1.0), yf, d)
    t = 0.0
    while t < t_end - 1e-14:
        vely = np.where(dens >= R.RHO_VAC, momy/np.maximum(dens,1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(0.5/max(rate,1e-300), t_end - t)
        dens, momy = advect_y2(dens, momy, dt, yf, d, w, use_limiter)
        t += dt
    if mode == "homologous":
        lam = 1.0 + a_diag*t_end
        exact = R.cell_average(lambda y: lam**(-d)*rho0(y/lam), yf, d)
    else:
        exact = R.cell_average(lambda y: ((y-t_end)/y)**(d-1)*rho0(y-t_end), yf, d)
    return np.sum(vol*np.abs(dens-exact))/np.sum(vol)

if __name__ == "__main__":
    for mode in ("linear", "homologous"):
        for ug, lab_grid in ((False, "log"), (True, "uniform")):
            for ul, lab_lim in ((True, "limiter"), (False, "no-limiter")):
                errs = [run(mode, 2, n, ul, ug) for n in (128, 256, 512)]
                orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2)
                print(f"{mode:10s} {lab_grid:8s} {lab_lim:11s}: L1={np.array2string(np.array(errs), precision=3)} orders={np.round(orders,3)}")
