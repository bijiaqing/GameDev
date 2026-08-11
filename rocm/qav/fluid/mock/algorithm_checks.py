#!/usr/bin/env python3

"""Fast CPU checks for the numerical building blocks used by the fluid solver

This program translates selected HIP formulas into NumPy and compares them
with independent mathematical references.  It is useful while changing one
algorithm because it runs without a GPU and exposes intermediate arrays easily.

These are supplementary checks.  A Python transcription can drift away from
the HIP source, so the cluster suite in ``test_common`` remains the
authoritative end-to-end validation of the production implementation.

Reading guide for someone less familiar with Python:

* a NumPy array stores one value per grid cell
* arithmetic on an array acts on every cell at once
* ``array[start:stop]`` selects a consecutive section of an array
* ``np.roll`` shifts a periodic array and supplies neighboring-cell values
* functions named ``test_*`` perform checks and call :func:`report`
* the block at the bottom runs the checks only when this file is executed
"""


import numpy as np

# These constants mirror their counterparts in the fluid solver
POS_LIMIT = 0.9
RHO_VAC = 1.0e-15
Y_MIN, Y_MAX = 0.5, 2.5

# Each entry is ``(test name, passed)`` and is summarized at the end
RESULTS = []


def report(name, ok, detail=""):
    """Record and print one test result in a uniform format"""

    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def compact_bump(value, center=1.4, half_width=0.4):
    """Return a smooth profile with finite support and no edge discontinuity

    The exponential is evaluated only where ``|u| < 1``.  Using ``np.where``
    directly would evaluate both branches first and produce harmless overflow
    warnings outside the support.
    """

    value = np.asarray(value)
    u = (value - center) / half_width
    result = np.zeros_like(value, dtype=float)
    active = np.abs(u) < 1.0
    result[active] = np.exp(1.0 - 1.0 / (1.0 - u[active]**2))
    return result


def x_cyclic_cn(q_in, dt, Dx, dx_len, force_n_sub=None):
    """Advance one periodic ring with CN and a Sherman-Morrison cyclic solve"""

    N = len(q_in)
    if force_n_sub is None:
        n_sub = max(1, int(np.ceil(dt*Dx/(dx_len*dx_len)/POS_LIMIT)))
    else:
        n_sub = force_n_sub
    dt_sub = dt/n_sub
    c = 0.5*dt_sub*Dx/(dx_len*dx_len)
    diag = 1.0 + 2.0*c
    wrap = -c
    gamma = -diag
    vlast = wrap/gamma
    diag0 = diag - gamma
    diagN = diag - wrap*vlast

    q = q_in.astype(float).copy()
    for _ in range(n_sub):
        rhs = c*np.roll(q, 1) + (1.0 - 2.0*c)*q + c*np.roll(q, -1)
        up = np.zeros(N)
        yv = rhs.copy()
        zv = np.zeros(N)
        zv[0] = gamma
        zv[-1] = wrap

        dc = diag0
        up[0] = -c/dc
        yv[0] /= dc
        zv[0] /= dc
        for i in range(1, N):
            dc = diag if i < N-1 else diagN
            piv = dc + c*up[i-1]
            up[i] = (-c/piv) if i < N-1 else 0.0
            yv[i] = (yv[i] + c*yv[i-1])/piv
            zv[i] = (zv[i] + c*zv[i-1])/piv
        for i in range(N-2, -1, -1):
            yv[i] -= up[i]*yv[i+1]
            zv[i] -= up[i]*zv[i+1]

        base_proj = yv[0] + vlast*yv[-1]
        corr_proj = zv[0] + vlast*zv[-1]
        corr = base_proj/(1.0 + corr_proj)
        q = yv - corr*zv
    return q

def dense_cyclic_cn_step(q, dt, Dx, dx_len):
    """Solve one CN step with a dense matrix to provide a slow reference"""

    N = len(q)
    c = 0.5*dt*Dx/(dx_len*dx_len)
    A = np.diag(np.full(N, 1.0 + 2.0*c))
    B = np.diag(np.full(N, 1.0 - 2.0*c))
    for i in range(N):
        A[i, (i-1) % N] -= c
        A[i, (i+1) % N] -= c
        B[i, (i-1) % N] += c
        B[i, (i+1) % N] += c
    return np.linalg.solve(A, B @ q)

def test_x_diffusion():
    """Check the periodic CN solver, conservation, accuracy, and positivity"""
    rng = np.random.default_rng(7)
    N = 128
    dx_len = 2.0*np.pi/N


    worst = 0.0
    for dtDx in (1.0e-4, 3.0e-2, 1.0, 40.0):
        dt, Dx = dtDx, 1.0
        q0 = rng.random(N) + 0.2
        n_sub = max(1, int(np.ceil(dt*Dx/(dx_len*dx_len)/POS_LIMIT)))
        q_fast = x_cyclic_cn(q0, dt, Dx, dx_len)

        q_ref = q0.copy()
        for _ in range(n_sub):
            q_ref = dense_cyclic_cn_step(q_ref, dt/n_sub, Dx, dx_len)
        worst = max(worst, np.max(np.abs(q_fast - q_ref)))
    report("T1a X-diffusion SM solve matches dense cyclic solve", worst < 1e-10,
           f"max abs diff = {worst:.3e}")


    qc = x_cyclic_cn(np.full(N, 2.5), 0.37, 1.0, dx_len)
    ok_const = np.max(np.abs(qc - 2.5)) < 1e-13
    q0 = rng.random(N) + 0.1
    q1 = x_cyclic_cn(q0, 0.41, 1.0, dx_len)
    ok_mass = abs(q1.sum() - q0.sum()) < 1e-12*N
    report("T1b X-diffusion preserves constant state and ring mass",
           ok_const and ok_mass,
           f"const err={np.max(np.abs(qc-2.5)):.2e}, mass err={abs(q1.sum()-q0.sum()):.2e}")


    m = 2
    errs = []
    Ns = [32, 64, 128, 256]
    for n in Ns:
        dx = 2.0*np.pi/n
        x  = (np.arange(n) + 0.5)*dx
        Dx = 0.05
        T  = 1.0/(Dx*m*m)
        dt = 0.05*dx*dx/Dx
        nsteps = int(round(T/dt))
        dt = T/nsteps
        q = 1.0 + 0.01*np.cos(m*x)
        for _ in range(nsteps):
            q = x_cyclic_cn(q, dt, Dx, dx)
        amp = 2.0*np.abs(np.mean((q - 1.0)*np.exp(-1j*m*x)))/0.01
        lam_num = -np.log(amp)/T
        errs.append(abs(lam_num - Dx*m*m)/(Dx*m*m))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    report("T1c X-diffusion Fourier decay rate converges at ~2nd order",
           np.all(orders > 1.7) and errs[-1] < errs[0],
           f"rate rel errs={np.array2string(np.array(errs), precision=2)}, orders={np.round(orders,2)}")


    q0 = np.zeros(N)
    q0[N//4] = 1.0
    dt, Dx = 1.0, 1.0
    q_sub  = x_cyclic_cn(q0, dt, Dx, dx_len)
    q_nosub = x_cyclic_cn(q0, dt, Dx, dx_len, force_n_sub=1)
    n_sub = max(1, int(np.ceil(dt*Dx/(dx_len*dx_len)/POS_LIMIT)))
    report("T1d X-diffusion positivity holds only with POS_LIMIT subcycling",
           q_sub.min() > -1e-12 and q_nosub.min() < -1e-6,
           f"n_sub={n_sub}, min(sub)={q_sub.min():.3e}, min(no sub)={q_nosub.min():.3e}")


# ======================================================================================================================
# Radial Crank-Nicolson diffusion
# ======================================================================================================================

def y_grid(N, d):
    """Construct logarithmic radial faces, centers, volumes, and face areas"""
    r  = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    yc = np.sqrt(yf[:-1]*yf[1:])
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    return r, yf, yc, vol, area

def y_diffusion(q_in, dt, D, yf, yc, vol, area, pos_limit=POS_LIMIT):
    """Advance radial diffusion with CN and a tridiagonal Thomas solve"""

    N = len(q_in)
    dlen = np.diff(yc)
    cn_i = np.zeros(N)
    cn_o = np.zeros(N)
    cn_i[1:]   = 0.5*dt*area[1:-1]*D/(dlen*vol[1:])
    cn_o[:-1]  = 0.5*dt*area[1:-1]*D/(dlen*vol[:-1])
    n_sub = max(1, int(np.ceil(np.max(cn_i + cn_o)/pos_limit)))
    cn_i /= n_sub
    cn_o /= n_sub

    lower = -cn_i
    upper = -cn_o
    diag = 1.0 - lower - upper

    q = q_in.astype(float).copy()
    for _ in range(n_sub):
        rhs = cn_i*np.roll(q, 1) + (1.0 - cn_i - cn_o)*q + cn_o*np.roll(q, -1)

        up = np.zeros(N)
        yv = rhs.copy()
        up[0] = upper[0]/diag[0]
        yv[0] /= diag[0]
        for i in range(1, N):
            piv = diag[i] - lower[i]*up[i-1]
            up[i] = (upper[i]/piv) if i < N-1 else 0.0
            yv[i] = (yv[i] - lower[i]*yv[i-1])/piv
        for i in range(N-2, -1, -1):
            yv[i] -= up[i]*yv[i+1]
        q = yv
    return q

def y_explicit_matrix(D, yf, yc, vol, area):
    """Build the semi-discrete radial diffusion matrix for eigenvalue checks"""

    N = len(yc)
    dlen = np.diff(yc)
    L = np.zeros((N, N))
    ai = np.zeros(N)
    ao = np.zeros(N)
    ai[1:]  = area[1:-1]*D/(dlen*vol[1:])
    ao[:-1] = area[1:-1]*D/(dlen*vol[:-1])
    for i in range(N):
        L[i, i] = -(ai[i] + ao[i])
        if i > 0:     L[i, i-1] = ai[i]
        if i < N-1:   L[i, i+1] = ao[i]
    return L

def shell_eigenvalue(d, n_rk=40000):
    """Find the first nonzero Neumann shell eigenvalue by shooting with RK4"""


    def shoot(k):
        h = (Y_MAX - Y_MIN)/n_rk
        Q, P = 1.0, 0.0
        y = Y_MIN
        for _ in range(n_rk):
            def f(y, Q, P): return P, -((d-1.0)/y)*P - k*k*Q
            k1q, k1p = f(y, Q, P)
            k2q, k2p = f(y+0.5*h, Q+0.5*h*k1q, P+0.5*h*k1p)
            k3q, k3p = f(y+0.5*h, Q+0.5*h*k2q, P+0.5*h*k2p)
            k4q, k4p = f(y+h,     Q+h*k3q,     P+h*k3p)
            Q += h*(k1q + 2*k2q + 2*k3q + k4q)/6
            P += h*(k1p + 2*k2p + 2*k3p + k4p)/6
            y += h
        return P

    ks = np.linspace(0.05, 8.0, 400)
    prev = shoot(ks[0])
    for k in ks[1:]:
        cur = shoot(k)
        if prev*cur < 0:
            lo, hi = k - (ks[1]-ks[0]), k
            for _ in range(80):
                mid = 0.5*(lo + hi)
                if shoot(lo)*shoot(mid) <= 0: hi = mid
                else: lo = mid
            return 0.5*(lo+hi)
        prev = cur
    raise RuntimeError("no eigenvalue bracketed")

def test_y_diffusion():
    """Check radial CN invariants and its cylindrical/spherical eigenrates"""
    for d in (2, 3):

        N = 128
        _, yf, yc, vol, area = y_grid(N, d)
        rng = np.random.default_rng(3)
        q0 = rng.random(N) + 0.1
        q1 = y_diffusion(q0, 0.7, 1.0e-2, yf, yc, vol, area)
        mass_err = abs(np.sum(q1*vol) - np.sum(q0*vol))/np.sum(q0*vol)
        qc = y_diffusion(np.full(N, 1.7), 0.7, 1.0e-2, yf, yc, vol, area)
        const_err = np.max(np.abs(qc - 1.7))
        qs = y_diffusion(np.where(yc < 1.5, 1.0, 0.0), 0.7, 1.0e-2, yf, yc, vol, area)
        report(f"T2a d={d} Y-diffusion constant/mass/positivity",
               mass_err < 1e-12 and const_err < 1e-13 and qs.min() > -1e-12,
               f"mass rel err={mass_err:.2e}, const err={const_err:.2e}, min={qs.min():.2e}")


        k1 = shell_eigenvalue(d)
        D = 1.0e-3
        errs = []
        Ns = [32, 64, 128, 256]
        for n in Ns:
            _, yf, yc, vol, area = y_grid(n, d)
            L = y_explicit_matrix(D, yf, yc, vol, area)
            ev = np.linalg.eigvals(L).real
            lam = ev[np.argmin(np.abs(ev - (-D*k1*k1)))]
            errs.append(abs(lam + D*k1*k1)/(D*k1*k1))
        orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
        report(f"T2b d={d} Y-diffusion eigenrate converges at ~2nd order (k1={k1:.6f})",
               np.all(orders > 1.7), f"rel errs={np.array2string(np.array(errs), precision=2)}, "
               f"orders={np.round(orders,2)}")


# ======================================================================================================================
# Drag and external-force source integration
# ======================================================================================================================

def source_update(u, ug, ts, Fn, Fnew, dt):
    """Integrate drag exactly while the external force changes linearly"""
    h = dt/ts
    relax = -np.expm1(-h)
    decay = 1.0 - relax
    if h < 1.0e-4:
        h2 = h*h
        h3 = h2*h
        wn  = dt*(0.5 - h/3.0 + h2/8.0  - h3/30.0)
        wn1 = dt*(0.5 - h/6.0 + h2/24.0 - h3/120.0)
    else:
        wn1 = ts*(h - relax)/h
        wn  = ts*relax - wn1
    return decay*u + relax*ug + wn*Fn + wn1*Fnew

def test_source():
    """Check source integration from weak drag through the stiff limit"""

    worst = 0.0
    for h in (1e-6, 1e-2, 1.0, 100.0, 1e6):
        ts = 0.37
        dt = h*ts
        u0, ug, F = 1.3, -0.4, 2.2
        un = source_update(u0, ug, ts, F, F, dt)
        ex = ug + F*ts + (u0 - ug - F*ts)*np.exp(-dt/ts)
        worst = max(worst, abs(un - ex))
    report("T3a source update exact for constant force at any dt/ts", worst < 1e-12,
           f"max abs err={worst:.2e}")


    worst = 0.0
    for h in (1e-5, 0.1, 2.0, 300.0):
        ts = 0.61
        dt = h*ts
        u0, ug, a0, a1 = 0.8, 0.2, -1.1, 0.7
        un = source_update(u0, ug, ts, a0, a0 + a1*dt, dt)
        E = np.exp(-dt/ts)
        ex = ug + (u0 - ug)*E + a0*ts*(1.0 - E) + a1*(ts*dt - ts*ts*(1.0 - E))
        worst = max(worst, abs(un - ex))
    report("T3b source update exact for linear force", worst < 1e-12,
           f"max abs err={worst:.2e}")


    ts = 1e-5
    dt = 1000.0*ts
    u = source_update(0.0, 1.0, ts, 0.0, 0.0, dt)
    u2 = source_update(u, 1.0, ts, 0.0, 0.0, dt)
    report("T3c stiff drag relaxes monotonically to gas (no CN -1 amplification)",
           abs(u - 1.0) < 1e-12 and abs(u2 - 1.0) < 1e-12,
           f"u1={u:.16f}, u2={u2:.16f}")


    ts = 1.0
    def weights(h):
        relax = -np.expm1(-h)
        if h < 1e-4:
            h2 = h*h
            h3 = h2*h
            return (h*ts*(0.5 - h/3.0 + h2/8.0 - h3/30.0),
                    h*ts*(0.5 - h/6.0 + h2/24.0 - h3/120.0))
        wn1 = ts*(h - relax)/h
        return ts*relax - wn1, wn1
    w_lo = weights(1e-4*(1 - 1e-9))
    w_hi = weights(1e-4*(1 + 1e-9))
    jump = max(abs(w_lo[0]-w_hi[0]), abs(w_lo[1]-w_hi[1]))/1e-4
    report("T3d force-weight branch continuous at h=1e-4", jump < 1e-8,
           f"relative jump={jump:.2e}")


# ======================================================================================================================
# PPM reconstruction on nonuniform finite-volume coordinates
# ======================================================================================================================

def ppm_cubic_face_weights(face_s, iface):
    """Derive four weights that reproduce cubics at one interior face"""
    face = face_s[iface]
    scale = max(face - face_s[iface-2], face_s[iface+2] - face)
    A = np.zeros((4, 5))
    for n in range(4):
        for j in range(4):
            ic = iface - 2 + j
            tl = (face_s[ic]   - face)/scale
            tu = (face_s[ic+1] - face)/scale
            A[n, j] = (tu**(n+1) - tl**(n+1))/((n+1)*(tu - tl))
        A[n, 4] = 1.0 if n == 0 else 0.0
    for col in range(4):
        piv = col + int(np.argmax(np.abs(A[col:, col])))
        A[[col, piv]] = A[[piv, col]]
        A[col] = A[col]/A[col, col]
        for row in range(4):
            if row == col: continue
            A[row] = A[row] - A[row, col]*A[col]
    return A[:, 4].copy()

def ppm_nonuniform_weights(face_s):
    """Precompute interior cubic weights and two-cell boundary weights"""
    ncells = len(face_s) - 1
    w = np.zeros((ncells + 1, 4))
    for iface in range(1, ncells):
        if 2 <= iface <= ncells - 2:
            w[iface] = ppm_cubic_face_weights(face_s, iface)
        else:
            cL = 0.5*(face_s[iface-1] + face_s[iface])
            cR = 0.5*(face_s[iface]   + face_s[iface+1])
            w[iface, 0] = (cR - face_s[iface])/(cR - cL)
            w[iface, 1] = (face_s[iface] - cL)/(cR - cL)
    return w

def ppm_edges_nonuniform(cell_val, w):
    """Reconstruct bounded face values from nonuniform cell averages"""
    n = len(cell_val)
    edge = np.zeros(n + 1)
    edge[0] = cell_val[0]
    for iface in range(1, n):
        if 2 <= iface <= n - 2:
            edge[iface] = w[iface] @ cell_val[iface-2:iface+2]
        else:
            edge[iface] = w[iface, 0]*cell_val[iface-1] + w[iface, 1]*cell_val[iface]
        lo = min(cell_val[iface-1], cell_val[iface])
        hi = max(cell_val[iface-1], cell_val[iface])
        edge[iface] = min(max(edge[iface], lo), hi)
    edge[n] = cell_val[n-1]
    return edge

def test_ppm_weights():
    """Check polynomial exactness and smooth reconstruction order"""


    worst = 0.0
    for geom in ("log2", "log3", "cos"):
        N = 33
        if geom == "cos":
            faces_z = np.linspace(0.15, 1.35, N+1)
            fs = -np.cos(faces_z)
        else:
            d = 2 if geom == "log2" else 3
            r = (Y_MAX/Y_MIN)**(1.0/N)
            fs = (Y_MIN*r**np.arange(N+1))**d/d
        w = ppm_nonuniform_weights(fs)
        coefs = np.array([1.0, 0.3, 0.2, 0.05])
        cellavg = sum(coefs[n]*(fs[1:]**(n+1) - fs[:-1]**(n+1))/((n+1)*(fs[1:]-fs[:-1]))
                    for n in range(4))
        edge = ppm_edges_nonuniform(cellavg, w)
        exact = sum(coefs[n]*fs**n for n in range(4))
        worst = max(worst, np.max(np.abs(edge[2:-2] - exact[2:-2])))
    report("T4a nonuniform PPM weights exact for cubics in volume coordinate",
           worst < 1e-11, f"max cubic-face err={worst:.2e}")


    errs = []
    Ns = [32, 64, 128, 256]
    qfun = lambda y: np.exp(-y)
    for n in Ns:
        d = 2
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        fs = yf**d/d
        w = ppm_nonuniform_weights(fs)

        cellavg = np.empty(n)
        for i in range(n):
            yy = np.linspace(yf[i], yf[i+1], 512)
            cellavg[i] = np.trapezoid(qfun(yy)*yy**(d-1), yy)/(fs[i+1]-fs[i])
        edge = ppm_edges_nonuniform(cellavg, w)
        errs.append(np.max(np.abs(edge[2:-2] - qfun(yf[2:-2]))))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    report("T4b interior face reconstruction ~4th order for smooth monotone data",
           np.all(orders > 3.4),
           f"errs={np.array2string(np.array(errs), precision=3)}, orders={np.round(orders,3)}")


# ======================================================================================================================
# Periodic FARGO plus PPM transport
# ======================================================================================================================

def ppm_edge_uni(qm1, q0, qp1, qp2):
    """Reconstruct one bounded PPM face on a uniform periodic grid"""
    qe = (7.0*(q0 + qp1) - (qm1 + qp2))/12.0
    return min(max(qe, min(q0, qp1)), max(q0, qp1))

def ppm_limit(q0, ql, qr):
    """Limit one parabolic cell profile so it introduces no new extrema"""
    dq = qr - ql
    q6 = 6.0*q0 - 3.0*(ql + qr)
    if (qr - q0)*(q0 - ql) <= 0.0:
        return q0, q0, 0.0, 0.0
    if dq*q6 > dq*dq:
        ql = 3.0*q0 - 2.0*qr
        dq = qr - ql
        q6 = 6.0*q0 - 3.0*(ql + qr)
    if -dq*q6 > dq*dq:
        qr = 3.0*q0 - 2.0*ql
        dq = qr - ql
        q6 = 6.0*q0 - 3.0*(ql + qr)
    return ql, qr, dq, q6

def ppm_face_value(edge, cell_val, iL, iR, up_left, cfl):
    """Average the limited PPM profile over an upwind domain of dependence"""
    ql, qr, dq, q6 = ppm_limit(cell_val[iL], edge[iL], edge[iR])
    if up_left:
        return qr - 0.5*cfl*(dq - (1.0 - 2.0*cfl/3.0)*q6)
    return ql + 0.5*cfl*(dq + (1.0 - 2.0*cfl/3.0)*q6)

def hll_2(sL, sR, dL, mL, dR, mR):
    """Return pressureless HLL fluxes for density and one momentum"""
    if sL >= 0.0 and sR >= 0.0:
        return sL*dL, sL*mL
    if sL <= 0.0 and sR <= 0.0:
        return sR*dR, sR*mR
    wL = min(sL, sR)
    wR = max(sL, sR)
    inv = 1.0/(wR - wL)
    fd = (wR*sL*dL - wL*sR*dR + wL*wR*(dR - dL))*inv
    fm = (wR*sL*mL - wL*sR*mR + wL*wR*(mR - mL))*inv
    return fd, fm

def advect_x(dens, momx, dt, Rc=1.0):
    """Advance one periodic ring with the compact FARGO-PPM transcription"""


    N = len(dens)
    dx = 2.0*np.pi/N
    velx = np.where(dens >= RHO_VAC, momx/np.maximum(dens, 1e-300), np.sqrt(Rc))
    avg = velx.mean()
    shift = avg*dt/(Rc*Rc*dx)
    n_shift = int(np.rint(shift))
    omega_shift = n_shift*dx/dt
    frame = Rc*Rc*omega_shift

    dens_s = np.roll(dens, n_shift)
    momx_s = np.roll(momx, n_shift)
    velx_s = np.roll(velx, n_shift)
    res = velx_s - frame

    ed = np.array([ppm_edge_uni(dens_s[(i-2) % N], dens_s[(i-1) % N], dens_s[i], dens_s[(i+1) % N])
                   for i in range(N)])
    ev = np.array([ppm_edge_uni(velx_s[(i-2) % N], velx_s[(i-1) % N], velx_s[i], velx_s[(i+1) % N])
                   for i in range(N)])

    fd = np.zeros(N)
    fm = np.zeros(N)
    for i in range(N):
        ip1 = (i+1) % N
        ip2 = (i+2) % N
        cflL = abs(res[i])  /(Rc*Rc)*dt/dx
        cflR = abs(res[ip1])/(Rc*Rc)*dt/dx
        dL = max(ppm_face_value(ed, dens_s, i,   ip1, True,  cflL), 0.0)
        dR = max(ppm_face_value(ed, dens_s, ip1, ip2, False, cflR), 0.0)
        vL = ppm_face_value(ev, velx_s, i,   ip1, True,  cflL)
        vR = ppm_face_value(ev, velx_s, ip1, ip2, False, cflR)
        oL = (vL - frame)/(Rc*Rc)
        oR = (vR - frame)/(Rc*Rc)
        fd[i], fm[i] = hll_2(oL, oR, dL, dL*vL, dR, dR*vR)

    scale = np.ones(N)
    for i in range(N):
        im1 = (i-1) % N
        leave = dt*(max(fd[i], 0.0) + max(-fd[im1], 0.0))
        avail = max(dens_s[i], 0.0)*dx
        allow = (1.0 - 1.0e-12)*avail
        if leave > allow and leave > 0.0:
            scale[i] = allow/leave

    fsd = np.zeros(N)
    fsm = np.zeros(N)
    for i in range(N):
        ip1 = (i+1) % N
        iup = i if fd[i] >= 0.0 else ip1
        fsd[i] = fd[i]*scale[iup]
        fsm[i] = fm[i]*scale[iup]

    dens_n = dens_s - dt*(fsd - np.roll(fsd, 1))/dx
    momx_n = momx_s - dt*(fsm - np.roll(fsm, 1))/dx
    neg = dens_n < 0.0
    dens_n[neg] = 0.0
    momx_n[neg] = 0.0
    return dens_n, momx_n

def test_x_advection():
    """Check periodic accuracy, conservation, and density positivity"""
    m = 2

    for frac in (0.0, 0.25, 0.5, 0.75):
        N = 128
        dx = 2.0*np.pi/N
        x = (np.arange(N) + 0.5)*dx
        ncell_shift = 3 + frac
        dt = ncell_shift*dx
        nsteps = int(round(2.0*np.pi/dt))

        dt = 2.0*np.pi/nsteps
        dens = 1.0 + 0.4*np.cos(m*x)
        momx = dens*1.0
        mass0 = dens.sum()*dx
        for _ in range(nsteps):
            dens, momx = advect_x(dens, momx, dt)
        exact = 1.0 + 0.4*np.cos(m*x)
        err = np.max(np.abs(dens - exact))
        merr = abs(dens.sum()*dx - mass0)/mass0
        report(f"T5a FARGO+PPM one-revolution, shift frac={frac}",
               err < 2e-2 and merr < 1e-12,
               f"Linf={err:.2e} after {nsteps} steps, mass rel err={merr:.2e}")


    errs = []
    Ns = [32, 64, 128, 256]
    for n in Ns:
        dx = 2.0*np.pi/n
        x = (np.arange(n) + 0.5)*dx
        dt = 3.25*dx
        nsteps = int(round(2.0*np.pi/dt))
        dt = 2.0*np.pi/nsteps
        dens = 1.0 + 0.4*np.cos(m*x)
        momx = dens.copy()
        for _ in range(nsteps):
            dens, momx = advect_x(dens, momx, dt)
        exact = 1.0 + 0.4*np.cos(m*x)
        errs.append(np.sum(np.abs(dens - exact))*dx/(2.0*np.pi))
    orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
    report("T5b FARGO+PPM smooth-mode order >= 2",
           np.all(orders > 1.9),
           f"L1={np.array2string(np.array(errs), precision=2)}, orders={np.round(orders,2)}")


    N = 256
    dx = 2.0*np.pi/N
    x = (np.arange(N) + 0.5)*dx
    dens = np.where((x > 1.0) & (x < 3.0), 1.0, 0.2)
    momx = dens.copy()
    dt = 2.0*dx
    mass0 = dens.sum()*dx
    for _ in range(int(round(2.0*np.pi/dt))):
        dens, momx = advect_x(dens, momx, dt)
    report("T5c positivity scaling: no negative density, mass conserved",
           dens.min() >= 0.0 and abs(dens.sum()*dx - mass0)/mass0 < 1e-12,
           f"min={dens.min():.3e}, mass rel err={abs(dens.sum()*dx-mass0)/mass0:.2e}")


# ======================================================================================================================
# Radial optical-depth quadrature
# ======================================================================================================================

def test_optdepth():
    """Compare cumulative optical depth with power-law antiderivatives"""
    kappa = 1.0e5


    for p in (1.0, -1.0, -3.0):
        errs_face = []
        Ns = [32, 64, 128, 256]
        for n in Ns:
            r = (Y_MAX/Y_MIN)**(1.0/n)
            yf = Y_MIN*r**np.arange(n+1)
            yc = np.sqrt(yf[:-1]*yf[1:])
            C = 1.0
            rho = C*yc**p
            tau_cell = kappa*rho*(yf[1:] - yf[:-1])
            tau_face = np.concatenate([[0.0], np.cumsum(tau_cell)])
            if p == -1.0:
                exact = kappa*C*np.log(yf/Y_MIN)
            else:
                exact = kappa*C*(yf**(p+1.0) - Y_MIN**(p+1.0))/(p+1.0)
            errs_face.append(np.max(np.abs(tau_face - exact))/max(1.0, np.max(np.abs(exact))))
        orders = np.log(np.array(errs_face[:-1])/np.array(errs_face[1:]))/np.log(2.0)
        report(f"T6 p={p:+.0f} cumulative face optical depth 2nd order",
               np.all(orders > 1.7),
               f"rel errs={np.array2string(np.array(errs_face), precision=3)}, orders={np.round(orders,3)}")

    errs_c = []
    for n in (32, 64, 128, 256):
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        yc = np.sqrt(yf[:-1]*yf[1:])
        rho = yc**1.0
        tau_cell = kappa*rho*(yf[1:] - yf[:-1])
        tau_face = np.concatenate([[0.0], np.cumsum(tau_cell)])
        tau_centre = 0.5*(tau_face[:-1] + tau_face[1:])
        exact_c = kappa*(yc**2 - Y_MIN**2)/2.0
        errs_c.append(np.max(np.abs(tau_centre - exact_c))/np.max(np.abs(exact_c)))
    orders = np.log(np.array(errs_c[:-1])/np.array(errs_c[1:]))/np.log(2.0)
    report("T6b cell-centred tau 2nd-order but not roundoff-exact",
           np.all(orders > 1.7) and errs_c[-1] > 1e-14,
           f"rel errs={np.array2string(np.array(errs_c), precision=3)}, orders={np.round(orders,3)}")

    # T6c: the C9 logarithmic-center rule tau_c = tau_i + f_c*(tau_o - tau_i), f_c = 1/(sqrt(r)+1),
    # is roundoff-exact for constant extinction (piecewise-constant cells are then exact) and
    # keeps the underlying second-order cell quadrature otherwise
    kappa_c = 1.0e5
    exact_const = []
    for n in (32, 64, 128, 256):
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        yc = np.sqrt(yf[:-1]*yf[1:])
        fc = 1.0/(np.sqrt(r) + 1.0)
        tau_cell = kappa_c*1.0*(yf[1:] - yf[:-1])
        tau_face = np.concatenate([[0.0], np.cumsum(tau_cell)])
        tau_c = tau_face[:-1] + fc*(tau_face[1:] - tau_face[:-1])
        exact_const.append(np.max(np.abs(tau_c - kappa_c*(yc - Y_MIN))))
    report("T6c1 logarithmic-center optical depth roundoff-exact for constant extinction",
           max(exact_const) < 1e-9, f"max abs err={max(exact_const):.2e}")

    errs_c2 = []
    for n in (32, 64, 128, 256):
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        yc = np.sqrt(yf[:-1]*yf[1:])
        fc = 1.0/(np.sqrt(r) + 1.0)
        rho = yc**1.0
        tau_cell = kappa_c*rho*(yf[1:] - yf[:-1])
        tau_face = np.concatenate([[0.0], np.cumsum(tau_cell)])
        tau_c = tau_face[:-1] + fc*(tau_face[1:] - tau_face[:-1])
        exact_c2 = kappa_c*(yc**2 - Y_MIN**2)/2.0
        errs_c2.append(np.max(np.abs(tau_c - exact_c2))/np.max(np.abs(exact_c2)))
    orders = np.log(np.array(errs_c2[:-1])/np.array(errs_c2[1:]))/np.log(2.0)
    report("T6c2 logarithmic-center optical depth 2nd order for a power-law profile",
           np.all(orders > 1.7),
           f"rel errs={np.array2string(np.array(errs_c2), precision=3)}, orders={np.round(orders,3)}")


# ======================================================================================================================
# Radial PPM transport and SSPRK comparison
# ======================================================================================================================

def advect_y(dens, momy, dt, yf, d, w):
    """Advance density and radial momentum with one forward-Euler operator evaluation"""

    N = len(dens)
    powy = float(d)
    yc = np.sqrt(yf[:-1]*yf[1:])
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    area = yf**(powy - 1.0)

    vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)

    ed = ppm_edges_nonuniform(dens, w)
    ev = ppm_edges_nonuniform(vely, w)

    F = np.zeros(N+1)
    Md = np.zeros(N+1)
    for iy in range(N-1):
        y_face = yf[iy+1]
        s_face = y_face**powy/powy
        s_inL  = yf[iy]**powy/powy
        s_outR = yf[iy+2]**powy/powy
        trL = y_face - abs(vely[iy])*dt
        trR = y_face + abs(vely[iy+1])*dt
        cflL = (s_face - trL**powy/powy)/(s_face - s_inL)
        cflR = (trR**powy/powy - s_face)/(s_outR - s_face)
        dL = max(ppm_face_value(ed, dens, iy,   iy+1, True,  cflL), 0.0)
        dR = max(ppm_face_value(ed, dens, iy+1, iy+2, False, cflR), 0.0)
        vL = ppm_face_value(ev, vely, iy,   iy+1, True,  cflL)
        vR = ppm_face_value(ev, vely, iy+1, iy+2, False, cflR)
        fd, fm = hll_2(vL, vR, dL, dL*vL, dR, dR*vR)
        F[iy+1], Md[iy+1] = fd, fm

    so = vely[N-1]
    F[N] = so*max(dens[N-1], 0.0) if so > 0.0 else 0.0
    Md[N] = F[N]*vely[N-1]
    si = vely[0]
    F[0] = si*max(dens[0], 0.0) if si < 0.0 else 0.0
    Md[0] = F[0]*vely[0]


    scale = np.ones(N)
    for i in range(N):
        leave = dt*(area[i+1]*max(F[i+1], 0.0) + area[i]*max(-F[i], 0.0))
        avail = max(dens[i], 0.0)*vol[i]
        allow = (1.0 - 1.0e-12)*avail
        if leave > allow and leave > 0.0:
            scale[i] = allow/leave
    Fs = F.copy()
    Ms = Md.copy()
    Fs[0] *= scale[0]
    Ms[0] *= scale[0]
    for i in range(1, N):
        iup = i-1 if F[i] >= 0.0 else i
        Fs[i] *= scale[iup]
        Ms[i] *= scale[iup]
    Fs[N] *= scale[N-1]
    Ms[N] *= scale[N-1]

    dens_n = dens - dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
    momy_n = momy - dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
    neg = dens_n < 0.0
    dens_n[neg] = 0.0
    momy_n[neg] = 0.0
    return dens_n, momy_n

def cell_average(fun, yf, d, n_quad=4000):
    """Numerically average a radial function over every finite-volume cell"""

    powy = float(d)
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    out = np.empty(len(vol))
    for i in range(len(vol)):
        yy = np.linspace(yf[i], yf[i+1], n_quad)
        out[i] = np.trapezoid(fun(yy)*yy**(d-1), yy)/vol[i]
    return out

def test_y_advection():
    """Check radial transport geometry and measure its CFL-dependent order"""
    a = 0.2
    t_end = 0.5
    lam = 1.0 + a*t_end
    rho0 = lambda y: np.exp(-((y - 1.6)/0.12)**2)

    def run(d, n, cfl):
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        vol = (yf[1:]**d - yf[:-1]**d)/d
        area = yf**(d-1)
        fs = yf**d/d
        w = ppm_nonuniform_weights(fs)
        dens = cell_average(rho0, yf, d)
        momy = cell_average(lambda y: rho0(y)*a*y, yf, d)
        mass0 = np.sum(dens*vol)
        t = 0.0
        while t < t_end - 1e-14:
            vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
            rate = np.max(np.abs(vely)*area[1:]/vol)
            dt = min(cfl/max(rate, 1e-300), t_end - t)
            dens, momy = advect_y(dens, momy, dt, yf, d, w)
            t += dt
        exact_rho = cell_average(lambda y: lam**(-d)*rho0(y/lam), yf, d)
        L1 = np.sum(vol*np.abs(dens - exact_rho))/np.sum(vol)
        merr = abs(np.sum(dens*vol) - mass0)/mass0
        return L1, merr


    for d in (2, 3):
        errs, mass_errs = [], []
        for n in (128, 256, 512):
            L1, merr = run(d, n, 0.05)
            errs.append(L1)
            mass_errs.append(merr)
        orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
        report(f"T7a d={d} homologous expansion spatial order ~2 (CFL=0.05), mass kept",
               np.all(orders > 1.7) and max(mass_errs) < 1e-10,
               f"L1={np.array2string(np.array(errs), precision=3)}, orders={np.round(orders,3)}, "
               f"mass rel err={max(mass_errs):.2e}")


    for d in (2,):
        errs = []
        for n in (128, 256, 512):
            L1, _ = run(d, n, 0.5)
            errs.append(L1)
        orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
        report(f"T7b d={d} homologous expansion at CFL=0.5 degrades toward 1st order (known finding)",
               np.all(orders < 1.8),
               f"L1={np.array2string(np.array(errs), precision=3)}, orders={np.round(orders,3)}")


def advect_y_rk(dens, momy, dt, yf, d, w, order):
    """Apply the radial forward-Euler operator with SSPRK2 or SSPRK3"""
    N = len(dens)
    powy = float(d)
    vol = (yf[1:]**powy - yf[:-1]**powy)/powy
    area = yf**(powy - 1.0)

    def L(dens, momy):
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        ed = ppm_edges_nonuniform(dens, w)
        ev = ppm_edges_nonuniform(vely, w)
        F = np.zeros(N+1)
        M = np.zeros(N+1)
        for iy in range(N-1):
            dL = max(ppm_face_value(ed, dens, iy, iy+1, True, 0.0), 0.0)
            dR = max(ppm_face_value(ed, dens, iy+1, iy+2, False, 0.0), 0.0)
            vL = ppm_face_value(ev, vely, iy, iy+1, True, 0.0)
            vR = ppm_face_value(ev, vely, iy+1, iy+2, False, 0.0)
            fd, fm = hll_2(vL, vR, dL, dL*vL, dR, dR*vR)
            F[iy+1], M[iy+1] = fd, fm
        so = vely[N-1]
        F[N] = so*max(dens[N-1], 0.0) if so > 0.0 else 0.0
        M[N] = F[N]*vely[N-1]
        si = vely[0]
        F[0] = si*max(dens[0], 0.0) if si < 0.0 else 0.0
        M[0] = F[0]*vely[0]
        scale = np.ones(N)
        for i in range(N):
            leave = dt*(area[i+1]*max(F[i+1], 0.0) + area[i]*max(-F[i], 0.0))
            allow = (1.0 - 1.0e-12)*max(dens[i], 0.0)*vol[i]
            if leave > allow and leave > 0.0:
                scale[i] = allow/leave
        Fs, Ms = F.copy(), M.copy()
        Fs[0] *= scale[0]
        Ms[0] *= scale[0]
        for i in range(1, N):
            iup = i-1 if F[i] >= 0.0 else i
            Fs[i] *= scale[iup]
            Ms[i] *= scale[iup]
        Fs[N] *= scale[N-1]
        Ms[N] *= scale[N-1]
        dd = -dt*(area[1:]*Fs[1:] - area[:-1]*Fs[:-1])/vol
        dm = -dt*(area[1:]*Ms[1:] - area[:-1]*Ms[:-1])/vol
        return dd, dm

    d1, m1 = L(dens, momy)
    dens1, momy1 = dens + d1, momy + m1
    dens1[dens1 < 0] = 0.0
    momy1[dens1 < 0] = 0.0
    if order == 2:
        d2, m2 = L(dens1, momy1)
        return 0.5*(dens + dens1 + d2), 0.5*(momy + momy1 + m2)
    d2, m2 = L(dens1, momy1)
    dens2 = 0.75*dens + 0.25*(dens1 + d2)
    momy2 = 0.75*momy + 0.25*(momy1 + m2)
    dens2[dens2 < 0] = 0.0
    momy2[dens2 < 0] = 0.0
    d3, m3 = L(dens2, momy2)
    return dens/3.0 + 2.0*(dens2 + d3)/3.0, momy/3.0 + 2.0*(momy2 + m3)/3.0

def _y_transport_case(N, d, rk, cfl_num=0.5, a=0.2, tend=0.25):
    """Run one smooth homologous-expansion case for the RK comparison"""
    r = (Y_MAX/Y_MIN)**(1.0/N)
    yf = Y_MIN*r**np.arange(N+1)
    vol = (yf[1:]**d - yf[:-1]**d)/d
    area = yf**(d-1)
    w = ppm_nonuniform_weights(yf**d/d)
    dens = cell_average(compact_bump, yf, d)
    momy = cell_average(lambda y: compact_bump(y)*a*y, yf, d)
    mass0 = np.sum(dens*vol)
    t = 0.0
    while t < tend - 1e-14:
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        rate = np.max(np.abs(vely)*area[1:]/vol)
        dt = min(cfl_num/max(rate, 1e-300), tend - t)
        dens, momy = advect_y_rk(dens, momy, dt, yf, d, w, rk)
        t += dt
    lam = 1.0 + a*tend
    exact = cell_average(lambda y: lam**(-d)*compact_bump(y/lam), yf, d)
    L1 = np.sum(vol*np.abs(dens - exact))/np.sum(vol)
    merr = abs(np.sum(dens*vol) - mass0)/mass0
    return L1, merr

def test_ssprk3_fix():
    """Document why SSPRK3 replaced SSPRK2 at the production CFL"""

    errs2 = [_y_transport_case(n, 2, 2)[0] for n in (64, 128, 256)]
    orders2 = np.log(np.array(errs2[:-1])/np.array(errs2[1:]))/np.log(2.0)

    errs3, merrs = [], []
    for n in (64, 128, 256):
        L1, me = _y_transport_case(n, 2, 3)
        errs3.append(L1)
        merrs.append(me)
    orders3 = np.log(np.array(errs3[:-1])/np.array(errs3[1:]))/np.log(2.0)
    ok = (orders3[-1] > 2.3) and (max(merrs) < 1e-12)
    report("T8 SSPRK(3,3) restores clean ~2nd order at CFL=0.5 (SSPRK2 degrades)",
           ok,
           f"SSPRK2 orders={np.round(orders2,3)}, SSPRK3 orders={np.round(orders3,3)}, "
           f"SSPRK3 L1={np.array2string(np.array(errs3), precision=3)}, mass err={max(merrs):.1e}")


# ======================================================================================================================
# radial invariant-domain transport
# ======================================================================================================================

def hll_flux_vector(speed_l, speed_r, dens_l, primitive_l, dens_r, primitive_r):
    """Return density plus three momentum fluxes from one pressureless HLL solve

    Each primitive vector contains ``(lx, vy, lz)``.  HLL uses one density
    flux and the same wave speeds for every conserved momentum.
    """

    flux = np.empty(4)
    flux[0], flux[1] = hll_2(
        speed_l, speed_r, dens_l, dens_l*primitive_l[0], dens_r, dens_r*primitive_r[0]
    )
    _, flux[2] = hll_2(
        speed_l, speed_r, dens_l, dens_l*primitive_l[1], dens_r, dens_r*primitive_r[1]
    )
    _, flux[3] = hll_2(
        speed_l, speed_r, dens_l, dens_l*primitive_l[2], dens_r, dens_r*primitive_r[2]
    )
    return flux


def local_bounds(values, index):
    """Find padded extrema in one cell and its available radial neighbors"""

    index_min = max(index - 1, 0)
    index_max = min(index + 2, len(values))
    value_min = np.min(values[index_min:index_max])
    value_max = np.max(values[index_min:index_max])
    padding = 1.0e-12*max(1.0, abs(value_min), abs(value_max))
    return value_min - padding, value_max + padding


def invariant_scale(state, correction, bounds):
    """Limit one correction so density stays positive and velocities stay bounded

    ``state`` contains density and three momenta.  Velocity bounds can be
    written as linear inequalities involving density and the matching momentum,
    allowing one correction scale to preserve all constraints at once.
    """

    dens, momx, momy, momz = state
    ddens, dmomx, dmomy, dmomz = correction
    momenta = (momx, momy, momz)
    changes = (dmomx, dmomy, dmomz)
    scale = 1.0

    def restrict(available, change):
        nonlocal scale
        if change >= 0.0:
            return
        candidate = (1.0 - 1.0e-12)*available/(-change) if available > 0.0 else 0.0
        scale = min(scale, candidate)

    restrict(dens, ddens)
    for momentum, change, (value_min, value_max) in zip(momenta, changes, bounds):
        restrict(momentum - value_min*dens, change - value_min*ddens)
        restrict(value_max*dens - momentum, value_max*ddens - change)

    return min(max(scale, 0.0), 1.0)


def advect_y_invariant(dens, momx, momy, momz, dt, yface, dimension, weights):
    """Mirror the production radial PPM-HLL-FCT update with SSPRK(3,3)

    Array layout is ``state[quantity, cell]``.  Quantity zero is density and
    quantities one through three are conserved momenta.  Each Euler stage first
    advances a robust cell-centered HLL state, then restores as much of the
    high-order PPM flux as the invariant bounds allow.
    """

    cell_count = len(dens)
    volume = (yface[1:]**dimension - yface[:-1]**dimension)/dimension
    area = yface**(dimension - 1.0)
    ycent = np.sqrt(yface[:-1]*yface[1:])
    state_initial = np.stack((dens, momx, momy, momz))

    def euler_step(state):
        # Work on a copy because near-vacuum recovery also repairs momenta
        stage = state.copy()
        dens_s, momx_s, momy_s, momz_s = stage

        active = dens_s >= RHO_VAC
        safe_dens = np.maximum(dens_s, 1.0e-300)
        lx = np.where(active, momx_s/safe_dens, np.sqrt(ycent))
        vy = np.where(active, momy_s/safe_dens, 0.0)
        lz = np.where(active, momz_s/safe_dens, 0.0)
        momx_s[~active] = dens_s[~active]*lx[~active]
        momy_s[~active] = 0.0
        momz_s[~active] = 0.0
        primitive = np.stack((lx, vy, lz))

        # Reconstruct density and all three primitives at radial faces
        face_values = [ppm_edges_nonuniform(values, weights) for values in (dens_s, lx, vy, lz)]
        flux_high = np.zeros((4, cell_count + 1))
        flux_low = np.zeros((4, cell_count + 1))

        for face in range(1, cell_count):
            left = face - 1
            right = face
            reconstructed = []
            for quantity, (faces, cells) in enumerate(zip(face_values, (dens_s, lx, vy, lz))):
                value_l = ppm_face_value(faces, cells, left, face, True, 0.0)
                value_r = ppm_face_value(faces, cells, right, face + 1, False, 0.0)
                if quantity == 0:
                    value_l = max(value_l, 0.0)
                    value_r = max(value_r, 0.0)
                reconstructed.append((value_l, value_r))

            dens_l, dens_r = reconstructed[0]
            primitive_l = np.array([pair[0] for pair in reconstructed[1:]])
            primitive_r = np.array([pair[1] for pair in reconstructed[1:]])
            flux_high[:, face] = hll_flux_vector(
                primitive_l[1], primitive_r[1], dens_l, primitive_l, dens_r, primitive_r
            )
            flux_low[:, face] = hll_flux_vector(
                vy[left], vy[right], dens_s[left], primitive[:, left], dens_s[right], primitive[:, right]
            )

        # Permit outflow at either radial boundary but prohibit inflow
        if vy[0] < 0.0:
            mass_flux = vy[0]*max(dens_s[0], 0.0)
            boundary_flux = np.concatenate(([mass_flux], mass_flux*primitive[:, 0]))
            flux_low[:, 0] = flux_high[:, 0] = boundary_flux
        if vy[-1] > 0.0:
            mass_flux = vy[-1]*max(dens_s[-1], 0.0)
            boundary_flux = np.concatenate(([mass_flux], mass_flux*primitive[:, -1]))
            flux_low[:, -1] = flux_high[:, -1] = boundary_flux

        # The cell-centered HLL update supplies the robust low-order state
        updated = stage - dt*(area[None, 1:]*flux_low[:, 1:]
                              - area[None, :-1]*flux_low[:, :-1])/volume[None, :]
        updated[:, updated[0] < 0.0] = 0.0

        # Restore the PPM-minus-HLL correction face by face using one shared scale
        antidiffusive = flux_high - flux_low
        for face in range(1, cell_count):
            left = face - 1
            right = face
            correction_l = -dt*area[face]*antidiffusive[:, face]/volume[left]
            correction_r = dt*area[face]*antidiffusive[:, face]/volume[right]
            bounds_l = [local_bounds(values, left) for values in primitive]
            bounds_r = [local_bounds(values, right) for values in primitive]
            scale = min(
                invariant_scale(updated[:, left], correction_l, bounds_l),
                invariant_scale(updated[:, right], correction_r, bounds_r),
            )
            updated[:, left] += scale*correction_l
            updated[:, right] += scale*correction_r

        return updated

    # SSPRK(3,3): two Euler evaluations, a convex combination, then a final Euler evaluation
    stage = euler_step(state_initial)
    stage = euler_step(stage)
    stage = 0.75*state_initial + 0.25*stage
    stage = euler_step(stage)
    result = state_initial/3.0 + 2.0*stage/3.0
    return tuple(result)


def test_invariant_limiter():
    """Check that the current limiter avoids low-density velocity/CFL defects"""

    def run(N, d=2, tend=0.25, a=0.2):
        r = (Y_MAX/Y_MIN)**(1.0/N)
        yf = Y_MIN*r**np.arange(N+1)
        vol = (yf[1:]**d - yf[:-1]**d)/d
        area = yf**(d-1)
        w = ppm_nonuniform_weights(yf**d/d)
        dens = cell_average(compact_bump, yf, d)
        momy = cell_average(lambda y: compact_bump(y)*a*y, yf, d)
        momx = 0.7*dens
        momz = 0.11*dens
        mass0 = np.sum(dens*vol)
        t = 0.0
        steps = 0
        while t < tend - 1e-14:
            vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
            rate = np.max(np.abs(vely)*area[1:]/vol)
            dt = min(0.5/max(rate, 1e-300), tend - t)
            dens, momx, momy, momz = advect_y_invariant(dens, momx, momy, momz, dt, yf, d, w)
            t += dt
            steps += 1
        vely = np.where(dens >= RHO_VAC, momy/np.maximum(dens, 1e-300), 0.0)
        lam = 1.0 + a*tend
        exact = cell_average(lambda y: lam**(-d)*compact_bump(y/lam), yf, d)
        L1 = np.sum(vol*np.abs(dens - exact))/np.sum(vol)
        merr = abs(np.sum(dens*vol) - mass0)/mass0
        vmax = np.max(np.abs(vely[dens > 1e-12])) if np.any(dens > 1e-12) else 0.0
        return L1, steps, merr, vmax

    L1_64, st64, me64, vm64 = run(64)
    L1_128, st128, me128, vm128 = run(128)
    L1_256, st256, me256, vm256 = run(256)
    orders = np.log(np.array([L1_64, L1_128, L1_256])[:-1]/np.array([L1_64, L1_128, L1_256])[1:])/np.log(2.0)
    ok = (st128 <= 16 and st256 <= 24
          and vm128 < 1.0 and vm256 < 1.0
          and max(me64, me128, me256) < 1e-12
          and orders[-1] > 2.0)
    report("T9 invariant-domain flux limiter fixes the low-density CFL defect",
           ok,
           f"steps=({st64},{st128},{st256}), max|vely|=({vm64:.2f},{vm128:.2f},{vm256:.2f}), "
           f"orders={np.round(orders,3)}, mass err={max(me64,me128,me256):.1e}")


# =============================================================================
# exact mesh primitives
# =============================================================================

def test_mesh_primitives():
    """Check exact logarithmic-radial and polar grid measures"""
    dy = (Y_MAX/Y_MIN)**(1.0/64.0)
    yface = lambda iy: Y_MIN*dy**iy
    ycent = lambda iy: Y_MIN*dy**(iy + 0.5)
    dy_len = lambda iy: yface(iy)*(dy - 1.0)
    for d in (2, 3):
        vol_y = lambda iy: yface(iy)**d*(dy**d - 1.0)/d
        # measure telescopes to the domain measure; faces/logarithmic center are ordered
        ok = abs(sum(vol_y(i) for i in range(64)) - (Y_MAX**d - Y_MIN**d)/d) < 1e-10
        ok = ok and all(yface(i) < ycent(i) < yface(i+1) for i in range(64))
        # cell width and center-to-center distances are exact
        ok = ok and all(abs(dy_len(i) - (yface(i+1) - yface(i))) < 1e-15 for i in range(64))
        ok = ok and all(abs(ycent(i)*(dy - 1.0)/dy - (ycent(i) - ycent(i-1))) < 1e-15 for i in range(1, 64))
        ok = ok and all(abs(ycent(i)*(dy - 1.0) - (ycent(i+1) - ycent(i))) < 1e-15 for i in range(63))
        report(f"T10 d={d} exact radial mesh primitives", ok, "telescoping, ordering, exact widths/distances")

    # compare exact cell width with the local logarithmic differential approximation y*log(dy)
    for n, tol in ((32, 0.031), (256, 0.004)):
        r = (Y_MAX/Y_MIN)**(1.0/n)
        yf = Y_MIN*r**np.arange(n+1)
        yy = np.sqrt(yf[:-1]*yf[1:])
        exact = yf[1:] - yf[:-1]
        former = yy*np.log(r)
        rel = np.abs(exact/former - 1.0)
        report(f"T10 dr exact; deviation from y*log(dy) below {tol:.1%} at N_Y={n}",
               rel.max() < tol, f"max relative deviation={rel.max():.3e}")

    # polar measure telescopes to the hemisphere
    Nz = 32
    dz = (np.pi/2.0)/Nz
    vol_z_sum = sum(np.cos(i*dz) - np.cos((i+1)*dz) for i in range(Nz))
    report("T10 polar measure telescopes to the hemisphere", abs(vol_z_sum - 1.0) < 1e-15,
           f"sum={vol_z_sum:.16f}")

# =============================================================================

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("=== GameDev fluid algorithm checks ===")
    test_x_diffusion()
    test_y_diffusion()
    test_source()
    test_ppm_weights()
    test_x_advection()
    test_optdepth()
    test_y_advection()
    test_ssprk3_fix()
    test_invariant_limiter()
    test_mesh_primitives()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    if n_pass != len(RESULTS):
        raise SystemExit(1)
