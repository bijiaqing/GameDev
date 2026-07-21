#!/usr/bin/env python3


import numpy as np
from review_mock_tests import (ppm_nonuniform_weights, ppm_edges_nonuniform,
                               Y_MIN, Y_MAX)

def cellavg_s(coefs, fs):
    return sum(coefs[n]*(fs[1:]**(n+1) - fs[:-1]**(n+1))/((n+1)*(fs[1:]-fs[:-1]))
               for n in range(len(coefs)))


worst = 0.0
for geom in ("log2", "log3", "cos"):
    N = 33
    if geom == "cos":
        fs = -np.cos(np.linspace(0.15, 1.35, N+1))
    else:
        d = 2 if geom == "log2" else 3
        r = (Y_MAX/Y_MIN)**(1.0/N)
        fs = (Y_MIN*r**np.arange(N+1))**d/d
    w = ppm_nonuniform_weights(fs)
    coefs = np.array([1.0, 0.3, 0.2, 0.05])
    cellavg = cellavg_s(coefs, fs)
    edge = ppm_edges_nonuniform(cellavg, w)
    exact = sum(coefs[n]*fs**n for n in range(4))
    worst = max(worst, np.max(np.abs(edge - exact)))
print(f"monotone cubic exactness (all faces): max err = {worst:.3e}")


def face_errors(qfun, n, label):
    d = 2
    r = (Y_MAX/Y_MIN)**(1.0/n)
    yf = Y_MIN*r**np.arange(n+1)
    fs = yf**d/d
    w = ppm_nonuniform_weights(fs)
    cellavg = np.empty(n)
    for i in range(n):
        yy = np.linspace(yf[i], yf[i+1], 2000)
        cellavg[i] = np.trapezoid(qfun(yy)*yy**(d-1), yy)/(fs[i+1]-fs[i])
    edge = ppm_edges_nonuniform(cellavg, w)
    err = np.abs(edge - qfun(yf))
    i_max = np.argmax(err[2:-2]) + 2
    print(f"  {label:24s} N={n:4d}  max interior err={err[2:-2].max():.3e} at face y={yf[i_max]:.4f}")
    return err[2:-2].max()

print("face reconstruction errors:")
for label, fun in (("monotone exp(-y)", lambda y: np.exp(-y)),
                   ("gaussian sig=0.24", lambda y: np.exp(-((y-1.6)/0.24)**2)),
                   ("gaussian sig=0.12", lambda y: np.exp(-((y-1.6)/0.12)**2))):
        errs = [face_errors(fun, n, label) for n in (64, 128, 256, 512)]
        orders = np.log(np.array(errs[:-1])/np.array(errs[1:]))/np.log(2.0)
        print(f"    -> {label}: errs={np.array2string(np.array(errs), precision=3)} orders={np.round(orders,3)}")
