#!/usr/bin/env python3

"""Validate resolved-vertical monodisperse production particle initialization"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


X_MIN, X_MAX = -math.pi, math.pi
ASPR_0 = 0.05
IDX_P = 2.0
IDX_Q = -1.0
STOKES_0 = 0.2
ALPHA = 1.0e-4


def norms(error: np.ndarray) -> dict[str, float]:
    """Return the standard deterministic particle-state norms"""

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error*error))),
        "linf": float(np.max(np.abs(error))),
    }


def analyze(out_dir: Path, resolution: int) -> dict:
    """Reconstruct gas rotation, viscous inflow, radial drift, settling, and spherical projection"""

    meta = json.loads((out_dir/f"meta_N{resolution}.json").read_text())
    nparticle = int(meta["np"])
    state = np.fromfile(out_dir/f"state_N{resolution}.dat", dtype=np.float64).reshape(6, nparticle)
    fraction = np.arange(nparticle)/(nparticle - 1.0)
    radius = 0.7 + 0.6*fraction
    height = 0.03*radius*(np.arange(nparticle) % 5 - 2.0)
    y = np.sqrt(radius*radius + height*height)
    z = np.arctan2(radius, height)
    x = X_MIN + (np.arange(nparticle) + 0.25)*(X_MAX - X_MIN)/nparticle

    h_g = ASPR_0*radius**(0.5*(IDX_Q + 1.0))
    omega = radius**-1.5
    velocity_k = radius*omega
    cylindrical_fraction = radius/y
    stratification = np.exp((cylindrical_fraction - 1.0)/(h_g*h_g))
    stokes = STOKES_0/radius**IDX_P/stratification
    eta = -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g
                + IDX_Q*(1.0 - cylindrical_fraction))
    velocity_gas_x = velocity_k*np.sqrt(np.maximum(1.0 - 2.0*eta, 0.0))

    nu = ALPHA*h_g*h_g*radius*radius*omega
    strat_log = (cylindrical_fraction - 1.0)/(h_g*h_g)
    grad_strat_r = cylindrical_fraction*(1.0 - cylindrical_fraction**2)/(h_g*h_g) \
        - (IDX_Q + 1.0)*strat_log
    grad_strat_z = -cylindrical_fraction*(1.0 - cylindrical_fraction**2)/(h_g*h_g)
    grad_rhog_r = IDX_P - 0.5*(IDX_Q + 3.0) + grad_strat_r
    grad_rhog_z = grad_strat_z
    term_r = 3.0*nu*((IDX_Q + 1.5) + grad_rhog_r + 0.5)
    term_z = IDX_Q*nu*(1.0 + grad_rhog_z)
    velocity_gas_r = -(term_r - term_z)/radius

    velocity_r = (
        velocity_gas_r + 2.0*stokes*(velocity_gas_x - velocity_k)
    )/(1.0 + stokes*stokes)
    velocity_x = velocity_gas_x - 0.5*stokes*velocity_r
    velocity_z = -stokes*omega*height
    velocity_y = velocity_r*np.sin(z) + velocity_z*np.cos(z)
    velocity_polar = velocity_r*np.cos(z) - velocity_z*np.sin(z)
    expected = np.stack((x, y, z, radius*velocity_x, velocity_y, y*velocity_polar))
    error = state - expected
    finite = bool(np.all(np.isfinite(state)))
    return {
        "case": meta["case"],
        "resolution": resolution,
        "reference_class": "production-initialization",
        "finite": finite,
        "errors": {"state": norms(error)},
        "passed": bool(finite and np.max(np.abs(error)) < 5.0e-12),
    }
