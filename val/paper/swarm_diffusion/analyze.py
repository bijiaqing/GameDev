#!/usr/bin/env python3
"""Compare radial density or concentration diffusion with its exact lognormal solution."""

import sys
sys.dont_write_bytecode = True
import argparse
import configparser
import json
import math
from pathlib import Path
import numpy as np

A = 1e-4
S0 = 0.05

def moments(t, stokes=0.0, mode="density", gas_power=-1.0):
    a = A/(1.0+stokes*stokes)
    drift = 2.0 + (gas_power if mode == "concentration" else 0.0)
    return drift*a*t, S0*S0+2*a*t

def reference_cdf(r, t, stokes=0.0, mode="density", gas_power=-1.0):
    mu, var = moments(t, stokes, mode, gas_power)
    z = (np.log(r)-mu)/math.sqrt(2*var)
    return np.fromiter((0.5*math.erfc(-float(x)) for x in z.flat),float,count=z.size).reshape(z.shape)

def reference_density(r, t, stokes=0.0, mode="density", gas_power=-1.0):
    """Surface density divided by total represented mass (R_c=1)."""
    mu, var = moments(t, stokes, mode, gas_power)
    return np.exp(-(np.log(r)-mu)**2/(2*var))/(2*math.pi*r*r*math.sqrt(2*math.pi*var))

def analyze(output, mode="density", legacy=False):
    if mode not in ("density", "concentration"): raise ValueError(mode)
    config=configparser.ConfigParser()
    with (output/'variables.txt').open() as f: config.read_file(f)
    values={k:config['PARAMETERS'].getfloat(k) for k in (
        'n_p','idx_p','idx_q','alpha','aspr_0','schmidt_r','save_max',
        'stokes_0','dt_out','dt_max','y_min','y_max')}
    expected={'n_p':1048576,'idx_q':0.5,'alpha':0.04,'aspr_0':0.05,'schmidt_r':1,'save_max':10}
    for key,value in expected.items():
        if not math.isclose(values[key],value,rel_tol=1e-7): raise ValueError(f'Not a paper diffusion model: {key}={values[key]}')
    stokes=values['stokes_0']; gas_power=values['idx_p']
    if legacy:
        if mode != 'density' or gas_power != 1.0 or stokes != 1.0:
            raise ValueError('Legacy reference is only for the original density ring pilot')
        stokes=0.0  # Original kernel used gas diffusivity with no St suppression.
    a=A/(1+stokes*stokes)
    if not math.isclose(values['dt_out'],100*(1+stokes*stokes),rel_tol=1e-7):
        raise ValueError('Unexpected output interval')
    if not math.isclose(values['dt_max'],1+stokes*stokes,rel_tol=1e-7):
        raise ValueError('Unexpected fixed timestep')
    n=int(values['n_p']); edges=np.geomspace(math.exp(-3),math.exp(3),121)
    records=[]
    for frame in range(11):
        path=output/f'particle_{frame:05d}.dat'
        data=np.fromfile(path,dtype=np.float64)
        if data.size!=6*n: raise ValueError(f'{path}: expected {6*n} doubles, got {data.size}')
        state=data.reshape(n,6); radius=state[:,1]
        if not np.all(np.isfinite(state)) or np.any(radius<=0): raise ValueError(f'{path}: nonfinite or inactive particle')
        if np.any(radius<values['y_min']) or np.any(radius>values['y_max']): raise ValueError(f'{path}: particle outside domain')
        t=frame*values['dt_out']; mu,var=moments(t,stokes,mode,gas_power); u=np.log(radius)
        cdf=reference_cdf(np.sort(radius),t,stokes,mode,gas_power)
        ks=max(float(np.max(np.arange(1,n+1)/n-cdf)),float(np.max(cdf-np.arange(n)/n)))
        counts,_=np.histogram(radius,bins=edges); area=math.pi*np.diff(edges**2)
        records.append({'frame':frame,'time':t,'A_t':a*t,'cdf_sup_error':ks,
            'log_mean':float(u.mean()),'exact_log_mean':mu,'log_variance':float(u.var()),'exact_log_variance':var,
            'log_mean_standard_error':math.sqrt(var/n),'max_abs_velocity':float(np.abs(state[:,3:]).max()),
            'minimum_radius':float(radius.min()),'maximum_radius':float(radius.max()),
            'surface_density_per_mass':(counts/n/area).tolist(),
            'exact_bin_average_per_mass':(np.diff(reference_cdf(edges,t,stokes,mode,gas_power))/area).tolist()})
    result={'legacy_gas_diffusivity':legacy,'mode':mode,'stokes':stokes,'gas_power':gas_power,'diffusivity_prefactor':a,
        'particle_count':n,'reference':'unbounded radial diffusion D=A R^2/(1+St^2)',
        'radial_bin_edges':edges.tolist(),'records':records,
        'limitations':'No pass threshold; finite boundaries reflect. Saved states do not reveal between-output boundary encounters.'}
    target=output/'analysis.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=('density','concentration'),required=True,
                        help='Must match the model DIFFUSE_CONCENTRATION flag')
    parser.add_argument('--legacy',action='store_true',help='Original ring_pilot output before St-dependent diffusivity')
    args=parser.parse_args()
    print(analyze(args.output,args.mode,args.legacy))
