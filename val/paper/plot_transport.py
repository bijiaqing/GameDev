#!/usr/bin/env python3
"""Plot current orbit and density/concentration diffusion outputs; requires NumPy and Matplotlib."""

import sys
sys.dont_write_bytecode = True
import argparse
import configparser
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':200,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']

def save(fig, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def orbit(data_root, destination, backend):
    t=np.arange(2001)*(2*np.pi/20)
    M=(t+np.pi)%(2*np.pi)-np.pi
    E=M.copy()
    for _ in range(12): E-=(E-0.5*np.sin(E)-M)/(1-0.5*np.cos(E))
    assert np.max(np.abs(E-0.5*np.sin(E)-M)) < 1e-13
    exact_xy=np.column_stack((np.cos(E)-0.5,np.sqrt(0.75)*np.sin(E)))
    exact_phi=np.unwrap(np.arctan2(exact_xy[:,1],exact_xy[:,0]))
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    records=[]
    for color,n in zip(COLORS,(80,160,320,640)):
        folder=data_root/f'ecc_orbit/out/orbit_dt_p{n}'
        if backend: folder/=backend
        cfg=configparser.ConfigParser();cfg.read(folder/'variables.txt');p=cfg['PARAMETERS']
        assert p.getint('n_p')==1 and p.getint('save_max')==2000
        assert np.isclose(p.getfloat('dt_out'),2*np.pi/20,rtol=1e-8)
        assert np.isclose(p.getfloat('dt_max'),2*np.pi/n,rtol=1e-8)
        data=[]
        for i in range(2001):
            v=np.fromfile(folder/f'particle_{i:05d}.dat',dtype=np.float64)
            assert v.shape==(6,) and np.isfinite(v).all()
            data.append(v)
        data=np.array(data); r=data[:,1]; phi=data[:,0]
        assert np.all((r>p.getfloat('y_min'))&(r<p.getfloat('y_max')))
        assert np.allclose(data[0],[0,.5,np.pi/2,np.sqrt(3),0,0],atol=1e-14)
        energy=0.5*np.sum(data[:,3:]**2,axis=1)-1/r
        eerr=np.abs((energy+.5)/.5)
        xy=r[:,None]*np.column_stack((np.cos(phi),np.sin(phi)))
        poserr=np.linalg.norm(xy-exact_xy,axis=1)
        phase=np.unwrap(phi)-exact_phi
        record={'steps_per_period':n,'max_relative_energy_error':float(eerr.max()),
          'rms_position_error_over_100P':float(np.sqrt(np.mean(poserr**2))),
          'final_position_error':float(poserr[-1]),'final_phase_error_rad':float(phase[-1]),
          'max_angular_momentum_error':float(np.max(np.abs(r*data[:,3]-np.sqrt(.75))))}
        records.append(record)
        ax[0].plot(t/(2*np.pi),np.maximum.accumulate(eerr),color=color,label=rf'$P/{n}$')
        ax[1].plot(t/(2*np.pi),phase,color=color)
    ax[0].set(yscale='log',ylim=(3e-5,1e-2),xlabel='Time / orbital period',ylabel='Maximum relative energy error so far',title='Energy conservation')
    ax[0].legend(title=r'$\Delta t$',fontsize=9)
    ax[1].set(xlabel='Time / orbital period',ylabel='Phase error [rad]',title='Accumulated phase error')
    step=1/np.array([x['steps_per_period'] for x in records])
    errors=np.array([x['max_relative_energy_error'] for x in records])
    ax[2].loglog(step,errors,'o-',color=COLORS[0],label='Maximum relative energy error')
    ax[2].loglog(step,errors[-1]*(step/step[-1])**2,'--',color='0.45',label=r'$\propto\Delta t^2$')
    ax[2].set(xlabel=r'$\Delta t/P$',ylabel='Error over 100 periods',title='Temporal convergence')
    ax[2].set_xticks(step, ['1/80','1/160','1/320','1/640'])
    ax[2].xaxis.set_minor_formatter(NullFormatter())
    ax[2].legend(fontsize=8)
    orders=np.log(errors[:-1]/errors[1:])/np.log(2)
    folder=destination/'ecc_orbit'
    if backend: folder/=backend
    save(fig,folder,'orbital_accuracy')
    (folder/'metrics.json').write_text(json.dumps({'records':records,'energy_orders':orders.tolist(),
      'reference_time':'Full-precision model schedule i*2*pi/20; variables.txt rounds DT_OUT.'},indent=2)+'\n')
    return records,orders.tolist()

def diffusion(data_root, destination):
    spec=importlib.util.spec_from_file_location('diffusion_analysis',ROOT/'diffusion/analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results={}
    folder=destination/'diffusion';folder.mkdir(parents=True,exist_ok=True)
    for mode in ('density','concentration'):
        for st in ('0','0p1','1','10'):
            model=f'{mode}_st{st}'
            for backend in ('cuda','rocm'):
                source=data_root/'diffusion/out'/model/backend
                target=folder/f'{model}_{backend}.json'
                result=json.loads(module.analyze(source,mode,target).read_text())
                results[(mode,st,backend)]=result
                print(f'Analyzed {model}/{backend}',flush=True)
    # St=1 represents the common diffusion age; every St is checked in the moment panels below.
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for axis,mode in zip(ax,('density','concentration')):
        for color,i in zip(COLORS,(0,1,5,10)):
            ref=results[(mode,'1','cuda')];rec=ref['records'][i]
            edges=np.array(ref['radial_bin_edges']);r=np.sqrt(edges[:-1]*edges[1:])
            expected=np.array(rec['exact_bin_average_per_mass'])
            use=(r>.12)&(r<6)
            axis.plot(r[use],expected[use],color=color,label=rf'$at={rec["A_t"]:g}$')
            for backend,marker in [('cuda','o'),('rocm','x')]:
                measured=np.array(results[(mode,'1',backend)]['records'][i]['surface_density_per_mass'])
                axis.plot(r[use],measured[use],marker,ms=3,mfc='none',color=color,alpha=.75)
        axis.set(xscale='log',xlim=(.12,6),xlabel=r'$R/R_0$',ylabel=r'$\Sigma_d/M_d$',title=f'{mode.capitalize()} diffusion, St=1')
        axis.set_xticks([.2,.5,1,2,5],['0.2','0.5','1','2','5'])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.legend(fontsize=8,title='Lines: exact bin averages\nCircles: CUDA; crosses: ROCm',title_fontsize=8)
    save(fig,folder,'ring_spreading')
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    summary=[]
    for row,mode in enumerate(('density','concentration')):
        for color,st in zip(COLORS,('0','0p1','1','10')):
            for backend,style in [('cuda','-'),('rocm','--')]:
                result=results[(mode,st,backend)];records=result['records'];n=result['particle_count']
                t=np.array([x['A_t'] for x in records]);var=np.array([x['exact_log_variance'] for x in records])
                mz=np.array([x['log_mean']-x['exact_log_mean'] for x in records])/np.sqrt(var/n)
                vz=np.array([x['log_variance']-x['exact_log_variance'] for x in records])/(var*np.sqrt(2/n))
                for col,z in enumerate((mz,vz)):
                    ax[row,col].plot(t,z,style,color=color,label=f'St={st.replace("p",".")} {backend.upper()}',lw=1.3)
                summary.append({'mode':mode,'stokes':result['stokes'],'backend':backend,
                    'max_abs_mean_standard_errors':float(np.max(np.abs(mz))),
                    'max_abs_variance_standard_errors':float(np.max(np.abs(vz))),
                    'final_cdf_error':records[-1]['cdf_sup_error'],
                    'max_cdf_error':max(r['cdf_sup_error'] for r in records),
                    'final_mean':records[-1]['log_mean'],'final_variance':records[-1]['log_variance'],
                    'max_abs_velocity':max(r['max_abs_velocity'] for r in records)})
        for col,stat in enumerate(('Mean','Variance')):
            ax[row,col].axhspan(-2,2,color='0.93',zorder=0)
            ax[row,col].axhline(0,color='.6',lw=.5)
            ax[row,col].set(xlabel=r'Diffusion age $at$',ylabel='Residual / sampling standard error',title=f'{mode.capitalize()}: {stat} of log radius')
        ax[row,1].legend(fontsize=7,ncol=2)
    save(fig,folder,'diffusion_moments')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True,help='Downloaded paper directory containing ecc_orbit and diffusion')
    parser.add_argument('--output',type=Path,default=ROOT/'out/transport_plots')
    args=parser.parse_args()
    summary={'orbit':{}}
    for backend in ('cuda','rocm'):
        records,orders=orbit(args.data_root,args.output,backend)
        summary['orbit'][backend]={'records':records,'energy_orders':orders}
    summary['diffusion']=diffusion(args.data_root,args.output)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
