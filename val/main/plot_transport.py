#!/usr/bin/env python3
"""Plot downloaded orbit and ring-pilot outputs; requires NumPy and Matplotlib."""
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

def orbit():
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
        folder=ROOT/f'swarm_orbit/outputs/orbit_dt_p{n}'
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
    folder=ROOT/'swarm_orbit/outputs/plots'
    save(fig,folder,'orbital_accuracy')
    (folder/'metrics.json').write_text(json.dumps({'records':records,'energy_orders':orders.tolist(),
      'reference_time':'Full-precision model schedule i*2*pi/20; variables.txt rounds DT_OUT.'},indent=2)+'\n')
    return records,orders.tolist()

def diffusion():
    path=ROOT/'swarm_diffusion/analyze.py'
    spec=importlib.util.spec_from_file_location('diffusion_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=ROOT/'swarm_diffusion/outputs/ring_pilot'
    result=json.loads(module.analyze(folder).read_text()); records=result['records']
    edges=np.array(result['radial_bin_edges']); centers=np.sqrt(edges[:-1]*edges[1:])
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    r=np.geomspace(.12,6,1000)
    for color,i in zip(COLORS,(0,1,5,10)):
        rec=records[i]; density=np.array(rec['surface_density_per_mass'])
        ax[0].plot(r,module.reference_density(r,rec['time']),color=color,label=f"At = {rec['A_t']:g}")
        use=(centers>.12)&(centers<6)&(density>1e-5)
        ax[0].plot(centers[use],density[use],'o',ms=3,mfc='none',color=color)
    ax[0].set(xscale='log',xlim=(.12,6),xlabel=r'$R/R_c$',ylabel=r'$\Sigma_d/M_d$',title='Ring spreading')
    ax[0].set_xticks([.2,.5,1,2,5],['0.2','0.5','1','2','5'])
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend(fontsize=8,loc='upper right',title='Lines: exact\nCircles: particle bin averages',title_fontsize=8)
    times=np.array([x['A_t'] for x in records]);var=np.array([x['exact_log_variance'] for x in records]);n=result['particle_count']
    mean_z=np.array([x['log_mean']-x['exact_log_mean'] for x in records])/np.sqrt(var/n)
    var_z=np.array([x['log_variance']-x['exact_log_variance'] for x in records])/(var*np.sqrt(2/n))
    ax[1].axhspan(-2,2,color='0.93',label=r'$\pm2$ sampling standard errors')
    ax[1].axhline(0,color='0.6',lw=.8)
    ax[1].plot(times,mean_z,'o-',color=COLORS[0],label=r'Mean of $\ln R$')
    ax[1].plot(times,var_z,'s-',color=COLORS[1],label=r'Variance of $\ln R$')
    ax[1].set(xlabel=r'$At$',ylabel='Residual / sampling standard error',title='Analytical moment comparison')
    ax[1].legend(fontsize=8)
    save(fig,folder/'plots','diffusion_accuracy')
    return {k:v for k,v in records[-1].items() if not isinstance(v,list)}

if __name__=='__main__':
    orbits,orders=orbit(); diff=diffusion()
    print(json.dumps({'orbit':orbits,'energy_orders':orders,'diffusion_final':diff},indent=2))
