#!/usr/bin/env python3
"""Eriksson Fig.14-style maps from representative-particle masses (grain radius)."""

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse,json,hashlib,os
p=argparse.ArgumentParser();p.add_argument('--outputs',type=Path,default=Path.home()/'Scratch/gamedev/campaign_diskevol');p.add_argument('--frame',type=int,default=110);args=p.parse_args()
here=Path(__file__).resolve().parent;out=here/'results'/f'figure14_output{args.frame}';out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR','/tmp/gamedev-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/gamedev-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator,LogFormatterMathtext,NullFormatter
AU=1.495978707e13;YEAR=31557600.
reference=here/'results/figure14_output110/reference_grains.sizes'
centers=np.loadtxt(reference)[:,1];ratio=(centers[-1]/centers[0])**(1/(len(centers)-1))
# The published size values are rounded; use the underlying logarithmic grid.
acenters=np.geomspace(centers[0],centers[-1],len(centers));ae=np.r_[acenters/np.sqrt(ratio),acenters[-1]*np.sqrt(ratio)]
re=np.geomspace(5.,50.,513);te=np.linspace(0,.2,129);rc=np.sqrt(re[1:]*re[:-1]);tc=(te[1:]+te[:-1])/2
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':11,'mathtext.fontset':'dejavuserif'})
fig=plt.figure(figsize=(22,4.4));gs=fig.add_gridspec(1,6,left=.042,right=.99,bottom=.30,top=.79,wspace=.28)
axes=[fig.add_subplot(gs[0,i]) for i in range(6)];cmap=plt.get_cmap('inferno').copy();cmap.set_bad('black');cmap.set_under('black')
report={'reference':'https://arxiv.org/pdf/2603.22550v2','frame':args.frame,'size_grid_source':'https://github.com/astrolinn/dustComparison/blob/main/2Dsims/cuDisc/alpha1.e-3/grains.sizes','method':'Mass per size bin; spherical radial bins. Radial maps integrate both hemispheres. Vertical maps average both hemispheres and integrate rho dr, using per-particle M/r^2 divided by 4*pi*delta(sin(theta)). No smoothing.','normalization_limits':{'Sigma':[1e-4,1e-1],'Sigma_r':[1e-5,1.]},'reference_plotting_script_available':False,'models':{}}
npz={}
for mi,search in enumerate(['kdtree','morton']):
 directory=args.outputs/'strong_turbulence/rocm'/search;file=directory/f'particle_{args.frame:05}.dat'
 pars={}
 for line in (directory/'variables.txt').read_text().splitlines():
  if '=' in line:
   k,v=line.split('=',1)
   try:pars[k.strip()]=float(v.strip())
   except ValueError:pass
 assert pars['ALPHA']==1e-3 and pars['N_P']==1048576
 x=np.fromfile(file,dtype='<f8').reshape(-1,8);assert len(x)==int(pars['N_P']) and np.isfinite(x).all()
 active=x[:,1]>0;x=x[active];r=x[:,1]/AU;theta=np.abs(np.pi/2-x[:,2]);a=x[:,6]/2;mass=x[:,7]*np.pi/6*pars['RHO_0']*x[:,6]**3
 assert (mass>0).all() and (a>0).all()
 select=(r>=5)&(r<=50)&(theta<=.2);r,theta,a,mass=[v[select] for v in (r,theta,a,mass)]
 h=np.histogram2d(r,a,bins=[re,ae],weights=mass)[0]
 area=np.pi*np.diff((re*AU)**2);sigma=h/area[:,None]
 assert np.isclose(h.sum(),mass.sum(),rtol=1e-12)
 def avg(coord,edges,w):
  den=np.histogram(coord,bins=edges,weights=w)[0];num=np.histogram(coord,bins=edges,weights=w*a)[0]
  return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 # Averages use the same integration measure as the respective density maps.
 mean_r=avg(r,re,mass)
 maps=[sigma];means=[mean_r]
 for mask in [r<15,r>25]:
  w=mass/(r*AU)**2*mask
  h=np.histogram2d(theta,a,bins=[te,ae],weights=w)[0]
  assert np.isclose(h.sum(),w.sum(),rtol=1e-12)
  maps.append(h/(4*np.pi*np.diff(np.sin(te)))[:,None]);means.append(avg(theta,te,w))
 axs=axes[mi*3:mi*3+3]
 im0=axs[0].pcolormesh(re,ae,np.ma.masked_less_equal(maps[0].T,0),norm=LogNorm(1e-4,1e-1),cmap=cmap,rasterized=True)
 axs[0].plot(rc,means[0],color='.8',lw=1.25);axs[0].set(xscale='log',yscale='log',xlim=(5,50),ylim=(5e-5,50),xlabel='Distance to star [au]',ylabel='Grain radius [cm]',title='Column density')
 axs[0].set_xticks([5,10,25,50]);axs[0].set_xticklabels(['5','10','25','50']);axs[0].xaxis.set_minor_formatter(NullFormatter())
 for boundary in [15,25]:axs[0].axvline(boundary,color='.8',lw=.8)
 for j,title in enumerate(['Inner disc: r < 15 au','Outer disc: r > 25 au'],1):
  im1=axs[j].pcolormesh(ae,te,np.ma.masked_less_equal(maps[j],0),norm=LogNorm(1e-5,1),cmap=cmap,rasterized=True)
  axs[j].plot(means[j],tc,color='.8',lw=1.25)
  axs[j].set(xscale='log',xlim=(5e-5,50),ylim=(0,.125),xlabel='Grain radius [cm]',ylabel=r'$\vartheta$ [rad]' if j==1 else '',title=title)
  axs[j].set_xticks([1e-3,1e-1,1e1]);axs[j].set_yticks([0,.05,.10]);axs[j].xaxis.set_major_formatter(LogFormatterMathtext())
  if j==2:axs[j].tick_params(labelleft=False)
 for ax in axs:ax.tick_params(which='both',direction='out');ax.set_facecolor('black')
 pos=[ax.get_position() for ax in axs]
 cb0=fig.add_axes([pos[0].x0,.15,pos[0].width,.026]);cb1=fig.add_axes([pos[1].x0,.15,pos[2].x1-pos[1].x0,.026])
 fig.colorbar(im0,cax=cb0,orientation='horizontal',ticks=[1e-4,1e-3,1e-2,1e-1]).set_label(r'$\Sigma$ [g cm$^{-2}$]')
 fig.colorbar(im1,cax=cb1,orientation='horizontal',ticks=[1e-5,1e-4,1e-3,1e-2,1e-1,1]).set_label(r'$\Sigma_r$ [g cm$^{-2}$]')
 time=args.frame*pars['DT_OUT']/YEAR
 fig.text((pos[0].x0+pos[-1].x1)/2,.89,('KD-tree' if search=='kdtree' else 'Morton')+r' — ROCm, $\alpha=10^{-3}$'+f', t = {time:,.0f} yr',ha='center',fontsize=13)
 report['models'][search]={'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'time_yr':time,'active_particles':int(active.sum()),'plotted_particles':len(mass),'plotted_mass_g':float(mass.sum()),'maximum_radius_cm':float(a.max())}
 for j,label in enumerate(['radial','inner','outer']):npz[search+'_'+label]=maps[j];npz[search+'_'+label+'_mean_radius_cm']=means[j]
fig.savefig(out/'strong_rocm_figure14.png',dpi=200);fig.savefig(out/'strong_rocm_figure14.svg');plt.close(fig)
np.savez_compressed(out/'binned_data.npz',radius_edges_au=re,angle_edges_rad=te,size_edges_cm=ae,**npz)
(out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['models'],indent=2))
