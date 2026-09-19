#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/gamedev-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/gamedev-cache')
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
root=Path.home()/'Scratch/gamedev';out=Path(__file__).resolve().parent/'results/mcdust_strong_comparison';out.mkdir(exist_ok=True)
AU=1.495978707e13
with h5py.File(root/'mcdust/setups/alpha1.e-3/swarms-00124.h5') as f:
 x=f['swarms/swarmsout'][0];attrs=dict(f.attrs);time=float(f['times/timesout'][0]);R=x['cylindrical radius [AU]'];Z=x['height above midplane [AU]'];radius=np.cbrt(3*x['mass of a particle [g]']/(4*np.pi*float(attrs['material_density_[g/cm3]'])))
 data={'mcdust':(R,Z,radius,np.full(len(R),float(attrs['mass_of_swarm[g]'])))}
for search in ['kdtree','morton']:
 for frame in [105,110,115]:
  x=np.fromfile(root/f'campaign_diskevol/strong_turbulence/rocm/{search}/particle_{frame:05}.dat').reshape(-1,8);x=x[x[:,1]>0]
  R=x[:,1]*np.sin(x[:,2])/AU;Z=x[:,1]*np.cos(x[:,2])/AU;radius=x[:,6]/2;mass=x[:,7]*4*np.pi/3*radius**3
  data[search+('' if frame==110 else f'_{frame}')]=(R,Z,radius,mass)
se=np.geomspace(5e-5/1.0559,50*1.0559,128);te=np.linspace(0,.2,129);re=np.geomspace(5,50,129)
report={'times_yr':{'mcdust':time,'gamedev':11000},'regions':{},'notes':['mcdust snapshot mass weights use mass_of_swarm, not uninitialized output fields.','Spherical r cuts and folded hemispheres match our Fig.14-style plot.','Current source/setup files are not a verified build record of the published mcdust snapshot.']};maps={};curves={};samples={}
def stats(a,w):
 order=np.argsort(a);return {'mass_g':float(w.sum()),'count':len(a),'mean_radius_um':float(np.average(a,weights=w)*1e4),'q50_q90_q99_um':(np.interp([.5,.9,.99],np.cumsum(w[order])/w.sum(),a[order])*1e4).tolist(),'fraction_radius_gt_100um':float(w[a>1e-2].sum()/w.sum())}
for name,(R,Z,a,w) in data.items():
 r=np.hypot(R,Z);t=np.abs(np.arctan2(Z,R));domain=(r>=5)&(r<=50)&(t<=.2)&(R>0)
 R,Z,a,w,r,t=[v[domain] for v in (R,Z,a,w,r,t)];samples[name]=(r,t,a,w)
 report['regions'][name]={}
 for reg,mask in {'inner':r<15,'outer':r>25,'outer_midplane':(r>25)&(t<.025),'outer_high':(r>25)&(t>=.075)&(t<.125)}.items():
  report['regions'][name][reg]=stats(a[mask],w[mask])
  report['regions'][name][reg]['mean_r_au']=float(np.average(r[mask],weights=w[mask]));report['regions'][name][reg]['rms_angle']=float(np.sqrt(np.average(t[mask]**2,weights=w[mask])))
 if name not in ['mcdust','kdtree','morton']:continue
 h=np.histogram2d(r,a,[re,se],weights=w)[0]/(np.pi*np.diff((re*AU)**2))[:,None];maps[name]=[h];curves[name]=[]
 for reg,mask in [('inner',r<15),('outer',r>25)]:
  weight=w/(r*AU)**2*mask;H=np.histogram2d(t,a,[te,se],weights=weight)[0]/(4*np.pi*np.diff(np.sin(te)))[:,None];maps[name].append(H)
  den=np.histogram(t,te,weights=weight)[0];num=np.histogram(t,te,weights=weight*a)[0];mean=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
  curves[name].append((mean,H.sum(axis=1)))
report['radial_shells']={}
for name,(r,t,a,w) in samples.items():
 if name not in ['mcdust','kdtree','morton']:continue
 report['radial_shells'][name]={}
 for lo,hi in [(25,30),(30,35),(35,40),(40,45),(45,50)]:
  report['radial_shells'][name][f'{lo}-{hi}']={}
  for label,mask in {'midplane':(t<.025),'high':(t>=.075)&(t<.125)}.items():
   m=(r>=lo)&(r<hi)&mask;report['radial_shells'][name][f'{lo}-{hi}'][label]=stats(a[m],w[m])
fig,ax=plt.subplots(3,3,figsize=(11,9),layout='constrained');cm=plt.get_cmap('inferno').copy();cm.set_bad('black')
for row,name in enumerate(['mcdust','kdtree','morton']):
 im0=ax[row,0].pcolormesh(re,se,np.ma.masked_less_equal(maps[name][0].T,0),norm=LogNorm(1e-4,1e-1),cmap=cm,rasterized=True);ax[row,0].set(xscale='log',yscale='log',xlim=(5,50),ylim=(5e-5,50),ylabel=name+'\nGrain radius [cm]')
 ax[row,0].set_xticks([5,10,25,50]);ax[row,0].set_xticklabels(['5','10','25','50']);ax[row,0].xaxis.set_minor_formatter(NullFormatter())
 for j in [1,2]:
  im1=ax[row,j].pcolormesh(se,te,np.ma.masked_less_equal(maps[name][j],0),norm=LogNorm(1e-5,1),cmap=cm,rasterized=True);ax[row,j].plot(curves[name][j-1][0],(te[1:]+te[:-1])/2,color='.8');ax[row,j].set(xscale='log',xlim=(5e-5,50),ylim=(0,.125),ylabel='Polar angle [rad]')
for j,title in enumerate(['Column density','Inner disc: r < 15 au','Outer disc: r > 25 au']):ax[0,j].set_title(title);ax[2,j].set_xlabel('Radius [au]' if j==0 else 'Grain radius [cm]')
fig.colorbar(im0,ax=ax[:,0],orientation='horizontal',label=r'$\Sigma$ [g cm$^{-2}$]',fraction=.035);fig.colorbar(im1,ax=ax[:,1:],orientation='horizontal',label=r'$\Sigma_r$ [g cm$^{-2}$]',fraction=.035)
fig.suptitle(f'Same binning: mcdust {time:,.1f} yr; GameDev 11,000 yr');fig.savefig(out/'matched_maps.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for name,color in [('mcdust','black'),('kdtree','tab:blue'),('morton','tab:orange')]:
 mean,density=curves[name][1];tc=(te[1:]+te[:-1])/2
 ax[0].plot(tc,mean*1e4,label=name,color=color);ax[1].plot(tc,density,label=name,color=color)
 r,t,a,w=samples[name];mask=r>25;h=np.histogram(a[mask],se,weights=w[mask])[0]/w[mask].sum()/np.diff(np.log10(se));ax[2].stairs(h,se*1e4,label=name,color=color)
ax[0].set(xlim=(0,.125),yscale='log',xlabel='Polar angle [rad]',ylabel='Mean grain radius [µm]',title='Outer-disc mean size');ax[1].set(xlim=(0,.125),yscale='log',xlabel='Polar angle [rad]',ylabel=r'$\int\rho_d\,dr$ [g cm$^{-2}$]',title='Outer-disc vertical density');ax[2].set(xscale='log',xlim=(.5,3000),xlabel='Grain radius [µm]',ylabel='Mass fraction per dex',title='Outer-disc size distribution');ax[0].legend()
fig.savefig(out/'outer_diagnostics.png',dpi=200);plt.close(fig)
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'times':report['times_yr'],'regions':report['regions']},indent=2))
