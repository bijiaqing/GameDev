#!/usr/bin/env python3
"""Compare GameDev/MCDUST campaigns and available cuDisc snapshots."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gamedev-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/gamedev-cache')
import argparse,json,configparser
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

AU=1.495978707e13; YEAR=31557600.
LABELS=['CUDA KD-tree','CUDA Morton','ROCm KD-tree','ROCm Morton','MCDUST 36 cores','MCDUST 72 cores']
COLORS=['#0072B2','#56B4E9','#D55E00','#E69F00','#888888','#000000']
SE=np.geomspace(1e-5,1e2,321)
RE=np.geomspace(5.,50.,513);TE=np.linspace(0,.2,129)

def parameters(path):
    c=configparser.ConfigParser();c.read(path)
    return c['PARAMETERS']

def load(path,kind,pars):
    if kind=='gamedev':
        x=np.fromfile(path,dtype='<f8').reshape(-1,8)
        assert len(x)==pars.getint('N_P') and np.isfinite(x).all()
        x=x[x[:,1]>0];r=x[:,1]/AU;theta=np.abs(np.pi/2-x[:,2]);a=x[:,6]/2
        w=x[:,7]*(4*np.pi/3)*pars.getfloat('RHO_0')*a**3
        time=int(path.stem.split('_')[-1])*pars.getfloat('DT_OUT')/YEAR
    else:
        with h5py.File(path) as f:
            x=f['swarms/swarmsout'][0];time=float(f['times/timesout'][0])
            R=x['cylindrical radius [AU]'];Z=x['height above midplane [AU]']
            r=np.hypot(R,Z);theta=np.abs(np.arctan2(Z,R))
            a=np.cbrt(3*x['mass of a particle [g]']/(4*np.pi*float(f.attrs['material_density_[g/cm3]'])))
            # npar is zero/uninitialized in these HDF5 files; use the documented constant swarm mass.
            w=np.full(len(x),float(f.attrs['mass_of_swarm[g]']))
    assert np.isfinite(r).all() and np.isfinite(a).all() and (a>0).all() and (w>0).all()
    total=float(w.sum());mask=(r>=5)&(r<=50)&(theta<=.2)
    return time,r[mask],theta[mask],a[mask],w[mask],total

def stats(a,w,t):
    order=np.argsort(a);cdf=np.cumsum(w[order])/w.sum()
    h=np.histogram(a,SE,weights=w)[0];assert np.isclose(h.sum(),w.sum())
    return {'mass_g':float(w.sum()),'particles':len(a),'mean_radius_cm':float(np.average(a,weights=w)),
            'q50_q90_q99_cm':np.interp([.5,.9,.99],cdf,a[order]).tolist(),
            'rms_angle':float(np.sqrt(np.average(t*t,weights=w))),
            'fraction_a_gt_100um':float(w[a>1e-2].sum()/w.sum()),'histogram':h}

def maps(r,t,a,w,ae):
    radial=np.histogram2d(r,a,[RE,ae],weights=w)[0]/(np.pi*np.diff((RE*AU)**2))[:,None]
    result=[radial];means=[]
    for coord,edges,weight in [(r,RE,w),(t,TE,w/(r*AU)**2*(r<15)),(t,TE,w/(r*AU)**2*(r>25))]:
        den=np.histogram(coord,edges,weights=weight)[0];num=np.histogram(coord,edges,weights=weight*a)[0]
        means.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
        if coord is t:
            h=np.histogram2d(t,a,[TE,ae],weights=weight)[0]
            result.append(h/(4*np.pi*np.diff(np.sin(TE)))[:,None])
    return result,means

def load_cudisc(directory,ae,quadrature=4):
    """Integrate piecewise-constant cells into the common spherical domain.

    Binary layout: cuDisc/cuDisc codes/python/fileIO.py and src/grid.cu.
    Two ghost cells per boundary, as specified in the supplied setup.
    Integrate azimuth and both hemispheres; split angles at native/output edges.
    """
    nr,nz=np.fromfile(directory/'2Dgrid.dat',dtype='<i4',count=2)
    grid=np.fromfile(directory/'2Dgrid.dat',dtype='<f8',offset=8)
    re=grid[:nr+1][2:-2]
    off=2*nr+1
    te=np.arctan(grid[off:off+nz+1])[2:-2]
    vol=grid[off+2*nz+1:].reshape(nr,nz)[2:-2,2:-2]
    expected=np.diff(re**3)[:,None]/3*np.diff(np.tan(te))[None,:]
    assert np.allclose(vol,expected,rtol=1e-10)
    path,=directory.glob('dens_*.dat')
    dims=np.fromfile(path,dtype='<i4',count=3);assert tuple(dims)==(nr,nz,len(ae)-1)
    raw=np.fromfile(path,dtype='<f8',offset=12).reshape(nr,4*(dims[2]+1)*nz+1)
    rho=raw[:,:-1].reshape(nr,nz,dims[2]+1,4)[2:-2,2:-2,1:,0]
    assert np.isfinite(rho).all() and (rho>=0).all()
    assert np.allclose(np.loadtxt(directory/'grains.sizes')[:,1],ae)
    sizes=np.cbrt((ae[:-1]**3+ae[1:]**3)/2)
    radial=np.zeros((len(RE)-1,len(sizes)))
    vertical=[np.zeros((len(TE)-1,len(sizes))) for _ in range(2)]
    mass=np.zeros((3,len(sizes)));angle2=mass.copy()
    nodes,weights=np.polynomial.legendre.leggauss(quadrature)
    edges=np.unique(np.r_[TE,te[(te>0)&(te<.2)]])
    for low,high in zip(edges[:-1],edges[1:]):
        mid=(low+high)/2;half=(high-low)/2
        j=np.searchsorted(te,mid)-1;k=np.searchsorted(TE,mid)-1
        den=rho[:,j,:]
        for theta,weight in zip(mid+half*nodes,half*weights):
            c=np.cos(theta);lo=re[:-1]/c;hi=re[1:]/c
            # Spherical radial-bin volumes; cylindrical native edges tilt with theta.
            left=np.maximum((RE[:-1]*AU)[:,None],lo)
            right=np.minimum((RE[1:]*AU)[:,None],hi)
            v=4*np.pi/3*c*weight*np.maximum(right**3-left**3,0)
            radial+=v@den
            for region,(rlo,rhi) in enumerate(((5,50),(5,15),(25,50))):
                l=np.maximum(lo,rlo*AU);h=np.minimum(hi,rhi*AU)
                dm=(4*np.pi/3*c*weight*np.maximum(h**3-l**3,0))@den
                mass[region]+=dm;angle2[region]+=dm*theta**2
                if region:vertical[region-1][k]+=(c*weight*np.maximum(h-l,0))@den
    radial/=np.pi*np.diff((RE*AU)**2)[:,None]
    vertical=[v/np.diff(np.sin(TE))[:,None] for v in vertical]
    mm=[radial,*vertical]
    means=[m@sizes/m.sum(axis=1) for m in mm]
    # Conservative log-size overlap avoids an artificial comb when plotting a discrete grid.
    overlap=np.maximum(0,np.minimum(np.log(SE[1:])[:,None],np.log(ae[1:]))-
                       np.maximum(np.log(SE[:-1])[:,None],np.log(ae[:-1]))) / np.diff(np.log(ae))
    assert np.allclose(overlap.sum(axis=0),1)
    regions={}
    for i,name in enumerate(('whole','inner','outer')):
        st=stats(sizes,mass[i],np.sqrt(angle2[i]/mass[i]))
        st.pop('particles');st['size_bins']=len(sizes)
        st['histogram']=overlap@mass[i];regions[name]=st
    mapped_mass=(radial*np.pi*np.diff((RE*AU)**2)[:,None]).sum()
    assert np.isclose(mapped_mass,mass[0].sum(),rtol=1e-12)
    time=np.loadtxt(directory/'2Dtimes.txt')[int(path.stem.split('_')[-1])]/YEAR
    return {'maps':(mm,means),'regions':regions,'time':float(time),'path':str(path)}

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=160)
    fig.savefig(out/(name+'.pdf'),dpi=160)
    plt.close(fig)

def figure14(data,ae,out,regime,time):
    fig,axs=plt.subplots(len(data),3,figsize=(12,2.8*len(data)),layout='constrained')
    cm=plt.get_cmap('inferno').copy();cm.set_bad('black');cm.set_under('black')
    for i,label in enumerate(data):
        m,means=data[label]['maps'];ax=axs[i]
        im0=ax[0].pcolormesh(RE,ae,np.ma.masked_less_equal(m[0].T,0),norm=LogNorm(1e-4,1e-1),cmap=cm,rasterized=True)
        ax[0].plot(np.sqrt(RE[:-1]*RE[1:]),means[0],color='.8',lw=1)
        ax[0].set(xscale='log',yscale='log',xlim=(5,50),ylim=(5e-5,50),ylabel=label+f' ({data[label]["time"]:,.0f} yr)\nGrain radius [cm]')
        ax[0].set_xticks([5,10,25,50],['5','10','25','50']);ax[0].xaxis.set_minor_formatter(NullFormatter())
        for cut in (15,25):ax[0].axvline(cut,color='.8',lw=.6)
        for j in (1,2):
            im1=ax[j].pcolormesh(ae,TE,np.ma.masked_less_equal(m[j],0),norm=LogNorm(1e-5,1),cmap=cm,rasterized=True)
            ax[j].plot(means[j],(TE[:-1]+TE[1:])/2,color='.8',lw=1)
            ax[j].set(xscale='log',xlim=(5e-5,50),ylim=(0,.125),ylabel=r'$\vartheta$ [rad]')
            ax[j].set_xticks([1e-3,1e-1,1e1]);ax[j].set_yticks([0,.05,.10])
        for a in ax:a.set_facecolor('black')
    for j,title in enumerate(['Vertical column density','Inner disc: r < 15 au','Outer disc: r > 25 au']):
        axs[0,j].set_title(title);axs[-1,j].set_xlabel('Distance to star [au]' if j==0 else 'Grain radius [cm]')
    fig.colorbar(im0,ax=axs[:,0],orientation='horizontal',fraction=.025,pad=.03,label=r'$\Sigma$ per size bin [g cm$^{-2}$]')
    fig.colorbar(im1,ax=axs[:,1:],orientation='horizontal',fraction=.025,pad=.03,label=r'$\Sigma_r$ per size bin [g cm$^{-2}$]')
    fig.suptitle(f'{regime.capitalize()} turbulence — GameDev/MCDUST {time:,.0f} yr; cuDisc {data['cuDisc']['time']:,.0f} yr\nFigure 14 layout and colour scales; grey: mass-weighted mean size',fontsize=14)
    save(fig,out,f'figure14_{regime}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'Scratch/gamedev')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/campaign_comparison')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    ref=Path(__file__).parent/'results/figure14_output110/reference_grains.sizes'
    ae=np.loadtxt(ref)[:,1]  # cuDisc writes size edges, not centers.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    report={'timing_basis':'Preserved snapshot modification times; elapsed from first to final output, excluding initialization; may include pauses and I/O. Not profiler timings.',
            'reference':'https://arxiv.org/html/2603.22550v2#S5.SS2',
            'geometry':'Common spherical domain 5<=r/AU<=50, |theta-pi/2|<=0.2. Radial maps integrate both hemispheres; vertical maps average folded hemispheres using swarm mass/r^2 /(4*pi*d(sin(theta))).',
            'comparison':'MCDUST is a separate numerical method, not an exact solution. CDF distance uses 320 fixed logarithmic size bins.', 'cudisc_notes':'Different stored times; initial MRN 0.5-1 um rather than 0.5 um monomers. Cell density integrated over common spherical domain with angular Gauss quadrature. Size histograms conservatively rebinned assuming uniform mass per log size within each native bin. No runtime inferred from one snapshot.', 'regimes':{}}
    perf,pax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    agree,aax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    arrays={}
    for row,(regime,alpha) in enumerate([('strong','1e-3'),('weak','1e-4')]):
        dirs=[args.root/(regime+'_turbulence')/b/s for b in ('cuda','rocm') for s in ('kdtree','morton')]
        dirs += [args.root/'mcdust/outputs'/f'alpha_{alpha}_{n}cores' for n in (36,72)]
        data={};res={};times_ref=None
        for index,(label,directory) in enumerate(zip(LABELS,dirs)):
            kind='gamedev' if index<4 else 'mcdust';pars=parameters(directory/'variables.txt') if index<4 else None
            paths=sorted(directory.glob('particle_*.dat' if index<4 else 'swarms-*.h5'))
            elapsed=np.array([p.stat().st_mtime-paths[0].stat().st_mtime for p in paths])/3600
            assert np.all(np.diff(elapsed)>0),directory
            history=[];times=[];regions={}
            for ii,path in enumerate(paths):
                time,r,t,a,w,total=load(path,kind,pars);times.append(time)
                st=stats(a,w,t);hist=st.pop('histogram')
                history.append({'time_yr':time,'elapsed_hours':float(elapsed[ii]),'all_active_mass_g':total,**st})
                if ii==len(paths)-1:
                    for region,mask in [('whole',np.ones(len(r),dtype=bool)),('inner',r<15),('outer',r>25)]:regions[region]=stats(a[mask],w[mask],t[mask])
                    data[label]={'maps':maps(r,t,a,w,ae),'regions':regions,'time':time}
            times=np.array(times)
            if times_ref is None:times_ref=times
            else:assert np.allclose(times,times_ref,rtol=0,atol=.01),(directory,times,times_ref)
            res[label]={'path':str(directory),'history':history,'elapsed_hours':float(elapsed[-1]),'final_time_yr':float(times[-1]),
                        'regions':{name:{k:v for k,v in st.items() if k!='histogram'} for name,st in regions.items()}}
            print(regime,label,'hours',round(elapsed[-1],3),flush=True)
            pax[row,0].plot(times,elapsed,'o-',color=COLORS[index],label=label,ms=3)
            pax[row,1].plot((times[:-1]+times[1:])/2,np.diff(elapsed)*3600/np.diff(times),'-o',color=COLORS[index],ms=3)
            for j,region in enumerate(('inner','outer')):
                h=regions[region]['histogram'];aax[row,j].stairs(h/h.sum()/np.diff(np.log10(SE)),SE,color=COLORS[index],label=label,lw=1.3)
                arrays[f'{regime}_{index}_{region}_mass']=h
            m=data[label]['maps'][0][2];aax[row,2].plot((TE[:-1]+TE[1:])/2,m.sum(axis=1),color=COLORS[index],label=label)
        cd=load_cudisc(args.root/'dustComparison/2Dsims/cuDisc'/alpha.replace('1e','alpha1.e'),ae)
        data['cuDisc']=cd
        res['cuDisc']={'path':cd['path'],'final_time_yr':cd['time'],'elapsed_hours':None,
            'regions':{name:{k:v for k,v in st.items() if k!='histogram'} for name,st in cd['regions'].items()}}
        for j,region in enumerate(('inner','outer')):
            h=cd['regions'][region]['histogram'];arrays[f'{regime}_6_{region}_mass']=h
            aax[row,j].stairs(h/h.sum()/np.diff(np.log10(SE)),SE,color='#009E73',label='cuDisc',lw=1.5)
        aax[row,2].plot((TE[:-1]+TE[1:])/2,cd['maps'][0][2].sum(axis=1),color='#009E73',label='cuDisc')
        for label in data:
            res[label]['distances']={}
            for baseline in ('CUDA KD-tree','MCDUST 72 cores'):
                res[label]['distances'][baseline]={}
                for region in ('whole','inner','outer'):
                    h=data[label]['regions'][region]['histogram'];refh=data[baseline]['regions'][region]['histogram']
                    res[label]['distances'][baseline][region]=float(np.max(np.abs(np.cumsum(h)/h.sum()-np.cumsum(refh)/refh.sum())))
        figure14(data,ae,args.out,regime,times[-1])
        report['regimes'][regime]=res
        for j in (0,1):
            pax[row,j].set(xlabel='Simulation time [yr]',title=regime.capitalize()+' turbulence',ylabel='Elapsed wall time [h]' if j==0 else 'Wall seconds / simulated year')
            aax[row,j].set(xscale='log',xlim=(5e-5,50),xlabel='Grain radius [cm]',ylabel='Mass fraction per dex',title=regime.capitalize()+(' — inner disc' if j==0 else ' — outer disc'))
        aax[row,2].set(yscale='log',xlim=(0,.125),xlabel=r'$\vartheta$ [rad]',ylabel=r'$\Sigma_r$ [g cm$^{-2}$]',title=regime.capitalize()+' — outer vertical profile')
    pax[0,0].legend(fontsize=8);aax[0,0].legend(fontsize=8)
    perf.suptitle('Performance from preserved output timestamps (initialization excluded)')
    agree.suptitle('Strong: GameDev/MCDUST 12,500 yr; cuDisc 11,134 yr | Weak: 25,000 yr; cuDisc 23,809 yr')
    save(perf,args.out,'performance');save(agree,args.out,'agreement')
    np.savez_compressed(args.out/'distributions.npz',size_edges_cm=SE,**arrays)
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
