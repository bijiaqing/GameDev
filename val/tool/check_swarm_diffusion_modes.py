#!/usr/bin/env python3
"""Host finite-difference check of the production diffusion closure and gas interpolant."""
from pathlib import Path
import os, subprocess, tempfile
repo=Path(__file__).resolve().parents[2]
env=dict(os.environ)
if Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk').exists():
    env['CPLUS_INCLUDE_PATH']='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
source=r'''
#include <cmath>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <vector>
using std::isfinite;
#define __host__
#define __device__
#define __forceinline__ inline
#define COAG_ALPHA .001
#define COAG_OUTPUTS 250
constexpr double NU=0.02;
struct double3 {double x,y,z;};
void atomicAdd(double *p,double v){*p+=v;}
#include <_diffusion.cuh>
#define __global__
struct dim{int x;};dim threadIdx{0},blockIdx{0},blockDim{1};
real curand_normal_double(curs*) {return 1.;}
#include "KERNEL_PATH"
#define TRANSPORT
#include "RATE_PATH"

std::vector<real> gas(N_G);
real value(real phi,real R,real Z,real size,bool density) {
 real y=N_Z>1?hypot(R,Z):R,z=N_Z>1?atan2(R,Z):M_PI/2;
#ifdef IMPORTGAS
 if(density)return _interp_field(gas.data(),_get_loc_x(phi),_get_loc_y(y),_get_loc_z(z));
#else
 if(density)return pow(R/R_0,IDX_P)*(N_Z>1?_get_gas_strat(R,Z,_get_hg(R))/(_get_hg(R)*R):1.);
#endif
 return _get_dust_diffusion(phi,y,z,size
#ifdef IMPORTGAS
 ,gas.data()
#endif
 ).nu;
}
int main(){
 swarm particle{};particle.position={0.,10.*AU,M_PI/2};curs rng{};

 for(int k=0;k<N_Z;++k)for(int j=0;j<N_Y;++j)for(int i=0;i<N_X;++i)
 gas[i+N_X*(j+N_Y*k)]=1.0+0.1*sin((i+.5)*_get_dx())+.03*j+.02*k;
 diffusion_pos(&particle,&rng,1.
#ifdef IMPORTGAS
 ,gas.data()
#endif
 );assert(isfinite(particle.position.y));
 real rate=0;
 dyn_rate_calc(&rate,&particle
#ifdef IMPORTGAS
 ,gas.data(),gas.data(),gas.data(),gas.data(),gas.data(),gas.data(),gas.data(),gas.data()
#endif
 );assert(isfinite(rate)&&rate>0);

 for(real ly:{.1,1.23,3.75,7.9})for(real lx:{.1,1.27,3.9})for(real lz:{.1,.83}) {
 real phi=X_MIN+lx*_get_dx(),y=Y_MIN*pow(_get_dy(),ly),z=N_Z>1?Z_MIN+lz*_get_dz():M_PI/2;
 real R=N_Z>1?y*sin(z):y,Z=N_Z>1?y*cos(z):0.;
 for(real size:{S_0,S_0*1.e5,S_0*1.e9}) {
 auto d=_get_dust_diffusion(phi,y,z,size
#ifdef IMPORTGAS
 ,gas.data()
#endif
 );
 real e=R*1.e-6, ep=1.e-6;
 auto logv=[&](real x,real r,real h){real q=log(value(x,r,h,size,false));
#ifdef DIFFUSE_CONCENTRATION
 q+=log(value(x,r,h,size,true));
#endif
 return q;};
 real gR=(logv(phi,R+e,Z)-logv(phi,R-e,Z))/(2*e)+1/R;
 real gZ=N_Z>1?(logv(phi,R,Z+e)-logv(phi,R,Z-e))/(2*e):0;
 real gp=(logv(phi+ep,R,Z)-logv(phi-ep,R,Z))/(2*ep);
 auto check=[](real a,real b){if(fabs(a-b)>2.e-6){printf("%.17g != %.17g\n",a,b);assert(false);}};
 check(R*d.drift_R_per_D,R*gR);check(R*d.drift_Z_per_D,R*gZ);check(d.drift_phi_per_D,gp);
 }
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);(tmp/'curand_kernel.h').write_text('struct curandState {};\n')
 (tmp/'check.cpp').write_text(source.replace('KERNEL_PATH',str(repo/'src/swarm/diffusion_pos.cu')).replace('RATE_PATH',str(repo/'src/swarm/dyn_rate_calc.cu')))
 (tmp/'swarm_kern.cuh').write_text('')
 (tmp/'gpu_compat.cuh').write_text('#define gpuRandNormalDouble curand_normal_double\n')
 (tmp/'_transport.cuh').write_text('#pragma once\nbool _is_particle_active(real,real){return true;}\nvoid _apply_diffusion_boundary(real&,real&,real&){}\n')
 const=(repo/'val/main/swarm_coagulation/common/const_defs.cuh').read_text()
 count=0
 for nz in [1,4]:
  (tmp/'const_defs.cuh').write_text(const.replace('N_X = 1, N_Y = 128, N_Z = 64',f'N_X = 4, N_Y = 8, N_Z = {nz}'))
  for mode in [[],['DIFFUSE_CONCENTRATION']]:
   for gas in [[],['CONST_ST'],['IMPORTGAS'],['IMPORTGAS','CONST_ST'],['CONST_NU'],['IMPORTGAS','CONST_NU']]:
    cmd=['c++','-std=c++17','-O2','-DDIFFUSION']+['-D'+f for f in mode+gas]+['-I'+str(tmp),'-I'+str(repo/'inc/swarm'),str(tmp/'check.cpp'),'-o',str(tmp/'check')]
    subprocess.run(cmd,check=True,env=env);subprocess.run([str(tmp/'check')],check=True);count+=1
 print(f'{count} particle closure configurations passed (host execution; no GPU qualification).')
