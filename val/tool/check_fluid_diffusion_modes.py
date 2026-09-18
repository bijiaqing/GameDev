#!/usr/bin/env python3
"""Execute all six production diffusion kernels serially on CPU; no GPU qualification."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PRE = r'''
#include <cmath>
#include <cassert>
#include <algorithm>
#include <array>
#include <iostream>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define DIFFUSION
using real = double;
constexpr int N_X=8,N_Y=9,N_G=N_X*N_Y*N_Z;
constexpr real G=1,M_S=1,R_0=1,ASPR_0=.2,IDX_Q=-.5,IDX_P=-1;
constexpr real ALPHA=.03,SCHMIDT_X=1.1,SCHMIDT_Y=1.3,SCHMIDT_Z=1.7;
constexpr real X_MIN=0,X_MAX=6.283185307179586,Y_MIN=1,Y_MAX=2;
constexpr real Z_MIN=1.32,Z_MAX=3.141592653589793-1.32;
constexpr real POS_LIMIT=.8,RHO_VAC=1.e-30;
real STOKES_0=.3;
struct dim {int x=0;}; dim threadIdx,blockIdx,blockDim{1};
void __syncthreads() {}
real shared_work[8*(N_X+N_Y+N_Z)];
'''
TEST = r'''
using Field=std::array<real,N_G>;
using Kernel=void(*)(real*,real*,real*,real*,real);
void close(real a,real b,real tol=3.e-11) {
 if(std::abs(a-b)>tol*std::max({1.,std::abs(a),std::abs(b)})) {
  std::cerr << a << " != " << b << '\n'; std::abort();
 }
}
void run(Kernel k,int axis,Field& d,Field& mx,Field& my,Field& mz,real dt) {
 int columns=axis==0?N_Y*N_Z:axis==1?N_X*N_Z:N_X*N_Y;
 for(blockIdx.x=0;blockIdx.x<columns;blockIdx.x++) k(d.data(),mx.data(),my.data(),mz.data(),dt);
}
real volume(int j) {return _get_vol_y(j/N_X%N_Y)*_get_vol_z(j/(N_X*N_Y));}
int main() {
 Kernel kernels[3][2]={{diffusion_xth,diffusion_xbl},{diffusion_yth,diffusion_ybl},{diffusion_zth,diffusion_zbl}};
 for(real st:{0.,.3,3.}) {
  STOKES_0=st;
  real R=1.4,Z=.12,h=_get_hg(R),s=_get_stokes(R,Z,h);
  close(_get_diffusivity(R,Z,h,SCHMIDT_Y),_get_nu(R,h)/(SCHMIDT_Y*(1+s*s)));
  for(int axis=0;axis<3;axis++)for(int profile=0;profile<3;profile++)for(real dt:{.001,300.}) {
   Field initial,mx0,my0,mz0;
   for(int j=0;j<N_G;j++) {
    real y=_get_ycent(j/N_X%N_Y),z=_get_zcent(j/(N_X*N_Y));
    initial[j]=_get_diffusion_weight(y,z)*(profile==0?1.:profile==1?1.+.6*std::sin(j*1.7):(j%13==0?1.:1.e-12));
    mx0[j]=2*initial[j];my0[j]=-3*initial[j];mz0[j]=.7*initial[j];
   }
   Field saved;
   for(int variant=0;variant<2;variant++) {
    auto d=initial,mx=mx0,my=my0,mz=mz0;
    run(kernels[axis][variant],axis,d,mx,my,mz,dt);
    real before=0,after=0;
    for(int j=0;j<N_G;j++) {
     assert(std::isfinite(d[j]) && d[j]>=0);
     before+=initial[j]*volume(j);after+=d[j]*volume(j);
     close(mx[j],2*d[j]);close(my[j],-3*d[j]);close(mz[j],.7*d[j]);
     if(profile==0)close(d[j],initial[j]);
     if(variant==1)close(d[j],saved[j]);
    }
    close(before,after);
    saved=d;
   }
  }
 }
 // Independent finite-volume flux derivative checks both the selected gradient and face D.
 STOKES_0=.3;
 for(int axis=0;axis<3;axis++) {
  if(axis==2 && N_Z==1)continue;
  Field d,mx{},my{},mz{},expected{};
  for(int j=0;j<N_G;j++)d[j]=1.+.2*std::sin(1.7*j);
  const real dt=1.e-7;
  for(int j=0;j<N_G;j++) {
   int ix=j%N_X,iy=j/N_X%N_Y,iz=j/(N_X*N_Y),next;
   real y=_get_ycent(iy),z=_get_zcent(iz),distance,area,vi,vj;
   if(axis==0) {
    next=j-ix+(ix+1)%N_X;
    distance=y*sin(z)*_get_dx();area=1.;vi=vj=distance;
   } else if(axis==1) {
    if(iy==N_Y-1)continue;
    next=j+N_X;distance=_get_ycent(iy+1)-y;
    area=_get_area_y(iy+1);vi=_get_vol_y(iy);vj=_get_vol_y(iy+1);
    y=_get_yface(iy+1);
   } else {
    if(iz==N_Z-1)continue;
    next=j+N_X*N_Y;distance=y*_get_dz();z=_get_zface(iz+1);
    area=sin(z);vi=y*_get_vol_z(iz);vj=y*_get_vol_z(iz+1);
   }
   real gc=_get_diffusion_weight(_get_ycent(iy),_get_zcent(iz));
   real gn=_get_diffusion_weight(_get_ycent(next/N_X%N_Y),_get_zcent(next/(N_X*N_Y)));
   real schmidt=axis==0?SCHMIDT_X:axis==1?SCHMIDT_Y:SCHMIDT_Z;
   real R=y*sin(z),st=_get_stokes(R,y*cos(z),_get_hg(R));
   real D=_get_nu(R,_get_hg(R))/(schmidt*(1+st*st));
   real flux=-area*_get_diffusion_weight(y,z)*D*(d[next]/gn-d[j]/gc)/distance;
   expected[j]-=flux/vi;expected[next]+=flux/vj;
  }
  auto before=d;
  run(kernels[axis][0],axis,d,mx,my,mz,dt);
  for(int j=0;j<N_G;j++)close((d[j]-before[j])/dt,expected[j],3.e-7);
 }
 // Verify the initial diffusion-balance velocity uses the selected concentration.
 Field d,vx,vy,vz;
 for(int j=0;j<N_G;j++)d[j]=_get_diffusion_weight(_get_ycent(j/N_X%N_Y),_get_zcent(j/(N_X*N_Y)));
 for(blockIdx.x=0;blockIdx.x<N_G;blockIdx.x++)init_vel_calc(vx.data(),vy.data(),vz.data(),d.data());
 for(int j=0;j<N_G;j++) {
  real z=_get_zcent(j/(N_X*N_Y)),y=_get_ycent(j/N_X%N_Y);
  if(N_Z>1)close(vz[j]/y,vy[j]*cos(z)/sin(z));else close(vz[j],0.);
 }
}
'''

def main():
    env = dict(os.environ)
    sdk = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1')
    if sdk.exists():
        env['CPLUS_INCLUDE_PATH'] = str(sdk)
    with tempfile.TemporaryDirectory(prefix='fluid-diffusion-') as tmp:
        tmp = Path(tmp)
        (tmp/'const_defs.cuh').write_text('')
        (tmp/'fluid_kern.cuh').write_text('')
        includes = ''.join(f'#include "{ROOT}/src/fluid/diffusion_{axis}{variant}.cu"\n'
                           for axis in 'xyz' for variant in ('th','bl'))
        includes += f'#include "{ROOT}/src/fluid/init_vel_calc.cu"\n'
        (tmp/'check.cpp').write_text(PRE + includes + TEST)
        for nz in (1, 7):
            for concentration in (False, True):
                cmd = ['c++', '-std=c++17', '-O2', f'-DN_Z={nz}', '-I'+str(tmp),
                       '-I'+str(ROOT/'inc/fluid'), str(tmp/'check.cpp'), '-o', str(tmp/'check')]
                if concentration:
                    cmd.append('-DDIFFUSE_CONCENTRATION')
                subprocess.run(cmd, check=True, env=env)
                subprocess.run([str(tmp/'check')], check=True)
    print(json.dumps({'status': 'pass', 'kernels': 6, 'modes': 2, 'dimensions': [2, 3],
                      'checks': ['equilibrium', 'mass conservation', 'positive density',
                                 'donor momentum', 'thread/block agreement', 'Stokes suppression',
                                 'initial diffusion balance', 'independent finite-volume flux'], 'native_gpu_execution': False}))

if __name__ == '__main__':
    main()
