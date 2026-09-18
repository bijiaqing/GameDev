#!/usr/bin/env python3
"""Check exact gas gradients, displacement moments, velocity preservation and routing."""
from pathlib import Path
import os,subprocess,json
lab=Path(__file__).resolve().parent;repo=lab.parents[2];model=lab/'common'
out=repo/'val/temp/swarm_coagulation/checks/concentration_diffusion';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
if Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk').exists():env['CPLUS_INCLUDE_PATH']='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
def function(file,name):
 s=file.read_text();a=s.index('real '+name+' (');b=s.index('{',a);j=b+1;depth=1
 while depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]+'\n'
phys=repo/'inc/swarm/param_phys.cuh';grid=repo/'inc/swarm/param_grid.cuh'
(out/'param_phys.cuh').write_text('#pragma once\n'+''.join(function(phys,n) for n in ['_get_omegaK','_get_hg','_get_nu','_get_gas_strat','_get_sigma_g','_get_stokes']))
with (out/'param_phys.cuh').open('a') as f:f.write(''.join(function(grid,n) for n in ['_get_cyl_R','_get_cyl_Z']))
(out/'swarm_kern.cuh').write_text('')
(out/'gpu_compat.cuh').write_text('#ifdef GAMEDEV_ROCM\n#define gpuRandNormalDouble hiprand_normal_double\n#else\n#define gpuRandNormalDouble curand_normal_double\n#endif\n')
(out/'_transport.cuh').write_text('bool _is_particle_active(real,real){return true;}\nvoid _apply_diffusion_boundary(real&,real&,real&){}\n')
(out/'curand_kernel.h').write_text('struct curandState {int counter;};\n')
(out/'hiprand').mkdir(exist_ok=True);(out/'hiprand/hiprand_kernel.h').write_text('struct hiprandState {int counter;};\n')
pre='''#include <cmath>
#include <cassert>
#include <cstdio>
#include <algorithm>
#define __device__
#define __host__
#define __forceinline__ inline
#define __global__
#define __shared__
#define DIFFUSION
#define MULTISIZE
#define DIFFUSE_CONCENTRATION
struct double3{double x,y,z;};
#include "const_defs.cuh"
struct dim{int x;};dim threadIdx{0},blockIdx{0},blockDim{1};
real normal(curs *r){return (++r->counter%2)?1.:-1.;}
#define curand_normal_double normal
#define hiprand_normal_double normal
'''
test=r'''
real loggas(real R,real Z){real h=_get_hg(R);return log(_get_sigma_g(R)/(h*R))+log(_get_gas_strat(R,Z,h));}
void close(real a,real b,real tol){if(fabs(a-b)>tol) {fprintf(stderr,"actual=%.17g expected=%.17g tolerance=%.17g\n",a,b,tol);assert(false);}}
real3 cart(const swarm &p){real R=p.position.y*sin(p.position.z);real vphi=p.velocity.x/R;
 real vr=p.velocity.y*sin(p.position.z)+p.velocity.z/p.position.y*cos(p.position.z);
 return {vr*cos(p.position.x)-vphi*sin(p.position.x),vr*sin(p.position.x)+vphi*cos(p.position.x),p.velocity.y*cos(p.position.z)-p.velocity.z/p.position.y*sin(p.position.z)};}
real diffusivity(real R,real Z,real size){real h=_get_hg(R),st=_get_stokes(R,Z,h,size);return _get_nu(R,h)/(1+st*st);}
int main(){
 for(real rau:{5.,10.,30.,49.})for(real zr:{-.15,-.03,0.,.03,.15})for(real target_st:{1.e-4,.1,1.,10.}){
  real R=rau*AU,Z=R*zr,h=_get_hg(R),eps=R*1.e-6;
  real gr=(loggas(R+eps,Z)-loggas(R-eps,Z))/(2*eps);
  real gz=(loggas(R,Z+eps)-loggas(R,Z-eps))/(2*eps);
  close(R*_get_diff_gradlnrho_R(R,Z,h),R*gr,1.e-7);
  close(R*_get_diff_gradlnrho_Z(R,Z,h),R*gz,1.e-7);
  if(Z!=0)assert(Z*_get_diff_gradlnrho_Z(R,Z,h)<0);
  real size=S_0*target_st/_get_stokes(R,Z,h,S_0);
  real D=diffusivity(R,Z,size),dt=.01*YEAR;
  real dR=(diffusivity(R+eps,Z,size)-diffusivity(R-eps,Z,size))/(2*eps);
  real dZ=(diffusivity(R,Z+eps,size)-diffusivity(R,Z-eps,size))/(2*eps);
  auto actual=_get_dust_diffusion(.3,hypot(R,Z),atan2(R,Z),size);
  close(actual.nu/_get_nu(R,h),1/(1+target_st*target_st),1.e-12);
  close(R*actual.drift_R_per_D,R*(dR/D+1/R+gr),2.e-6);
  close(R*actual.drift_Z_per_D,R*(dZ/D+gz),2.e-6);
  real expectedR=dt*(dR+D/R+D*gr),expectedZ=dt*(dZ+D*gz);
  real dr[2],dz[2];
  for(int j=0;j<2;++j){
   swarm p{};p.par_size=size;p.position={.3,hypot(R,Z),atan2(R,Z)};p.velocity={R*123.,456.,p.position.y*789.};
   auto old=cart(p);curs rng{j};diffusion_pos(&p,&rng,dt);auto now=cart(p);
   close(old.x,now.x,1.e-9);close(old.y,now.y,1.e-9);close(old.z,now.z,1.e-9);assert(rng.counter==j+2);
   dr[j]=p.position.y*sin(p.position.z)-R;dz[j]=p.position.y*cos(p.position.z)-Z;
  }
  close((dr[0]+dr[1])/2,expectedR,R*1.e-10);
  close((dz[0]+dz[1])/2,expectedZ,R*1.e-10);
  close(pow((dr[0]-dr[1])/2,2)/(2*D*dt),1.,1.e-9);
  close(pow((dz[0]-dz[1])/2,2)/(2*D*dt),1.,1.e-9);
 }
}
'''
report={'native_gpu_execution':False,'backend_checks':{}}
for case in ['weak_turbulence','strong_turbulence']:
 for backend,ext in [('cuda','cu'),('rocm','hip')]:
  src=repo/'src/swarm/diffusion_pos.cu'
  cpp=out/(case+'_'+backend+'.cpp');cpp.write_text('#define GAMEDEV_'+backend.upper()+'\n'+pre+'\n#include "'+str(src)+'"\n'+test)
  exe=out/(case+'_'+backend)
  subprocess.run(['c++','-std=c++17','-O2','-I'+str(out),'-I'+str(lab/'models'/case),'-I'+str(model),'-I'+str(repo/'inc/swarm'),str(cpp),'-o',str(exe)],check=True,env=env)
  subprocess.run([str(exe)],check=True)
  report['backend_checks'][case+'_'+backend]={'stokes_values':[1e-4,.1,1.,10.],'diffusivity_gradient_finite_difference':True,'gas_gradient_finite_difference':True,'drift_and_noise_moments':True,'cartesian_velocity_preserved':True,'unchanged_rng_draw_count':True}
source=(repo/'src/swarm/dyn_rate_calc.cu').read_text()
assert 'diff_R*diffusion.drift_R_per_D' in source
assert 'diff_Z*diffusion.drift_Z_per_D' in source
report['diffusion_drift_in_timestep_controller']=True
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
