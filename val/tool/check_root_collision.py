#!/usr/bin/env python3
"""Host differential checks of actual pair physics and serial kernel paths; no GPU qualification."""
from pathlib import Path
import os,re,subprocess,sys
repo=Path(__file__).resolve().parents[2]
lab=repo/'val/main/swarm_coagulation'
model=repo/'inc/swarm';parent=lab/'common'
constants=lab/'models/weak_turbulence'
out=repo/'val/temp/swarm_coagulation/checks/root_collision';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
if Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk').exists():
 env['CPLUS_INCLUDE_PATH']='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
def function(path,name):
 s=path.read_text();a=s.index('real '+name+' (');b=s.index('{',a);depth=1;j=b+1
 while depth:
  depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]+'\n'
phys=repo/'inc/swarm/param_phys.cuh';grid=repo/'inc/swarm/param_grid.cuh';collision=repo/'inc/swarm/_collision.cuh'
(out/'curand_kernel.h').write_text('struct curandState { unsigned long long value=17; };\n')
stubs=r'''
#define CUDA_CHECK(x) do { (void)(x); } while(0)
#define HIP_CHECK CUDA_CHECK
#define CUDA_KERNEL_CHECK(x) do{}while(0)
#define HIP_KERNEL_CHECK CUDA_KERNEL_CHECK
int cudaMalloc(void**,size_t){return 0;}
int cudaFree(void*){return 0;}
int cudaMemcpy(void*,const void*,size_t,int){return 0;}
int cudaMemset(void*,int,size_t){return 0;}
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
'''
pre=r'''
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <iostream>
using std::isfinite;using std::min;
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define COLLISION
#define COL_DIAGNOSTICS
#define COLLISION_QUERY_LOCAL
#define GAMEDEV_CUDA
struct double3 {double x,y,z;};
#include "const_defs.cuh"
struct dim {int x=0;};dim threadIdx,blockIdx,blockDim{1};
void __syncthreads() {}
unsigned long long __double_as_longlong(double v) {unsigned long long x;memcpy(&x,&v,8);return x;}
double __longlong_as_double(long long x) {double v;memcpy(&v,&x,8);return v;}
template<class T> T atomicCAS(T* p,T a,T b) {T old=*p;if(old==a)*p=b;return old;}
template<class T,class U> T atomicAdd(T* p,U x) {T old=*p;*p+=x;return old;}
template<class T> void atomicMax(T* p,T x) {if(x>*p)*p=x;}
void atomicOr(unsigned int* p,unsigned int x) {*p|=x;}
double curand_uniform_double(curs* p) {p->value=p->value*6364136223846793005ULL+1;return ((p->value>>11)+1.)/9007199254740993.;}
enum KernelType {CONSTANT_KERNEL,LINEAR_KERNEL,PRODUCT_KERNEL,CUSTOM_KERNEL};
int _get_col_idx_old(int i){return i;}
int _get_col_image(int){return 0;}
int _get_col_offset(int i,int k){return i*N_K+k;}
'''
for name in ('_get_cyl_R','_get_cyl_Z'):pre+=function(grid,name)
for name in ('_get_grain_mass','_get_omegaK','_get_hg','_get_eta','_get_gas_strat','_get_sigma_g','_get_cs','_get_alpha','_get_stokes'):pre+=function(phys,name)
for name in ('_get_vrel_b','_get_re_inv_sqrt','_get_vrel_t','_get_vrel_pair'):pre+=function(collision,name)
(out/'_col_cache.cuh').write_text('');(out/'_collision.cuh').write_text('')
# Parse the actual controller and kernels with GPU launch syntax removed.
host_header=out/'_col_chain_host.cuh'
host_header.write_text(re.sub(r'<<<.*?>>>','',(model/'_col_chain.cuh').read_text()))
pre+=stubs+'\n#include "'+str(host_header)+'"\n'
old=parent.joinpath('_col_chain.cuh').read_text()
a=old.index('template <KernelType kernel>');b=old.index('// freeze the partner reservoir',a)
pre+=old[a:b].replace('_get_col_chain_rate','parent_rate')
a=old.index('__global__\nvoid col_chain_run');b=old.index('// aggregate predicted moments',a)
pre+=old[a:b].replace('col_chain_run','parent_chain').replace('_get_col_chain_rate','parent_rate')
test=r'''
void close(double a,double b){assert(std::abs(a-b)<=3.e-13*std::max({1.e-100,std::abs(a),std::abs(b)}));}
int main(){
 // Actual production formulas across positions, grain sizes and turbulent regimes.
 for(double R:{5.,10.,50.})for(double zr:{-.18,0.,.18}) {
  swarm p{};p.position={0,R*AU*sqrt(1+zr*zr),atan2(1.,zr)};
  auto e=cache_query_environment(p);
  for(double si:{1.e-4,1.e-3,.01,.1,1.,10.,100.})for(double sj:{1.e-4,.001,.1,1.,100.}) {
   close(cached_pair_velocity(e,si,sj),_get_vrel_pair(&p,si,sj,0,0,0));
   close(cached_stokes(e,si),_get_stokes(R*AU,R*AU*zr,_get_hg(R*AU),si));
  }
 }
 // Execute the actual old/new kernels serially with all neighbor lanes in thread 0.
 // This checks arithmetic/state/RNG paths, not GPU barriers or concurrency.
 for(double grain:{1.e-4,.1,10.})for(double q:{1.e-7,.01,1.})for(double scale:{0.,.001,1.,40.}) {
  swarm initial[2]{};
  for(auto &p:initial){p.position={0,10*AU,M_PI/2};p.par_numr=1.e20;}
  initial[0].par_size=grain;initial[1].par_size=grain*cbrt(q);
  int ids[1]={0},neighbors[N_K],spatial[2]={0,0};for(int k=0;k<N_K;k++)neighbors[k]=k%2;
  double sizes[2]={initial[0].par_size,initial[1].par_size},nums[2]={1.e20,1.e20},measure[2]={1.e35,1.e35};
  unsigned char active[2]={1,1};query_environment e[2]={cache_query_environment(initial[0]),cache_query_environment(initial[1])};
  cached_rate_moments cache[2];double rates[2]{},first[2]{},second[2]{};
  col_bath_rate(ids,1,rates,first,second,initial,neighbors,measure,active,sizes,nums,1.,e,cache);
  assert(rates[0]>0);double step[1]={scale/rates[0]};
  struct State {swarm p[2];curs rng[2];int error[2]{},events[2]{},unfinished=0,queue[2]{},flag=0;unsigned char complete[2]{};double time[2]{},hazard[2]{},j1[2]{},j2[2]{},jm[2]{};event_work work[2]{};};
  State a{},b{};for(int i=0;i<2;++i)a.p[i]=b.p[i]=initial[i];
  event_work *output_work=nullptr;
#ifdef COL_DIAGNOSTICS
  output_work=b.work;
#endif
  int loops=0;
  do {
   assert(++loops<1000);a.unfinished=b.unfinished=0;
   parent_chain(ids,1,a.p,a.rng,a.error,&a.unfinished,a.time,a.events,a.complete,a.hazard,a.j1,a.j2,a.jm,neighbors,measure,active,sizes,nums,1.,step,spatial,a.queue,&a.flag,a.work,e,cache);
   col_chain_run(ids,1,b.p,b.rng,b.error,&b.unfinished,b.time,b.events,b.complete,b.hazard,b.j1,b.j2,b.jm,neighbors,measure,active,sizes,nums,1.,step,spatial,b.queue,&b.flag,output_work,e,cache);
   assert(a.flag==b.flag && a.flag==0 && a.unfinished==b.unfinished);
   assert(a.events[0]==b.events[0] && a.rng[0].value==b.rng[0].value && a.complete[0]==b.complete[0]);
   close(a.p[0].par_size,b.p[0].par_size);close(a.p[0].par_numr,b.p[0].par_numr);
   close(a.time[0],b.time[0]);close(a.hazard[0],b.hazard[0]);close(a.j1[0],b.j1[0]);close(a.j2[0],b.j2[0]);close(a.jm[0],b.jm[0]);
#ifdef COL_DIAGNOSTICS
   for(int k=0;k<EVENT_CATEGORIES;k++){assert(a.work[0].count[k]==b.work[0].count[k]);close(a.work[0].log_mass[k],b.work[0].log_mass[k]);}
#endif
  }while(a.unfinished);
 }
 std::cout<<"Cached physics and parent/new event/no-event/continuation states and RNG agree in host checks.\n";
}
'''
cpp=out/'check.cpp';cpp.write_text(pre+test);exe=out/'check'
subprocess.run(['c++','-std=c++17','-O2','-I'+str(out),'-I'+str(constants),'-I'+str(model),str(cpp),'-o',str(exe)],check=True,env=env)
subprocess.run([str(exe)],check=True)
# Production omits event bookkeeping but must preserve particle state and RNG.
cpp=out/'check_no_diagnostics.cpp';cpp.write_text((pre+test).replace('#define COL_DIAGNOSTICS',''))
exe=out/'check_no_diagnostics'
subprocess.run(['c++','-std=c++17','-O2','-I'+str(out),'-I'+str(constants),'-I'+str(model),str(cpp),'-o',str(exe)],check=True,env=env)
subprocess.run([str(exe)],check=True)

for backend in ('cuda','rocm'):
 for search in ('kdtree','morton'):
  plan=subprocess.check_output(['make','-Bn','-C',str(repo),'MODEL=swarm_fiducial','GPU_BACKEND='+backend,'COLLISION_SEARCH='+search,'GPU_FLAGS=-DCOLLISION -DMULTISIZE -DCODE_UNIT'],text=True)
  assert str(repo/'src/swarm/swarm_runtime.cu') in plan
print('Root backend routing passed.')
# Type-check the actual host launch arguments for both backends with launch syntax removed.

runtime=(repo/'src/swarm/swarm_runtime.cu').read_text()
start=runtime.index('evolve_local_collisions(')
controller_call=runtime[start:runtime.index(');',start)+2]
driver=r'''
void check_driver() {
 double duration=1.,total_dust_mass=1.,clock_sim=0,clock_dyn=0,dt_col=0;
 int count_col=0;const int col_raw_count=_get_col_raw_count();
 local_workspace local("/tmp/");bool local_geometry_valid=false;col_controller_summary col_summary;
 swarm *dev_particle=nullptr;curs *dev_rngstate=nullptr;
 int *dev_col_spatial=nullptr,*dev_col_neighbor=nullptr,*dev_col_events=nullptr;
 int *dev_col_count=nullptr,*dev_col_binmap=nullptr,*dev_col_error=nullptr,*dev_col_unfinished=nullptr;
 unsigned char *dev_col_active=nullptr,*dev_col_complete=nullptr;
 real *dev_size_old=nullptr,*dev_numr_old=nullptr,*dev_col_time=nullptr,*dev_col_rate=nullptr;
 real *dev_col_hazard=nullptr,*dev_col_jump1_int=nullptr,*dev_col_jump2_int=nullptr,*dev_col_jumpmax_int=nullptr,*dev_col_measure=nullptr;
 col_rate_bin *dev_col_ratebin=nullptr;col_audit_accum *dev_col_audit=nullptr;
'''+controller_call+'\n}\n'
(out/'hiprand').mkdir(exist_ok=True)
(out/'hiprand/hiprand_kernel.h').write_text('struct hiprandState { unsigned long long value=17; };\n')
for backend in ('cuda','rocm'):
 source=pre+driver
 if backend=='rocm':source=source.replace('#define GAMEDEV_CUDA','#define GAMEDEV_ROCM').replace('curand_uniform_double','hiprand_uniform_double')
 cpp=out/(backend+'_syntax.cpp');cpp.write_text(source)
 subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(out),'-I'+str(constants),'-I'+str(model),str(cpp)],check=True,env=env)
 # Also type-check the production controller with diagnostics compiled out.
 source=source.replace('#define COL_DIAGNOSTICS','')
 cpp=out/(backend+'_production_syntax.cpp');cpp.write_text(source)
 subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(out),'-I'+str(constants),'-I'+str(model),str(cpp)],check=True,env=env)
print('CUDA/ROCm launch-argument host-stub syntax checks passed; these do not compile GPU device code.')

# Compile alternate root physics paths; imported interpolation is a signature stub.
extra="""
real _get_loc_x(real x){return x;} real _get_loc_y(real y){return y;} real _get_loc_z(real z){return z;}
real _interp_field(const real* d,real,real y,real){return d[0]*(1.+y/AU);}
real3 _get_cart_vel(const swarm &p,int=0){return p.velocity;}
constexpr real REYNOLDS_0=1.e6;
"""
for mode in ('analytic','code_unit','imported','imported_2d','imported_code','constant_st','query_2d'):
 for backend in ('cuda','rocm'):
  source=pre.split('template <KernelType kernel>')[0]+driver
  source=source.replace('real _get_grain_mass (',extra+'\nreal _get_grain_mass (',1)
  if mode in ('analytic','code_unit','imported'):source=source.replace('#define COLLISION_QUERY_LOCAL','')
  if mode in ('code_unit','imported_code'):source='#define CODE_UNIT\n'+source
  if mode.startswith('imported'):source='#define IMPORTGAS\n'+source;source=source.replace('swarm *dev_particle=nullptr;curs *dev_rngstate=nullptr;','swarm *dev_particle=nullptr;curs *dev_rngstate=nullptr;real *dev_gas_dens=nullptr;')
  if mode=='constant_st':source='#define CONST_ST\n'+source
  config=out/mode;config.mkdir(exist_ok=True)
  values=(constants/'const_defs.cuh').read_text().replace('../../common/const_defs.cuh',str(lab/'common/const_defs.cuh'))
  if mode in ('query_2d','imported_2d'):values=(lab/'common/const_defs.cuh').read_text().replace('N_Z = 64','N_Z = 1');values='#define COAG_ALPHA 1.e-4\n#define COAG_OUTPUTS 250\n'+values
  (config/'const_defs.cuh').write_text(values)
  if backend=='rocm':source=source.replace('#define GAMEDEV_CUDA','#define GAMEDEV_ROCM').replace('curand_uniform_double','hiprand_uniform_double')
  cpp=out/(mode+'_'+backend+'.cpp');cpp.write_text(source)
  subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(out),'-I'+str(config),'-I'+str(model),str(cpp)],check=True,env=env)
  if backend=='cuda':
   physics=source.split('#include "'+str(host_header)+'"')[0]
   physics+=r'''
int main(){
 swarm p[2]{};p[0].position={.1,10*AU,1.4};p[1].position={2.,30*AU,.9};
 real density[1]={1.e-12};
 auto speed=[&](){return _get_vrel_pair(p,.01,.1,0,1,2
 #ifdef IMPORTGAS
 ,density
 #endif
 );};
 real a=speed();assert(isfinite(a)&&a>0.);
 p[1].position={-1.,50*AU,2.};p[0].velocity={1.e20,-1.e20,1.e20};p[1].velocity={-1.e20,1.e20,-1.e20};
 assert(speed()==a); // Neither partner environment nor resolved velocities enter.
 p[0].position.y=20*AU;assert(speed()!=a);
 #ifdef IMPORTGAS
 a=speed();density[0]*=2.;assert(speed()!=a);
 #endif
}
'''
   numerical=out/(mode+'_local.cpp');numerical.write_text(physics);binary=out/(mode+'_local')
   subprocess.run(['c++','-std=c++17','-O2','-I'+str(out),'-I'+str(config),'-I'+str(model),str(numerical),'-o',str(binary)],check=True,env=env)
   subprocess.run([str(binary)],check=True)
print('Query-local numerical checks and CUDA/ROCm host-stub syntax passed for analytic, code-unit, imported 2D/3D, imported code-unit, constant-St and 2D paths.')
# Controller records are per group; launch counts are per wave, not per record.
import ast,json,math,hashlib
for backend in ('cuda','rocm'):
 path=repo/'val'/backend/'swarm/test_common/run_chain.py'
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_controller')
 scope={'Path':Path,'json':json,'math':math,'digest':lambda p:hashlib.sha256(p.read_bytes()).hexdigest()}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
 d={k:1. for k in ('minimum_duration','maximum_duration','minimum_limit_scale','maximum_f','maximum_e','maximum_touched','maximum_events','maximum_g','maximum_g_upper','maximum_d_bath')}
 record={k:1. for k in ('duration','limit_before','limit_after','max_f','max_e','max_touched','max_events','max_g','max_g_upper','d_bath')}
 record.update(operator=1,bath=1,group=0,merged_bins=1,persistent_overshoot=False)
 d.update(schema=2,operator_count=1,bath_count=8,wave_count=2,continuation_launches=2,persistent_overshoots=0,baths=[record]*8)
 f=out/'controller_check.json';f.write_text(json.dumps(d));assert scope['validate_controller'](f,False)['passed'];assert not scope['validate_controller'](f,True)['passed']
 d['continuation_launches']=3;f.write_text(json.dumps(d));assert scope['validate_controller'](f,True)['passed']
 d['continuation_launches']=1;f.write_text(json.dumps(d));assert not scope['validate_controller'](f,False)['passed']
print('Local group/wave diagnostic validation passed.')
