// CPU setup/initializer checks only; no GPU transport or collision execution.
#include <algorithm>
#include <cassert>
#include <iostream>
using std::min;
using std::max;
struct double3 { double x,y,z; };
inline double atomicAdd(double *p,double value) { double old=*p; *p+=value; return old; }
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
struct index3 { int x; };
index3 threadIdx{0}, blockIdx{0}, blockDim{64};
#include "initial_profile.hpp"
#include "../../../../src/comm/swarm/particle_init.cu"
int _get_col_image_shift(int image) { return image==1 ? -1 : (image==2 ? 1 : 0); }
#include <collision_velocity_check.hpp>
int main()
{
    assert(N_X==1 && N_Z>1 && COAG_KERNEL==3 && N_P==1048576);
    assert(SAVE_MAX==(ALPHA==1e-3 ? 125 : 250) && DT_OUT/YEAR==100.0);
    assert(SAVE_MAX*DT_OUT/YEAR==(ALPHA==1e-3 ? 12500.0 : 25000.0));
    assert(INIT_SMIN/2==5e-5 && INIT_SMAX/2==1e-4 && V_FRAG==100.0);
    assert(std::abs(STOKES_0 - M_PI*RHO_0*(S_0/2)/(2*SIGMA_0))<1e-20);
    double cs = _get_cs(AU,ASPR_0);
    assert(std::abs(cs*cs*M_MOL/1.380649e-16/209.7926358245702-1)<1e-13);
    double integral = initial_integral();
    assert(std::abs(initial_integral(256)/integral-1)<1e-7);
    double mean_ref = initial_integral(512,1)/integral;
    double var_ref = initial_integral(512,2)/integral-mean_ref*mean_ref;
    std::mt19937 rng(0);
    const int count=100000;
    double sum=0, mean_cos=0;
    for (int i=0;i<count;++i) {
        double r,theta;
        sample_initial_position(rng,r,theta);
        assert(r>=Y_MIN && r<=Y_MAX && theta>=Z_MIN && theta<=Z_MAX);
        sum+=r/AU; mean_cos+=std::cos(theta);
        // Verify the sampled density against the actual production gas stratification.
        double R=r*std::sin(theta),Z=r*std::cos(theta),h=_get_hg(R);
        double rho_shape=std::pow(R/AU,IDX_P-0.5*(IDX_Q+3))*_get_gas_strat(R,Z,h);
        double weight=rho_shape*(r/AU)*(r/AU)*std::sin(theta);
        assert(std::abs(weight/initial_weight(r/AU,theta)-1)<1e-11);
    }
    assert(std::abs(sum/count-mean_ref)<5*std::sqrt(var_ref/count));
    assert(std::abs(mean_cos/count)<0.001);
    double mass=initial_dust_mass();
    double x=0,r=10*AU,theta=M_PI/2-0.05,size=INIT_SMAX,bank=mass;
    swarm p{};
    particle_init(&p,&x,&r,&theta,&size,&bank,1,1.0);
    assert(p.position.y==r && p.par_size==size);
    assert(std::abs(p.par_numr*_get_grain_mass(size)*N_P/mass-1)<1e-14);
    assert(std::isfinite(p.velocity.x) && std::isfinite(p.velocity.y) && std::isfinite(p.velocity.z));
    // Reconstruct cylindrical vertical velocity: above the midplane it must settle down.
    double vZ=p.velocity.y*std::cos(theta)-p.velocity.z/r*std::sin(theta);
    assert(vZ<0);
    swarm pair[2]={p,p};
    double equal=_get_vrel_pair(pair,size,size,0,1,0);
    double brownian=_get_vrel_b(r*std::sin(theta),size,size,_get_hg(r*std::sin(theta)));
    assert(std::abs(equal/brownian-1)<1e-12); // tiny equal grains: no differential drift/turbulence
    double speed=_get_vrel_pair(pair,size,0.5*size,0,1,0);
    pair[1].position={1.0,40*AU,M_PI/2+0.15};
    pair[0].velocity={1e30,-1e30,1e30};
    pair[1].velocity={-1e30,1e30,-1e30};
    assert(_get_vrel_pair(pair,size,0.5*size,0,1,2)==speed);
    double R=r*std::sin(theta), h=_get_hg(R);
    double strat=_get_gas_strat(R,r*std::cos(theta),h);
    assert(std::abs(_get_re_inv_sqrt(R,ALPHA,_get_sigma_g(R)*strat)
                   /_get_re_inv_sqrt(R,ALPHA,_get_sigma_g(R))*std::sqrt(strat)-1)<1e-12);
    // At the midplane, isolate drift against the analytic St=1 and St=0.1 result.
    pair[0].position={0,10*AU,M_PI/2};
    R=10*AU; h=_get_hg(R);
    double unit_st_size=S_0/_get_stokes(R,0,h,S_0);
    double total=_get_vrel_pair(pair,unit_st_size,0.1*unit_st_size,0,1,0);
    double vb=_get_vrel_b(R,unit_st_size,0.1*unit_st_size,h);
    double vt=_get_vrel_t(R,1,0.1,h,_get_sigma_g(R));
    double vn=-_get_eta(R,0,h)*R*_get_omegaK(R);
    double radial=2*vn*(0.5-0.1/1.01), azimuthal=vn*(0.5-1/1.01);
    double expected=radial*radial+azimuthal*azimuthal+vb*vb+vt*vt;
    assert(std::abs(total*total/expected-1)<1e-12);
    assert(azimuthal*azimuthal>0.01*expected);
    std::cout << "{\"alpha\":" << ALPHA << ",\"dust_mass_g\":" << mass
              << ",\"initialization_passed\":true}\n";
}
