// Shared query environment, refresh limits and collision outcomes.
#ifndef LAB_QUERY_ENVIRONMENT_CUH
#define LAB_QUERY_ENVIRONMENT_CUH
#if !defined(COLLISION_QUERY_LOCAL) || defined(IMPORTGAS) || defined(CODE_UNIT) || defined(CONST_ST)
#error This cache is for the analytic query-local benchmark only
#endif
static_assert(N_Z>1 && COAG_KERNEL==CUSTOM_KERNEL);
struct query_environment {
    real Z, omega, vn, radial, strat, cs, re_inv_sqrt, vg_sq;
};
// Positions and gas are fixed throughout one collision half-operator.
__device__ __forceinline__ query_environment cache_query_environment(const swarm &p) {
    real R=_get_cyl_R(p.position.y,p.position.z);
    real Z=_get_cyl_Z(p.position.y,p.position.z),h=_get_hg(R);
    real omega=_get_omegaK(R),cs=_get_cs(R,h),alpha=_get_alpha(R,h);
    real strat=_get_gas_strat(R,Z,h);
    return {Z,omega,-_get_eta(R,Z,h)*R*omega,pow(R/R_0,IDX_P),strat,cs,
        _get_re_inv_sqrt(R,alpha,_get_sigma_g(R)*strat),1.5*alpha*cs*cs};
}
__device__ __forceinline__ real cached_stokes(const query_environment &e,real size) {
    real st=STOKES_0*(size/S_0); st/=e.radial; st/=e.strat; return st;
}
// Identical regime algebra to _get_vrel_t; only gas coefficients are precomputed.
__device__ __forceinline__ real cached_turbulence(const query_environment &e,real stokes_i,real stokes_j) {
    real re_inv_sqrt=e.re_inv_sqrt,vg_sq=e.vg_sq;
    real stokes_large, stokes_small, eps;
    
    if (stokes_i >= stokes_j)
    {
        stokes_large = stokes_i;
        stokes_small = stokes_j;
    } 
    else 
    {
        stokes_large = stokes_j;
        stokes_small = stokes_i;
    }
    
    eps = stokes_small / stokes_large;
    
    // y_a = t_star / t_stop = 1.6 is the solution to y_star when St << 1
    // y_s is an empirical polynomial fit to the exact solution of y_star (eq. 21d)
    real y_a = 1.6;
    real y_s = 1.6015125;
    
    // taken from DustPy
    y_s += -0.63119577*stokes_large;
    y_s +=  0.32938936*stokes_large*stokes_large;
    y_s += -0.29847604*stokes_large*stokes_large*stokes_large;

    real vrel_sq = 0.0;
    
    if (stokes_large < 0.2*re_inv_sqrt)
    {
        // regime 1: very small particles (t_stop_large << t_small) following eq. 27
        
        vrel_sq = vg_sq*(stokes_large - stokes_small)*(stokes_large - stokes_small) / re_inv_sqrt;
    }
    else if (stokes_large < re_inv_sqrt / y_a)
    {
        // regime 2: transition near t_small boundary (t_stop_large ~ t_small) following eq. 26
        
        vrel_sq = vg_sq*(stokes_large - stokes_small) / (stokes_large + stokes_small);
        vrel_sq *= (stokes_large / (1.0 + re_inv_sqrt / stokes_large) - stokes_small / (1.0 + re_inv_sqrt / stokes_small));
    }
    else if (stokes_large < 5.0*re_inv_sqrt)
    {
        // regime 3: intermediate coupling (t_small < t_stop_large < 5*t_small)
        
        real coeff = 0.0;
        // coefficient of delta_VI^2  following eq. 17
        coeff  = (stokes_large - stokes_small) / (stokes_large + stokes_small);
        coeff *= (stokes_large / (1.0 + y_a) - stokes_small*stokes_small / (stokes_small + y_a*stokes_large));
        // coefficient of delta_VII^2 following eq. 18
        coeff += 2.0*(y_a*stokes_large - re_inv_sqrt) + stokes_large / (1.0 + y_a);
        coeff -= stokes_large*stokes_large / (stokes_large + re_inv_sqrt);
        coeff += stokes_small*stokes_small / (y_a*stokes_large + stokes_small);
        coeff -= stokes_small*stokes_small / (stokes_small + re_inv_sqrt);
        
        vrel_sq = vg_sq*coeff;
    }
    else if (stokes_large < 0.2)
    {
        // regime 4: fully intermediate regime (5t_small < t_stop_large < 0.2t_large) following eq. 28
        
        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_a - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_a) + eps*eps*eps / (y_a + eps)));
    }
    else if (stokes_large < 1.0)
    {
        // regime 5: transition near t_large boundary (0.2t_large < t_stop_large < t_large) 
        // following eq. 28, but uses the empirical y_s fit instead of the fixed y_a = 1.6
        
        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_s - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_s) + eps*eps*eps / (y_s + eps)));
    }
    else
    {
        // regime 6: heavy particles (t_stop_large >= t_large) following eq. 29
        
        vrel_sq = vg_sq*(1.0 / (1.0 + stokes_large) + 1.0 / (1.0 + stokes_small));
    }

    if (vrel_sq < 0.0)
    {
        printf("ERROR: negative vrel_sq in _get_vrel_t\n");
        assert(false);
    }

    return sqrt(vrel_sq);
}


__device__ __forceinline__ real cached_pair_velocity(const query_environment &e,real size_i,real size_j) {
    real si=cached_stokes(e,size_i),sj=cached_stokes(e,size_j);
    real fi=1.0/(1.0+si*si),fj=1.0/(1.0+sj*sj);
    real dvr=2.0*e.vn*(si*fi-sj*fj),dvphi=e.vn*(fi-fj);
    real dvz=e.Z*e.omega*(fmin(si,0.5)-fmin(sj,0.5));
    real vt=cached_turbulence(e,si,sj);
    real mi=_get_grain_mass(size_i),mj=_get_grain_mass(size_j);
    real vb=fmin(sqrt(8.0*e.cs*e.cs*M_MOL*(mi+mj)/(M_PI*mi*mj)),e.cs);
    return sqrt(dvr*dvr+dvphi*dvphi+dvz*dvz+vt*vt+vb*vb);
}
struct cached_rate_moments { real rate,first,second,maximum; };
#endif

#ifndef CHANGE_REFRESH_LIMIT_HPP
#define CHANGE_REFRESH_LIMIT_HPP
#include <algorithm>
#include <cmath>
#include <stdexcept>

struct change_bound { double duration; int reason; };

// A = mass-weighted mean absolute log-size jump rate.
// B = mass-weighted second log-size jump moment rate (not variance of the mean).
// Frozen-rate estimates: mean accumulated absolute change = h*A;
// compound-Poisson fluctuation scale = sqrt(h*B). Bound each by epsilon.
inline change_bound change_limit(double A, double B, double epsilon, double horizon) {
    if (!std::isfinite(A) || A<0 || !std::isfinite(B) || B<0
        || !std::isfinite(epsilon) || !(epsilon>0)
        || !std::isfinite(horizon) || !(horizon>0))
        throw std::runtime_error("invalid change-based refresh inputs");
    change_bound result{horizon,0};
    if (A>0 && epsilon/A<result.duration) result={epsilon/A,1};
    if (B>0 && epsilon*epsilon/B<result.duration) result={epsilon*epsilon/B,2};
    if (!(result.duration>0)) throw std::runtime_error("change-based timestep underflow");
    return result;
}
#endif

#ifndef LAB_EVENT_WORK_CUH
#define LAB_EVENT_WORK_CUH
constexpr int EVENT_CATEGORIES=7;
struct event_work { unsigned long long count[EVENT_CATEGORIES]; real log_mass[EVENT_CATEGORIES]; };
__host__ __device__ inline void record_event_work(event_work &work, int category, real log_mass) {
    ++work.count[category];
    work.log_mass[category]+=log_mass;
}
#endif

#ifndef LAB_GROUPED_STICKING_CUH
#define LAB_GROUPED_STICKING_CUH
// Only tiny sticking projectiles; each packet adds at most 0.01% target mass.
__host__ __device__ inline real sticking_packet(real q, bool fragmentation) {
    return !fragmentation && q>0.0 && q<=1.e-6
        ? fmax(1.0,floor(1.e-4/q)) : 1.0;
}
__host__ __device__ inline real sticking_mass_ratio(real size_i, real size_j) {
    real ratio=size_j/size_i;
    return ratio*ratio*ratio;
}
// ponytail: packets preserve frozen-state mean mass growth but inflate variance;
// lower the 1e-4 packet bound if distribution comparisons show a bias.
#endif

#ifndef LAB_EROSION_OUTCOME_CUH
#define LAB_EROSION_OUTCOME_CUH


// Return sampled-rate / physical-rate and conditional absolute log-diameter moments.
// Erosion is the superposition of remnant packets and ungrouped debris transitions.
__host__ __device__ inline real erosion_outcome_moments(real si, real sj,
    bool high_speed, real &mean, real &second, real &maximum) {
    real q=sticking_mass_ratio(si,sj);
    real G=sticking_packet(q,false);
    if (!high_speed) {
        mean=log1p(G*q)/3.0; second=mean*mean; maximum=mean;
        return 1.0/G;
    }
    if (q<=0.1) {
        real remnant=(1.0-q)/G, debris=q, factor=remnant+debris;
        real jr=-log1p(-G*q)/3.0, jd=-log(q)/3.0;
        mean=(remnant*jr+debris*jd)/factor;
        second=(remnant*jr*jr+debris*jd*jd)/factor;
        maximum=fmax(jr,jd);
        return factor;
    }
    // Fragment diameter: [sqrt(s_min)+U*(sqrt(si)-sqrt(s_min))]^2.
    // For L=log(si/s_min)/2, integrate -2 log(y) over y in [exp(-L),1].
    real L=0.5*log(si/INIT_SMIN);
    if (L<1.e-3) {
        // Series avoid cancellation when the target is near the monomer floor.
        mean=L-L*L/6.0+L*L*L*L/360.0;
        second=L*L*(4.0/3.0-L/3.0+L*L/90.0+L*L*L/180.0);
    } else {
        real tail=exp(-L)/(-expm1(-L));
        mean=2.0*(1.0-L*tail);
        second=8.0-(4.0*L*L+8.0*L)*tail;
    }
    maximum=2.0*L;
    return 1.0;
}

// Categories: four sticking q bins, fragmentation, remnant erosion, debris erosion.
// u is used only for high-speed events; the caller supplies one independent draw.
__host__ __device__ inline real sample_erosion_outcome(real si, real sj,
    bool high_speed, real u, int &category, real &log_mass) {
    real q=sticking_mass_ratio(si,sj);
    real G=sticking_packet(q,false);
    if (!high_speed) {
        category=q<=1.e-6?0:q<=1.e-4?1:q<=1.e-2?2:3;
        log_mass=log1p(G*q);
        return cbrt(si*si*si+G*sj*sj*sj);
    }
    if (q<=0.1) {
        real factor=(1.0-q)/G+q;
        if (u<q/factor) {
            category=6; log_mass=log(q);
            return sj;
        }
        category=5; log_mass=log1p(-G*q);
        return si*cbrt(1.0-G*q);
    }
    category=4;
    real lower=sqrt(INIT_SMIN);
    real root=lower+u*(sqrt(si)-lower);
    real size=fmin(si,fmax(INIT_SMIN,root*root));
    log_mass=3.0*log(size/si);
    return size;
}
#endif

