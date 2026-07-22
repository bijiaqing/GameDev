#ifndef HELPERS_COLLISION_CUH
#define HELPERS_COLLISION_CUH

#ifdef COLLISION

#include <cassert>      // for assert

#include <const.cuh>
#include <helpers_paramgrid.cuh>
#include <helpers_interpval.cuh>
#include <helpers_paramphys.cuh>

#include "cukd/knn.h"   // for cukd::cct::knn, cukd::HeapCandidateList

using candidatelist = cukd::HeapCandidateList<N_K>;

enum KernelType { 
    CONSTANT_KERNEL = 0, 
    LINEAR_KERNEL   = 1, 
    PRODUCT_KERNEL  = 2,
    CUSTOM_KERNEL   = 3,
};

// =========================================================================================================================
// Collision Helper Functions (Orders 0-5)
// Purpose: Relative velocity components and collision kernel calculations
// =========================================================================================================================

// Order: O1 | Dependencies: _get_eta [O0], _get_omegaK [O0]
// Purpose: Radial drift-induced relative velocity
__device__ __forceinline__
real _get_vrel_x (real R, real St_i, real St_j, real h_g)
{
    real v_drift = -_get_eta(R, 0.0, h_g)*_get_omegaK(R)*R;
    real vrel_x = v_drift*(1.0 / (1.0 + St_i*St_i) - 1.0 / (1.0 + St_j*St_j));
    
    return abs(vrel_x);
}

// Order: O0 | Dependencies: None (direct velocity difference)
// Purpose: Vertical velocity difference
__device__ __forceinline__
real _get_vrel_y (real vy_i, real vy_j)
{
    return abs(vy_i - vy_j);
}

// Order: O3 | Dependencies: _get_hd [O2], _get_omegaK [O0]
// Purpose: Vertical settling-induced relative velocity
__device__ __forceinline__
real _get_vrel_z (real R_i, real R_j, real St_i, real St_j)
{
    real h_di = _get_hd(R_i, St_i);
    real h_dj = _get_hd(R_j, St_j);

    real R = 0.5*(R_i + R_j);

    return abs(min(St_i, 0.5)*h_di - min(St_j, 0.5)*h_dj)*_get_omegaK(R)*R;
}

#ifndef CODE_UNIT
// Order: O2 | Dependencies: _get_cs [O1], _get_grain_mass [O0]
// Flags: Only compiled in cgs units (not CODE_UNIT)
// Purpose: Brownian motion-induced relative velocity
__device__ __forceinline__
real _get_vrel_b (real R, real s_i, real s_j, real h_g)
{
    // Brownian motion-induced relative velocity v = sqrt(8*k_B*T*(m_i+m_j) / (pi*m_i*m_j))
    // here we take c_s^2 = k_B*T / mmw_gas
    
    real c_s = _get_cs(R, h_g);

    real m_i = _get_grain_mass(s_i);
    real m_j = _get_grain_mass(s_j);

    real vrel_b = sqrt(8.0*c_s*c_s*M_MOL*(m_i + m_j) / (M_PI*m_i*m_j));

    return min(vrel_b, c_s);
}
#endif // NOT CODE_UNIT

// Order: O2 | Dependencies: _get_sigma_g [O0], _get_alpha [O1]
// Purpose: Calculate inverse sqrt of Reynolds number for turbulent relative velocity
__device__ __forceinline__
real _get_ReInvSqrt (real R, real alpha)
{
    real Re = 1.0;
    real sigma = _get_sigma_g(R);
    
    #ifndef CODE_UNIT
    Re = 0.5*alpha*sigma*X_SEC / M_MOL;
    #else  // CODE_UNIT
    real alpha_0 = _get_alpha(R_0, ASPR_0);
    Re = RE_0*(alpha / alpha_0)*(sigma / SIGMA_0);
    #endif // NOT CODE_UNIT

    return 1.0 / sqrt(Re);
}

// Order: O3 | Dependencies: _get_cs [O1], _get_alpha [O1], _get_ReInvSqrt [O2]
// Purpose: Turbulence-induced relative velocity (Ormel & Cuzzi 2007)
__device__ __forceinline__
real _get_vrel_t (real R, real St_i, real St_j, real h_g)
{
    // turbulence-induced relative velocity based on Ormel & Cuzzi (2007), A&A, 466, 413
    // part of code adapted from DustPy (Stammler & Birnstiel 2022, ApJ, 935, 35), also see:
    // https://github.com/stammler/dustpy/blob/48e6c05b2b9c2a91ca35a0945f3138dc3aa34685/dustpy/std/dust.f90#L1402
    // important concepts mentioned in OC07 and used (but not necessarily coded) here include:
    // (0)  Class I  eddies (t_k > t_stop): particles follow eddy motions systematically
    //      Class II eddies (t_k < t_stop): eddies fluctuate too fast, act as random kicks to particles
    // (1)  t_large: turnover timescale of the largest  eddies (integral scale)
    //      t_large = omega_K^(-1)                                          (page 413, section 2)
    // (2)  t_small: turnover timescale of the smallest eddies (Kolmogorov scale)
    //      t_small = t_eta = Re^(-1/2)*t_large                             (page 414, section 2)
    // (3)  t_k: turnover timescale of arbitrary eddy k with spacial scale l = 1 / k and velocity V(k)
    //      t_k = l / V(k) = (k*V(k))^(-1)                                  (page 413, section 2)
    // (4)  t_cross: eddy crossing timescale due to particle-eddy relative velocity
    //      t_cross = l / V_rel = (k*V_rel(k))^(-1)                         (page 414, section 2)
    // (5)  t_stop: particle stopping time (friction timescale)
    //      t_stop = St*t_large = St*omega_K^(-1)                           (page 413, section 2)
    // (6)  t_star: boundary eddy turnover time separating Class I and Class II eddies
    //      t_star = 1.6*t_stop for St << 1                                 (eq. 21d)
    //      t_star = (t_stop^(-1) - t_cross^(-1))^(-1)                      (eq. 3)
    // (7)  v_large: turbulent velocity at the largest  eddy (integral scale)
    //      v_large = c_s*alpha^(1/2)                                       (page 416, section 3.3)
    // (8)  v_small: turbulent velocity at the smallest eddy (Kolmogorov scale)
    //      v_small = Re^(-1/4)*v_large                                     (page 417, section 3.4.1)
    
    real c_s = _get_cs(R, h_g);
    real alpha = _get_alpha(R, h_g);
    real ReInvSqrt = _get_ReInvSqrt(R, alpha);

    // comes from normalizing the power spectrum                            (page 415, section 3.2)
    real v_gas2 = 1.5*alpha*c_s*c_s;

    real St_large, St_small, eps;
    
    if (St_i >= St_j) 
    {
        St_large = St_i;
        St_small = St_j;
    } 
    else 
    {
        St_large = St_j;
        St_small = St_i;
    }
    
    eps = St_small / St_large;
    
    // y_a = t_star / t_stop = 1.6 is the solution to y_star when St << 1
    // y_s is an empirical polynomial fit to the exact solution of y_star (eq. 21d)
    real y_a = 1.6;
    real y_s = 1.6015125;
    
    // taken from DustPy
    y_s += -0.63119577*St_large;
    y_s +=  0.32938936*St_large*St_large;
    y_s += -0.29847604*St_large*St_large*St_large;

    real vrel_sq = 0.0;
    
    if (St_large < 0.2*ReInvSqrt) 
    {
        // regime 1: very small particles (t_stop_large << t_small) following eq. 27
        
        vrel_sq = v_gas2*(St_large - St_small)*(St_large - St_small) / ReInvSqrt;
    }
    else if (St_large < ReInvSqrt / y_a)
    {
        // regime 2: transition near t_small boundary (t_stop_large ~ t_small) following eq. 26
        
        vrel_sq = v_gas2*(St_large - St_small) / (St_large + St_small);
        vrel_sq *= (St_large / (1.0 + ReInvSqrt / St_large) - St_small / (1.0 + ReInvSqrt / St_small));
    }
    else if (St_large < 5.0*ReInvSqrt) 
    {
        // regime 3: intermediate coupling (t_small < t_stop_large < 5*t_small)
        
        real coeff = 0.0;
        // coefficient of delta_VI^2  following eq. 17
        coeff  = (St_large - St_small) / (St_large + St_small);
        coeff *= (St_large / (1.0 + y_a) - St_small*St_small / (St_small + y_a*St_large));
        // coefficient of delta_VII^2 following eq. 18
        coeff += 2.0*(y_a*St_large - ReInvSqrt) + St_large / (1.0 + y_a);
        coeff -= St_large*St_large / (St_large + ReInvSqrt);
        coeff += St_small*St_small / (y_a*St_large + St_small);
        coeff -= St_small*St_small / (St_small + ReInvSqrt);
        
        vrel_sq = v_gas2*coeff;
    }
    else if (St_large < 0.2*ReInvSqrt)
    {
        // regime 4: fully intermediate regime (5t_small < t_stop_large < 0.2t_large) following eq. 28
        
        vrel_sq = v_gas2*St_large;
        vrel_sq *= (2.0*y_a - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_a) + eps*eps*eps / (y_a + eps)));
    }
    else if (St_large < 1.0)
    {
        // regime 5: transition near t_large boundary (0.2t_large < t_stop_large < t_large) 
        // following eq. 28, but uses the empirical y_s fit instead of the fixed y_a = 1.6
        
        vrel_sq = v_gas2*St_large;
        vrel_sq *= (2.0*y_s - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_s) + eps*eps*eps / (y_s + eps)));
    }
    else
    {
        // regime 6: heavy particles (t_stop_large >= t_large) following eq. 29
        
        vrel_sq = v_gas2*(1.0 / (1.0 + St_large) + 1.0 / (1.0 + St_small));
    }

    if (vrel_sq < 0.0)
    {
        printf("ERROR: negative vrel_sq in _get_vrel_t\n");
        assert(false);
    }
 
    return sqrt(vrel_sq);
}

// Order: O4 | Dependencies: _get_hg [O0], _get_St [O3], _get_vrel_x [O1], _get_vrel_y [O0], _get_vrel_z [O3], _get_vrel_b [O2], _get_vrel_t [O3]
// Flags: IMPORTGAS enables gas density for Stokes number calculation
// Purpose: Calculate total relative velocity from all sources (drift, settling, Brownian, turbulence)
__device__ __forceinline__
real _get_vrel (const swarm *dev_particle, int idx_old_i, int idx_old_j
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
)
{
    #ifdef IMPORTGAS
    real x_i = dev_particle[idx_old_i].position.x;
    real x_j = dev_particle[idx_old_j].position.x;
    #endif // IMPORTGAS
    
    real y_i = dev_particle[idx_old_i].position.y;
    real y_j = dev_particle[idx_old_j].position.y;
    
    real z_i = dev_particle[idx_old_i].position.z;
    real z_j = dev_particle[idx_old_j].position.z;
    
    real vy_i = dev_particle[idx_old_i].velocity.y;
    real vy_j = dev_particle[idx_old_j].velocity.y;
    
    #ifdef MULTISIZE
    real s_i = dev_particle[idx_old_i].par_size;
    real s_j = dev_particle[idx_old_j].par_size;
    #else
    real s_i = S_0;
    real s_j = S_0;
    #endif // MULTISIZE
    
    real R_i = y_i*sin(z_i);
    real R_j = y_j*sin(z_j);
    
    real Z_i = y_i*cos(z_i);
    real Z_j = y_j*cos(z_j);
    
    real R = 0.5*(R_i + R_j);

    real h_g = _get_hg(R);

    real St_i = _get_St(R_i, Z_i, s_i, h_g
        #ifdef IMPORTGAS
        , x_i, y_i, z_i, dev_gasdens
        #endif
    );
    real St_j = _get_St(R_j, Z_j, s_j, h_g
        #ifdef IMPORTGAS
        , x_j, y_j, z_j, dev_gasdens
        #endif
    );

    real vrel_sq = 0.0;
    
    // Calculate velocity components with pre-calculated values
    real vrel_x = _get_vrel_x(R, St_i, St_j, h_g);
    vrel_sq += vrel_x*vrel_x;
    
    real vrel_y = _get_vrel_y(vy_i, vy_j);
    vrel_sq += vrel_y*vrel_y;

    real vrel_z = _get_vrel_z(R_i, R_j, St_i, St_j);
    vrel_sq += vrel_z*vrel_z;

    // brownian motion will not be considered when code units are used
    // for turbulent motion, Reynolds number will be prescribed when code units are used

    #ifndef CODE_UNIT
    real vrel_b = _get_vrel_b(R, s_i, s_j, h_g);
    vrel_sq += vrel_b*vrel_b; // Brownian motion
    #endif // NOT CODE_UNIT

    real vrel_t = _get_vrel_t(R, St_i, St_j, h_g);
    vrel_sq += vrel_t*vrel_t; // turbulence
    
    return sqrt(vrel_sq);
}

// =========================================================================================================================
// Collision Rate Calculation (Order 5)
// Purpose: Calculate collision rate between particle pair using coagulation kernel
// =========================================================================================================================

// Order: O5 | Dependencies: _get_grain_mass [O0], _get_vrel [O4]
// Purpose: Compute collision rate λ_ij = N_j * K_ij for particle pair
template <KernelType kernel> __device__ __forceinline__
real _get_col_rate_ij (const swarm *dev_particle, int idx_old_i, int idx_old_j
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
)
{
    // text mainly from Drazkowska et al. 2013:
    // we assume that a limited number n representative particles represent all N physical particles
    // each representative particle i describes a swarm of N_i identical physical particles
    // as n << N, we only need to consider the collisions between representative and non-representative particles
    // the probability of a collision between particles i and j is determined as 
    // lambda_ij = N_j * K_ij / V, where K_ij is the coagulation kernel and V is the volume of the cell
    
    real lam_ij = LAMBDA_0; // dimension issue saved for later
    #ifdef MULTISIZE
    real numr_j = dev_particle[idx_old_j].par_numr;
    #else
    real numr_j = M_D / N_P;
    #endif // MULTISIZE

    if constexpr (kernel == CONSTANT_KERNEL)
    {
        lam_ij *= 1.0; // by definition
        
        return lam_ij*numr_j;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        #ifdef MULTISIZE
        real s_i = dev_particle[idx_old_i].par_size;
        real s_j = dev_particle[idx_old_j].par_size;
        #else
        real s_i = S_0;
        real s_j = S_0;
        #endif // MULTISIZE

        // m_i + m_j
        lam_ij *= 0.5*(_get_grain_mass(s_i) + _get_grain_mass(s_j));
        
        return lam_ij*numr_j;
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        #ifdef MULTISIZE
        real s_i = dev_particle[idx_old_i].par_size;
        real s_j = dev_particle[idx_old_j].par_size;
        #else
        real s_i = S_0;
        real s_j = S_0;
        #endif // MULTISIZE
        
        // m_i * m_j
        lam_ij *= _get_grain_mass(s_i)*_get_grain_mass(s_j);
        
        return lam_ij*numr_j;
    }
    else if constexpr (kernel == CUSTOM_KERNEL)
    {
        // K_ij should be v_ik * sigma_ik, 
        // where v_ik is the relative velocity between particle i and j
        // and sigma_ik is the collisional cross section between particle i and j
        
        #ifdef MULTISIZE
        real s_i = dev_particle[idx_old_i].par_size;
        real s_j = dev_particle[idx_old_j].par_size;
        #else
        real s_i = S_0;
        real s_j = S_0;
        #endif // MULTISIZE
        
        real v_rel_ij = _get_vrel(dev_particle, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gasdens
            #endif
        );
        real sigma_ij = M_PI*(s_i + s_j)*(s_i + s_j) / 4.0;
        
        lam_ij *= v_rel_ij*sigma_ij;
        
        return lam_ij*numr_j;
    }
    else
    {
        if (threadIdx.x == 0 && blockIdx.x == 0) // KernelType is a compile-time constant
        {
            printf("ERROR: Invalid COAG_KERNEL value = %d \n", static_cast<int>(kernel));
        }
        
        assert(false);
        return 0.0; // unreachable, but prevents compiler warning
    }
}

#endif // COLLISION

// =========================================================================================================================

#endif // NOT HELPERS_COLLISION_CUH
