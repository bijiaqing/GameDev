#ifndef PARAM_PHYS_CUH
#define PARAM_PHYS_CUH

#include <const_defs.cuh>

// =========================================================================================================================
// orbital and vertically isothermal gas profiles

__device__ __forceinline__
real _get_omegaK (real R)
{ return sqrt(G*M_S / R / R / R); }

__device__ __forceinline__
real _get_hg (real R)
{ return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0)); }

// calculate the radial pressure-support parameter used by the gas velocity
__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{ return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z))); }

// calculate exact vertically isothermal hydrostatic stratification relative to the midplane
__device__ __forceinline__
real _get_gas_strat (real R, real Z, real h_g)
{ return exp((R / sqrt(R*R + Z*Z) - 1.0) / (h_g*h_g)); }

// =========================================================================================================================
// gas density and Epstein stopping-time profiles

__device__ __forceinline__
real _get_sigma_g (real R)
{ return SIGMA_0*pow(R / R_0, IDX_P); }

__device__ __forceinline__
real _get_rhog (real R, real Z, real h_g)
{
    real H_g = h_g*R;
    real sigma_g = _get_sigma_g(R);
    real rhog_mid = sigma_g / (sqrt(2.0*M_PI)*H_g);

    return rhog_mid*_get_gas_strat(R, Z, h_g);
}

__device__ __forceinline__
real _get_stokes (real R, real Z, real h_g)
{
    real stokes = STOKES_0;
    stokes /= pow(R / R_0, IDX_P);          // apply inverse midplane surface-density scaling
    stokes /= _get_gas_strat(R, Z, h_g);    // apply inverse vertical gas-stratification scaling

    return stokes;
}

#ifdef DIFFUSION
// turbulent transport coefficients
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifndef CONST_NU  // CONST_ALPHA
    real nu = ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    return nu;
    #else             // CONST_NU
        return NU;
    #endif // CONST_NU
}

#ifdef VISC_FLOW
// calculate the cylindrical radial gas velocity from equation 41 of Kanagawa et al. 2017
__device__ __forceinline__
real _get_visc_vel (real R, real Z, real h_g)
{
    real nu = _get_nu(R, h_g);

    #ifdef CONST_NU
    real grad_nu_R = 0.0;
    #else  // CONST_ALPHA
    real grad_nu_R = IDX_Q + 1.5;
    #endif // CONST_NU

    // use the vertically integrated equation 9 in the vertically integrated 2D model
    if (N_Z == 1) return -3.0*nu*(grad_nu_R + IDX_P + 0.5) / R;

    real y = sqrt(R*R + Z*Z);
    real cyl_frac = R / y;
    real strat = (cyl_frac - 1.0) / (h_g*h_g);

    real grad_strat_R = cyl_frac*(1.0 - cyl_frac*cyl_frac) / (h_g*h_g);
    grad_strat_R -= (IDX_Q + 1.0)*strat;
    real grad_strat_Z = -cyl_frac*(1.0 - cyl_frac*cyl_frac) / (h_g*h_g);

    real grad_rhog_R = IDX_P - 0.5*(IDX_Q + 3.0) + grad_strat_R;
    real grad_rhog_Z = grad_strat_Z;

    real term_R = 3.0*nu*(grad_nu_R + grad_rhog_R + 0.5);
    real term_Z = IDX_Q*nu*(1.0 + grad_rhog_Z);

    return -(term_R - term_Z) / R;
}
#endif // VISC_FLOW

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU  // CONST_ALPHA
    return ALPHA;
    #else             // CONST_NU
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #endif // CONST_NU
}
#endif // DIFFUSION

// calculate the density-diffusion equilibrium scale height used by 3D initialization
__device__ __forceinline__
real _get_hd (real R, real h_g)
{
    real H_g = h_g*R;

    #ifdef DIFFUSION
    real alpha_z = _get_alpha(R, h_g) / SCHMIDT_Z;
    real stokes_mid = _get_stokes(R, 0.0, h_g);
    return H_g*sqrt(alpha_z / stokes_mid);
    #else
    return H_g;
    #endif
}

// =========================================================================================================================

#endif
