#ifndef PARAM_PHYS_CUH
#define PARAM_PHYS_CUH

#include <const.cuh>

// =========================================================================================================================

__device__ __forceinline__
real _get_omegaK (real R)
{ return sqrt(G*M_S / R / R / R); }

__device__ __forceinline__
real _get_hg (real R)
{ return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0)); }

// the pressure gradient parameter η for gas velocity
__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{ return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z))); }

// the true gas stratification parameter in hydrostatic equilibrium (vertically isothermal)
__device__ __forceinline__
real _get_gas_strat (real R, real Z, real h_g)
{ return exp((R / sqrt(R*R + Z*Z) - 1.0) / (h_g*h_g)); }

// =========================================================================================================================

__device__ __forceinline__
real _get_sigma_g (real R)
{ return SIGMA_0*pow(R / R_0, IDX_P); }

__device__ __forceinline__
real _get_rhog (real R, real Z, real h_g)
{
    real H_g = h_g*R;
    real sigma_g = _get_sigma_g(R);
    real rho_mid = sigma_g / (sqrt(2.0*M_PI)*H_g);

    return rho_mid*_get_gas_strat(R, Z, h_g);
}

__device__ __forceinline__
real _get_stokes (real R, real Z, real h_g)
{
    real stokes = STOKES_0;
    stokes /= pow(R / R_0, IDX_P);          // radial correction for gas density and sound speed
    stokes /= _get_gas_strat(R, Z, h_g);    // vertical correction for gas stratification

    return stokes;
}

#ifdef DIFFUSION
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifndef CONST_NU    // const alpha, variable nu
    real nu = ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    return nu;
    #else               // const nu, variable alpha
    return NU;
    #endif
}

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU    // const alpha, variable nu
    return ALPHA;
    #else               // const nu, variable alpha
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #endif
}
#endif

// calculate the density-diffusion scale height used by initialization and the 2D radiation closure
__device__ __forceinline__
real _get_hd (real R, real h_g)
{
    real H_g = h_g*R;

    #ifdef DIFFUSION
    real alpha_z = _get_alpha(R, h_g) / SC_Z;
    real stokes_mid = _get_stokes(R, 0.0, h_g);
    return H_g*sqrt(alpha_z / stokes_mid);
    #else
    return H_g;
    #endif
}

// =========================================================================================================================

#endif
