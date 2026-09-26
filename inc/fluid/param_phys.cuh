#ifndef GAMEDEV_FLUID_PARAM_PHYS_CUH
#define GAMEDEV_FLUID_PARAM_PHYS_CUH

#include <const_defs.cuh>

// =====================================================================================================================
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

// =====================================================================================================================
// Epstein stopping-time profile

// scale the reference midplane Stokes number by inverse surface density and vertical stratification
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
        #else  // CONST_NU
        return NU;
        #endif // CONST_NU
}

// return the local dust diffusivity, including finite-Stokes suppression in both flux modes
__device__ __forceinline__
real _get_diffusivity (real R, real Z, real h_g, real schmidt)
{
    real stokes = _get_stokes(R, Z, h_g);
    return _get_nu(R, h_g) / (schmidt*(1.0 + stokes*stokes));
}

// return the concentration weight w proportional to rho_g, or Sigma_g in 2D, and w = 1 for density diffusion
// its normalization cancels between face weights and concentration
__device__ __forceinline__
real _get_diffusion_weight (real y, real z)
{
    #ifdef DIFFUSE_CONCENTRATION
    real R = y*sin(z);
    if (N_Z == 1) return pow(R / R_0, IDX_P);
    real h_g = _get_hg(R);
    return pow(R / R_0, IDX_P - 0.5*(IDX_Q + 3.0))*_get_gas_strat(R, y*cos(z), h_g);
    #else  // !DIFFUSE_CONCENTRATION
    return 1.0;
    #endif // DIFFUSE_CONCENTRATION
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

// recover the local alpha equivalent of the selected viscosity prescription
__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU  // CONST_ALPHA
    return ALPHA;
    #else  // CONST_NU
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #endif // CONST_NU
}
#endif // DIFFUSION

// return the midplane, small-height settling-equilibrium dust scale height used by 3D initialization
__device__ __forceinline__
real _get_hd (real R, real h_g)
{
    real H_g = h_g*R;

    #ifdef DIFFUSION
    real alpha_z = _get_alpha(R, h_g) / SCHMIDT_Z;
    real stokes_mid = _get_stokes(R, 0.0, h_g);
    alpha_z /= 1.0 + stokes_mid*stokes_mid;
    #ifdef DIFFUSE_CONCENTRATION
    return H_g*sqrt(alpha_z / (stokes_mid + alpha_z));
    #else  // !DIFFUSE_CONCENTRATION
    return H_g*sqrt(alpha_z / stokes_mid);
    #endif // DIFFUSE_CONCENTRATION
    #else  // !DIFFUSION
    return H_g;
    #endif // DIFFUSION
}

// =====================================================================================================================

#endif // GAMEDEV_FLUID_PARAM_PHYS_CUH
