#ifndef PARAM_PHYS_CUH
#define PARAM_PHYS_CUH

#if defined(IMPORTGAS) && !defined(CONST_ST)
#include <cassert>      // assert
#endif // IMPORTGAS and not CONST_ST

#include <const_defs.cuh>
#include <param_grid.cuh>
#include <swarm_grid.cuh>

// =========================================================================================================================
// grain and disk profiles
// =========================================================================================================================

__device__ __forceinline__
real _get_grain_mass (real s)
{ return M_PI*RHO_0*s*s*s / 6.0; }

__device__ __forceinline__
real _get_omegaK (real R)
{ return sqrt(G*M_S / R / R / R); }

__device__ __forceinline__
real _get_hg (real R)
{ return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0)); }

// calculate the dimensionless radial pressure-support parameter at cylindrical position R,Z
__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{ return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z))); }

// calculate the exact vertically isothermal spherical gas stratification relative to the midplane
__device__ __forceinline__
real _get_rhog_strat (real R, real Z, real h_g)
{ return exp((R / sqrt(R*R + Z*Z) - 1.0) / (h_g*h_g)); }

#ifdef COLLISION
__device__ __forceinline__
real _get_sigma_g (real R)
{ return SIGMA_0*pow(R / R_0, IDX_P); }
#endif // COLLISION

// =========================================================================================================================
// thermodynamic and turbulent transport profiles
// =========================================================================================================================

// calculate the vertically isothermal sound speed
__device__ __forceinline__
real _get_cs (real R, real h_g)
{ return h_g*_get_omegaK(R)*R; }

#if defined(DIFFUSION) || defined(COLLISION)
// calculate the kinematic viscosity from the configured constant-nu or alpha prescription
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifdef CONST_NU
    return NU;
    #else  // CONST_ALPHA
    real nu = ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    return nu;
    #endif // CONST_NU
}
#endif // DIFFUSION || COLLISION

#ifdef COLLISION
// calculate the local dimensionless turbulence strength from the configured viscosity prescription
__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifdef CONST_NU
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #else  // CONST_ALPHA
    return ALPHA;
    #endif // CONST_NU
}

// calculate the settled dust aspect ratio used by vertically integrated collision rates
__device__ __forceinline__
real _get_hd (real R, real St)
{
    // evaluate h_g at the individual particle radius rather than at the pair midpoint
    
    real h_g = _get_hg(R);
    real alpha_Z = _get_alpha(R, h_g) / SCHMIDT_Z;
    
    return h_g*sqrt(alpha_Z / (alpha_Z + St));
}
#endif // COLLISION

// =========================================================================================================================
// stopping-time coupling
// =========================================================================================================================

// calculate the local Stokes number from grain size and the analytic or imported gas density
__device__ __forceinline__
real _get_St (real R, real Z, real s, real h_g
    #ifdef IMPORTGAS
    , real x, real y, real z, const real *dev_gas_dens
    #endif
)
{
    real St = ST_0*(s / S_0);

    #ifndef CONST_ST
    #ifdef IMPORTGAS
    if (dev_gas_dens != nullptr)
    {
        // scale the reference midplane Stokes number by the local Epstein-drag dependence
        real loc_x = _get_loc_x(x);
        real loc_y = _get_loc_y(y);
        real loc_z = _get_loc_z(z);
        
        real rhog = _interp_field(dev_gas_dens, loc_x, loc_y, loc_z);
        if (rhog <= 0.0)
        {
            printf("ERROR: Invalid gas density rhog = %e at (x,y,z) = (%e,%e,%e)\n", rhog, x, y, z);
            assert(false);
        }
        
        St *= SIGMA_0 / (rhog*sqrt(2.0*M_PI)*h_g*R);
    }
    else
    #endif // IMPORTGAS
    {
        // apply the radial surface-density scaling and exact spherical vertical stratification
        St /= pow(R / R_0, IDX_P);
        St /= _get_rhog_strat(R, Z, h_g);
    }
    #endif // NOT CONST_ST

    return St;
}

// =========================================================================================================================

#endif // PARAM_PHYS_CUH
