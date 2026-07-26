#ifndef HELPERS_PARAMPHYS_CUH
#define HELPERS_PARAMPHYS_CUH

#if defined(IMPORTGAS) && !defined(CONST_ST)
#include <cassert>
#endif // IMPORTGAS and not CONST_ST

#include <const.cuh>
#include <helpers_paramgrid.cuh>
#include <helpers_interpval.cuh>

// =========================================================================================================================
// Basic Disk Parameter Functions (Order 0)
// Purpose: Fundamental physical quantities from constants only
// =========================================================================================================================

// Order: O0 | Dependencies: None (only constants: M_PI, RHO_0)
__device__ __forceinline__
real _get_grain_mass (real s)
{
    return M_PI*RHO_0*s*s*s / 6.0;
}

// Order: O0 | Dependencies: None (only constants: G, M_S)
__device__ __forceinline__
real _get_omegaK (real R)
{
    return sqrt(G*M_S / R / R / R);
}

// Order: O0 | Dependencies: None (only constants: ASPR_0, R_0, IDX_Q)
__device__ __forceinline__
real _get_hg (real R)
{
    return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0));
}

// Order: O0 | Dependencies: None (only constants: IDX_P, IDX_Q)
__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{
    return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z)));
}

#ifdef COLLISION
// Order: O0 | Dependencies: None (only constants: SIGMA_0, R_0, IDX_P)
// Flags: Only compiled when COLLISION is enabled
__device__ __forceinline__
real _get_sigma_g (real R)
{
    return SIGMA_0*pow(R / R_0, IDX_P);
}
#endif // COLLISION

// =========================================================================================================================
// Derived Disk Parameter Functions (Order 1)
// Purpose: Physical quantities derived from Order 0 functions
// =========================================================================================================================

// Order: O1 | Dependencies: _get_omegaK [O0]
__device__ __forceinline__
real _get_cs (real R, real h_g)
{
    return h_g*_get_omegaK(R)*R;
}

#if defined(DIFFUSION) || defined(COLLISION)
// Order: O1 | Dependencies: _get_omegaK [O0]
// Flags: Only compiled when DIFFUSION or COLLISION is enabled
// Conditional: Returns constant NU if CONST_NU defined, else calculates from ALPHA
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifndef CONST_NU
    return ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    #else  // CONST_NU
    return NU;
    #endif // NOT CONST_NU
}
#endif // DIFFUSION || COLLISION

#ifdef COLLISION
// Order: O1 | Dependencies: _get_omegaK [O0]
// Flags: Only compiled when COLLISION is enabled
// Conditional: Returns constant ALPHA if not CONST_NU, else calculates from NU
__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU
    return ALPHA;
    #else  // CONST_NU
    return NU / (h_g*h_g*R*R*_get_omegaK(R));
    #endif // NOT CONST_NU
}

// Order: O2 | Dependencies: _get_hg [O0], _get_nu [O1]
// Flags: Only compiled when COLLISION is enabled
__device__ __forceinline__
real _get_hd (real R, real St)
{
    // h_g cannot be passed by a parameter here because _get_hd is for individual particles
    // whereas h_g is, in most cases, evaluated at the midpoint between particles
    
    real h_g = _get_hg(R);
    real delta_Z = _get_nu(R, h_g) / SC_Z;
    
    return h_g*sqrt(delta_Z / (delta_Z + St));
}
#endif // COLLISION

// =========================================================================================================================
// Stokes Number Calculation (Order 3)
// Purpose: Calculate particle Stokes number with optional gas density interpolation
// =========================================================================================================================

// Order: O3 | Dependencies: _get_loc_x/y/z [O0], _interp_field [O2]
// Flags: IMPORTGAS adds gas field interpolation; CONST_ST disables all scaling but grain size 
__device__ __forceinline__
real _get_St (real R, real Z, real s, real h_g
    #ifdef IMPORTGAS
    , real x, real y, real z, const real *dev_gasdens
    #endif
)
{
    real St = ST_0*(s / S_0);

    #ifndef CONST_ST
    #ifdef IMPORTGAS
    if (dev_gasdens != nullptr)
    {
        // Use imported gas density
        real loc_x = _get_loc_x(x);
        real loc_y = _get_loc_y(y);
        real loc_z = _get_loc_z(z);
        
        real rhog  = _interp_field(dev_gasdens, loc_x, loc_y, loc_z);
        real rhog0 = SIGMA_0 / (sqrt(2.0*M_PI)*h_g*R);
        
        if (rhog <= 0.0)
        {
            printf("ERROR: Invalid gas density rhog = %e at (x,y,z) = (%e,%e,%e)\n", rhog, x, y, z);
            assert(false);
        }
        
        St *= rhog0 / rhog;
    }
    else
    #endif // IMPORTGAS
    {
        // Analytical gas density profile
        St /= pow(R / R_0, IDX_P);           // scale with radial   profile for volumetric gas density and sound speed
        St /= exp(-Z*Z / (2.0*h_g*h_g*R*R)); // scale with vertical profile for volumetric gas density
    }
    #endif // NOT CONST_ST

    return St;
}

// =========================================================================================================================

#endif // HELPERS_PARAMPHYS_CUH
