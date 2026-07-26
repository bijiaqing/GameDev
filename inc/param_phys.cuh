#ifndef PARAM_PHYS_CUH
#define PARAM_PHYS_CUH

#ifdef IMPORTGAS
#include <cassert>      // assert
#endif // IMPORTGAS

#include <const_defs.cuh>
#include <param_grid.cuh>
#include <swarm_grid.cuh>

// =========================================================================================================================
// grain and disk profiles
// =========================================================================================================================

__device__ __forceinline__
real _get_grain_mass (real size)
{ return M_PI*RHO_0*size*size*size / 6.0; }

#ifdef MULTISIZE
// calculate the represented-mass weight implied by the sampled swarm-size distribution
__host__ __device__ __forceinline__
real _get_mass_weight (real size)
{
    real weight = 1.0;

    #ifdef RADIATION
    if (INIT_SMIN != INIT_SMAX)
    {
        weight *= size;
        weight *= pow(INIT_SMIN, -0.5) - pow(INIT_SMAX, -0.5);
        weight /= pow(INIT_SMAX,  0.5) - pow(INIT_SMIN,  0.5);
    }
    #endif // RADIATION

    return weight;
}
#endif // MULTISIZE

__host__ __device__ __forceinline__
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
real _get_gas_strat (real R, real Z, real h_g)
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

// calculate the local dimensionless turbulence strength from the configured viscosity prescription
__host__ __device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifdef CONST_NU
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #else  // CONST_ALPHA
    return ALPHA;
    #endif // CONST_NU
}
#endif // DIFFUSION || COLLISION

#ifdef VISC_ACCRETION
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

    real grad_strat_R =  cyl_frac*(1.0 - cyl_frac*cyl_frac) / (h_g*h_g) - (IDX_Q + 1.0)*strat;
    real grad_strat_Z = -cyl_frac*(1.0 - cyl_frac*cyl_frac) / (h_g*h_g);

    real grad_rhog_R = IDX_P - 0.5*(IDX_Q + 3.0) + grad_strat_R;
    real grad_rhog_Z = grad_strat_Z;

    real term_R = 3.0*nu*(grad_nu_R + grad_rhog_R + 0.5);
    real term_Z = IDX_Q*nu*(1.0 + grad_rhog_Z);

    return -(term_R - term_Z) / R;
}
#endif // VISC_ACCRETION

// =========================================================================================================================
// stopping-time coupling
// =========================================================================================================================

// calculate the local Stokes number from grain size and the analytic or imported gas density
__device__ __forceinline__
real _get_stokes (real R, real Z, real h_g, real size
    #ifdef IMPORTGAS
    , real x, real y, real z, const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    real stokes = STOKES_0*(size / S_0);

    #ifdef IMPORTGAS
    // calibrate the imported density to the analytical reference midplane at R_0
    assert(dev_gas_dens != nullptr);

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    real rhog = _interp_field(dev_gas_dens, loc_x, loc_y, loc_z);
    if (rhog <= 0.0)
    {
        printf("ERROR: Invalid gas density rhog = %e at (x,y,z) = (%e,%e,%e)\n", rhog, x, y, z);
        assert(false);
    }

    real H_g0 = ASPR_0*R_0;
    real rhog_0 = SIGMA_0 / (sqrt(2.0*M_PI)*H_g0);
    real H_g = h_g*R;

    stokes *= rhog_0*H_g0 / (rhog*H_g);
    #else  // ANALYTIC_GAS
    #ifndef CONST_ST
    // apply the radial surface-density scaling and exact spherical vertical stratification
    stokes /= pow(R / R_0, IDX_P);
    stokes /= _get_gas_strat(R, Z, h_g);
    #endif // NOT CONST_ST
    #endif // IMPORTGAS

    return stokes;
}

// =========================================================================================================================

#endif // PARAM_PHYS_CUH
