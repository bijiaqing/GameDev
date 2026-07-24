#ifndef SWARM_KERN_CUH
#define SWARM_KERN_CUH

#if defined(DIFFUSION) && !defined(TRANSPORT)
#error "DIFFUSION requires TRANSPORT"
#endif // DIFFUSION && !TRANSPORT

#if defined(RADIATION) && !defined(TRANSPORT)
#error "RADIATION requires TRANSPORT"
#endif // RADIATION && !TRANSPORT

#if defined(VISC_ACCRETION) && !defined(DIFFUSION)
#error "VISC_ACCRETION requires DIFFUSION"
#endif // VISC_ACCRETION && !DIFFUSION

#if defined(VISC_ACCRETION) && defined(IMPORTGAS)
#error "VISC_ACCRETION cannot be combined with imported gas velocities"
#endif // VISC_ACCRETION && IMPORTGAS

#if defined(COLLISION) && !defined(MULTISIZE)
#error "COLLISION requires MULTISIZE because collision outcomes evolve grain size and represented grain number"
#endif // COLLISION && !MULTISIZE

#if !defined(TRANSPORT) && !defined(COLLISION)
#error "No evolution module is enabled"
#endif // !TRANSPORT && !COLLISION

#if defined(LOGTIMING) && defined(LOGOUTPUT)
#error "LOGTIMING and LOGOUTPUT cannot be enabled simultaneously"
#endif // LOGTIMING && LOGOUTPUT

#if defined(LOGTIMING) && defined(TRANSPORT)
#error "LOGTIMING is not compatible with TRANSPORT"
#endif // LOGTIMING && TRANSPORT

#if defined(LOGTIMING) && defined(SAVE_DENS)
#error "LOGTIMING is not compatible with SAVE_DENS"
#endif // LOGTIMING && SAVE_DENS

#include <const_defs.cuh>

// =========================================================================================================================
// particle initialization
// =========================================================================================================================

__global__ void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z
    #ifdef MULTISIZE
    , const real *dev_random_s, real mass_norm
    #endif // MULTISIZE
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
);

// =========================================================================================================================
// imported gas interpolation
// =========================================================================================================================

#ifdef IMPORTGAS
__global__ void gas_lerp_calc (real *dev_gas_dens, real *dev_gas_velx, real *dev_gas_vely, real *dev_gas_velz,
    const real *dev_gas_dens_next, const real *dev_gas_velx_next,
    const real *dev_gas_vely_next, const real *dev_gas_velz_next, real blend);
#endif // IMPORTGAS

// =========================================================================================================================
// particle-to-grid dust density
// =========================================================================================================================

#ifdef SAVE_DENS
__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_depo (real *dev_dustdens, const swarm *dev_particle, real total_dust_mass);
__global__ void dustdens_calc (real *dev_dustdens);
#endif // SAVE_DENS

// =========================================================================================================================
// representative-particle collisions and random states
// =========================================================================================================================

#ifdef COLLISION
__global__ void col_snap_save (real *dev_size_old, real *dev_numr_old, const swarm *dev_particle);
__global__ void col_rate_calc (real *dev_col_rate, real *dev_col_dist, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const tree *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0
);
__global__ void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const real *dev_col_dist, const real *dev_size_old, const real *dev_numr_old,
    const tree *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col
);
__global__ void col_tree_init (tree *dev_col_tree, const swarm *dev_particle);
#endif // COLLISION

#if defined(COLLISION) || defined(DIFFUSION)
__global__ void rngstate_init (curs *dev_rngstate, int seed = 1);
#endif // COLLISION || DIFFUSION

// =========================================================================================================================
// particle transport and runtime timestep
// =========================================================================================================================

#ifdef TRANSPORT
__global__ void dt_rates_calc (real *dev_dt_rates, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz
    , const real *dev_gas_velx_next, const real *dev_gas_vely_next, const real *dev_gas_velz_next
    #endif // IMPORTGAS
);

#ifdef RADIATION
__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_depo (real *dev_optdepth, const swarm *dev_particle, real total_dust_mass);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_csum (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);
__global__ void ssa_substep_1 (swarm *dev_particle, real dt);
__global__ void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth,
    #ifdef IMPORTGAS
    const real *dev_gas_dens, const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz,
    #endif // IMPORTGAS
    real beta_taper,
    real dt
);
#else  // NO RADIATION
__global__ void ssa_transport (swarm *dev_particle,
    #ifdef IMPORTGAS
    const real *dev_gas_dens, const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz,
    #endif // IMPORTGAS
    real dt
);
#endif // RADIATION
#endif // TRANSPORT

// =========================================================================================================================
// turbulent spatial diffusion
// =========================================================================================================================

#ifdef DIFFUSION
__global__ void diffusion_pos (swarm *dev_particle, curs *dev_rngstate, real dt);
#endif // DIFFUSION

// =========================================================================================================================

#endif // SWARM_KERN_CUH
