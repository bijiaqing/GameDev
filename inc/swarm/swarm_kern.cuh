#ifndef GAMEDEV_SWARM_KERN_CUH
#define GAMEDEV_SWARM_KERN_CUH

#include <gpu.cuh>

#include <const_defs.cuh>

#if defined(DIFFUSION) && !defined(TRANSPORT)
#error "DIFFUSION requires TRANSPORT"
#endif // DIFFUSION && !TRANSPORT

#if defined(RADIATION) && !defined(TRANSPORT)
#error "RADIATION requires TRANSPORT"
#endif // RADIATION && !TRANSPORT

#if defined(PR_EFFECT) && !defined(RADIATION)
#error "PR_EFFECT requires RADIATION"
#endif // PR_EFFECT && !RADIATION

#if defined(VISC_FLOW) && !defined(DIFFUSION)
#error "VISC_FLOW requires DIFFUSION"
#endif // VISC_FLOW && !DIFFUSION

#if defined(VISC_FLOW) && defined(IMPORTGAS)
#error "VISC_FLOW cannot be combined with imported gas velocities"
#endif // VISC_FLOW && IMPORTGAS

#if defined(IMPORTGAS) && defined(CONST_ST)
#error "CONST_ST cannot be combined with IMPORTGAS because imported gas density determines the local Stokes number"
#endif // IMPORTGAS && CONST_ST

#if defined(COLLISION) && !defined(MULTISIZE)
#error "COLLISION requires MULTISIZE because collision outcomes evolve grain size and represented grain number"
#endif // COLLISION && !MULTISIZE

#if defined(COLLISION) && (defined(COLLISION_KDTREE) == defined(COLLISION_MORTON))
#error "COLLISION requires exactly one of COLLISION_KDTREE or COLLISION_MORTON"
#endif // COLLISION backend selection

#ifdef COL_CHAIN
#error "COL_CHAIN is obsolete; COLLISION now selects frozen-bath collisions by default"
#endif // COL_CHAIN

#if defined(BERNOULLI) && !defined(COLLISION)
#error "BERNOULLI requires COLLISION"
#endif // BERNOULLI && !COLLISION

#if defined(KNN_CACHE) && !defined(BERNOULLI)
#error "KNN_CACHE requires BERNOULLI"
#endif // KNN_CACHE && !BERNOULLI

#if defined(COLLISION) && !defined(BERNOULLI)
static_assert(COL_BATH_TPB > 0 && COL_BATH_TPB <= 1024,
    "frozen-bath collisions require 0 < COL_BATH_TPB <= 1024");
static_assert(COL_EVENT_CAP > 0, "frozen-bath collisions require COL_EVENT_CAP > 0");
static_assert(COL_BIN_X > 0 && COL_BIN_Y > 0 && COL_BIN_Z > 0 && COL_BIN_S > 0,
    "frozen-bath controller-bin counts must be positive");
static_assert(COL_BIN_MIN > 0, "frozen-bath collisions require COL_BIN_MIN > 0");
static_assert(COL_BATH_MAX > 0.0, "frozen-bath collisions require COL_BATH_MAX > 0");
static_assert(COL_BATH_EPS > 0.0 && COL_BATH_EPS < 1.0,
    "frozen-bath collisions require 0 < COL_BATH_EPS < 1");
static_assert(COL_BATH_ALPHA > 0.0 && COL_BATH_ALPHA < 1.0,
    "frozen-bath collisions require 0 < COL_BATH_ALPHA < 1");
#endif // COLLISION && !BERNOULLI

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

#ifdef COLLISION_MORTON
#include <morton/morton_types.cuh>
#endif // COLLISION_MORTON

// =====================================================================================================================
// particle initialization
// =====================================================================================================================

__global__ void particle_init (swarm *dev_particle, const real *dev_randposx, const real *dev_randposy,
    const real *dev_randposz
    #ifdef MULTISIZE
    , const real *dev_randsize, const real *dev_mass_bank, int mass_bin_count, real mass_norm
    #endif // MULTISIZE
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
);

// =====================================================================================================================
// imported gas interpolation
// =====================================================================================================================

#ifdef IMPORTGAS
__global__ void gas_lerp_calc (real *dev_gas_dens, real *dev_gas_velx, real *dev_gas_vely, real *dev_gas_velz,
    const real *dev_gas_dens_next, const real *dev_gas_velx_next,
    const real *dev_gas_vely_next, const real *dev_gas_velz_next, real gas_blend);
#endif // IMPORTGAS

// =====================================================================================================================
// particle-to-grid dust density
// =====================================================================================================================

#ifdef SAVE_DENS
__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_depo (real *dev_dustdens, const swarm *dev_particle, real total_dust_mass);
__global__ void dustdens_calc (real *dev_dustdens);
#endif // SAVE_DENS

// =====================================================================================================================
// representative-particle collisions and random states
// =====================================================================================================================

#ifdef COLLISION
__global__ void col_snap_save (real *dev_size_old, real *dev_numr_old, const swarm *dev_particle);
__global__ void colstate_flag (const swarm *dev_particle, int *dev_bad_part);
__global__ void inf_rate_flag (const real *dev_col_rate, const real *dev_col_dist, int *dev_bad_part);

#ifdef COLLISION_KDTREE
__global__ void col_rate_calc (real *dev_col_rate, real *dev_col_dist, const swarm *dev_particle,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0
);
__global__ void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const real *dev_col_dist, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0,
    real dt_col
);
__global__ void col_site_init (kdtree_node *dev_kdtree_node, unsigned char *dev_col_active,
    const swarm *dev_particle, int *dev_bad_part);
#else  // COLLISION_MORTON
__global__ void col_rate_calc (real *dev_col_rate, real *dev_col_dist, unsigned int *dev_morton_overflow,
    const swarm *dev_particle, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_morton_point, morton_view morton_data, bool unique_ids,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0
);
__global__ void col_event_run (swarm *dev_particle, curs *dev_rngstate,
    const real *dev_col_rate, const real *dev_col_dist, unsigned int *dev_morton_overflow,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_morton_point, morton_view morton_data, bool unique_ids,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col
);
__global__ void col_site_init (float3 *dev_morton_point, float *dev_morton_posx, float *dev_search_dist,
    unsigned char *dev_col_active, const swarm *dev_particle, int *dev_bad_part);
#endif // COLLISION_KDTREE

#ifdef KNN_CACHE
__global__ void col_rate_calc (real *dev_col_rate, const swarm *dev_particle,
    const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0
);
__global__ void col_event_run (swarm *dev_particle, curs *dev_rngstate,
    const real *dev_col_rate, const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col
);
#endif // KNN_CACHE
#endif // COLLISION

#if defined(COLLISION) || defined(DIFFUSION)
__global__ void rngstate_init (curs *dev_rngstate, int seed = 1);
#endif // COLLISION || DIFFUSION

// =====================================================================================================================
// particle transport and runtime timestep
// =====================================================================================================================

#ifdef TRANSPORT
__global__ void dyn_rate_calc (real *dev_dyn_rate, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz
    , const real *dev_gas_velx_next, const real *dev_gas_vely_next, const real *dev_gas_velz_next
    , const real *dev_gas_dens, const real *dev_gas_dens_next
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

// =====================================================================================================================
// turbulent spatial diffusion
// =====================================================================================================================

#ifdef DIFFUSION
__global__ void diffusion_pos (swarm *dev_particle, curs *dev_rngstate, real dt
#ifdef IMPORTGAS
    , const real *dev_gas_dens
#endif // IMPORTGAS
);
#endif // DIFFUSION

// =====================================================================================================================

#endif // GAMEDEV_SWARM_KERN_CUH
