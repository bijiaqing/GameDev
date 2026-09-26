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

#if defined(DIFFUSE_CONCENTRATION) && !defined(DIFFUSION)
#error "DIFFUSE_CONCENTRATION requires DIFFUSION"
#endif // DIFFUSE_CONCENTRATION && !DIFFUSION

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

#if defined(COL_CHAIN) || defined(BERNOULLI) || defined(KNN_CACHE)
#error "COL_CHAIN, BERNOULLI, and KNN_CACHE were removed; COLLISION always uses the frozen-bath chain"
#endif // COL_CHAIN || BERNOULLI || KNN_CACHE


#ifdef COLLISION
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
#endif // COLLISION

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
__global__ void colstate_flag (const swarm *dev_particle, int *dev_bad_part);

#ifdef COLLISION_KDTREE
__global__ void col_site_init (kdtree_node *dev_kdtree_node, unsigned char *dev_col_active,
    const swarm *dev_particle, int *dev_bad_part);
#else  // COLLISION_MORTON
__global__ void col_site_init (float3 *dev_morton_point, float *dev_morton_posx, float *dev_search_dist,
    unsigned char *dev_col_active, const swarm *dev_particle, int *dev_bad_part);
#endif // COLLISION_KDTREE

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
// collision chain: bath controller, local scheduler, and frozen-bath event chain
// =====================================================================================================================

#ifdef COLLISION
#include <_col_types.cuh>

__global__
void col_size_zero ();

__global__
void col_size_scan (const int *ids, int count, const swarm *particles,
    const int *spatial, const unsigned char *active);

__global__
void col_size_bnds ();

__global__
void col_bath_init (const int *owner_ids, int owner_count, real *dev_size_old, real *dev_numr_old, real *dev_col_time,
    int *dev_col_events, unsigned char *dev_col_complete, const swarm *dev_particle);

__global__
void col_space_bin (int *dev_col_spatial, const swarm *dev_particle);

__global__
void col_bath_rate (const int *owner_ids, int owner_count, real *dev_col_rate, real *change_rate, real *second_rate,
    const swarm *dev_particle, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const query_environment *environment, cached_rate_moments *cached);

__global__
void col_count_bin (const int *owner_ids, int owner_count, int *dev_col_count, const swarm *dev_particle,
    const int *dev_col_spatial, const unsigned char *dev_col_active);

__global__
void col_rate_bins (const int *owner_ids, int owner_count, col_rate_bin *dev_col_bin, const swarm *dev_particle,
    const real *dev_col_rate, const real *change_rate, const real *second_rate, const int *dev_col_spatial,
    const int *dev_col_binmap,
    const unsigned char *dev_col_active);

__global__
void col_skip_scan (const int *ids, int count, curs *rng,
    const unsigned char *active, const real *measure, const int *spatial, const real *steps,
    const cached_rate_moments *cached, real *time, int *events, unsigned char *complete,
    real *hazard, real *jump1, real *jump2, real *jumpmax, int *queue, int *queued);

__global__
void col_chain_run (const int *owner_ids, int owner_count, swarm *dev_particle, curs *dev_rngstate, int *dev_col_error,
    int *dev_col_unfinished, real *dev_col_time, int *dev_col_events,
    unsigned char *dev_col_complete, real *dev_col_hazard,
    real *dev_col_jump1_int, real *dev_col_jump2_int, real *dev_col_jumpmax_int,
    const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const real *group_step, const int *spatial,
    int *unfinished_ids, int *error_flag, event_work *work,
    const query_environment *environment, const cached_rate_moments *cached);

__global__
void col_audit_bin (const int *owner_ids, int owner_count, col_audit_accum *dev_col_bin, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const real *dev_col_rate,
    const real *dev_col_hazard, const real *dev_col_jump1_int,
    const real *dev_col_jump2_int, const real *dev_col_jumpmax_int,
    const int *dev_col_events, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active, const real *group_step);

#ifdef COL_QUERY_ENV_CACHE
__global__
void col_env_cache (query_environment *env, const swarm *particle);
#endif // COL_QUERY_ENV_CACHE

__global__
void col_dep_graph (unsigned int *edges, const int *spatial,
    const int *neighbors, const unsigned char *active);

__global__
void col_comp_zero (const int *ids, int count, real *hazard,
    real *jump1, real *jump2, real *jumpmax);

__global__
void col_event_sum (const event_work *work, event_work *sum);

#endif // COLLISION

// =====================================================================================================================

#endif // GAMEDEV_SWARM_KERN_CUH
