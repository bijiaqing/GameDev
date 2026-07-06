#ifndef GRAFFITI_KERN_CUH 
#define GRAFFITI_KERN_CUH

#include <const.cuh>

// =========================================================================================================================
// Kernel declarations (organized by compiler flags)
// =========================================================================================================================

// always compiled
__global__ void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z
    #ifdef MULTISIZE
    , const real *dev_random_s
    #endif // MULTISIZE
);

// =========================================================================================================================

#ifdef SAVE_DENS
__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_scat (real *dev_dustdens, const swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);
#endif // SAVE_DENS

// =========================================================================================================================

#ifdef COLLISION
__global__ void col_rate_init (real *dev_col_rate, real *dev_col_expt, real *dev_col_rand);
__global__ void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
__global__ void col_flag_calc (int *dev_col_flag, curs *dev_rs_grids, real *dev_col_rand, const real *dev_col_rate, real dt_col);
__global__ void col_proc_exec (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, const real *dev_col_rand, 
    const int *dev_col_flag, const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
__global__ void col_tree_init (tree *dev_col_tree, const swarm *dev_particle);
__global__ void rs_grids_init (curs *dev_rs_grids, int seed = 0);
#endif // COLLISION

#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 1);
#endif // COLLISION or DIFFUSION

// =========================================================================================================================

#ifdef TRANSPORT
#ifdef RADIATION
__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_scat (real *dev_optdepth, const swarm *dev_particle);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_csum (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);
__global__ void ssa_substep_1 (swarm *dev_particle, real dt);
__global__ void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx, const real *dev_gasvely, const real *dev_gasvelz
    #endif
);
#else  // NO RADIATION
__global__ void ssa_transport (swarm *dev_particle, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx, const real *dev_gasvely, const real *dev_gasvelz
    #endif
);
#endif // RADIATION

// =========================================================================================================================

#ifdef DIFFUSION
__global__ void diffusion_pos (swarm *dev_particle, curs *dev_rs_swarm, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
__global__ void diffusion_vel (swarm *dev_particle, curs *dev_rs_swarm, real dt);
#endif // DIFFUSION
#endif // TRANSPORT

// =========================================================================================================================

#endif // NOT GRAFFITI_KERN_CUH
