#ifndef GRAFFITI_KERN_CUH 
#define GRAFFITI_KERN_CUH

#include <const.cuh>

#if defined(COLLISION) && !defined(MULTISIZE)
#error "COLLISION requires MULTISIZE because collision outcomes evolve grain size and represented grain number"
#endif

// =========================================================================================================================
// particle initialization

__global__ void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z
    #ifdef MULTISIZE
    , const real *dev_random_s
    #endif // MULTISIZE
);

// =========================================================================================================================
// imported gas interpolation

#ifdef IMPORTGAS
__global__ void gas_interp_calc (real *dev_gasdens, real *dev_gasvelx, real *dev_gasvely, real *dev_gasvelz,
    const real *dev_gasdens_next, const real *dev_gasvelx_next,
    const real *dev_gasvely_next, const real *dev_gasvelz_next, real blend);
#endif

// =========================================================================================================================
// particle-to-grid dust density

#ifdef SAVE_DENS
__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_scat (real *dev_dustdens, const swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);
#endif // SAVE_DENS

// =========================================================================================================================
// representative-particle collisions and random states

#ifdef COLLISION
__global__ void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
__global__ void col_proc_exec (swarm *dev_particle, const swarm *dev_particle_old, curs *dev_rs_swarm, real dt_col,
    const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
__global__ void col_tree_init (tree *dev_col_tree, const swarm *dev_particle);
#endif // COLLISION

#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 1);
#endif // COLLISION || (TRANSPORT && DIFFUSION)

// =========================================================================================================================
// particle transport and runtime timestep

#ifdef TRANSPORT
__global__ void dt_rate_calc (real *dev_dt_rate, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx,
      const real *dev_gasvely, const real *dev_gasvelz,
      const real *dev_gasdens_next, const real *dev_gasvelx_next,
      const real *dev_gasvely_next, const real *dev_gasvelz_next
    #endif
);

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
// turbulent spatial diffusion

#ifdef DIFFUSION
__global__ void diffusion_pos (swarm *dev_particle, curs *dev_rs_swarm, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
);
#endif // DIFFUSION
#endif // TRANSPORT

// =========================================================================================================================

#endif // NOT GRAFFITI_KERN_CUH
