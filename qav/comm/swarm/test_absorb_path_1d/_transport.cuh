#ifndef QAV_ABSORB_PATH_TRANSPORT_CUH
#define QAV_ABSORB_PATH_TRANSPORT_CUH

// retain the production staggered drift, boundary policy, active-state predicate, and public transport kernel
// replace only disk forces and drag so every radial path has an exact constant-velocity crossing time
#define _ssa_substep_2 _qav_production_ssa_substep_2
#include "../../../../inc/comm/swarm/_transport.cuh"
#undef _ssa_substep_2

#define QAV_CONSTANT_RADIAL_PATH 1

__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i,
    real x_1, real y_1, real z_1, real &x_j, real &y_j, real &z_j,
    real &lx_j, real &vy_j, real &lz_j
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz, const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    (void)size;
    (void)beta;
    (void)lx_i;
    (void)lz_i;

    x_j = 0.5*(X_MIN + X_MAX);
    y_j = y_1 + 0.5*vy_i*dt;
    z_j = 0.5*M_PI;
    lx_j = 0.0;
    vy_j = vy_i;
    lz_j = 0.0;
}

#endif

