#ifndef VAL_DRAG_PATH_TRANSPORT_CUH
#define VAL_DRAG_PATH_TRANSPORT_CUH

// import the production transport header under a private name, then replace only its second SSA substep in this test
// retain the production first drift, launch interface, boundaries, and ordering in the public ssa_transport kernel
#define _ssa_substep_2 _val_production_ssa_substep_2
#include "../../../../inc/swarm/_transport.cuh"
#undef _ssa_substep_2

#define VAL_CONSTANT_DRAG_ACTIVE 1
#define VAL_DRAG_GAS_VY 0.15
#define VAL_DRAG_FORCE_Y -0.08

// replace disk-dependent forces and stopping times with constant coefficients that have an exact velocity and path solution
// call the production SSA stages with constant gas velocity, stopping time, and radial force
__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i,
    real x_1, real y_1, real z_1, real &x_j, real &y_j, real &z_j,
    real &lx_j, real &vy_j, real &lz_j
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz, const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    _ssa_advance(dt, size, 0.0, VAL_DRAG_GAS_VY, 0.0, beta, lx_i, vy_i, lz_i,
        x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j,
        [] (real, real, real, real, real, real, real &g, real &c, real &t) {
            g=VAL_DRAG_FORCE_Y; c=0.0; t=0.0;
        });
}

#endif // VAL_DRAG_PATH_TRANSPORT_CUH
