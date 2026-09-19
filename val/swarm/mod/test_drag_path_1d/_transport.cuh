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
// keep the velocity update exact while testing second-order convergence of the retained production staggered drift
__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i,
    real x_1, real y_1, real z_1, real &x_j, real &y_j, real &z_j,
    real &lx_j, real &vy_j, real &lz_j
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz, const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    (void)beta;
    (void)lx_i;
    (void)lz_i;
    // reuse par_size as a controlled stopping-time label; no physical size-to-Stokes conversion is part of this test
    real stopping_time = size;
    real equilibrium = VAL_DRAG_GAS_VY + VAL_DRAG_FORCE_Y*stopping_time;
    vy_j = equilibrium + (vy_i - equilibrium)*exp(-dt/stopping_time);
    lx_j = 0.0;
    lz_j = 0.0;
    y_j = y_1 + 0.5*vy_j*dt;
    x_j = 0.5*(X_MIN + X_MAX);
    z_j = 0.5*M_PI;
}

#endif // VAL_DRAG_PATH_TRANSPORT_CUH
