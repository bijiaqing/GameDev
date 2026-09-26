#ifndef VAL_ORBIT_ECC_TRANSPORT_CUH
#define VAL_ORBIT_ECC_TRANSPORT_CUH

// import the production transport header under a private name, then replace only its second SSA substep in this test
// retain the production first drift, launch interface, boundaries, and ordering in the public ssa_transport kernel
#define _ssa_substep_2 _val_production_ssa_substep_2
#include "../../../../inc/swarm/_transport.cuh"
#undef _ssa_substep_2

#define VAL_ZERO_DRAG_ACTIVE 1

// remove gas relaxation so Kepler's equation provides an exact eccentric-orbit reference
// preserve the production force evaluation and staggered position update as the algorithm under test
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
    _ssa_advance<true>(dt, 1.0, 0.0, 0.0, 0.0, beta, lx_i, vy_i, lz_i,
        x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j,
        [](real y, real z, real R, real lx, real lz, real b, real &g, real &c, real &t)
        {
            _get_force_term(y, z, R, lx, lz, b, g, c, t);
        });
}

#endif // VAL_ORBIT_ECC_TRANSPORT_CUH
