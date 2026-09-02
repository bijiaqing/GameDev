#ifndef VAL_ORBIT_ECC_TRANSPORT_CUH
#define VAL_ORBIT_ECC_TRANSPORT_CUH

// import the production transport header under a private name, then replace only its second SSA substep in this test
// retain the production first drift, launch interface, boundaries, and ordering in the public ssa_transport kernel
#define _ssa_substep_2 _val_production_ssa_substep_2
#include "../../../../inc/comm/swarm/_transport.cuh"
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
    // grain size affects production drag only, so it is intentionally inactive in this drag-free specialization
    (void)size;
    real R_1 = _get_cyl_R(y_1, z_1);
    real grav_y1, cent_y1, torq_z1;
    _get_force_term(y_1, z_1, R_1, lx_i, lz_i, beta, grav_y1, cent_y1, torq_z1);

    real lx_1 = lx_i;
    real vy_1 = vy_i + 0.5*dt*(grav_y1 + cent_y1);
    real lz_1 = lz_i + 0.5*dt*torq_z1;

    real grav_y2, cent_y2, torq_z2;
    _get_force_term(y_1, z_1, R_1, lx_1, lz_1, beta, grav_y2, cent_y2, torq_z2);
    lx_j = lx_i;
    vy_j = vy_i + dt*(grav_y2 + cent_y2);
    lz_j = lz_i + dt*torq_z2;

    y_j = y_1 + 0.5*vy_j*dt;
    if constexpr (N_Z == 1)
    {
        z_j = 0.5*M_PI;
        lz_j = 0.0;
        x_j = (N_X > 1) ? x_1 + 0.5*lx_j*dt / y_1 / y_j : 0.5*(X_MIN + X_MAX);
        return;
    }

    z_j = z_1 + 0.5*lz_j*dt / y_1 / y_j;
    x_j = x_1 + 0.5*lx_j*dt / y_1 / y_j / sin(z_1) / sin(z_j);
}

#endif // VAL_ORBIT_ECC_TRANSPORT_CUH
