#ifndef PAPER_ORBIT_TRANSPORT_CUH
#define PAPER_ORBIT_TRANSPORT_CUH
// Keep production state access, boundaries, forces, and first SSA drift.
#define _ssa_substep_2 _production_ssa_substep_2
#include "../../../../inc/swarm/_transport.cuh"
#undef _ssa_substep_2

// Exact zero-drag limit for this planar test; finite St cannot express zero drag.
__device__ __forceinline__
void _ssa_substep_2(real dt, real size, real beta, real lx_i, real vy_i, real lz_i,
    real x_1, real y_1, real z_1, real &x_j, real &y_j, real &z_j,
    real &lx_j, real &vy_j, real &lz_j)
{
    static_assert(N_Z == 1, "Planar orbital benchmark");
    real gravity, centrifugal, torque;
    _get_force_term(y_1, z_1, y_1, lx_i, lz_i, beta, gravity, centrifugal, torque);
    lx_j = lx_i;
    vy_j = vy_i + dt*(gravity + centrifugal);
    lz_j = 0.0;
    y_j = y_1 + 0.5*vy_j*dt;
    z_j = 0.5*M_PI;
    x_j = x_1 + 0.5*lx_j*dt/y_1/y_j;
}
#endif
