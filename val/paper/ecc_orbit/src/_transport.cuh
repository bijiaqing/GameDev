#ifndef PAPER_ORBIT_TRANSPORT_CUH
#define PAPER_ORBIT_TRANSPORT_CUH
// keep production state access, boundaries, forces, and the first SSA drift
#define _ssa_substep_2 _production_ssa_substep_2
#include "../../../../inc/swarm/_transport.cuh"
#undef _ssa_substep_2

// replace only the second SSA stage by its exact zero-drag limit, which no finite St can express
__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i,
    real x_1, real y_1, real z_1, real &x_j, real &y_j, real &z_j,
    real &lx_j, real &vy_j, real &lz_j)
{
    static_assert(N_Z == 1, "Planar orbital benchmark");
    _ssa_advance<true>(dt, 1.0, 0.0, 0.0, 0.0, beta, lx_i, vy_i, lz_i,
        x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j,
        [](real y, real z, real R, real lx, real lz, real b, real &g, real &c, real &t)
        {
            _get_force_term(y, z, R, lx, lz, b, g, c, t);
        });
}
#endif // !PAPER_ORBIT_TRANSPORT_CUH
