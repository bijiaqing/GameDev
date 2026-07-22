#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>
#include <helpers_transport.cuh>

// =========================================================================================================================
// kernel: ssa_substep_1
// drift particles to midpoint positions before reconstructing the radiation optical-depth field
//
// parallelization: one thread per representative particle
// =========================================================================================================================

__global__
void ssa_substep_1 (swarm *dev_particle, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        real x_i, y_i, z_i;
        real x_1, y_1, z_1;
        
        real lx_i, vy_i, lz_i;

        // retain the initial velocity while replacing the stored position by its midpoint value
        _load_particle(dev_particle, idx, x_i, y_i, z_i, lx_i, vy_i, lz_i);
        _ssa_substep_1(dt, x_i, y_i, z_i, lx_i, vy_i, lz_i, x_1, y_1, z_1);
        _if_out_of_box(x_1, y_1, z_1, lx_i, vy_i, lz_i);
        _save_particle(dev_particle, idx, x_1, y_1, z_1, lx_i, vy_i, lz_i);
    }
}

// =========================================================================================================================

#endif // TRANSPORT && RADIATION
