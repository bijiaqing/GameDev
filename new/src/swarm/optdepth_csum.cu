#ifdef RADIATION

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: optdepth_csum
// integrate radial optical-depth increments from the inner boundary to every outer cell face
// =========================================================================================================================

__global__
void optdepth_csum (real *dev_optdepth)
{
    int idx_ray = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ray >= N_X*N_Z) return;

    int ix = idx_ray % N_X;
    int iz = idx_ray / N_X;

    // accumulate one independent radial ray at fixed azimuth and polar angle
    for (int iy = 1; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
    }
}

// =========================================================================================================================

#endif // RADIATION
