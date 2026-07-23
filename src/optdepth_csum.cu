#ifdef RADIATION

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: optdepth_csum
// integrate radial optical-depth increments from the inner boundary to every outer cell face
// =========================================================================================================================

__global__
void optdepth_csum (real *dev_optdepth)
{
    int idx_y = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx_y >= N_X*N_Z) return;

    int idx_x = idx_y % N_X;
    int idx_z = idx_y / N_X;

    // accumulate one independent radial ray at fixed azimuth and polar angle
    for (int i = 1; i < N_Y; i++)
    {
        int idx_cell = idx_z*N_X*N_Y + i*N_X + idx_x;
        dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
    }
}

// =========================================================================================================================

#endif // RADIATION
