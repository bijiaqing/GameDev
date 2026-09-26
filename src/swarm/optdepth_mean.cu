#ifdef RADIATION

#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: optdepth_mean
// replace every azimuthal ring of optical depth with its ring average
// =====================================================================================================================

__global__
void optdepth_mean (real *dev_optdepth)
{
    int idx_ring = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;

    real optdepth_sum = 0.0;

    // sum one independent azimuthal ring at fixed radius and polar angle
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        optdepth_sum += dev_optdepth[idx_cell];
    }

    real optdepth_mean = optdepth_sum / N_X;

    // broadcast the ring mean to every azimuthal cell
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_optdepth[idx_cell] = optdepth_mean;
    }
}

// =====================================================================================================================

#endif // RADIATION
