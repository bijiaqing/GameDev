#ifdef RADIATION

#include <fluid_kern.cuh>

__global__
void optdepth_csum (real *dev_optdepth)
{
    int idx_ray = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ray >= N_X*N_Z) return;

    int ix = idx_ray % N_X;
    int iz = idx_ray / N_X;

    // accumulate local contributions outward along one radial ray
    for (int iy = 1; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
    }
}

#endif
