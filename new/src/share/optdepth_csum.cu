#ifdef RADIATION

#include <fluid_kern.cuh>







__global__
void optdepth_csum (real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Z) return;

    int ix = idx % N_X;
    int iz = idx / N_X;

    for (int iy = 1; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_optdepth[ic] += dev_optdepth[ic - N_X];
    }
}



#endif
