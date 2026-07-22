#ifdef COLLISION

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: rs_grids_init
// Purpose: Initialize cuRAND random number generator states for each grid cell
// Dependencies: curand (curand_init)
// =========================================================================================================================

__global__
void rs_grids_init (curs *dev_rs_grids, int seed)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {
        curand_init(seed, idx, 0, &dev_rs_grids[idx]);
    }
}

// =========================================================================================================================

#endif // COLLISION
