#ifdef COLLISION

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: col_rate_init
// Purpose: Initialize collision rate arrays to zero
// Dependencies: None (simple initialization kernel)
// =========================================================================================================================

__global__
void col_rate_init (real *dev_col_rate, real *dev_col_expt, real *dev_col_rand)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {
        dev_col_rate[idx] = 0.0;
        dev_col_expt[idx] = 0.0;
        dev_col_rand[idx] = 0.0;
    }
}

// =========================================================================================================================

#endif // COLLISION
