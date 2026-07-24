#ifdef SAVE_DENS

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dustdens_init
// clear the particle-mass accumulation grid before scattering
// =========================================================================================================================

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    dev_dustdens[idx] = 0.0;
}

// =========================================================================================================================

#endif // SAVE_DENS
