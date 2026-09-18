#ifdef SAVE_DENS

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dustdens_init
// clear the particle-mass accumulation grid before scattering
// =========================================================================================================================

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    dev_dustdens[idx_cell] = 0.0;
}

// =========================================================================================================================

#endif // SAVE_DENS
