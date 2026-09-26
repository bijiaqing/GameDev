#ifdef RADIATION

#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: optdepth_init
// clear the opacity-weighted mass grid before particle scattering
// =====================================================================================================================

__global__
void optdepth_init (real *dev_optdepth)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    dev_optdepth[idx_cell] = 0.0;
}

// =====================================================================================================================

#endif // RADIATION
