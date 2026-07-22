#ifdef SAVE_DENS

#include <graffiti_kern.cuh>
#include <helpers_paramgrid.cuh>  // for _get_grid_volume

// =========================================================================================================================
// Kernel: dustdens_calc
// Purpose: Normalize dust density by grid cell volume
// Dependencies: helpers_paramgrid.cuh (_get_grid_volume)
// =========================================================================================================================

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {	
        real volume = _get_grid_volume(idx);
        dev_dustdens[idx] /= volume;
    }
}

// =========================================================================================================================

#endif // SAVE_DENS
