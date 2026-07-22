#ifdef SAVE_DENS

#include <graffiti_kern.cuh>
#include <helpers_paramgrid.cuh>  // for _get_grid_volume

// =========================================================================================================================
// kernel: dustdens_calc
// convert accumulated dust mass in every cell to volume or area density
// =========================================================================================================================

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {	
        real volume = _get_grid_volume(idx); // use the cell measure of the active spatial dimension
        dev_dustdens[idx] /= volume;
    }
}

// =========================================================================================================================

#endif // SAVE_DENS
