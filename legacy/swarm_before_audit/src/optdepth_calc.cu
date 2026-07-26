#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>
#include <helpers_paramgrid.cuh>  // for _get_grid_volume

// =========================================================================================================================
// Kernel: optdepth_calc
// Purpose: Normalize optical depth by grid cell volume and prepare for radial integration
// Dependencies: helpers_paramgrid.cuh (_get_grid_volume)
// =========================================================================================================================

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {	
        real y0, dy;
        real volume = _get_grid_volume(idx, &y0, &dy);
        dev_optdepth[idx] /= volume;
        dev_optdepth[idx] *= y0*(dy - 1.0); // prepare for radial integration
    }
}

// =========================================================================================================================

#endif // RADIATION
