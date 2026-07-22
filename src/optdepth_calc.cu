#ifdef RADIATION

#include <graffiti_kern.cuh>
#include <paramgrid.cuh>  // for _get_grid_volume

// =========================================================================================================================
// kernel: optdepth_calc
// convert scattered extinction cross section to one radial cell's optical-depth increment
// =========================================================================================================================

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real y0, dy;
    real volume = _get_grid_volume(idx, &y0, &dy);
    dev_optdepth[idx] /= volume;

    dev_optdepth[idx] *= y0*(dy - 1.0); // integrate extinction density across the logarithmic radial cell
}

// =========================================================================================================================

#endif // RADIATION
