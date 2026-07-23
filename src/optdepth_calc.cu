#ifdef RADIATION

#include <param_grid.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: optdepth_calc
// convert scattered extinction cross section to one radial cell's optical-depth increment
// =========================================================================================================================

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real volume = _get_vol_x()*_get_vol_y(iy)*_get_vol_z(iz);
    real dr = _get_yedge(iy)*(_get_dy() - 1.0);
    dev_optdepth[idx] /= volume;

    dev_optdepth[idx] *= dr; // integrate extinction density across the logarithmic radial cell
}

// =========================================================================================================================

#endif // RADIATION
