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
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    real cell_measure = _get_vol_x()*_get_vol_y(iy)*_get_vol_z(iz);
    real dr = _get_yedge(iy)*(_get_dy() - 1.0);
    dev_optdepth[idx_cell] /= cell_measure;

    dev_optdepth[idx_cell] *= dr; // integrate extinction density across the logarithmic radial cell
}

// =========================================================================================================================

#endif // RADIATION
