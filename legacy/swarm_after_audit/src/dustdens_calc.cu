#ifdef SAVE_DENS

#include <param_grid.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dustdens_calc
// convert accumulated dust mass in every cell to volume or area density
// =========================================================================================================================

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    real cell_measure = _get_vol_x()*_get_vol_y(iy)*_get_vol_z(iz);
    dev_dustdens[idx_cell] /= cell_measure;
}

// =========================================================================================================================

#endif // SAVE_DENS
