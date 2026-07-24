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
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real volume = _get_vol_x()*_get_vol_y(iy)*_get_vol_z(iz);
    dev_dustdens[idx] /= volume;
}

// =========================================================================================================================

#endif // SAVE_DENS
