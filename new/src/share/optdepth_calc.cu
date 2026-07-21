#ifdef RADIATION

#include <fluid_kern.cuh>
#include <param_grid.cuh>








__global__
void optdepth_calc (real *dev_optdepth, const real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;

    real dy = _get_dy();
    real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
    real dy_len = y0*(dy - 1.0);

    dev_optdepth[idx] = KAPPA_0*dev_dustdens[idx]*dy_len;
}



#endif
