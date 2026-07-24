#ifdef RADIATION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

__global__
void optdepth_calc (real *dev_optdepth, const real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;

    real dr = _get_yedge(iy)*(_get_dy() - 1.0);

    real extinction_dens = dev_dustdens[idx];
    if (N_Z == 1)
    {
        real yc = _get_ycent(iy);
        real H_g = _get_hg(yc)*yc;

        // close the vertically integrated disk with a well-mixed gas-scale-height profile
        extinction_dens /= sqrt(2.0*M_PI)*H_g;
    }

    // store the local radial optical-depth contribution of one cell
    dev_optdepth[idx] = KAPPA_0*extinction_dens*dr;
}

#endif
