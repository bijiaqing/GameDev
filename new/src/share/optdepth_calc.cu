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

    real dy = _get_dy();
    real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
    real dy_len = y0*(dy - 1.0);

    real extinction_dens = dev_dustdens[idx];
    if (N_Z == 1)
    {
        real yc = Y_MIN*pow(dy, iy + 0.5);
        real h_g = _get_hg(yc);
        real H_d = _get_hd(yc, h_g);

        // reconstruct midplane volume density from the evolved surface density
        extinction_dens /= sqrt(2.0*M_PI)*H_d;
    }

    // store the local radial optical-depth contribution of one cell
    dev_optdepth[idx] = KAPPA_0*extinction_dens*dy_len;
}

#endif
