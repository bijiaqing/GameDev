#include <curand_kernel.h> // curand_init, curand_normal_double, curandState

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

__global__
void init_rho_calc (real *dev_dustdens, const real *dev_initdens)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int ix = idx_cell % N_X;
    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X * N_Y);

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);

    real R = y*sin(z);
    real Z = y*cos(z);

    real h_g = _get_hg(R);

    real H_d = _get_hd(R, h_g);

    // interpolate the convolved dust surface density at cylindrical radius
    real sigma_d = 0.0;
    if (R >= Y_MIN && R <= Y_MAX)
    {
        real du = (Y_MAX - Y_MIN) / static_cast<real>(N_Y);
        int iu = static_cast<int>((R - Y_MIN) / du);
        if (iu >= N_Y) iu = N_Y - 1;

        real frac_u = (R - (Y_MIN + iu*du)) / du;
        sigma_d = (1.0 - frac_u)*dev_initdens[iu] + frac_u*dev_initdens[iu + 1];
    }

    real dens;
    if (N_Z == 1)
    {
        // evolve the vertically integrated dust surface density in a 2D disk
        dens = sigma_d;
    }
    else
    {
        // embed the surface profile with the density-diffusion equilibrium
        dens = sigma_d*exp(-0.5*Z*Z/(H_d*H_d)) / (sqrt(2.0*M_PI)*H_d);
    }

    // apply azimuthal density noise shared across radius and polar angle
    curandState rng;
    curand_init(static_cast<unsigned long long>(ix), 0ULL, 0ULL, &rng);
    real xi = curand_normal_double(&rng);
    dens = fmax(dens*(1.0 + 0.1*xi), 0.0);

    dev_dustdens[idx_cell] = dens;
}
