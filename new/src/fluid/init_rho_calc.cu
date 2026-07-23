#include <curand_kernel.h> // curand_init, curand_normal_double, curandState

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

__global__
void init_rho_calc (real *dev_dustdens, const real *dev_initdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int ix = idx % N_X;
    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X * N_Y);

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real h_g = _get_hg(Rc);

    real H_g = h_g*Rc;
    real H_d = _get_hd(Rc, h_g);

    // interpolate the convolved dust surface density at cylindrical radius
    real sigma_d = 0.0;
    if (Rc >= Y_MIN && Rc <= Y_MAX)
    {
        real du = (Y_MAX - Y_MIN) / static_cast<real>(N_Y);
        int iu = static_cast<int>((Rc - Y_MIN) / du);
        if (iu >= N_Y) iu = N_Y - 1;

        real frac_u = (Rc - (Y_MIN + iu*du)) / du;
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
        // embed the surface profile with the settled dust-to-gas vertical stratification
        real sigma_g = _get_sigma_g(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);
        real ratio_mid = sigma_d*H_g / (sigma_g*H_d);
        real settle_exp = exp(-0.5*Zc*Zc*(1.0/(H_d*H_d) - 1.0/(H_g*H_g)));
        dens = rhog*ratio_mid*settle_exp;
    }

    // apply azimuthal density noise shared across radius and polar angle
    curandState rng;
    curand_init(static_cast<unsigned long long>(ix), 0ULL, 0ULL, &rng);
    real xi = curand_normal_double(&rng);
    dens = fmax(dens*(1.0 + 0.1*xi), 0.0);

    dev_dustdens[idx] = dens;
}
