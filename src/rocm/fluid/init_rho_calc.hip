#include <hiprand/hiprand_kernel.h> // hiprand_init, hiprand_normal_double, hiprandState

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// interpolate the host-convolved surface-density profile on its uniform cylindrical-radius axis
__device__ __forceinline__
real initdens_lerp (real R, const real *dev_initdens)
{
    real R_min = _get_init_Rmin();
    if (R < R_min || R > Y_MAX) return 0.0;

    real dR = (Y_MAX - R_min) / static_cast<real>(N_Y);
    int idx_src = static_cast<int>((R - R_min) / dR);
    if (idx_src >= N_Y) idx_src = N_Y - 1;

    real R_src = R_min + static_cast<real>(idx_src)*dR;
    real frac_R = (R - R_src) / dR;

    return (1.0 - frac_R)*dev_initdens[idx_src] + frac_R*dev_initdens[idx_src + 1];
}

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
    real sigma_d = initdens_lerp(R, dev_initdens);

    real rhod;
    if (N_Z == 1)
    {
        // evolve the vertically integrated dust surface density in a 2D disk
        rhod = sigma_d;
    }
    else
    {
        // embed the surface profile with the density-diffusion equilibrium
        rhod = sigma_d*exp(-0.5*Z*Z/(H_d*H_d)) / (sqrt(2.0*M_PI)*H_d);
    }

    // apply azimuthal density noise shared across radius and polar angle
    hiprandState rngstate;
    hiprand_init(static_cast<unsigned long long>(ix), 0ULL, 0ULL, &rngstate);
    real noise_x = hiprand_normal_double(&rngstate);
    rhod = fmax(rhod*(1.0 + 0.1*noise_x), 0.0);

    dev_dustdens[idx_cell] = rhod;
}
