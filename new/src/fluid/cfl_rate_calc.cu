#include <fluid_kern.cuh>
#include <param_grid.cuh>

__global__
void cfl_rate_calc (
    real *dev_cfl_rate, const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_Y*N_Z) return;

    int iy = idx % N_Y;
    int iz = idx / N_Y;

    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();

    real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;
    real Rc = yc*sin(zc);

    real pow_y = _get_powy();
    real vol_y = (pow(y0*dy, pow_y) - pow(y0, pow_y)) / pow_y;

    // construct radial and polar inverse length scales from finite-volume geometry
    real cfl_invlen_y = pow(y0*dy, pow_y - 1.0) / vol_y;
    real cfl_invlen_z = 0.0;

    if (N_Z > 1)
    {
        real z0 = Z_MIN + static_cast<real>(iz)*dz;
        real z1 = z0 + dz;

        real vol_z = cos(z0) - cos(z1);
        real sin_max = fmax(sin(z0), sin(z1));
        if (z0 <= 0.5*M_PI && z1 >= 0.5*M_PI) sin_max = 1.0;

        cfl_invlen_z = sin_max / (yc*vol_z);
    }

    real velx_avg = 0.0;
    bool ring_finite = true;

    // validate one ring and average azimuthal specific angular momentum for the FARGO frame
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        real dens = dev_dustdens[ic];
        real momx = dev_dustmomx[ic];
        real momy = dev_dustmomy[ic];
        real momz = dev_dustmomz[ic];
        real velx = dev_dustvelx[ic];
        real vely = dev_dustvely[ic];
        real velz = dev_dustvelz[ic];

        bool cell_finite = isfinite(dens);
        cell_finite = cell_finite && isfinite(velx) && isfinite(vely) && isfinite(velz);
        cell_finite = cell_finite && isfinite(momx) && isfinite(momy) && isfinite(momz);
        if (!cell_finite) ring_finite = false;

        velx_avg += velx;
    }

    velx_avg /= static_cast<real>(N_X);

    // force timestep rejection when any ring state is nonfinite
    if (!ring_finite || !isfinite(velx_avg))
    {
        for (int ix = 0; ix < N_X; ix++)
        {
            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_cfl_rate[ic] = INFINITY;
        }
        return;
    }

    // store the largest directional transport rate for each non-vacuum cell
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        if (dev_dustdens[ic] < RHO_VAC)
        {
            dev_cfl_rate[ic] = 0.0;
            continue;
        }

        real velx = dev_dustvelx[ic];
        real vely = dev_dustvely[ic];
        real velz = dev_dustvelz[ic];

        // convert angular primitives to residual azimuthal and linear polar speeds
        real speed_z = velz / yc;

        real omega_res = (velx - velx_avg) / fmax(Rc*Rc, 1.0e-30);

        real cfl_rate = 0.0;
        cfl_rate = fmax(cfl_rate, fabs(omega_res) / dx);
        cfl_rate = fmax(cfl_rate, fabs(vely)*cfl_invlen_y);
        cfl_rate = fmax(cfl_rate, fabs(speed_z)*cfl_invlen_z);

        dev_cfl_rate[ic] = isfinite(cfl_rate) ? cfl_rate : INFINITY;
    }
}
