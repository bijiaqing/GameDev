#include <fluid_kern.cuh>
#include <param_grid.cuh>
#ifdef VISC_ACCRETION
#include <param_phys.cuh>
#endif // VISC_ACCRETION

__global__
void cfl_rate_calc (
    real *dev_cfl_rates, const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz)
{
    int idx_ring = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;

    real dx = _get_dx();

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);

    real vol_y = _get_vol_y(iy);

    // construct radial and polar inverse length scales from finite-volume geometry
    real cfl_invlen_y = _get_area_y(iy + 1) / vol_y;
    real cfl_invlen_z = 0.0;

    if (N_Z > 1)
    {
        real z_i = _get_zedge(iz);
        real z_o = _get_zedge(iz + 1);

        real vol_z = _get_vol_z(iz);
        real sin_max = fmax(sin(z_i), sin(z_o));
        if (z_i <= 0.5*M_PI && z_o >= 0.5*M_PI) sin_max = 1.0;

        cfl_invlen_z = sin_max / (y*vol_z);
    }

    real lx_avg = 0.0;
    bool ring_finite = true;

    // validate one ring and average azimuthal specific angular momentum for the FARGO frame
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        real dens = dev_dustdens[idx_cell];
        real mx = dev_dustmomx[idx_cell];
        real my = dev_dustmomy[idx_cell];
        real mz = dev_dustmomz[idx_cell];
        real lx = dev_dustvelx[idx_cell];
        real vy = dev_dustvely[idx_cell];
        real lz = dev_dustvelz[idx_cell];

        bool cell_finite = isfinite(dens);
        cell_finite = cell_finite && isfinite(lx) && isfinite(vy) && isfinite(lz);
        cell_finite = cell_finite && isfinite(mx) && isfinite(my) && isfinite(mz);
        if (!cell_finite) ring_finite = false;

        lx_avg += lx;
    }

    lx_avg /= static_cast<real>(N_X);

    // force timestep rejection when any ring state is nonfinite
    if (!ring_finite || !isfinite(lx_avg))
    {
        for (int ix = 0; ix < N_X; ix++)
        {
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_cfl_rates[idx_cell] = INFINITY;
        }
        return;
    }

    // store the largest directional transport rate for each non-vacuum cell
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        if (dev_dustdens[idx_cell] < RHO_VAC)
        {
            dev_cfl_rates[idx_cell] = 0.0;
            continue;
        }

        real lx = dev_dustvelx[idx_cell];
        real vy = dev_dustvely[idx_cell];
        real lz = dev_dustvelz[idx_cell];

        // convert angular primitives to residual azimuthal and linear polar speeds
        real vel_z = lz / y;

        real omega_res = (lx - lx_avg) / fmax(R*R, 1.0e-30);

        real cfl_rate = 0.0;
        cfl_rate = fmax(cfl_rate, fabs(omega_res) / dx);
        cfl_rate = fmax(cfl_rate, fabs(vy)*cfl_invlen_y);
        cfl_rate = fmax(cfl_rate, fabs(vel_z)*cfl_invlen_z);

        #ifdef VISC_ACCRETION
        // include the analytic gas target velocity before a stiff source update transfers it to the dust
        real Z = y*cos(z);
        real h_g = _get_hg(R);
        real vgas_R = _get_visc_vel(R, Z, h_g);
        cfl_rate = fmax(cfl_rate, fabs(vgas_R*sin(z))*cfl_invlen_y);
        cfl_rate = fmax(cfl_rate, fabs(vgas_R*cos(z))*cfl_invlen_z);
        #endif // VISC_ACCRETION

        dev_cfl_rates[idx_cell] = isfinite(cfl_rate) ? cfl_rate : INFINITY;
    }
}
