#include <cmath>

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: cfl_rate_calc
// Purpose: compute the CFL-limiting rate [1/time] at each cell
//
// Parallelisation: one thread per (Y,Z) ring
// Each thread first computes the same ring-mean angular momentum used by FARGO,
// then writes the CFL rate for every X cell in that ring
//
// Rate contributions (advection only -- diffusion is handled by implicit CN solves):
//   Advection Y (radial)         :  max(y_face^(d-1))*|v_y| / Delta V_y
//   Advection Z (polar)          :  max(sin(z_face))*|v_z|/(y*Delta cos(z))
//   Advection X (FARGO residual) :  |Delta Omega| / dx
//                                   Delta Omega = (velx - mean_ring(velx)) / Rc^2
//
// dt_CFL = CFL_NUM / max_over_cells( rate[cell] )
// =========================================================================================================================

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

    if (!ring_finite || !isfinite(velx_avg))
    {
        for (int ix = 0; ix < N_X; ix++)
        {
            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_cfl_rate[ic] = INFINITY;
        }
        return;
    }

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
        real speed_z = velz / yc;

        real omega_res = (velx - velx_avg) / fmax(Rc*Rc, 1.0e-30);

        real cfl_rate = 0.0;
        cfl_rate = fmax(cfl_rate, fabs(omega_res) / dx);
        cfl_rate = fmax(cfl_rate, fabs(vely)*cfl_invlen_y);
        cfl_rate = fmax(cfl_rate, fabs(speed_z)*cfl_invlen_z);

        dev_cfl_rate[ic] = isfinite(cfl_rate) ? cfl_rate : INFINITY;
    }
}

// =========================================================================================================================

__global__
void finite_verify (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif
    int *dev_badstate
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    bool finite = isfinite(dev_dustdens[idx]);
    finite = finite && isfinite(dev_dustmomx[idx]) && isfinite(dev_dustmomy[idx]) && isfinite(dev_dustmomz[idx]);
    finite = finite && isfinite(dev_dustvelx[idx]) && isfinite(dev_dustvely[idx]) && isfinite(dev_dustvelz[idx]);

    #ifdef RADIATION
    finite = finite && isfinite(dev_optdepth[idx]);
    #endif

    if (!finite) atomicCAS(dev_badstate, 0, idx + 1);
}

// =========================================================================================================================
