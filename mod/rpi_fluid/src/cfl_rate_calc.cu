#include <cmath>

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: cfl_rate_calc
// Purpose: compute the CFL-limiting rate [1/time] at each cell.
//
// Parallelisation: one thread per (Y,Z) ring.  Each thread first computes the same ring-mean
// angular momentum used by FARGO, then writes the CFL rate for every X cell in that ring.
//
// Rate contributions (advection only -- diffusion is handled by implicit CN solves):
//   Advection Y (radial)         :  max(r_face^(d-1))*|v_r| / Delta V_r
//   Advection Z (polar)          :  max(sin(theta_face))*|v_theta|/(r*Delta cos(theta))
//   Advection X (FARGO residual) :  |Delta Omega| / dphi
//                                  Delta Omega = (l_phi - mean_ring(l_phi)) / Rc^2
//
// dt_CFL = CFL_NUM / max_over_cells( rate[cell] )
// =========================================================================================================================

__global__
void cfl_rate_calc (real *dev_cfl_rate, 
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz)
{
    int idx_ring = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;

    real dx = (N_X > 1) ? _get_dx() : 1.0;
    real dy =             _get_dy();
    real dz = (N_Z > 1) ? _get_dz() : 1.0;

    real y0 = Y_MIN*pow(dy, static_cast<real>(iy));         // inner face radius
    real yc = Y_MIN*pow(dy, static_cast<real>(iy) + 0.5);   // cell-center radius
    real zc = (N_Z > 1) ? Z_MIN + (iz + 0.5)*dz : 0.5*(Z_MIN + Z_MAX);
    real Rc = yc*sin(zc);
    
    // Volume-coordinate CFL rates match the geometry-aware PPM tracing in Y and Z
    real pow_y = _get_powy();
    real vol_y = (pow(y0*dy, pow_y) - pow(y0, pow_y)) / pow_y;
    real rate_geom_y = pow(y0*dy, pow_y - 1.0) / vol_y;

    real rate_geom_z = 0.0;
    if (N_Z > 1)
    {
        real z0 = Z_MIN + iz*dz;
        real z1 = z0 + dz;
        real vol_z = cos(z0) - cos(z1);
        real sin_max = fmax(sin(z0), sin(z1));
        if (z0 <= 0.5*M_PI && z1 >= 0.5*M_PI) sin_max = 1.0;
        rate_geom_z = sin_max / (yc*vol_z);
    }
    
    int idx_base = iy*N_X + iz*N_X*N_Y;

    // Match f_advection_x exactly: its FARGO reference is the arithmetic ring mean,
    // including the Keplerian fallback already stored in vacuum cells by mom_recover
    real lx_avg = 0.0;
    bool ring_finite = true;
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx = idx_base + ix;
        real rho = dev_dustdens[idx];
        real momx = dev_dustmomx[idx];
        real momy = dev_dustmomy[idx];
        real momz = dev_dustmomz[idx];
        real lx = dev_dustvelx[idx];
        real vy = dev_dustvely[idx];
        real lz = dev_dustvelz[idx];

        if (!isfinite(rho) || !isfinite(momx) || !isfinite(momy) || !isfinite(momz)
            || !isfinite(lx) || !isfinite(vy) || !isfinite(lz))
        {
            ring_finite = false;
        }

        lx_avg += lx;
    }
    lx_avg /= static_cast<real>(N_X);

    if (!ring_finite || !isfinite(lx_avg))
    {
        for (int ix = 0; ix < N_X; ix++)
            dev_cfl_rate[idx_base + ix] = INFINITY;
        return;
    }

    // ---- Advection rates only ----
    // The nearest-integer FARGO permutation leaves at most half a cell of mean residual
    // displacement.  CFL_NUM=0.5 constrains the additional differential-ring displacement,
    // so their sum remains no larger than one cell for a full X sweep.
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx = idx_base + ix;
        if (dev_dustdens[idx] < RHO_VAC)
        {
            dev_cfl_rate[idx] = 0.0;
            continue;
        }

        real lx = dev_dustvelx[idx];
        real vy = dev_dustvely[idx];
        real lz = dev_dustvelz[idx];
        real vz = lz / yc;
        real d_omega = (N_X > 1) ? ((lx - lx_avg) / fmax(Rc*Rc, 1.0e-30)) : 0.0;

        real cfl_rate = 0.0;
        cfl_rate = fmax(cfl_rate, fabs(d_omega) / dx);
        cfl_rate = fmax(cfl_rate, fabs(vy)*rate_geom_y);
        cfl_rate = fmax(cfl_rate, fabs(vz)*rate_geom_z);

        dev_cfl_rate[idx] = isfinite(cfl_rate) ? cfl_rate : INFINITY;
    }
}

// =========================================================================================================================

__global__
void state_finite_check (int *dev_bad_state,
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz
    #ifdef RADIATION
    , const real *dev_optdepth, bool check_optdepth
    #endif
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    bool finite = isfinite(dev_dustdens[idx])
               && isfinite(dev_dustmomx[idx]) && isfinite(dev_dustmomy[idx]) && isfinite(dev_dustmomz[idx])
               && isfinite(dev_dustvelx[idx]) && isfinite(dev_dustvely[idx]) && isfinite(dev_dustvelz[idx]);

    #ifdef RADIATION
    if (check_optdepth) finite = finite && isfinite(dev_optdepth[idx]);
    #endif

    if (!finite) atomicCAS(dev_bad_state, 0, idx + 1);
}

// =========================================================================================================================
