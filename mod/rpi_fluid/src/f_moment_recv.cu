#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_moment_recv
// Purpose: Recover dust velocity from momentum density after each PPM advection sweep.
//          v_d = mom / rho_d.
//
// Two-branch recovery:
//   rho >= RHO_VAC  →  vel = mom/rho exactly
//   rho <  RHO_VAC  →  vacuum: lx = lx_K, vy = 0, lz = 0, with momenta reset consistently
//
// The vacuum branch prevents a genuine numerical error:
//   When a cell drains to rho ≈ 0 (inner disk cleared by radiation), conservative advection
//   sets mx ≈ 0.  Without the branch, lx ≈ 0 gives a retrograde FARGO
//   residual (0 - lx_K)/Rc ≈ -Omega_K*Rc ≈ -2.83 that crashes the CFL by a factor ~300.
//   Setting lx = lx_K (exact Keplerian) gives zero FARGO residual.
//   Setting vy = 0 prevents unphysical radial drift in a cell with no dust.
//
// Called after conservative advection and diffusion stages whenever primitive velocity must be
// recovered from the updated density and momentum state.
//
// Kernel f_moment_sync (below) does the inverse: mom = rho*v_d.  It also applies this same
// vacuum convention, so initialization, restart, and source updates cannot leave a stale vacuum
// velocity in the arrays used by FARGO or CFL.
// =========================================================================================================================

__global__
void f_moment_sync (const real *dev_dustdens, real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real rho = dev_dustdens[idx];
    real velx = dev_dustvelx[idx];
    real vely = dev_dustvely[idx];
    real velz = dev_dustvelz[idx];

    if (rho < RHO_VAC)
    {
        int iy = (idx / N_X) % N_Y;
        int iz = idx / (N_X * N_Y);
        real yc = Y_MIN*pow(_get_dy(), iy + 0.5);
        real zc = (N_Z > 1) ? Z_MIN + (iz + 0.5)*_get_dz() : 0.5*(Z_MIN + Z_MAX);
        real Rc = yc*sin(zc);

        velx = sqrt(G*M_S*fmax(Rc, 0.0));
        vely = 0.0;
        velz = 0.0;

        dev_dustvelx[idx] = velx;
        dev_dustvely[idx] = vely;
        dev_dustvelz[idx] = velz;
    }

    dev_dustmomx[idx] = rho*velx;
    dev_dustmomy[idx] = rho*vely;
    dev_dustmomz[idx] = rho*velz;
}

__global__
void f_moment_recv (const real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real rho = dev_dustdens[idx];

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X * N_Y);
    
    real yc = Y_MIN*pow(_get_dy(), iy + 0.5);
    real zc = (N_Z > 1) ? Z_MIN + (iz + 0.5)*_get_dz() : 0.5*(Z_MIN + Z_MAX);
    real Rc = yc*sin(zc);

    real momx = dev_dustmomx[idx];
    real momy = dev_dustmomy[idx];
    real momz = dev_dustmomz[idx];
    real velx, vely, velz;

    _recover_dust_state(rho, Rc, momx, momy, momz, velx, vely, velz);

    dev_dustmomx[idx] = momx;
    dev_dustmomy[idx] = momy;
    dev_dustmomz[idx] = momz;
    dev_dustvelx[idx] = velx;
    dev_dustvely[idx] = vely;
    dev_dustvelz[idx] = velz;
}

// =========================================================================================================================
