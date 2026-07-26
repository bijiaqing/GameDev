#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_moment_getv
// Purpose: Recover the dust primitive variables from their conserved densities after each advection or diffusion stage.
//          velx = momx/dens, vely = momy/dens, velz = momz/dens.
//
// Two-branch recovery:
//   dens >= RHO_VAC  →  recover every primitive exactly from its conserved density
//   dens <  RHO_VAC  →  vacuum: velx = velx_K, vely = 0, velz = 0, with conserved fields reset consistently
//
// The vacuum branch prevents a genuine numerical error:
//   When a cell drains to dens ≈ 0 (inner disk cleared by radiation), conservative advection
//   sets momx ≈ 0.  Without the branch, velx ≈ 0 gives a retrograde FARGO
//   residual (0 - velx_K)/Rc ≈ -Omega_K*Rc ≈ -2.83 that crashes the CFL by a factor ~300.
//   Setting velx = velx_K (exact Keplerian) gives zero FARGO residual.
//   Setting vely = 0 prevents unphysical radial drift in a cell with no dust.
//
// Called whenever primitive variables must be recovered from the updated conserved
// state. f_moment_setv performs the inverse operation for cells at or above RHO_VAC.
// =========================================================================================================================

__global__
void f_moment_getv (const real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real dens = dev_dustdens[idx];

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;
    real Rc = yc*sin(zc);

    real momx = dev_dustmomx[idx];
    real momy = dev_dustmomy[idx];
    real momz = dev_dustmomz[idx];

    real velx, vely, velz;
    _recover_dust_state(dens, Rc, momx, momy, momz, velx, vely, velz);

    dev_dustmomx[idx] = momx;
    dev_dustmomy[idx] = momy;
    dev_dustmomz[idx] = momz;
    dev_dustvelx[idx] = velx;
    dev_dustvely[idx] = vely;
    dev_dustvelz[idx] = velz;
}

// =========================================================================================================================
