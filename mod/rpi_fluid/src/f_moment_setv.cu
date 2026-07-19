#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_moment_setv
// Purpose: Enforce the common vacuum state, then set momx = dens*velx, momy = dens*vely, and momz = dens*velz.
//          This is the inverse of f_moment_getv for cells at or above RHO_VAC.
//
// Applying the vacuum convention here ensures that initialization, restart, and source updates
// cannot leave stale vacuum primitives in the arrays used by FARGO or the CFL calculation.
// =========================================================================================================================

__global__
void f_moment_setv (const real *dev_dustdens, real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real dens  = dev_dustdens[idx];
    real velx = dev_dustvelx[idx];
    real vely = dev_dustvely[idx];
    real velz = dev_dustvelz[idx];

    if (dens < RHO_VAC)
    {
        int iy = (idx / N_X) % N_Y;
        int iz = idx / (N_X*N_Y);

        real dy = _get_dy();
        real dz = _get_dz();

        real yc = Y_MIN*pow(dy, iy + 0.5);
        real zc = Z_MIN + (iz + 0.5)*dz;
        real Rc = yc*sin(zc);

        velx = sqrt(G*M_S*fmax(Rc, 0.0));
        vely = 0.0;
        velz = 0.0;

        dev_dustvelx[idx] = velx;
        dev_dustvely[idx] = vely;
        dev_dustvelz[idx] = velz;
    }

    dev_dustmomx[idx] = dens*velx;
    dev_dustmomy[idx] = dens*vely;
    dev_dustmomz[idx] = dens*velz;
}

// =========================================================================================================================
