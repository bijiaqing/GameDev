#include <advection.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

__global__
void momentum_getv (const real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
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

    // recover primitive quantities and repair the conserved fallback state in near-vacuum cells
    real velx, vely, velz;
    _recover_dust_state(dens, Rc, momx, momy, momz, velx, vely, velz);

    // write the synchronized conserved and primitive states to global memory
    dev_dustmomx[idx] = momx;
    dev_dustmomy[idx] = momy;
    dev_dustmomz[idx] = momz;
    dev_dustvelx[idx] = velx;
    dev_dustvely[idx] = vely;
    dev_dustvelz[idx] = velz;
}
