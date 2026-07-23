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

    real yc = _get_ycent(iy);
    real zc = _get_zcent(iz);
    real Rc = yc*sin(zc);

    real mx = dev_dustmomx[idx];
    real my = dev_dustmomy[idx];
    real mz = dev_dustmomz[idx];

    // recover primitive quantities and repair the conserved fallback state in near-vacuum cells
    real lx, vy, lz;
    _recover_dust_state(dens, Rc, mx, my, mz, lx, vy, lz);

    // write the synchronized conserved and primitive states to global memory
    dev_dustmomx[idx] = mx;
    dev_dustmomy[idx] = my;
    dev_dustmomz[idx] = mz;
    dev_dustvelx[idx] = lx;
    dev_dustvely[idx] = vy;
    dev_dustvelz[idx] = lz;
}
