#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

__global__
void momentum_getv (const real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    real dens = dev_dustdens[idx_cell];

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);

    real mx = dev_dustmomx[idx_cell];
    real my = dev_dustmomy[idx_cell];
    real mz = dev_dustmomz[idx_cell];

    // recover primitive quantities and repair the conserved fallback state in near-vacuum cells
    real lx, vy, lz;
    _recover_dust_state(dens, R, mx, my, mz, lx, vy, lz);

    // write the synchronized conserved and primitive states to global memory
    dev_dustmomx[idx_cell] = mx;
    dev_dustmomy[idx_cell] = my;
    dev_dustmomz[idx_cell] = mz;
    dev_dustvelx[idx_cell] = lx;
    dev_dustvely[idx_cell] = vy;
    dev_dustvelz[idx_cell] = lz;
}
