#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =====================================================================================================================
// kernel: momentum_setv
// purpose: rebuild conserved momentum from density and primitives, resetting near-vacuum primitives first
//
// parallelization: one thread per grid cell
// =====================================================================================================================

__global__
void momentum_setv (const real *dev_dustdens, real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    real rhod = dev_dustdens[idx_cell];
    real lx = dev_dustvelx[idx_cell];
    real vy = dev_dustvely[idx_cell];
    real lz = dev_dustvelz[idx_cell];

    // reset near-vacuum primitives to the fallback state
    if (rhod < RHO_VAC)
    {
        int iy = (idx_cell / N_X) % N_Y;
        int iz = idx_cell / (N_X*N_Y);

        real y = _get_ycent(iy);
        real z = _get_zcent(iz);
        real R = y*sin(z);

        lx = sqrt(G*M_S*fmax(R, 0.0));
        vy = 0.0;
        lz = 0.0;

        dev_dustvelx[idx_cell] = lx;
        dev_dustvely[idx_cell] = vy;
        dev_dustvelz[idx_cell] = lz;
    }

    // rebuild conserved momentum from density and synchronized primitives
    dev_dustmomx[idx_cell] = rhod*lx;
    dev_dustmomy[idx_cell] = rhod*vy;
    dev_dustmomz[idx_cell] = rhod*lz;
}
