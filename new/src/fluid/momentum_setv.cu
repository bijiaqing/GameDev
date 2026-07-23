#include <fluid_kern.cuh>
#include <param_grid.cuh>

__global__
void momentum_setv (const real *dev_dustdens, real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real dens = dev_dustdens[idx];
    real lx = dev_dustvelx[idx];
    real vy = dev_dustvely[idx];
    real lz = dev_dustvelz[idx];

    // reset near-vacuum primitives to the fallback state
    if (dens < RHO_VAC)
    {
        int iy = (idx / N_X) % N_Y;
        int iz = idx / (N_X*N_Y);

        real yc = _get_ycent(iy);
        real zc = _get_zcent(iz);
        real Rc = yc*sin(zc);

        lx = sqrt(G*M_S*fmax(Rc, 0.0));
        vy = 0.0;
        lz = 0.0;

        dev_dustvelx[idx] = lx;
        dev_dustvely[idx] = vy;
        dev_dustvelz[idx] = lz;
    }

    // rebuild conserved momentum from density and synchronized primitives
    dev_dustmomx[idx] = dens*lx;
    dev_dustmomy[idx] = dens*vy;
    dev_dustmomz[idx] = dens*lz;
}
