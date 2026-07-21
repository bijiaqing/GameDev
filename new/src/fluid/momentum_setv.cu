#include <fluid_kern.cuh>
#include <param_grid.cuh>










__global__
void momentum_setv (const real *dev_dustdens, real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
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
