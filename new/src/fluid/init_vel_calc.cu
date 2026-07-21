#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>













#ifdef DIFFUSION
static __device__ __forceinline__
real _get_init_ratio (const real *dev_dustdens, int idx, real yc, real zc)
{
    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);
    real h_g = _get_hg(Rc);
    real rhog = _get_rhog(Rc, Zc, h_g);

    return dev_dustdens[idx] / rhog;
}
#endif

__global__
void init_vel_calc (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real h_g = _get_hg(Rc);
    real omega = _get_omegaK(Rc);
    real v_K = Rc*omega;
    real eta = _get_eta(Rc, Zc, h_g);
    real vg_x = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(Rc, Zc, h_g);

    real v_R = 2.0*stokes*(vg_x - v_K) / (1.0 + stokes*stokes);
    real v_x = vg_x - 0.5*stokes*v_R;

    real speed_z_diff = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        real Dz = _get_nu(Rc, h_g) / SC_Z;
        real ratio = _get_init_ratio(dev_dustdens, idx, yc, zc);
        real grad_ratio;

        if (iz == 0)
        {
            int idx_next = idx + N_X*N_Y;
            real ratio_next = _get_init_ratio(dev_dustdens, idx_next, yc, zc + dz);
            grad_ratio = (ratio_next - ratio) / dz;
        }
        else if (iz == N_Z - 1)
        {
            int idx_prev = idx - N_X*N_Y;
            real ratio_prev = _get_init_ratio(dev_dustdens, idx_prev, yc, zc - dz);
            grad_ratio = (ratio - ratio_prev) / dz;
        }
        else
        {
            int idx_prev = idx - N_X*N_Y;
            int idx_next = idx + N_X*N_Y;
            real ratio_prev = _get_init_ratio(dev_dustdens, idx_prev, yc, zc - dz);
            real ratio_next = _get_init_ratio(dev_dustdens, idx_next, yc, zc + dz);
            grad_ratio = (ratio_next - ratio_prev) / (2.0*dz);
        }

        if (ratio > 0.0) speed_z_diff = Dz*grad_ratio / (yc*ratio);
    }
    #endif

    real v_y = v_R*sin(zc);
    real v_z = v_R*cos(zc) + speed_z_diff;

    dev_dustvelx[idx] = Rc*v_x;
    dev_dustvely[idx] = v_y;
    dev_dustvelz[idx] = yc*v_z;
}
