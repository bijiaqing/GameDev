#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

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

    // construct the steady drag-coupled azimuthal and cylindrical radial drift
    real h_g = _get_hg(Rc);
    real omega = _get_omegaK(Rc);
    real v_K = Rc*omega;
    real eta = _get_eta(Rc, Zc, h_g);
    real vg_x = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(Rc, Zc, h_g);

    real vgas_R = 0.0;
    #ifdef VISC_ACCRETION
    vgas_R = _get_visc_vel(Rc, Zc, h_g);
    #endif // VISC_ACCRETION

    real v_R = (vgas_R + 2.0*stokes*(vg_x - v_K)) / (1.0 + stokes*stokes);
    real v_x = vg_x - 0.5*stokes*v_R;

    real speed_z_diff = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        // balance the initialized density gradient with spherical polar diffusion
        real Dz = _get_nu(Rc, h_g) / SC_Z;
        real dens = dev_dustdens[idx];
        real grad_dens;

        // differentiate density with one-sided boundary and centred interior stencils
        if (iz == 0)
        {
            int idx_next = idx + N_X*N_Y;
            real dens_next = dev_dustdens[idx_next];
            grad_dens = (dens_next - dens) / dz;
        }
        else if (iz == N_Z - 1)
        {
            int idx_prev = idx - N_X*N_Y;
            real dens_prev = dev_dustdens[idx_prev];
            grad_dens = (dens - dens_prev) / dz;
        }
        else
        {
            int idx_prev = idx - N_X*N_Y;
            int idx_next = idx + N_X*N_Y;
            real dens_prev = dev_dustdens[idx_prev];
            real dens_next = dev_dustdens[idx_next];
            grad_dens = (dens_next - dens_prev) / (2.0*dz);
        }

        if (dens > 0.0) speed_z_diff = Dz*grad_dens / (yc*dens);
    }
    #endif

    // project cylindrical radial drift and add the spherical polar diffusion balance
    real v_y = v_R*sin(zc);
    real v_z = (N_Z > 1) ? v_R*cos(zc) + speed_z_diff : 0.0;

    dev_dustvelx[idx] = Rc*v_x;
    dev_dustvely[idx] = v_y;
    dev_dustvelz[idx] = yc*v_z;
}
