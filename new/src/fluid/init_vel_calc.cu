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
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    real dz = _get_dz();

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);

    real R = y*sin(z);
    real Z = y*cos(z);

    // construct the steady drag-coupled azimuthal and cylindrical radial drift
    real h_g = _get_hg(R);
    real omega = _get_omegaK(R);
    real v_K = R*omega;
    real eta = _get_eta(R, Z, h_g);
    real vgas_x = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(R, Z, h_g);

    real vgas_R = 0.0;
    #ifdef VISC_ACCRETION
    vgas_R = _get_visc_vel(R, Z, h_g);
    #endif // VISC_ACCRETION

    real vel_R = (vgas_R + 2.0*stokes*(vgas_x - v_K)) / (1.0 + stokes*stokes);
    real vel_x = vgas_x - 0.5*stokes*vel_R;

    real vel_z_diff = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        // balance the initialized density gradient with spherical polar diffusion
        real diff_z = _get_nu(R, h_g) / SCHMIDT_Z;
        real dens = dev_dustdens[idx_cell];
        real grad_dens;

        // differentiate density with one-sided boundary and centred interior stencils
        if (iz == 0)
        {
            int idx_next = idx_cell + N_X*N_Y;
            real dens_next = dev_dustdens[idx_next];
            grad_dens = (dens_next - dens) / dz;
        }
        else if (iz == N_Z - 1)
        {
            int idx_prev = idx_cell - N_X*N_Y;
            real dens_prev = dev_dustdens[idx_prev];
            grad_dens = (dens - dens_prev) / dz;
        }
        else
        {
            int idx_prev = idx_cell - N_X*N_Y;
            int idx_next = idx_cell + N_X*N_Y;
            real dens_prev = dev_dustdens[idx_prev];
            real dens_next = dev_dustdens[idx_next];
            grad_dens = (dens_next - dens_prev) / (2.0*dz);
        }

        if (dens > 0.0) vel_z_diff = diff_z*grad_dens / (y*dens);
    }
    #endif

    // project cylindrical radial drift and add the spherical polar diffusion balance
    real vel_y = vel_R*sin(z);
    real vel_z = (N_Z > 1) ? vel_R*cos(z) + vel_z_diff : 0.0;

    dev_dustvelx[idx_cell] = R*vel_x;
    dev_dustvely[idx_cell] = vel_y;
    dev_dustvelz[idx_cell] = y*vel_z;
}
