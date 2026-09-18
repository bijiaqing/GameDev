#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

__global__
void init_vel_calc (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif // DIFFUSION
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
    real vx_g = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(R, Z, h_g);

    real vR_g = 0.0;
    #ifdef VISC_FLOW
    vR_g = _get_visc_vel(R, Z, h_g);
    #endif // VISC_FLOW

    real vR = (vR_g + 2.0*stokes*(vx_g - v_K)) / (1.0 + stokes*stokes);
    real vx = vx_g - 0.5*stokes*vR;

    real vz_diff = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        // balance the initialized selected diffusion gradient with spherical polar diffusion
        real diff_z = _get_diffusivity(R, Z, h_g, SCHMIDT_Z);
        real gas = _get_diffusion_weight(y, z);
        real rhod = dev_dustdens[idx_cell] / gas;
        real grad_rhod;

        // differentiate the selected density or concentration with one-sided boundary and centred interior stencils
        if (iz == 0)
        {
            int idx_next = idx_cell + N_X*N_Y;
            real rhod_next = dev_dustdens[idx_next] / _get_diffusion_weight(y, _get_zcent(iz + 1));
            grad_rhod = (rhod_next - rhod) / dz;
        }
        else if (iz == N_Z - 1)
        {
            int idx_prev = idx_cell - N_X*N_Y;
            real rhod_prev = dev_dustdens[idx_prev] / _get_diffusion_weight(y, _get_zcent(iz - 1));
            grad_rhod = (rhod - rhod_prev) / dz;
        }
        else
        {
            int idx_prev = idx_cell - N_X*N_Y;
            int idx_next = idx_cell + N_X*N_Y;
            real rhod_prev = dev_dustdens[idx_prev] / _get_diffusion_weight(y, _get_zcent(iz - 1));
            real rhod_next = dev_dustdens[idx_next] / _get_diffusion_weight(y, _get_zcent(iz + 1));
            grad_rhod = (rhod_next - rhod_prev) / (2.0*dz);
        }

        if (rhod > 0.0) vz_diff = diff_z*grad_rhod / (y*rhod);
    }
    #endif // DIFFUSION

    // project cylindrical radial drift and add the spherical polar diffusion balance
    real vy = vR*sin(z);
    real vz = (N_Z > 1) ? vR*cos(z) + vz_diff : 0.0;

    dev_dustvelx[idx_cell] = R*vx;
    dev_dustvely[idx_cell] = vy;
    dev_dustvelz[idx_cell] = y*vz;
}
