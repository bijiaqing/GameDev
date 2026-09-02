#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffusion_zth
// purpose: polar diffusion of dust density with spherical-geometry conservative momentum transport
//
// parallelization: one thread per azimuthal-radial column with a serial loop over N_Z polar cells
//
// per call:
//   1 polar Crank-Nicolson coefficient construction with zero boundary fluxes
//   2 positivity-controlled subcycling and tridiagonal solution
//   3 time-centred diffusive mass flux construction
//   4 donor-state momentum transport with the diffusing mass
// =========================================================================================================================

__global__
void diffusion_zth (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Y) return;
    if (N_Z == 1) return;
    if (dt <= 0.0) return;

    int ix = idx_col % N_X;
    int iy = idx_col / N_X;

    real dz = _get_dz();

    real y = _get_ycent(iy);

    // load dust density along one polar column
    real rhod[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        rhod[iz] = dev_dustdens[idx_cell];
    }

    // assemble full-step Crank-Nicolson face couplings and measure the largest local coefficient sum
    real cn_lower[N_Z], cn_diag[N_Z], cn_upper[N_Z];
    real max_cn_sum = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z_i = _get_zface(iz);
        real z_o = _get_zface(iz + 1);

        real vol_z  = _get_vol_z(iz);
        real dz_len = y*dz;

        real diff_zi = 0.0;
        if (iz > 0)
        {
            real R_i = y*sin(z_i);

            real h_gi = _get_hg(R_i);
            diff_zi = _get_nu(R_i, h_gi) / SCHMIDT_Z;
        }

        real diff_zo = 0.0;
        if (iz < N_Z - 1)
        {
            real R_o = y*sin(z_o);

            real h_go = _get_hg(R_o);
            diff_zo = _get_nu(R_o, h_go) / SCHMIDT_Z;
        }

        real cn_i = (iz > 0)       ? (0.5*dt*sin(z_i)*diff_zi / (y*dz_len*vol_z)) : 0.0;
        real cn_o = (iz < N_Z - 1) ? (0.5*dt*sin(z_o)*diff_zo / (y*dz_len*vol_z)) : 0.0;

        cn_lower[iz] = -cn_i;
        cn_upper[iz] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
    }

    // choose positivity-controlled substeps and rescale the implicit matrix coefficients
    int sub_count = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
    if (sub_count < 1) sub_count = 1;

    real dt_sub = dt / static_cast<real>(sub_count);
    real inv_sub_count = 1.0 / static_cast<real>(sub_count);
    for (int iz = 0; iz < N_Z; iz++)
    {
        cn_lower[iz] *= inv_sub_count;
        cn_upper[iz] *= inv_sub_count;
        cn_diag[iz] = 1.0 - cn_lower[iz] - cn_upper[iz];
    }

    // advance density and momentum through each diffusion substep
    real rhod_work[N_Z];
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        {
            real upper_work[N_Z], rhod_rhs[N_Z];

            // build the explicit Crank-Nicolson right-hand side with zero boundary gradients
            for (int iz = 0; iz < N_Z; iz++)
            {
                real cn_i = -cn_lower[iz];
                real cn_o = -cn_upper[iz];

                real rhod_prev = (iz > 0)       ? rhod[iz - 1] : rhod[iz];
                real rhod_next = (iz < N_Z - 1) ? rhod[iz + 1] : rhod[iz];

                rhod_rhs[iz] = cn_i*rhod_prev + (1.0 - cn_i - cn_o)*rhod[iz] + cn_o*rhod_next;
            }

            // initialize the Thomas forward elimination
            upper_work[0] = cn_upper[0]  / cn_diag[0];
            rhod_work[0] = rhod_rhs[0] / cn_diag[0];

            // eliminate the lower diagonal of the implicit system
            for (int iz = 1; iz < N_Z; iz++)
            {
                real pivot = cn_diag[iz] - cn_lower[iz]*upper_work[iz - 1];

                upper_work[iz] = (iz < N_Z - 1) ? (cn_upper[iz] / pivot) : 0.0;
                rhod_work[iz] = (rhod_rhs[iz] - cn_lower[iz]*rhod_work[iz - 1]) / pivot;
            }

            // back-substitute the density solution
            for (int iz = N_Z - 2; iz >= 0; iz--)
            {
                rhod_work[iz] -= upper_work[iz]*rhod_work[iz + 1];
            }
        }

        {
            real mass_flux[N_Z], moment_flux[N_Z];

            // reconstruct time-centred outward diffusive mass fluxes with zero boundary fluxes
            for (int iz = 0; iz < N_Z; iz++)
            {
                if (iz == N_Z - 1)
                {
                    mass_flux[iz] = 0.0;
                    continue;
                }

                real vol_z = _get_vol_z(iz);

                real cn_o = -cn_upper[iz];

                mass_flux[iz]  = -(cn_o*y*vol_z / dt_sub);
                mass_flux[iz] *= (rhod[iz + 1] - rhod[iz]) + (rhod_work[iz + 1] - rhod_work[iz]);
            }

            // combine each face mass flux with the donor azimuthal primitive quantity
            for (int iz = 0; iz < N_Z; iz++)
            {
                int iz_up = (mass_flux[iz] >= 0.0) ? iz : iz + 1;
                if (iz == N_Z - 1) iz_up = iz;

                real z_up = _get_zcent(iz_up);
                real R_up = y*sin(z_up);
                real rhod_up = rhod[iz_up];

                int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
                real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / rhod_up : sqrt(G*M_S*fmax(R_up, 0.0));

                moment_flux[iz] = mass_flux[iz]*lx_up;
            }

            // update azimuthal momentum from the spherical conservative flux divergence
            for (int iz = 0; iz < N_Z; iz++)
            {
                real vol_z = _get_vol_z(iz);
                real flux_i = (iz > 0) ? moment_flux[iz - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomx[idx_cell] -= dt_sub*(moment_flux[iz] - flux_i) / (y*vol_z);
            }

            // combine each face mass flux with the donor radial velocity
            for (int iz = 0; iz < N_Z; iz++)
            {
                int iz_up = (mass_flux[iz] >= 0.0) ? iz : iz + 1;
                if (iz == N_Z - 1) iz_up = iz;

                real rhod_up = rhod[iz_up];

                int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
                real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / rhod_up : 0.0;

                moment_flux[iz] = mass_flux[iz]*vy_up;
            }

            // update radial momentum from the spherical conservative flux divergence
            for (int iz = 0; iz < N_Z; iz++)
            {
                real vol_z = _get_vol_z(iz);
                real flux_i = (iz > 0) ? moment_flux[iz - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomy[idx_cell] -= dt_sub*(moment_flux[iz] - flux_i) / (y*vol_z);
            }

            // combine each face mass flux with the donor polar primitive quantity
            for (int iz = 0; iz < N_Z; iz++)
            {
                int iz_up = (mass_flux[iz] >= 0.0) ? iz : iz + 1;
                if (iz == N_Z - 1) iz_up = iz;

                real rhod_up = rhod[iz_up];

                int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
                real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / rhod_up : 0.0;

                moment_flux[iz] = mass_flux[iz]*lz_up;
            }

            // update polar momentum and accept the density solution for this substep
            for (int iz = 0; iz < N_Z; iz++)
            {
                real vol_z = _get_vol_z(iz);
                real flux_i = (iz > 0) ? moment_flux[iz - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomz[idx_cell] -= dt_sub*(moment_flux[iz] - flux_i) / (y*vol_z);

                rhod[iz] = rhod_work[iz];
            }
        }
    }

    // store the final dust density
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = rhod[iz];
    }
}

#endif // DIFFUSION
