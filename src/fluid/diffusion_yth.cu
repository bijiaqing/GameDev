#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffusion_yth
// purpose: radial diffusion of dust density with geometry-aware conservative momentum transport
//
// parallelization: one thread per azimuthal-polar column with a serial loop over N_Y radial cells
//
// per call:
//   1 radial Crank-Nicolson coefficient construction with zero boundary fluxes
//   2 positivity-controlled subcycling and tridiagonal solution
//   3 time-centred diffusive mass flux construction
//   4 donor-state momentum transport with the diffusing mass
// =========================================================================================================================

__device__ __forceinline__
real _thread_dr_cent_i (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0) / _get_dy(); }

__device__ __forceinline__
real _thread_dr_cent_o (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0); }

__global__
void diffusion_yth (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Z) return;
    if (dt <= 0.0) return;

    int ix = idx_col % N_X;
    int iz = idx_col / N_X;

    real z = _get_zcent(iz);

    // load dust density along one radial column
    real rhod[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        rhod[iy] = dev_dustdens[idx_cell];
    }

    // assemble full-step Crank-Nicolson face couplings and measure the largest local coefficient sum
    real cn_lower[N_Y], cn_diag[N_Y], cn_upper[N_Y];
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y_i = _get_yface(iy);
        real y_o = _get_yface(iy + 1);

        real vol_y = _get_vol_y(iy);

        real dr_i = _thread_dr_cent_i(iy);
        real dr_o = _thread_dr_cent_o(iy);

        real R_i = y_i*sin(z);
        real h_gi = _get_hg(R_i);
        real diff_yi = _get_diffusivity(R_i, y_i*cos(z), h_gi, SCHMIDT_Y);

        real R_o = y_o*sin(z);
        real h_go = _get_hg(R_o);
        real diff_yo = _get_diffusivity(R_o, y_o*cos(z), h_go, SCHMIDT_Y);

        real area_i = _get_area_y(iy);
        real area_o = _get_area_y(iy + 1);

        real cn_i = (iy > 0)       ? (0.5*dt*area_i*diff_yi / (dr_i*vol_y)) : 0.0;
        real cn_o = (iy < N_Y - 1) ? (0.5*dt*area_o*diff_yo / (dr_o*vol_y)) : 0.0;

        // Solve concentration with gas-weighted face conductances; the diagonal
        // is also the density outgoing sum used by positivity subcycling.
        real gas = _get_diffusion_weight(_get_ycent(iy), z);
        cn_i *= _get_diffusion_weight(y_i, z) / gas;
        cn_o *= _get_diffusion_weight(y_o, z) / gas;
        cn_lower[iy] = -cn_i;
        cn_upper[iy] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
    }

    // choose positivity-controlled substeps and rescale the implicit matrix coefficients
    int sub_count = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
    if (sub_count < 1) sub_count = 1;

    real dt_sub = dt / static_cast<real>(sub_count);
    real inv_sub_count = 1.0 / static_cast<real>(sub_count);
    for (int iy = 0; iy < N_Y; iy++)
    {
        cn_lower[iy] *= inv_sub_count;
        cn_upper[iy] *= inv_sub_count;

        cn_diag[iy] = 1.0 - cn_lower[iy] - cn_upper[iy];
    }

    // advance density and momentum through each diffusion substep
    real rhod_work[N_Y];
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        {
            real upper_work[N_Y], rhod_rhs[N_Y];

            // Build the RHS for density (weight=1) or concentration; boundary fluxes vanish.
            for (int iy = 0; iy < N_Y; iy++)
            {
                real cn_i = -cn_lower[iy];
                real cn_o = -cn_upper[iy];

                real rhod_prev = (iy > 0)       ? rhod[iy - 1] / _get_diffusion_weight(_get_ycent(iy - 1), z) : rhod[iy] / _get_diffusion_weight(_get_ycent(iy), z);
                real rhod_next = (iy < N_Y - 1) ? rhod[iy + 1] / _get_diffusion_weight(_get_ycent(iy + 1), z) : rhod[iy] / _get_diffusion_weight(_get_ycent(iy), z);

                rhod_rhs[iy] = cn_i*rhod_prev + (1.0 - cn_i - cn_o)*rhod[iy] / _get_diffusion_weight(_get_ycent(iy), z) + cn_o*rhod_next;
            }

            // initialize the Thomas forward elimination
            upper_work[0] = cn_upper[0]  / cn_diag[0];
            rhod_work[0] = rhod_rhs[0] / cn_diag[0];

            // eliminate the lower diagonal of the implicit system
            for (int iy = 1; iy < N_Y; iy++)
            {
                real pivot = cn_diag[iy] - cn_lower[iy]*upper_work[iy - 1];

                upper_work[iy] = (iy < N_Y - 1) ? (cn_upper[iy] / pivot) : 0.0;
                rhod_work[iy] = (rhod_rhs[iy] - cn_lower[iy]*rhod_work[iy - 1]) / pivot;
            }

            // Back-substitute the selected density or concentration solution.
            for (int iy = N_Y - 2; iy >= 0; iy--)
            {
                rhod_work[iy] -= upper_work[iy]*rhod_work[iy + 1];
            }
        }

        {
            real mass_flux[N_Y], moment_flux[N_Y];

            // Reconstruct mass flux from old density/weight and the solved diffused variable, with zero boundary fluxes
            for (int iy = 0; iy < N_Y; iy++)
            {
                if (iy == N_Y - 1)
                {
                    mass_flux[iy] = 0.0;
                    continue;
                }

                real vol_y = _get_vol_y(iy);

                real cn_o = -cn_upper[iy];

                mass_flux[iy]  = -(_get_diffusion_weight(_get_ycent(iy), z)*cn_o*vol_y / dt_sub);
                mass_flux[iy] *= (rhod[iy + 1] / _get_diffusion_weight(_get_ycent(iy + 1), z) - rhod[iy] / _get_diffusion_weight(_get_ycent(iy), z)) + (rhod_work[iy + 1] - rhod_work[iy]);
            }

            // bound total outward mass from each old radial donor using its finite-volume measure
            for (int iy = 0; iy < N_Y; iy++)
            {
                real flux_i = (iy > 0) ? mass_flux[iy - 1] : 0.0;
                real out_rate = fmax(mass_flux[iy], 0.0) + fmax(-flux_i, 0.0);
                real mass = fmax(rhod[iy], 0.0)*_get_vol_y(iy);
                rhod_work[iy] = (out_rate > 0.0)
                    ? fmin(1.0, POS_LIMIT*mass / (dt_sub*out_rate)) : 1.0;
            }

            for (int iy = 0; iy < N_Y - 1; iy++)
            {
                int iy_up = (mass_flux[iy] >= 0.0) ? iy : iy + 1;
                mass_flux[iy] *= rhod_work[iy_up];
            }

            // reconstruct accepted density from the limited zero-flux face transfers
            for (int iy = 0; iy < N_Y; iy++)
            {
                real flux_i = (iy > 0) ? mass_flux[iy - 1] : 0.0;
                rhod_work[iy] = rhod[iy]
                    - dt_sub*(mass_flux[iy] - flux_i) / _get_vol_y(iy);
            }

            // combine each face mass flux with the donor azimuthal primitive quantity
            for (int iy = 0; iy < N_Y; iy++)
            {
                int iy_up = (mass_flux[iy] >= 0.0) ? iy : iy + 1;
                if (iy == N_Y - 1) iy_up = iy;

                real y_up = _get_ycent(iy_up);
                real R_up = y_up*sin(z);
                real rhod_up = rhod[iy_up];

                int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
                real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / rhod_up : sqrt(G*M_S*fmax(R_up, 0.0));

                moment_flux[iy] = mass_flux[iy]*lx_up;
            }

            // update azimuthal momentum from the geometry-aware conservative flux divergence
            for (int iy = 0; iy < N_Y; iy++)
            {
                real vol_y = _get_vol_y(iy);
                real flux_i = (iy > 0) ? moment_flux[iy - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomx[idx_cell] -= dt_sub*(moment_flux[iy] - flux_i) / vol_y;
            }

            // combine each face mass flux with the donor radial velocity
            for (int iy = 0; iy < N_Y; iy++)
            {
                int iy_up = (mass_flux[iy] >= 0.0) ? iy : iy + 1;
                if (iy == N_Y - 1) iy_up = iy;

                real rhod_up = rhod[iy_up];

                int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
                real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / rhod_up : 0.0;

                moment_flux[iy] = mass_flux[iy]*vy_up;
            }

            // update radial momentum from the geometry-aware conservative flux divergence
            for (int iy = 0; iy < N_Y; iy++)
            {
                real vol_y = _get_vol_y(iy);
                real flux_i = (iy > 0) ? moment_flux[iy - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomy[idx_cell] -= dt_sub*(moment_flux[iy] - flux_i) / vol_y;
            }

            // combine each face mass flux with the donor polar primitive quantity
            for (int iy = 0; iy < N_Y; iy++)
            {
                int iy_up = (mass_flux[iy] >= 0.0) ? iy : iy + 1;
                if (iy == N_Y - 1) iy_up = iy;

                real rhod_up = rhod[iy_up];

                int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
                real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / rhod_up : 0.0;

                moment_flux[iy] = mass_flux[iy]*lz_up;
            }

            // update polar momentum and accept the density solution for this substep
            for (int iy = 0; iy < N_Y; iy++)
            {
                real vol_y = _get_vol_y(iy);
                real flux_i = (iy > 0) ? moment_flux[iy - 1] : 0.0;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dev_dustmomz[idx_cell] -= dt_sub*(moment_flux[iy] - flux_i) / vol_y;

                rhod[iy] = rhod_work[iy];
            }
        }
    }

    // store the final dust density
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = rhod[iy];
    }
}

#endif // DIFFUSION
