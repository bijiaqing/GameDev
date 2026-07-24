#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffus_y_calc
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
real _get_dr_cc_i (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0) / _get_dy(); }

__device__ __forceinline__
real _get_dr_cc_o (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0); }

__global__
void diffus_y_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Z) return;
    if (dt <= 0.0) return;

    int ix = idx_col % N_X;
    int iz = idx_col / N_X;

    real z = _get_zcent(iz);

    // load dust density along one radial column
    real dens[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dens[iy] = dev_dustdens[idx_cell];
    }

    // assemble full-step Crank-Nicolson face couplings and measure the largest local coefficient sum
    real cn_lower[N_Y], cn_diag[N_Y], cn_upper[N_Y], dens_rhs[N_Y];
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y_i = _get_yedge(iy);
        real y_o = _get_yedge(iy + 1);

        real vol_y = _get_vol_y(iy);

        real dr_i = _get_dr_cc_i(iy);
        real dr_o = _get_dr_cc_o(iy);

        real R_i = y_i*sin(z);
        real h_i = _get_hg(R_i);
        real diff_y_i = _get_nu(R_i, h_i) / SCHMIDT_Y;

        real R_o = y_o*sin(z);
        real h_o = _get_hg(R_o);
        real diff_y_o = _get_nu(R_o, h_o) / SCHMIDT_Y;

        real area_i = _get_area_y(iy);
        real area_o = _get_area_y(iy + 1);

        real cn_i = (iy > 0)       ? (0.5*dt*area_i*diff_y_i / (dr_i*vol_y)) : 0.0;
        real cn_o = (iy < N_Y - 1) ? (0.5*dt*area_o*diff_y_o / (dr_o*vol_y)) : 0.0;

        cn_lower[iy] = -cn_i;
        cn_upper[iy] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
    }

    // choose positivity-controlled substeps and rescale the implicit matrix coefficients
    int n_sub = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
    if (n_sub < 1) n_sub = 1;

    real dt_sub = dt / static_cast<real>(n_sub);
    real inv_n_sub = 1.0 / static_cast<real>(n_sub);
    for (int iy = 0; iy < N_Y; iy++)
    {
        cn_lower[iy] *= inv_n_sub;
        cn_upper[iy] *= inv_n_sub;

        cn_diag[iy] = 1.0 - cn_lower[iy] - cn_upper[iy];
    }

    // advance density and momentum through each diffusion substep
    real upper_work[N_Y], dens_work[N_Y];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        // build the explicit Crank-Nicolson right-hand side with zero boundary gradients
        for (int iy = 0; iy < N_Y; iy++)
        {
            real cn_i = -cn_lower[iy];
            real cn_o = -cn_upper[iy];

            real dens_prev = (iy > 0)       ? dens[iy - 1] : dens[iy];
            real dens_next = (iy < N_Y - 1) ? dens[iy + 1] : dens[iy];

            dens_rhs[iy] = cn_i*dens_prev + (1.0 - cn_i - cn_o)*dens[iy] + cn_o*dens_next;
        }

        // initialize the Thomas forward elimination
        upper_work[0] = cn_upper[0]  / cn_diag[0];
        dens_work[0] = dens_rhs[0] / cn_diag[0];

        // eliminate the lower diagonal of the implicit system
        for (int iy = 1; iy < N_Y; iy++)
        {
            real pivot = cn_diag[iy] - cn_lower[iy]*upper_work[iy - 1];

            upper_work[iy] = (iy < N_Y - 1) ? (cn_upper[iy] / pivot) : 0.0;
            dens_work[iy] = (dens_rhs[iy] - cn_lower[iy]*dens_work[iy - 1]) / pivot;
        }

        // back-substitute the density solution
        for (int iy = N_Y - 2; iy >= 0; iy--)
        {
            dens_work[iy] -= upper_work[iy]*dens_work[iy + 1];
        }

        // reconstruct time-centred outward diffusive mass fluxes with zero boundary fluxes
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y - 1)
            {
                upper_work[iy] = 0.0;
                continue;
            }

            real vol_y = _get_vol_y(iy);

            real cn_o = -cn_upper[iy];

            upper_work[iy]  = -(cn_o*vol_y / dt_sub);
            upper_work[iy] *= (dens[iy + 1] - dens[iy]) + (dens_work[iy + 1] - dens_work[iy]);
        }

        // combine each face mass flux with the donor azimuthal primitive quantity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real y_up = _get_ycent(iy_up);
            real R_up = y_up*sin(z);
            real dens_up = dens[iy_up];

            int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real lx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / dens_up : sqrt(G*M_S*fmax(R_up, 0.0));

            dens_rhs[iy] = upper_work[iy]*lx_up;
        }

        // update azimuthal momentum from the geometry-aware conservative flux divergence
        for (int iy = 0; iy < N_Y; iy++)
        {
            real vol_y = _get_vol_y(iy);
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[idx_cell] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;
        }

        // combine each face mass flux with the donor radial velocity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real dens_up = dens[iy_up];

            int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real vy_up = (dens_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / dens_up : 0.0;

            dens_rhs[iy] = upper_work[iy]*vy_up;
        }

        // update radial momentum from the geometry-aware conservative flux divergence
        for (int iy = 0; iy < N_Y; iy++)
        {
            real vol_y = _get_vol_y(iy);
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[idx_cell] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;
        }

        // combine each face mass flux with the donor polar primitive quantity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real dens_up = dens[iy_up];

            int idx_cell_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real lz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / dens_up : 0.0;

            dens_rhs[iy] = upper_work[iy]*lz_up;
        }

        // update polar momentum and accept the density solution for this substep
        for (int iy = 0; iy < N_Y; iy++)
        {
            real vol_y = _get_vol_y(iy);
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[idx_cell] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;

            dens[iy] = dens_work[iy];
        }
    }

    // store the final dust density
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = dens[iy];
    }
}

#endif
