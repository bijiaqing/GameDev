#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffus_z_calc
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
void diffus_z_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
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
    real dens[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dens[iz] = dev_dustdens[idx_cell];
    }

    // assemble full-step Crank-Nicolson face couplings and measure the largest local coefficient sum
    real cn_lower[N_Z], cn_diag[N_Z], cn_upper[N_Z], dens_rhs[N_Z];
    real max_cn_sum = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z_i = _get_zedge(iz);
        real z_o = _get_zedge(iz + 1);

        real vol_z  = _get_vol_z(iz);
        real dz_len = y*dz;

        real diff_z_i = 0.0;
        if (iz > 0)
        {
            real R_i = y*sin(z_i);

            real h_i = _get_hg(R_i);
            diff_z_i = _get_nu(R_i, h_i) / SCHMIDT_Z;
        }

        real diff_z_o = 0.0;
        if (iz < N_Z - 1)
        {
            real R_o = y*sin(z_o);

            real h_o = _get_hg(R_o);
            diff_z_o = _get_nu(R_o, h_o) / SCHMIDT_Z;
        }

        real cn_i = (iz > 0)       ? (0.5*dt*sin(z_i)*diff_z_i / (y*dz_len*vol_z)) : 0.0;
        real cn_o = (iz < N_Z - 1) ? (0.5*dt*sin(z_o)*diff_z_o / (y*dz_len*vol_z)) : 0.0;

        cn_lower[iz] = -cn_i;
        cn_upper[iz] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
    }

    // choose positivity-controlled substeps and rescale the implicit matrix coefficients
    int n_sub = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
    if (n_sub < 1) n_sub = 1;

    real dt_sub = dt / static_cast<real>(n_sub);
    real inv_n_sub = 1.0 / static_cast<real>(n_sub);
    for (int iz = 0; iz < N_Z; iz++)
    {
        cn_lower[iz] *= inv_n_sub;
        cn_upper[iz] *= inv_n_sub;
        cn_diag[iz] = 1.0 - cn_lower[iz] - cn_upper[iz];
    }

    // advance density and momentum through each diffusion substep
    real upper_work[N_Z], dens_work[N_Z];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        // build the explicit Crank-Nicolson right-hand side with zero boundary gradients
        for (int iz = 0; iz < N_Z; iz++)
        {
            real cn_i = -cn_lower[iz];
            real cn_o = -cn_upper[iz];

            real dens_prev = (iz > 0)       ? dens[iz - 1] : dens[iz];
            real dens_next = (iz < N_Z - 1) ? dens[iz + 1] : dens[iz];

            dens_rhs[iz] = cn_i*dens_prev + (1.0 - cn_i - cn_o)*dens[iz] + cn_o*dens_next;
        }

        // initialize the Thomas forward elimination
        upper_work[0] = cn_upper[0]  / cn_diag[0];
        dens_work[0] = dens_rhs[0] / cn_diag[0];

        // eliminate the lower diagonal of the implicit system
        for (int iz = 1; iz < N_Z; iz++)
        {
            real pivot = cn_diag[iz] - cn_lower[iz]*upper_work[iz - 1];

            upper_work[iz] = (iz < N_Z - 1) ? (cn_upper[iz] / pivot) : 0.0;
            dens_work[iz] = (dens_rhs[iz] - cn_lower[iz]*dens_work[iz - 1]) / pivot;
        }

        // back-substitute the density solution
        for (int iz = N_Z - 2; iz >= 0; iz--)
        {
            dens_work[iz] -= upper_work[iz]*dens_work[iz + 1];
        }

        // reconstruct time-centred outward diffusive mass fluxes with zero boundary fluxes
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                upper_work[iz] = 0.0;
                continue;
            }

            real vol_z = _get_vol_z(iz);

            real cn_o = -cn_upper[iz];

            upper_work[iz]  = -(cn_o*y*vol_z / dt_sub);
            upper_work[iz] *= (dens[iz + 1] - dens[iz]) + (dens_work[iz + 1] - dens_work[iz]);
        }

        // combine each face mass flux with the donor azimuthal primitive quantity
        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real z_up = _get_zcent(iz_up);
            real R_up = y*sin(z_up);
            real dens_up = dens[iz_up];

            int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real lx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / dens_up : sqrt(G*M_S*fmax(R_up, 0.0));

            dens_rhs[iz] = upper_work[iz]*lx_up;
        }

        // update azimuthal momentum from the spherical conservative flux divergence
        for (int iz = 0; iz < N_Z; iz++)
        {
            real vol_z = _get_vol_z(iz);
            real flux_i = (iz > 0) ? dens_rhs[iz - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[idx_cell] -= dt_sub*(dens_rhs[iz] - flux_i) / (y*vol_z);
        }

        // combine each face mass flux with the donor radial velocity
        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real dens_up = dens[iz_up];

            int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real vy_up = (dens_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / dens_up : 0.0;

            dens_rhs[iz] = upper_work[iz]*vy_up;
        }

        // update radial momentum from the spherical conservative flux divergence
        for (int iz = 0; iz < N_Z; iz++)
        {
            real vol_z = _get_vol_z(iz);
            real flux_i = (iz > 0) ? dens_rhs[iz - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[idx_cell] -= dt_sub*(dens_rhs[iz] - flux_i) / (y*vol_z);
        }

        // combine each face mass flux with the donor polar primitive quantity
        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real dens_up = dens[iz_up];

            int idx_cell_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real lz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / dens_up : 0.0;

            dens_rhs[iz] = upper_work[iz]*lz_up;
        }

        // update polar momentum and accept the density solution for this substep
        for (int iz = 0; iz < N_Z; iz++)
        {
            real vol_z = _get_vol_z(iz);
            real flux_i = (iz > 0) ? dens_rhs[iz - 1] : 0.0;

            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[idx_cell] -= dt_sub*(dens_rhs[iz] - flux_i) / (y*vol_z);

            dens[iz] = dens_work[iz];
        }
    }

    // store the final dust density
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = dens[iz];
    }
}

#endif
