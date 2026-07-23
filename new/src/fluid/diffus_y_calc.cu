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

__global__
void diffus_y_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Z) return;
    if (dt <= 0.0) return;

    int ix = idx % N_X;
    int iz = idx / N_X;

    real dy = _get_dy();
    real dz = _get_dz();

    // select cylindrical radial geometry in 2D and spherical radial geometry in 3D
    real pow_y = _get_powy();

    real zc = Z_MIN + (iz + 0.5)*dz;

    // load dust density along one radial column
    real dens[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dens[iy] = dev_dustdens[ic];
    }

    // assemble full-step Crank-Nicolson face couplings and measure the largest local coefficient sum
    real cn_lower[N_Y], cn_diag[N_Y], cn_upper[N_Y], dens_rhs[N_Y];
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
        real y1 = Y_MIN*pow(dy, static_cast<real>(iy + 1));
        real yc = Y_MIN*pow(dy, iy + 0.5);

        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

        real dy_len_i = yc*(dy - 1.0) / dy;
        real dy_len_o = yc*(dy - 1.0);

        real Rc_i = y0*sin(zc);
        real h_i = _get_hg(Rc_i);
        real Dy_i = _get_nu(Rc_i, h_i) / SC_Y;

        real Rc_o = y1*sin(zc);
        real h_o = _get_hg(Rc_o);
        real Dy_o = _get_nu(Rc_o, h_o) / SC_Y;

        real area_i = pow(y0, pow_y - 1.0);
        real area_o = pow(y1, pow_y - 1.0);

        real cn_i = (iy > 0)       ? (0.5*dt*area_i*Dy_i / (dy_len_i*vol_y)) : 0.0;
        real cn_o = (iy < N_Y - 1) ? (0.5*dt*area_o*Dy_o / (dy_len_o*vol_y)) : 0.0;

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

            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

            real cn_o = -cn_upper[iy];

            upper_work[iy]  = -(cn_o*vol_y / dt_sub);
            upper_work[iy] *= (dens[iy + 1] - dens[iy]) + (dens_work[iy + 1] - dens_work[iy]);
        }

        // combine each face mass flux with the donor azimuthal primitive quantity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real y_up = Y_MIN*pow(dy, iy_up + 0.5);
            real Rc_up = y_up*sin(zc);
            real dens_up = dens[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(Rc_up, 0.0));

            dens_rhs[iy] = upper_work[iy]*velx_up;
        }

        // update azimuthal momentum from the geometry-aware conservative flux divergence
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[ic] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;
        }

        // combine each face mass flux with the donor radial velocity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real dens_up = dens[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            dens_rhs[iy] = upper_work[iy]*vely_up;
        }

        // update radial momentum from the geometry-aware conservative flux divergence
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[ic] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;
        }

        // combine each face mass flux with the donor polar primitive quantity
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real dens_up = dens[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            dens_rhs[iy] = upper_work[iy]*velz_up;
        }

        // update polar momentum and accept the density solution for this substep
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? dens_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[ic] -= dt_sub*(dens_rhs[iy] - flux_i) / vol_y;

            dens[iy] = dens_work[iy];
        }
    }

    // store the final dust density
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = dens[iy];
    }
}

#endif
