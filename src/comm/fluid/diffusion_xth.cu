#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffusion_xth
// purpose: periodic azimuthal diffusion of dust density with conservative momentum transport
//
// parallelization: one thread per radial-polar ring with a serial loop over N_X azimuthal cells
//
// per call:
//   1 positivity-controlled Crank-Nicolson subcycling
//   2 cyclic tridiagonal solution by the Sherman-Morrison formula
//   3 time-centred diffusive mass flux construction
//   4 donor-state momentum transport with the diffusing mass
// =========================================================================================================================

__global__
void diffusion_xth (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_ring = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;
    if (dt <= 0.0) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;

    real dx = _get_dx();

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);

    real R = y*sin(z);
    real dx_len = R*dx;

    real h_g = _get_hg(R);
    real diff_x = _get_nu(R, h_g) / SCHMIDT_X;

    // load dust density along one azimuthal ring
    real rhod[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        rhod[ix] = dev_dustdens[idx_cell];
    }

    // limit each Crank-Nicolson substep by the configured positivity coefficient
    int sub_count = static_cast<int>(ceil(dt*diff_x / (dx_len*dx_len) / POS_LIMIT));
    if (sub_count < 1) sub_count = 1;

    real dt_sub   = dt / static_cast<real>(sub_count);
    real cn_coeff = 0.5*dt_sub*diff_x / (dx_len*dx_len);
    real cn_diag  = 1.0 + 2.0*cn_coeff;
    real cn_wrap  = -cn_coeff;

    // construct the Sherman-Morrison reduction of the periodic cyclic system
    real sm_gamma = -cn_diag;
    real sm_vlast = cn_wrap / sm_gamma;
    real sm_diag0 = cn_diag - sm_gamma;
    real sm_diagN = cn_diag - cn_wrap*sm_vlast;

    // advance density and momentum through each diffusion substep
    real rhod_work[N_X];
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        {
            real cycle_work[N_X], upper_work[N_X];

            // build the explicit Crank-Nicolson right-hand side and cyclic correction vector
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                int ixp1 = (ix + 1)       % N_X;

                rhod_work[ix] = cn_coeff*rhod[ixm1] + (1.0 - 2.0*cn_coeff)*rhod[ix] + cn_coeff*rhod[ixp1];
                cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
            }

            // initialize the modified Thomas forward elimination
            real diag_cur  = sm_diag0;
            upper_work[0]  = -cn_coeff / diag_cur;
            rhod_work[0] /= diag_cur;
            cycle_work[0] /= diag_cur;

            // eliminate the lower diagonal for the density and cyclic correction systems
            for (int ix = 1; ix < N_X; ix++)
            {
                diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
                real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
                upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
                rhod_work[ix] = (rhod_work[ix] + cn_coeff*rhod_work[ix - 1]) / pivot;
                cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
            }

            // back-substitute both modified tridiagonal systems
            for (int ix = N_X - 2; ix >= 0; ix--)
            {
                rhod_work[ix] -= upper_work[ix]*rhod_work[ix + 1];
                cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
            }

            // restore the periodic corner coupling with the Sherman-Morrison correction
            real base_proj  = rhod_work[0] + sm_vlast*rhod_work[N_X - 1];
            real corr_proj  = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
            real corr_scale = base_proj / (1.0 + corr_proj);

            // apply the cyclic correction to the provisional density solution
            for (int ix = 0; ix < N_X; ix++)
            {
                rhod_work[ix] -= corr_scale*cycle_work[ix];
            }
        }

        {
            real mass_flux[N_X], moment_flux[N_X];

            // reconstruct the time-centred diffusive mass flux at every periodic face
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixp1 = (ix + 1) % N_X;
                mass_flux[ix] = -0.5*diff_x*((rhod[ixp1] - rhod[ix]) + (rhod_work[ixp1] - rhod_work[ix])) / dx_len;
            }

            // bound the complete outgoing transfer from each old donor before changing any face
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                real out_rate = fmax(mass_flux[ix], 0.0) + fmax(-mass_flux[ixm1], 0.0);
                rhod_work[ix] = (out_rate > 0.0)
                    ? fmin(1.0, POS_LIMIT*fmax(rhod[ix], 0.0)*dx_len / (dt_sub*out_rate)) : 1.0;
            }

            // scale each face once with the factor belonging to its raw-flux donor
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixp1 = (ix + 1) % N_X;
                int ix_up = (mass_flux[ix] >= 0.0) ? ix : ixp1;
                mass_flux[ix] *= rhod_work[ix_up];
            }

            // reconstruct accepted density from the same limited flux used by momentum
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                rhod_work[ix] = rhod[ix] - dt_sub*(mass_flux[ix] - mass_flux[ixm1]) / dx_len;
            }

            // combine each face mass flux with the donor azimuthal primitive quantity
            for (int ix = 0; ix < N_X; ix++)
            {
                int ix_up = (mass_flux[ix] >= 0.0) ? ix : (ix + 1) % N_X;
                real rhod_up = rhod[ix_up];

                int idx_cell_up = ix_up + iy*N_X + iz*N_X*N_Y;
                real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / rhod_up : sqrt(G*M_S*fmax(R, 0.0));

                moment_flux[ix] = mass_flux[ix]*lx_up;
            }

            // update azimuthal momentum from the conservative flux divergence
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

                dev_dustmomx[idx_cell] -= dt_sub*(moment_flux[ix] - moment_flux[ixm1]) / dx_len;
            }

            // combine each face mass flux with the donor radial velocity
            for (int ix = 0; ix < N_X; ix++)
            {
                int ix_up = (mass_flux[ix] >= 0.0) ? ix : (ix + 1) % N_X;
                real rhod_up = rhod[ix_up];

                int idx_cell_up = ix_up + iy*N_X + iz*N_X*N_Y;
                real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / rhod_up : 0.0;

                moment_flux[ix] = mass_flux[ix]*vy_up;
            }

            // update radial momentum from the conservative flux divergence
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

                dev_dustmomy[idx_cell] -= dt_sub*(moment_flux[ix] - moment_flux[ixm1]) / dx_len;
            }

            // combine each face mass flux with the donor polar primitive quantity
            for (int ix = 0; ix < N_X; ix++)
            {
                int ix_up = (mass_flux[ix] >= 0.0) ? ix : (ix + 1) % N_X;
                real rhod_up = rhod[ix_up];

                int idx_cell_up = ix_up + iy*N_X + iz*N_X*N_Y;
                real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / rhod_up : 0.0;

                moment_flux[ix] = mass_flux[ix]*lz_up;
            }

            // update polar momentum and accept the density solution for this substep
            for (int ix = 0; ix < N_X; ix++)
            {
                int ixm1 = (ix - 1 + N_X) % N_X;
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

                dev_dustmomz[idx_cell] -= dt_sub*(moment_flux[ix] - moment_flux[ixm1]) / dx_len;
                rhod[ix] = rhod_work[ix];
            }
        }
    }

    // store the final dust density
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = rhod[ix];
    }
}

#endif // DIFFUSION
