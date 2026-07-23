#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffus_x_calc
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
void diffus_x_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_Y*N_Z) return;
    if (dt <= 0.0) return;

    int iy = idx % N_Y;
    int iz = idx / N_Y;

    real dx = _get_dx();

    real yc = _get_ycent(iy);
    real zc = _get_zcent(iz);

    real Rc = yc*sin(zc);
    real dx_len = Rc*dx;

    real h_g = _get_hg(Rc);
    real Dx = _get_nu(Rc, h_g) / SCHMIDT_X;

    // load dust density along one azimuthal ring
    real dens[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dens[ix] = dev_dustdens[ic];
    }

    // limit each Crank-Nicolson substep by the configured positivity coefficient
    int n_sub = static_cast<int>(ceil(dt*Dx / (dx_len*dx_len) / POS_LIMIT));
    if (n_sub < 1) n_sub = 1;

    real dt_sub   = dt / static_cast<real>(n_sub);
    real cn_coeff = 0.5*dt_sub*Dx / (dx_len*dx_len);
    real cn_diag  = 1.0 + 2.0*cn_coeff;
    real cn_wrap  = -cn_coeff;

    // construct the Sherman-Morrison reduction of the periodic cyclic system
    real sm_gamma = -cn_diag;
    real sm_vlast = cn_wrap / sm_gamma;
    real sm_diag0 = cn_diag - sm_gamma;
    real sm_diagN = cn_diag - cn_wrap*sm_vlast;

    // advance density and momentum through each diffusion substep
    real dens_work[N_X], cycle_work[N_X], upper_work[N_X];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        // build the explicit Crank-Nicolson right-hand side and cyclic correction vector
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ixp1 = (ix + 1)       % N_X;

            dens_work[ix] = cn_coeff*dens[ixm1] + (1.0 - 2.0*cn_coeff)*dens[ix] + cn_coeff*dens[ixp1];
            cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
        }

        // initialize the modified Thomas forward elimination
        real diag_cur  = sm_diag0;
        upper_work[0]  = -cn_coeff / diag_cur;
        dens_work[0] /= diag_cur;
        cycle_work[0] /= diag_cur;

        // eliminate the lower diagonal for the density and cyclic correction systems
        for (int ix = 1; ix < N_X; ix++)
        {
            diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
            real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
            upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
            dens_work[ix] = (dens_work[ix] + cn_coeff*dens_work[ix - 1]) / pivot;
            cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
        }

        // back-substitute both modified tridiagonal systems
        for (int ix = N_X - 2; ix >= 0; ix--)
        {
            dens_work[ix] -= upper_work[ix]*dens_work[ix + 1];
            cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
        }

        // restore the periodic corner coupling with the Sherman-Morrison correction
        real base_proj  = dens_work[0] + sm_vlast*dens_work[N_X - 1];
        real corr_proj  = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
        real corr_scale = base_proj / (1.0 + corr_proj);

        // apply the cyclic correction to the provisional density solution
        for (int ix = 0; ix < N_X; ix++)
        {
            dens_work[ix] -= corr_scale*cycle_work[ix];
        }

        // reconstruct the time-centred diffusive mass flux at every periodic face
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixp1 = (ix + 1) % N_X;
            upper_work[ix] = -0.5*Dx*((dens[ixp1] - dens[ix]) + (dens_work[ixp1] - dens_work[ix])) / dx_len;
        }

        // combine each face mass flux with the donor azimuthal primitive quantity
        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = dens[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real lx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(Rc, 0.0));

            cycle_work[ix] = upper_work[ix]*lx_up;
        }

        // update azimuthal momentum from the conservative flux divergence
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomx[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        // combine each face mass flux with the donor radial velocity
        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = dens[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real vy_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*vy_up;
        }

        // update radial momentum from the conservative flux divergence
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomy[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        // combine each face mass flux with the donor polar primitive quantity
        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = dens[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real lz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*lz_up;
        }

        // update polar momentum and accept the density solution for this substep
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomz[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
            dens[ix] = dens_work[ix];
        }
    }

    // store the final dust density
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = dens[ix];
    }
}

#endif
