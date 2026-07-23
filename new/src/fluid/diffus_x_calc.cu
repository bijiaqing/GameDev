#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: diffus_x_calc
// purpose: periodic azimuthal diffusion of the dust-to-gas ratio with conservative momentum transport
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
    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real dx_len = Rc*dx;

    real h_g  = _get_hg(Rc);
    real gasdens = _get_gasdens(Rc, Zc, h_g);
    real Dx   = _get_nu(Rc, h_g) / SC_X;

    // load the dust-to-gas ratio along one azimuthal ring
    real ratio[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        ratio[ix] = dev_dustdens[ic] / gasdens;
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

    // advance the ratio and momentum through each diffusion substep
    real ratio_work[N_X], cycle_work[N_X], upper_work[N_X];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        // build the explicit Crank-Nicolson right-hand side and cyclic correction vector
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ixp1 = (ix + 1)       % N_X;

            ratio_work[ix] = cn_coeff*ratio[ixm1] + (1.0 - 2.0*cn_coeff)*ratio[ix] + cn_coeff*ratio[ixp1];
            cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
        }

        // initialize the modified Thomas forward elimination
        real diag_cur  = sm_diag0;
        upper_work[0]  = -cn_coeff / diag_cur;
        ratio_work[0] /= diag_cur;
        cycle_work[0] /= diag_cur;

        // eliminate the lower diagonal for the ratio and cyclic correction systems
        for (int ix = 1; ix < N_X; ix++)
        {
            diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
            real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
            upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
            ratio_work[ix] = (ratio_work[ix] + cn_coeff*ratio_work[ix - 1]) / pivot;
            cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
        }

        // back-substitute both modified tridiagonal systems
        for (int ix = N_X - 2; ix >= 0; ix--)
        {
            ratio_work[ix] -= upper_work[ix]*ratio_work[ix + 1];
            cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
        }

        // restore the periodic corner coupling with the Sherman-Morrison correction
        real base_proj  = ratio_work[0] + sm_vlast*ratio_work[N_X - 1];
        real corr_proj  = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
        real corr_scale = base_proj / (1.0 + corr_proj);

        // apply the cyclic correction to the provisional ratio solution
        for (int ix = 0; ix < N_X; ix++)
        {
            ratio_work[ix] -= corr_scale*cycle_work[ix];
        }

        // reconstruct the time-centred diffusive mass flux at every periodic face
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixp1 = (ix + 1) % N_X;
            upper_work[ix] = -0.5*Dx*gasdens*((ratio[ixp1] - ratio[ix]) + (ratio_work[ixp1] - ratio_work[ix])) / dx_len;
        }

        // combine each face mass flux with the donor azimuthal primitive quantity
        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = gasdens*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(Rc, 0.0));

            cycle_work[ix] = upper_work[ix]*velx_up;
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
            real dens_up = gasdens*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*vely_up;
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
            real dens_up = gasdens*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*velz_up;
        }

        // update polar momentum and accept the ratio solution for this substep
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomz[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
            ratio[ix] = ratio_work[ix];
        }
    }

    // recover dust density from the final dust-to-gas ratio
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = gasdens*ratio[ix];
    }
}

#endif
