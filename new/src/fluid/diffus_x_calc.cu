#ifdef DIFFUSION

#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>























































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
    real rhog = _get_rhog(Rc, Zc, h_g);
    real Dx   = _get_nu(Rc, h_g) / SC_X;


    real ratio[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        ratio[ix] = dev_dustdens[ic] / rhog;
    }





    int n_sub = static_cast<int>(ceil(dt*Dx / (dx_len*dx_len) / POS_LIMIT));
    if (n_sub < 1) n_sub = 1;


    real dt_sub   = dt / static_cast<real>(n_sub);
    real cn_coeff = 0.5*dt_sub*Dx / (dx_len*dx_len);
    real cn_diag  = 1.0 + 2.0*cn_coeff;
    real cn_wrap  = -cn_coeff;


    real sm_gamma = -cn_diag;
    real sm_vlast = cn_wrap / sm_gamma;
    real sm_diag0 = cn_diag - sm_gamma;
    real sm_diagN = cn_diag - cn_wrap*sm_vlast;



    real ratio_work[N_X], cycle_work[N_X], upper_work[N_X];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ixp1 = (ix + 1)       % N_X;

            ratio_work[ix] = cn_coeff*ratio[ixm1] + (1.0 - 2.0*cn_coeff)*ratio[ix] + cn_coeff*ratio[ixp1];
            cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
        }


        real diag_cur  = sm_diag0;
        upper_work[0]  = -cn_coeff / diag_cur;
        ratio_work[0] /= diag_cur;
        cycle_work[0] /= diag_cur;

        for (int ix = 1; ix < N_X; ix++)
        {
            diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
            real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
            upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
            ratio_work[ix] = (ratio_work[ix] + cn_coeff*ratio_work[ix - 1]) / pivot;
            cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
        }

        for (int ix = N_X - 2; ix >= 0; ix--)
        {
            ratio_work[ix] -= upper_work[ix]*ratio_work[ix + 1];
            cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
        }




        real base_proj  = ratio_work[0] + sm_vlast*ratio_work[N_X - 1];
        real corr_proj  = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
        real corr_scale = base_proj / (1.0 + corr_proj);

        for (int ix = 0; ix < N_X; ix++)
        {
            ratio_work[ix] -= corr_scale*cycle_work[ix];
        }



        for (int ix = 0; ix < N_X; ix++)
        {
            int ixp1 = (ix + 1) % N_X;
            upper_work[ix] = -0.5*Dx*rhog*((ratio[ixp1] - ratio[ix]) + (ratio_work[ixp1] - ratio_work[ix])) / dx_len;
        }




        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(Rc, 0.0));

            cycle_work[ix] = upper_work[ix]*velx_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomx[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*vely_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomy[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*velz_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomz[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
            ratio[ix] = ratio_work[ix];
        }
    }

    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = rhog*ratio[ix];
    }
}



#endif
