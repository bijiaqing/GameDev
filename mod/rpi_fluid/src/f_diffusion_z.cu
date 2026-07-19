#ifdef DIFFUSION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// ---- Z direction (polar) -- Implicit CN TDMA, NB_Z blocks, 1 thread per (ix,iy) column ----
// Turbulent dust flux: J_z = -D_z*rho_g*(1/y)*d(rho_d/rho_g)/dz
// The minimum conservative momentum closure carries upwind-cell specific momentum with the
// time-centred CN mass flux at each polar face

__global__
void f_diffusion_z (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Y) return;
    if (N_Z == 1) return;
    if (dt <= 0.0) return;

    int ix = idx % N_X;
    int iy = idx / N_X;

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);

    // convert rho_d to q = rho_d/rho_g using the full analytic volumetric gas density
    real ratio[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = Z_MIN + (iz + 0.5)*dz;
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        int ic = ix + iy*N_X + iz*N_X*N_Y;
        ratio[iz] = dev_dustdens[ic] / rhog;
    }

    // store the full-step off-diagonal coefficients first
    // integrating the spherical Z divergence gives
    //   coeff = 0.5*dt*sin(z_face)*D_face*rho_g_face/(y^2*dz*dcos*rho_g_cell)
    // CN is subcycled until cn_i + cn_o <= 0.9 in every cell,
    // which makes the explicit-side coefficients nonnegative and avoids a non-conservative post-solve positivity clamp
    real cn_lower[N_Z], cn_diag[N_Z], cn_upper[N_Z], ratio_rhs[N_Z];
    real max_cn_sum = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z0 = Z_MIN + static_cast<real>(iz)*dz;
        real z1 = z0 + dz;

        real vol_z  = cos(z0) - cos(z1);
        real dz_len = yc*dz;

        real zc = z0 + 0.5*dz;
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g  = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        // evaluate only active interior faces
        // this avoids calling the analytic disk helpers at R=0 when a no-flux boundary lies on the polar axis
        real rhog_i = 0.0, Dz_i = 0.0;
        if (iz > 0)
        {
            real R_i = yc*sin(z0);
            real Z_i = yc*cos(z0);

            real h_i = _get_hg(R_i);
            rhog_i = _get_rhog(R_i, Z_i, h_i);
            Dz_i = _get_nu(R_i, h_i) / SC_Z;
        }

        real rhog_o = 0.0, Dz_o = 0.0;
        if (iz < N_Z - 1)
        {
            real R_o = yc*sin(z1);
            real Z_o = yc*cos(z1);

            real h_o = _get_hg(R_o);
            rhog_o = _get_rhog(R_o, Z_o, h_o);
            Dz_o = _get_nu(R_o, h_o) / SC_Z;
        }

        // no-flux BC at both polar-domain boundaries; for HALFDISK the outer boundary is normally the midplane
        // yc*dz_len = y^2*dz supplies both spherical metric factors
        real cn_i = (iz > 0)       ? (0.5*dt*sin(z0)*Dz_i*rhog_i / (yc*dz_len*vol_z*rhog)) : 0.0;
        real cn_o = (iz < N_Z - 1) ? (0.5*dt*sin(z1)*Dz_o*rhog_o / (yc*dz_len*vol_z*rhog)) : 0.0;

        cn_lower[iz] = -cn_i;
        cn_upper[iz] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
    }

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

    real upper_work[N_Z], ratio_work[N_Z];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int iz = 0; iz < N_Z; iz++)
        {
            real cn_i = -cn_lower[iz];
            real cn_o = -cn_upper[iz];

            real ratio_prev = (iz > 0)       ? ratio[iz - 1] : ratio[iz];
            real ratio_next = (iz < N_Z - 1) ? ratio[iz + 1] : ratio[iz];

            ratio_rhs[iz] = cn_i*ratio_prev + (1.0 - cn_i - cn_o)*ratio[iz] + cn_o*ratio_next;
        }

        // Thomas algorithm: forward sweep and backward substitution
        upper_work[0] = cn_upper[0]  / cn_diag[0];
        ratio_work[0] = ratio_rhs[0] / cn_diag[0];

        for (int iz = 1; iz < N_Z; iz++)
        {
            real pivot = cn_diag[iz] - cn_lower[iz]*upper_work[iz - 1];

            upper_work[iz] = (iz < N_Z - 1) ? (cn_upper[iz] / pivot) : 0.0;
            ratio_work[iz] = (ratio_rhs[iz] - cn_lower[iz]*ratio_work[iz - 1]) / pivot;
        }

        // keep ratio as q_old and complete q_new in ratio_work so the CN face flux can be reconstructed after the solve
        for (int iz = N_Z - 2; iz >= 0; iz--)
        {
            ratio_work[iz] -= upper_work[iz]*ratio_work[iz + 1];
        }

        // area-weighted polar mass flux sin(z)*J_z at each outer face
        // its divergence uses the same y*vol_z volume factor as the density CN operator
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                upper_work[iz] = 0.0;
                continue;
            }

            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);

            real zc = z0 + 0.5*dz;
            real Rc = yc*sin(zc);
            real Zc = yc*cos(zc);

            real h_g  = _get_hg(Rc);
            real rhog = _get_rhog(Rc, Zc, h_g);

            real cn_o = -cn_upper[iz];

            upper_work[iz]  = -(cn_o*yc*vol_z*rhog / dt_sub);
            upper_work[iz] *= (ratio[iz + 1] - ratio[iz]) + (ratio_work[iz + 1] - ratio_work[iz]);
        }

        // ratio_rhs is no longer needed, so reuse it for one momentum face flux at a time
        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real z_up = Z_MIN + (iz_up + 0.5)*dz;
            real R_up = yc*sin(z_up);
            real Z_up = yc*cos(z_up);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iz_up];

            int ic_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(R_up, 0.0));

            ratio_rhs[iz] = upper_work[iz]*velx_up;
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);
            real flux_i = (iz > 0) ? ratio_rhs[iz - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[ic] -= dt_sub*(ratio_rhs[iz] - flux_i) / (yc*vol_z);
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real z_up = Z_MIN + (iz_up + 0.5)*dz;
            real R_up = yc*sin(z_up);
            real Z_up = yc*cos(z_up);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iz_up];

            int ic_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            ratio_rhs[iz] = upper_work[iz]*vely_up;
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);
            real flux_i = (iz > 0) ? ratio_rhs[iz - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[ic] -= dt_sub*(ratio_rhs[iz] - flux_i) / (yc*vol_z);
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (upper_work[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z - 1) iz_up = iz;

            real z_up = Z_MIN + (iz_up + 0.5)*dz;
            real R_up = yc*sin(z_up);
            real Z_up = yc*cos(z_up);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iz_up];

            int ic_up = ix + iy*N_X + iz_up*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            ratio_rhs[iz] = upper_work[iz]*velz_up;
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);
            real flux_i = (iz > 0) ? ratio_rhs[iz - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[ic] -= dt_sub*(ratio_rhs[iz] - flux_i) / (yc*vol_z);

            ratio[iz] = ratio_work[iz];
        }
    }

    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = Z_MIN + (iz + 0.5)*dz;
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = rhog*ratio[iz];
    }
}

// =========================================================================================================================

#endif // DIFFUSION
