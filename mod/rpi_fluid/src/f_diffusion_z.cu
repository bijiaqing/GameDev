#ifdef DIFFUSION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// ---- Z direction (polar) -- Implicit CN TDMA, NB_Z blocks, 1 thread per (ix,iy) column ----
// Turbulent dust flux: J_theta = -D_theta*rho_g*(1/r)*d(rho_d/rho_g)/dtheta.
// The minimum conservative momentum closure carries donor-cell specific momentum with the
// time-centred CN mass flux at each polar face.

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

    real dy  = _get_dy();
    real dz_grid = _get_dz();
    real yc     = Y_MIN*pow(dy, iy + 0.5);

    // Convert rho_d to q=rho_d/rho_g using the full analytic volumetric gas density.
    real ratio[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = Z_MIN + dz_grid*(static_cast<real>(iz) + 0.5);
        real R_c = yc*sin(zc);
        real Z_c = yc*cos(zc);
        real h_g = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_g);
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        ratio[iz] = dev_dustdens[idx_cell] / rhog_c;
    }

    // Store the full-step off-diagonal coefficients first.  Integrating the spherical theta
    // divergence gives
    //   coeff = 0.5*dt*sin(theta_face)*D_face*rho_g_face
    //           /(r^2*dtheta*dcos*rho_g_cell).
    // CN is subcycled until cn_in + cn_out <= 0.9 in every cell, which makes the explicit-side
    // coefficients nonnegative and avoids a non-conservative post-solve positivity clamp.
    real cn_lower[N_Z], cn_diag[N_Z], cn_upper[N_Z], ratio_rhs[N_Z];
    real max_cn_sum = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z_in   = Z_MIN + dz_grid*static_cast<real>(iz);
        real z_out  = z_in + dz_grid;
        real dcos   = cos(z_in) - cos(z_out);    // > 0
        real dr_arc = yc*dz_grid;                 // arc distance between cell centres

        real zc = z_in + 0.5*dz_grid;
        real R_c = yc*sin(zc);
        real Z_c = yc*cos(zc);
        real h_c = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_c);

        // Evaluate only active interior faces.  This avoids calling the analytic disk helpers
        // at R=0 when a no-flux boundary lies on the polar axis.
        real rhog_in = 0.0, diff_in = 0.0;
        if (iz > 0)
        {
            real R_in = yc*sin(z_in);
            real Z_in = yc*cos(z_in);
            real h_in = _get_hg(R_in);
            rhog_in = _get_rhog(R_in, Z_in, h_in);
            diff_in = _get_nu(R_in, h_in) / SC_Z;
        }

        real rhog_out = 0.0, diff_out = 0.0;
        if (iz < N_Z-1)
        {
            real R_out = yc*sin(z_out);
            real Z_out = yc*cos(z_out);
            real h_out = _get_hg(R_out);
            rhog_out = _get_rhog(R_out, Z_out, h_out);
            diff_out = _get_nu(R_out, h_out) / SC_Z;
        }

        // No-flux BC at both polar-domain boundaries; for HALFDISK the outer boundary is
        // normally the midplane.
        // yc*dr_arc = r^2*dtheta supplies both spherical metric factors.
        real cn_in  = (iz > 0)     ? (0.5*dt*sin(z_in)*diff_in*rhog_in
                                      / (yc*dr_arc*dcos*rhog_c)) : 0.0;
        real cn_out = (iz < N_Z-1) ? (0.5*dt*sin(z_out)*diff_out*rhog_out
                                      / (yc*dr_arc*dcos*rhog_c)) : 0.0;

        cn_lower[iz] = -cn_in;
        cn_upper[iz] = -cn_out;
        max_cn_sum = fmax(max_cn_sum, cn_in + cn_out);
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

    real upper_mod[N_Z], ratio_mod[N_Z];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int iz = 0; iz < N_Z; iz++)
        {
            real cn_in  = -cn_lower[iz];
            real cn_out = -cn_upper[iz];
            real ratio_prev = (iz > 0)     ? ratio[iz-1] : ratio[iz];
            real ratio_next = (iz < N_Z-1) ? ratio[iz+1] : ratio[iz];
            ratio_rhs[iz] = cn_in*ratio_prev
                          + (1.0 - cn_in - cn_out)*ratio[iz]
                          + cn_out*ratio_next;
        }

        // ---- Thomas algorithm: forward sweep and backward substitution ----
        upper_mod[0] = cn_upper[0] / cn_diag[0];
        ratio_mod[0] = ratio_rhs[0] / cn_diag[0];

        for (int iz = 1; iz < N_Z; iz++)
        {
            real pivot = cn_diag[iz] - cn_lower[iz]*upper_mod[iz-1];
            upper_mod[iz] = (iz < N_Z-1) ? (cn_upper[iz] / pivot) : 0.0;
            ratio_mod[iz] = (ratio_rhs[iz] - cn_lower[iz]*ratio_mod[iz-1]) / pivot;
        }

        // Keep ratio as q_old and complete q_new in ratio_mod so the CN face flux can be
        // reconstructed after the solve.
        for (int iz = N_Z-2; iz >= 0; iz--)
            ratio_mod[iz] -= upper_mod[iz]*ratio_mod[iz+1];

        // Area-weighted polar mass flux sin(theta)*J_theta at each outer face.  Its
        // divergence uses the same r*dcos volume factor as the density CN operator.
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z-1)
            {
                upper_mod[iz] = 0.0;
                continue;
            }

            real z_in = Z_MIN + dz_grid*static_cast<real>(iz);
            real z_out = z_in + dz_grid;
            real dcos = cos(z_in) - cos(z_out);
            real zc = z_in + 0.5*dz_grid;
            real R_c = yc*sin(zc);
            real Z_c = yc*cos(zc);
            real h_c = _get_hg(R_c);
            real rhog_c = _get_rhog(R_c, Z_c, h_c);
            real cn_out = -cn_upper[iz];
            upper_mod[iz] = -(cn_out*yc*dcos*rhog_c / dt_sub)*
                ((ratio[iz+1] - ratio[iz]) + (ratio_mod[iz+1] - ratio_mod[iz]));
        }

        // ratio_rhs is no longer needed, so reuse it for one momentum face flux at a time.
        for (int iz = 0; iz < N_Z; iz++)
        {
            int donor = (upper_mod[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z-1) donor = iz;
            real z_donor = Z_MIN + dz_grid*(donor + 0.5);
            real R_donor = yc*sin(z_donor);
            real Z_donor = yc*cos(z_donor);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + iy*N_X + donor*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomx[idx_donor] / rho_donor
                                                    : sqrt(G*M_S*fmax(R_donor, 0.0));
            ratio_rhs[iz] = upper_mod[iz]*spec_mom;
        }
        for (int iz = 0; iz < N_Z; iz++)
        {
            real z_in = Z_MIN + dz_grid*static_cast<real>(iz);
            real z_out = z_in + dz_grid;
            real dcos = cos(z_in) - cos(z_out);
            real flux_in = (iz > 0) ? ratio_rhs[iz-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[idx_cell] -= dt_sub*(ratio_rhs[iz] - flux_in) / (yc*dcos);
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            int donor = (upper_mod[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z-1) donor = iz;
            real z_donor = Z_MIN + dz_grid*(donor + 0.5);
            real R_donor = yc*sin(z_donor);
            real Z_donor = yc*cos(z_donor);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + iy*N_X + donor*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomy[idx_donor] / rho_donor : 0.0;
            ratio_rhs[iz] = upper_mod[iz]*spec_mom;
        }
        for (int iz = 0; iz < N_Z; iz++)
        {
            real z_in = Z_MIN + dz_grid*static_cast<real>(iz);
            real z_out = z_in + dz_grid;
            real dcos = cos(z_in) - cos(z_out);
            real flux_in = (iz > 0) ? ratio_rhs[iz-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[idx_cell] -= dt_sub*(ratio_rhs[iz] - flux_in) / (yc*dcos);
        }

        for (int iz = 0; iz < N_Z; iz++)
        {
            int donor = (upper_mod[iz] >= 0.0) ? iz : iz + 1;
            if (iz == N_Z-1) donor = iz;
            real z_donor = Z_MIN + dz_grid*(donor + 0.5);
            real R_donor = yc*sin(z_donor);
            real Z_donor = yc*cos(z_donor);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + iy*N_X + donor*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomz[idx_donor] / rho_donor : 0.0;
            ratio_rhs[iz] = upper_mod[iz]*spec_mom;
        }
        for (int iz = 0; iz < N_Z; iz++)
        {
            real z_in = Z_MIN + dz_grid*static_cast<real>(iz);
            real z_out = z_in + dz_grid;
            real dcos = cos(z_in) - cos(z_out);
            real flux_in = (iz > 0) ? ratio_rhs[iz-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[idx_cell] -= dt_sub*(ratio_rhs[iz] - flux_in) / (yc*dcos);
            ratio[iz] = ratio_mod[iz];
        }
    }

    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = Z_MIN + dz_grid*(static_cast<real>(iz) + 0.5);
        real R_c = yc*sin(zc);
        real Z_c = yc*cos(zc);
        real h_c = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_c);
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = rhog_c*ratio[iz];
    }
}

// ====================================================================

#endif // DIFFUSION
