#ifdef DIFFUSION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// ---- Y direction (radial) -- Implicit CN TDMA, NB_Y blocks, 1 thread per (ix,iz) column ----
// Face areas and cell volumes use the same radial geometry:
//   pow_y=2: radial-only or azimuthal-radial disk (face area proportional to r)
//   pow_y=3: radial-colatitude or full 3D spherical grid (face area proportional to r^2)
// Turbulent dust flux: J_r = -D_r*rho_g*d(rho_d/rho_g)/dr.
// The minimum conservative momentum closure carries donor-cell specific momentum with the
// time-centred CN mass flux at each radial face.

__global__
void f_diffusion_y (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Z) return;
    if (dt <= 0.0) return;

    int ix = idx % N_X;
    int iz = idx / N_X;

    real dy = _get_dy();
    real pow_y = _get_powy();

    real zc = (N_Z > 1) ? (Z_MIN + (iz + 0.5)*_get_dz()) : 0.5*(Z_MIN + Z_MAX);

    // Convert rho_d to q=rho_d/rho_g at cell centres.
    real ratio[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        real r_c = Y_MIN*pow(dy, iy + 0.5);
        real R_c = r_c*sin(zc);
        real Z_c = r_c*cos(zc);
        real h_g = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_g);
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        ratio[iy] = dev_dustdens[idx_cell] / rhog_c;
    }

    // Store the full-step off-diagonal coefficients first.  CN is linearly stable for any dt,
    // but a sufficient positivity condition is cn_in + cn_out <= 1 in every cell.  Equal
    // substeps enforce that condition without a mass-destroying clamp after the solve.
    real cn_lower[N_Y], cn_diag[N_Y], cn_upper[N_Y], ratio_rhs[N_Y];
    real max_cn_sum = 0.0;

    for (int iy = 0; iy < N_Y; iy++)
    {
        real y0    = Y_MIN*pow(dy, static_cast<real>(iy)); // inner face radius
        real r_in  = y0;
        real r_out = y0*dy;                                // outer face radius
        real r_c   = Y_MIN*pow(dy, iy + 0.5);             // cell centre (geometric mean)
        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

        // Centre-to-centre distances (for gradient at each face)
        real dr_out = r_c*(dy - 1.0);          // r_c(iy+1) - r_c(iy)
        real dr_in  = r_c / dy*(dy - 1.0); // r_c(iy)   - r_c(iy-1)

        // Analytic gas density at the cell centre and radial faces.
        real R_c = r_c*sin(zc);
        real Z_c = r_c*cos(zc);
        real h_c = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_c);

        real R_out = r_out*sin(zc);
        real Z_out = r_out*cos(zc);
        real h_out = _get_hg(R_out);
        real rhog_out = _get_rhog(R_out, Z_out, h_out);
        real diff_out = _get_nu(R_out, h_out) / SC_Y;

        real R_in = r_in*sin(zc);
        real Z_in = r_in*cos(zc);
        real h_in = _get_hg(R_in);
        real rhog_in = _get_rhog(R_in, Z_in, h_in);
        real diff_in = _get_nu(R_in, h_in) / SC_Y;

        // CN half-step coefficients (zero at boundaries -> no-flux BC).  Dividing the
        // conservative rho_d equation by cell-centred rho_g gives
        //   coeff = 0.5*dt*area*D_face*rho_g_face/(dr*volume*rho_g_cell).
        real area_out = pow(r_out, pow_y - 1.0);
        real area_in  = pow(r_in,  pow_y - 1.0);
        real cn_out = (iy < N_Y-1) ? (0.5*dt*area_out*diff_out*rhog_out / (dr_out*vol_y*rhog_c)) : 0.0;
        real cn_in  = (iy > 0)     ? (0.5*dt*area_in*diff_in*rhog_in   / (dr_in*vol_y*rhog_c))  : 0.0;

        cn_lower[iy] = -cn_in;
        cn_upper[iy] = -cn_out;
        max_cn_sum = fmax(max_cn_sum, cn_in + cn_out);
    }

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

    real upper_mod[N_Y], ratio_mod[N_Y];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            real cn_in  = -cn_lower[iy];
            real cn_out = -cn_upper[iy];
            real ratio_prev = (iy > 0)     ? ratio[iy-1] : ratio[iy];
            real ratio_next = (iy < N_Y-1) ? ratio[iy+1] : ratio[iy];
            ratio_rhs[iy] = cn_in*ratio_prev
                          + (1.0 - cn_in - cn_out)*ratio[iy]
                          + cn_out*ratio_next;
        }

        // ---- Thomas algorithm: forward sweep and backward substitution ----
        upper_mod[0] = cn_upper[0] / cn_diag[0];
        ratio_mod[0] = ratio_rhs[0] / cn_diag[0];

        for (int iy = 1; iy < N_Y; iy++)
        {
            real pivot = cn_diag[iy] - cn_lower[iy]*upper_mod[iy-1];
            upper_mod[iy] = (iy < N_Y-1) ? (cn_upper[iy] / pivot) : 0.0;
            ratio_mod[iy] = (ratio_rhs[iy] - cn_lower[iy]*ratio_mod[iy-1]) / pivot;
        }

        // Keep ratio as q_old and complete q_new in ratio_mod so the CN face flux can be
        // reconstructed after the solve.
        for (int iy = N_Y-2; iy >= 0; iy--)
            ratio_mod[iy] -= upper_mod[iy]*ratio_mod[iy+1];

        // Area-weighted radial mass flux A*J at each outer face.  Recovering it from cn_out
        // guarantees that its divergence is exactly the conservative CN density increment.
        // The last outer face is the no-flux domain boundary.
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y-1)
            {
                upper_mod[iy] = 0.0;
                continue;
            }

            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real r_c = Y_MIN*pow(dy, iy + 0.5);
            real R_c = r_c*sin(zc);
            real Z_c = r_c*cos(zc);
            real h_c = _get_hg(R_c);
            real rhog_c = _get_rhog(R_c, Z_c, h_c);
            real cn_out = -cn_upper[iy];
            upper_mod[iy] = -(cn_out*vol_y*rhog_c / dt_sub)*
                ((ratio[iy+1] - ratio[iy]) + (ratio_mod[iy+1] - ratio_mod[iy]));
        }

        // ratio_rhs is no longer needed, so reuse it for one momentum face flux at a time.
        for (int iy = 0; iy < N_Y; iy++)
        {
            int donor = (upper_mod[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y-1) donor = iy;
            real r_donor = Y_MIN*pow(dy, donor + 0.5);
            real R_donor = r_donor*sin(zc);
            real Z_donor = r_donor*cos(zc);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + donor*N_X + iz*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomx[idx_donor] / rho_donor
                                                    : sqrt(G*M_S*fmax(R_donor, 0.0));
            ratio_rhs[iy] = upper_mod[iy]*spec_mom;
        }
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_in = (iy > 0) ? ratio_rhs[iy-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[idx_cell] -= dt_sub*(ratio_rhs[iy] - flux_in) / vol_y;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            int donor = (upper_mod[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y-1) donor = iy;
            real r_donor = Y_MIN*pow(dy, donor + 0.5);
            real R_donor = r_donor*sin(zc);
            real Z_donor = r_donor*cos(zc);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + donor*N_X + iz*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomy[idx_donor] / rho_donor : 0.0;
            ratio_rhs[iy] = upper_mod[iy]*spec_mom;
        }
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_in = (iy > 0) ? ratio_rhs[iy-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[idx_cell] -= dt_sub*(ratio_rhs[iy] - flux_in) / vol_y;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            int donor = (upper_mod[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y-1) donor = iy;
            real r_donor = Y_MIN*pow(dy, donor + 0.5);
            real R_donor = r_donor*sin(zc);
            real Z_donor = r_donor*cos(zc);
            real h_donor = _get_hg(R_donor);
            real rho_donor = _get_rhog(R_donor, Z_donor, h_donor)*ratio[donor];
            int idx_donor = ix + donor*N_X + iz*N_X*N_Y;
            real spec_mom = (rho_donor >= RHO_VAC) ? dev_dustmomz[idx_donor] / rho_donor : 0.0;
            ratio_rhs[iy] = upper_mod[iy]*spec_mom;
        }
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_in = (iy > 0) ? ratio_rhs[iy-1] : 0.0;
            int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[idx_cell] -= dt_sub*(ratio_rhs[iy] - flux_in) / vol_y;
            ratio[iy] = ratio_mod[iy];
        }
    }

    // ---- Write back ----
    for (int iy = 0; iy < N_Y; iy++)
    {
        real r_c = Y_MIN*pow(dy, iy + 0.5);
        real R_c = r_c*sin(zc);
        real Z_c = r_c*cos(zc);
        real h_c = _get_hg(R_c);
        real rhog_c = _get_rhog(R_c, Z_c, h_c);
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = rhog_c*ratio[iy];
    }
}

// ====================================================================

#endif // DIFFUSION
