#ifdef DIFFUSION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// ---- Y direction (radial) -- Implicit CN TDMA, NB_Y blocks, 1 thread per (ix,iz) column ----
// Face areas and cell volumes use the same radial geometry:
//   pow_y=2: 2D azimuthal-radial disk (face area proportional to y)
//   pow_y=3: full 3D spherical grid (face area proportional to y^2)
// In 3D each line holds z fixed, so y is the spherical radial coordinate throughout the solve.
// Turbulent dust flux: J_y = -D_y*rho_g*d(rho_d/rho_g)/dy
// The minimum conservative momentum closure carries upwind-cell specific momentum with the
// time-centred CN mass flux at each radial face

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
    real dz = _get_dz();

    real pow_y = _get_powy();

    real zc = Z_MIN + (iz + 0.5)*dz;

    // convert rho_d to q = rho_d/rho_g at cell centres
    real ratio[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        real yc = Y_MIN*pow(dy, iy + 0.5);
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g  = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        int ic = ix + iy*N_X + iz*N_X*N_Y;

        ratio[iy] = dev_dustdens[ic] / rhog;
    }

    // store the full-step off-diagonal coefficients first
    // CN is linearly stable for any dt, but a sufficient positivity condition is cn_i + cn_o <= 1 in every cell
    // equal substeps enforce that condition without a mass-destroying clamp after the solve
    real cn_lower[N_Y], cn_diag[N_Y], cn_upper[N_Y], ratio_rhs[N_Y];
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
        real y1 = Y_MIN*pow(dy, static_cast<real>(iy + 1));
        real yc = Y_MIN*pow(dy, iy + 0.5);

        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

        // centre-to-centre distances (for gradient at each face)
        real dy_len_i = yc*(dy - 1.0) / dy;   // yc(iy)   - yc(iy - 1)
        real dy_len_o = yc*(dy - 1.0);        // yc(iy+1) - yc(iy)

        // analytic gas density at the cell centre and radial faces
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g  = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        real R_i = y0*sin(zc);
        real Z_i = y0*cos(zc);
        real h_i = _get_hg(R_i);
        real rhog_i = _get_rhog(R_i, Z_i, h_i);
        real Dy_i = _get_nu(R_i, h_i) / SC_Y;

        real R_o = y1*sin(zc);
        real Z_o = y1*cos(zc);
        real h_o = _get_hg(R_o);
        real rhog_o = _get_rhog(R_o, Z_o, h_o);
        real Dy_o = _get_nu(R_o, h_o) / SC_Y;

        // CN half-step coefficients (zero at boundaries -> no-flux BC)
        // dividing the conservative rho_d equation by cell-centred rho_g gives
        // coeff = 0.5*dt*area*D_face*rho_g_face/(dy_len*volume*rho_g_cell)
        real area_i = pow(y0, pow_y - 1.0);
        real area_o = pow(y1, pow_y - 1.0);

        real cn_i = (iy > 0)       ? (0.5*dt*area_i*Dy_i*rhog_i / (dy_len_i*vol_y*rhog)) : 0.0;
        real cn_o = (iy < N_Y - 1) ? (0.5*dt*area_o*Dy_o*rhog_o / (dy_len_o*vol_y*rhog)) : 0.0;

        cn_lower[iy] = -cn_i;
        cn_upper[iy] = -cn_o;

        max_cn_sum = fmax(max_cn_sum, cn_i + cn_o);
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

    real upper_work[N_Y], ratio_work[N_Y];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            real cn_i = -cn_lower[iy];
            real cn_o = -cn_upper[iy];

            real ratio_prev = (iy > 0)       ? ratio[iy - 1] : ratio[iy];
            real ratio_next = (iy < N_Y - 1) ? ratio[iy + 1] : ratio[iy];

            ratio_rhs[iy] = cn_i*ratio_prev + (1.0 - cn_i - cn_o)*ratio[iy] + cn_o*ratio_next;
        }

        // Thomas algorithm: forward sweep and backward substitution
        upper_work[0] = cn_upper[0]  / cn_diag[0];
        ratio_work[0] = ratio_rhs[0] / cn_diag[0];

        for (int iy = 1; iy < N_Y; iy++)
        {
            real pivot = cn_diag[iy] - cn_lower[iy]*upper_work[iy - 1];

            upper_work[iy] = (iy < N_Y - 1) ? (cn_upper[iy] / pivot) : 0.0;
            ratio_work[iy] = (ratio_rhs[iy] - cn_lower[iy]*ratio_work[iy - 1]) / pivot;
        }

        // keep ratio as q_old and complete q_new in ratio_work so the CN face flux can be reconstructed after the solve
        for (int iy = N_Y - 2; iy >= 0; iy--)
        {
            ratio_work[iy] -= upper_work[iy]*ratio_work[iy + 1];
        }

        // area-weighted radial mass flux A*J at each outer face
        // recovering it from cn_o guarantees that its divergence is exactly the conservative CN density increment
        // the last outer face is the no-flux domain boundary
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y - 1)
            {
                upper_work[iy] = 0.0;
                continue;
            }

            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

            real yc = Y_MIN*pow(dy, iy + 0.5);
            real Rc = yc*sin(zc);
            real Zc = yc*cos(zc);

            real h_g  = _get_hg(Rc);
            real rhog = _get_rhog(Rc, Zc, h_g);
            real cn_o = -cn_upper[iy];

            upper_work[iy]  = -(cn_o*vol_y*rhog / dt_sub);
            upper_work[iy] *= (ratio[iy + 1] - ratio[iy]) + (ratio_work[iy + 1] - ratio_work[iy]);
        }

        // ratio_rhs is no longer needed, so reuse it for one momentum face flux at a time.
        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real y_up = Y_MIN*pow(dy, iy_up + 0.5);
            real R_up = y_up*sin(zc);
            real Z_up = y_up*cos(zc);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(R_up, 0.0));

            ratio_rhs[iy] = upper_work[iy]*velx_up;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? ratio_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomx[ic] -= dt_sub*(ratio_rhs[iy] - flux_i) / vol_y;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real y_up = Y_MIN*pow(dy, iy_up + 0.5);
            real R_up = y_up*sin(zc);
            real Z_up = y_up*cos(zc);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            ratio_rhs[iy] = upper_work[iy]*vely_up;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? ratio_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomy[ic] -= dt_sub*(ratio_rhs[iy] - flux_i) / vol_y;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            int iy_up = (upper_work[iy] >= 0.0) ? iy : iy + 1;
            if (iy == N_Y - 1) iy_up = iy;

            real y_up = Y_MIN*pow(dy, iy_up + 0.5);
            real R_up = y_up*sin(zc);
            real Z_up = y_up*cos(zc);

            real h_up = _get_hg(R_up);
            real dens_up = _get_rhog(R_up, Z_up, h_up)*ratio[iy_up];

            int ic_up = ix + iy_up*N_X + iz*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            ratio_rhs[iy] = upper_work[iy]*velz_up;
        }

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real flux_i = (iy > 0) ? ratio_rhs[iy - 1] : 0.0;

            int ic = ix + iy*N_X + iz*N_X*N_Y;
            dev_dustmomz[ic] -= dt_sub*(ratio_rhs[iy] - flux_i) / vol_y;

            ratio[iy] = ratio_work[iy];
        }
    }

    for (int iy = 0; iy < N_Y; iy++)
    {
        real yc = Y_MIN*pow(dy, iy + 0.5);
        real Rc = yc*sin(zc);
        real Zc = yc*cos(zc);

        real h_g  = _get_hg(Rc);
        real rhog = _get_rhog(Rc, Zc, h_g);

        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = rhog*ratio[iy];
    }
}

// =========================================================================================================================

#endif // DIFFUSION
