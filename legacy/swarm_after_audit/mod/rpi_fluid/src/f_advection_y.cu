#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_advection_y
// Purpose: PPM (Colella-Woodward 1984) radial advection of dens, momx, momy, and momz
//          Primitive PPM reconstruction with a shared conservative HLL flux
//          Outflow BC at both radial boundaries
//          Logarithmic grid
//
// Parallelisation: NB_Y blocks, 1 thread per (ix, iz) column, sequential loop over N_Y cells
//
// Density and velocity are reconstructed as primitive interface states
// The face solver then converts each left/right pair into one conservative HLL flux for density and all momenta
// High-order PPM/HLL corrections are convex-limited from a first-order HLL state so density stays
// positive and all three specific momenta remain within the local stage bounds
//
// SSPRK(3,3) method of lines:
//   Stage 1: U^(1) = U^n + dt*L(U^n)
//   Stage 2: U^(2) = 0.75*U^n + 0.25*(U^(1) + dt*L(U^(1)))
//   Result:  U^(n+1) = (1/3)*U^n + (2/3)*(U^(2) + dt*L(U^(2)))
// L uses instantaneous limited PPM face states and the HLL flux. This time-centres both normal
// velocity divergence and radial geometric dilution without a separate trace correction.
//
// FV update (spherical radial geometry):
//   dens_iy^{n+1} = dens_iy^n - dt*(y_out^(pow_y-1)*J_out - y_in^(pow_y-1)*J_in) / vol_y
//   pow_y = 2 for a 2D azimuthal-radial disk; pow_y = 3 for the full 3D spherical grid
// =========================================================================================================================

__global__
void f_advection_y (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_weight_y, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Z) return;

    int ix = idx % N_X;
    int iz = idx / N_X;

    real dy = _get_dy();
    real dz = _get_dz();

    real zc = Z_MIN + (iz + 0.5)*dz;

    real pow_y = _get_powy();

    real dens[N_Y], momx[N_Y], momy[N_Y], momz[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iy] = dev_dustdens[ic];
        momx[iy] = dev_dustmomx[ic];
        momy[iy] = dev_dustmomy[ic];
        momz[iy] = dev_dustmomz[ic];
    }

    for (int stage = 0; stage < 3; stage++)
    {
        // Recover primitives from this RK stage. The HLL solve converts them back to one common
        // conservative flux for density and all momenta.
        real velx[N_Y], vely[N_Y], velz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            real yc = Y_MIN*pow(dy, iy + 0.5);
            real Rc = yc*sin(zc);

            _recover_dust_state(dens[iy], Rc, momx[iy], momy[iy], momz[iy], velx[iy], vely[iy], velz[iy]);
        }

        // pass 1: PPM edge values
        real edge_dens[N_Y + 1], edge_velx[N_Y + 1], edge_vely[N_Y + 1], edge_velz[N_Y + 1];

        _ppm_edges_nonuniform(dens, dev_weight_y, edge_dens, N_Y);
        _ppm_edges_nonuniform(velx, dev_weight_y, edge_velx, N_Y);
        _ppm_edges_nonuniform(vely, dev_weight_y, edge_vely, N_Y);
        _ppm_edges_nonuniform(velz, dev_weight_y, edge_velz, N_Y);

        // pass 2: face fluxes
        real flux_dens[N_Y], flux_momx[N_Y], flux_momy[N_Y], flux_momz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y - 1) // outer boundary: outflow only
            {
                real speed_ob = vely[iy];
                real outflow = (speed_ob > 0.0) ? 1.0 : 0.0;

                flux_dens[iy] = outflow*speed_ob*fmax(dens[iy], 0.0);
                flux_momx[iy] = flux_dens[iy]*velx[iy];
                flux_momy[iy] = flux_dens[iy]*vely[iy];
                flux_momz[iy] = flux_dens[iy]*velz[iy];
                edge_dens[iy] = edge_velx[iy] = edge_vely[iy] = edge_velz[iy] = 0.0;

                continue;
            }

            // Instantaneous limited PPM states. SSPRK(3,3) supplies the temporal centring.
            real dens_L = fmax(_ppm_face_value(edge_dens, dens, iy,     iy + 1, true,  0.0), 0.0);
            real dens_R = fmax(_ppm_face_value(edge_dens, dens, iy + 1, iy + 2, false, 0.0), 0.0);
            real velx_L =      _ppm_face_value(edge_velx, velx, iy,     iy + 1, true,  0.0);
            real velx_R =      _ppm_face_value(edge_velx, velx, iy + 1, iy + 2, false, 0.0);
            real vely_L =      _ppm_face_value(edge_vely, vely, iy,     iy + 1, true,  0.0);
            real vely_R =      _ppm_face_value(edge_vely, vely, iy + 1, iy + 2, false, 0.0);
            real velz_L =      _ppm_face_value(edge_velz, velz, iy,     iy + 1, true,  0.0);
            real velz_R =      _ppm_face_value(edge_velz, velz, iy + 1, iy + 2, false, 0.0);

            _pressureless_hll_flux(
                vely_L, vely_R,
                dens_L, velx_L, vely_L, velz_L,
                dens_R, velx_R, vely_R, velz_R,
                flux_dens[iy], flux_momx[iy], flux_momy[iy], flux_momz[iy]
            );

            // First-order cell-centred HLL is the invariant-domain base flux. The edge arrays are
            // no longer needed at this face, so reuse their storage for the high-minus-low correction.
            real flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low;
            _pressureless_hll_flux(
                vely[iy], vely[iy + 1],
                dens[iy], velx[iy], vely[iy], velz[iy],
                dens[iy + 1], velx[iy + 1], vely[iy + 1], velz[iy + 1],
                flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low
            );

            edge_dens[iy] = flux_dens[iy] - flux_dens_low;
            edge_velx[iy] = flux_momx[iy] - flux_momx_low;
            edge_vely[iy] = flux_momy[iy] - flux_momy_low;
            edge_velz[iy] = flux_momz[iy] - flux_momz_low;
            flux_dens[iy] = flux_dens_low;
            flux_momx[iy] = flux_momx_low;
            flux_momy[iy] = flux_momy_low;
            flux_momz[iy] = flux_momz_low;
        }

        // inner radial boundary (ib): outflow only
        real speed_ib = vely[0];
        real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
        real flux_momx_ib = flux_dens_ib*velx[0];
        real flux_momy_ib = flux_dens_ib*vely[0];
        real flux_momz_ib = flux_dens_ib*velz[0];

        // Positivity-safe first-order update with spherical radial geometry
        real area_ratio = pow(dy, pow_y - 1.0);  // ratio of outer to inner radial face areas

        {
            real y0 = Y_MIN;
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real area_i = pow(y0, pow_y - 1.0);

            dens[0] -= dt*area_i*(area_ratio*flux_dens[0] - flux_dens_ib) / vol_y;
            momx[0] -= dt*area_i*(area_ratio*flux_momx[0] - flux_momx_ib) / vol_y;
            momy[0] -= dt*area_i*(area_ratio*flux_momy[0] - flux_momy_ib) / vol_y;
            momz[0] -= dt*area_i*(area_ratio*flux_momz[0] - flux_momz_ib) / vol_y;

            if (dens[0] < 0.0) dens[0] = momx[0] = momy[0] = momz[0] = 0.0;
        }

        for (int iy = 1; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real area_i = pow(y0, pow_y - 1.0);

            dens[iy] -= dt*area_i*(area_ratio*flux_dens[iy] - flux_dens[iy - 1]) / vol_y;
            momx[iy] -= dt*area_i*(area_ratio*flux_momx[iy] - flux_momx[iy - 1]) / vol_y;
            momy[iy] -= dt*area_i*(area_ratio*flux_momy[iy] - flux_momy[iy - 1]) / vol_y;
            momz[iy] -= dt*area_i*(area_ratio*flux_momz[iy] - flux_momz[iy - 1]) / vol_y;

            if (dens[iy] < 0.0) dens[iy] = momx[iy] = momy[iy] = momz[iy] = 0.0;
        }

        // Add the PPM antidiffusive correction one face at a time. Each face uses the minimum
        // admissible fraction from its two neighbours, preserving one conservative shared flux.
        for (int iy = 0; iy < N_Y - 1; iy++)
        {
            real y_face = Y_MIN*pow(dy, static_cast<real>(iy + 1));
            real area_f = pow(y_face, pow_y - 1.0);
            real vol_L = pow(y_face / dy, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
            real vol_R = pow(y_face, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;

            real corr_dens_L = -dt*area_f*edge_dens[iy] / vol_L;
            real corr_momx_L = -dt*area_f*edge_velx[iy] / vol_L;
            real corr_momy_L = -dt*area_f*edge_vely[iy] / vol_L;
            real corr_momz_L = -dt*area_f*edge_velz[iy] / vol_L;
            real corr_dens_R =  dt*area_f*edge_dens[iy] / vol_R;
            real corr_momx_R =  dt*area_f*edge_velx[iy] / vol_R;
            real corr_momy_R =  dt*area_f*edge_vely[iy] / vol_R;
            real corr_momz_R =  dt*area_f*edge_velz[iy] / vol_R;

            real velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L;
            real velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R;
            _local_bounds(velx, iy,     N_Y, velx_min_L, velx_max_L);
            _local_bounds(vely, iy,     N_Y, vely_min_L, vely_max_L);
            _local_bounds(velz, iy,     N_Y, velz_min_L, velz_max_L);
            _local_bounds(velx, iy + 1, N_Y, velx_min_R, velx_max_R);
            _local_bounds(vely, iy + 1, N_Y, vely_min_R, vely_max_R);
            _local_bounds(velz, iy + 1, N_Y, velz_min_R, velz_max_R);

            real scale_L = _invariant_scale(
                dens[iy], momx[iy], momy[iy], momz[iy],
                corr_dens_L, corr_momx_L, corr_momy_L, corr_momz_L,
                velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L
            );
            real scale_R = _invariant_scale(
                dens[iy + 1], momx[iy + 1], momy[iy + 1], momz[iy + 1],
                corr_dens_R, corr_momx_R, corr_momy_R, corr_momz_R,
                velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R
            );
            real scale = fmin(scale_L, scale_R);

            dens[iy] += scale*corr_dens_L;
            momx[iy] += scale*corr_momx_L;
            momy[iy] += scale*corr_momy_L;
            momz[iy] += scale*corr_momz_L;
            dens[iy + 1] += scale*corr_dens_R;
            momx[iy + 1] += scale*corr_momx_R;
            momy[iy + 1] += scale*corr_momy_R;
            momz[iy + 1] += scale*corr_momz_R;
        }

        // After the second forward-Euler evaluation, form the second Shu-Osher stage.
        // The device arrays remain U^n throughout the kernel, so no extra line-local state is needed.
        if (stage == 1)
        {
            for (int iy = 0; iy < N_Y; iy++)
            {
                int ic = ix + iy*N_X + iz*N_X*N_Y;

                dens[iy] = 0.75*dev_dustdens[ic] + 0.25*dens[iy];
                momx[iy] = 0.75*dev_dustmomx[ic] + 0.25*momx[iy];
                momy[iy] = 0.75*dev_dustmomy[ic] + 0.25*momy[iy];
                momz[iy] = 0.75*dev_dustmomz[ic] + 0.25*momz[iy];
            }
        }
    }

    // Complete the final Shu-Osher convex combination; the device arrays still contain U^n.
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = (1.0/3.0)*dev_dustdens[ic] + (2.0/3.0)*dens[iy];
        dev_dustmomx[ic] = (1.0/3.0)*dev_dustmomx[ic] + (2.0/3.0)*momx[iy];
        dev_dustmomy[ic] = (1.0/3.0)*dev_dustmomy[ic] + (2.0/3.0)*momy[iy];
        dev_dustmomz[ic] = (1.0/3.0)*dev_dustmomz[ic] + (2.0/3.0)*momz[iy];
    }
}

// =========================================================================================================================
