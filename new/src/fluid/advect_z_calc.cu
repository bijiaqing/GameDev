#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advect_z_calc
// purpose: polar transport with nonuniform PPM, pressureless HLL fluxes, boundary fluxes, and invariant-domain limiting
//
// parallelization: one thread per azimuthal-radial column with a serial loop over N_Z polar cells
//
// per call:
//   1 three SSPRK(3,3) forward-Euler evaluations
//   2 PPM high-order and cell-centred low-order HLL flux construction
//   3 spherical-geometry low-order conservative update
//   4 invariant-domain-limited antidiffusive correction
// =========================================================================================================================

__global__
void advect_z_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_z, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Y) return;
    if (N_Z == 1) return;

    int ix = idx_col % N_X;
    int iy = idx_col / N_X;

    real y = _get_ycent(iy);

    // load one polar column from global memory
    real dens[N_Z], mx[N_Z], my[N_Z], mz[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        dens[iz] = dev_dustdens[idx_cell];
        mx[iz] = dev_dustmomx[idx_cell];
        my[iz] = dev_dustmomy[idx_cell];
        mz[iz] = dev_dustmomz[idx_cell];
    }

    // advance three forward-Euler operator evaluations for SSPRK(3,3)
    for (int stage = 0; stage < 3; stage++)
    {
        // recover primitive quantities from the current stage state
        real lx[N_Z], vy[N_Z], lz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            real z = _get_zcent(iz);
            real R = y*sin(z);

            _recover_dust_state(dens[iz], R, mx[iz], my[iz], mz[iz], lx[iz], vy[iz], lz[iz]);
        }

        // reconstruct PPM face values in the spherical polar finite-volume coordinate
        real face_dens[N_Z + 1], face_lx[N_Z + 1], face_vy[N_Z + 1], face_lz[N_Z + 1];

        _ppm_faces_nonuniform(dens, dev_ppm_weight_z, face_dens, N_Z);
        _ppm_faces_nonuniform(lx, dev_ppm_weight_z, face_lx, N_Z);
        _ppm_faces_nonuniform(vy, dev_ppm_weight_z, face_vy, N_Z);
        _ppm_faces_nonuniform(lz, dev_ppm_weight_z, face_lz, N_Z);

        // compute interior face fluxes and the configured outer boundary flux
        real flux_dens[N_Z], flux_mx[N_Z], flux_my[N_Z], flux_mz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                #ifdef HALFDISK
                // impose zero flux at the reflecting midplane boundary
                flux_dens[iz] = flux_mx[iz] = flux_my[iz] = flux_mz[iz] = 0.0;
                #else
                // permit outward transport and suppress inflow at the outer polar boundary
                real speed_ob = lz[iz] / y;
                flux_dens[iz] = (speed_ob > 0.0) ? speed_ob*fmax(dens[iz], 0.0) : 0.0;
                flux_mx[iz] = flux_dens[iz]*lx[iz];
                flux_my[iz] = flux_dens[iz]*vy[iz];
                flux_mz[iz] = flux_dens[iz]*lz[iz];
                #endif
                face_dens[iz] = face_lx[iz] = face_vy[iz] = face_lz[iz] = 0.0;

                continue;
            }

            // reconstruct high-order PPM states at the interior polar face
            // use zero PPM tracing fraction because SSPRK supplies temporal integration
            real dens_L = fmax(_ppm_face_value(face_dens, dens, iz,     iz + 1, true,  0.0), 0.0);
            real dens_R = fmax(_ppm_face_value(face_dens, dens, iz + 1, iz + 2, false, 0.0), 0.0);
            real lx_L =      _ppm_face_value(face_lx, lx, iz,     iz + 1, true,  0.0);
            real lx_R =      _ppm_face_value(face_lx, lx, iz + 1, iz + 2, false, 0.0);
            real vy_L =      _ppm_face_value(face_vy, vy, iz,     iz + 1, true,  0.0);
            real vy_R =      _ppm_face_value(face_vy, vy, iz + 1, iz + 2, false, 0.0);
            real lz_L =      _ppm_face_value(face_lz, lz, iz,     iz + 1, true,  0.0);
            real lz_R =      _ppm_face_value(face_lz, lz, iz + 1, iz + 2, false, 0.0);

            _pressureless_hll_flux(
                lz_L / y, lz_R / y,
                dens_L, lx_L, vy_L, lz_L,
                dens_R, lx_R, vy_R, lz_R,
                flux_dens[iz], flux_mx[iz], flux_my[iz], flux_mz[iz]
            );

            // compute the low-order HLL flux from adjacent cell-centred states
            real flux_dens_low, flux_mx_low, flux_my_low, flux_mz_low;
            _pressureless_hll_flux(
                lz[iz] / y, lz[iz + 1] / y,
                dens[iz], lx[iz], vy[iz], lz[iz],
                dens[iz + 1], lx[iz + 1], vy[iz + 1], lz[iz + 1],
                flux_dens_low, flux_mx_low, flux_my_low, flux_mz_low
            );

            // retain low-order fluxes and store high-minus-low differences for the antidiffusive correction
            face_dens[iz] = flux_dens[iz] - flux_dens_low;
            face_lx[iz] = flux_mx[iz] - flux_mx_low;
            face_vy[iz] = flux_my[iz] - flux_my_low;
            face_lz[iz] = flux_mz[iz] - flux_mz_low;
            
            flux_dens[iz] = flux_dens_low;
            flux_mx[iz] = flux_mx_low;
            flux_my[iz] = flux_my_low;
            flux_mz[iz] = flux_mz_low;
        }

        // permit outward transport and suppress inflow at the inner polar boundary
        real speed_ib = lz[0] / y;
        real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
        real flux_mx_ib = flux_dens_ib*lx[0];
        real flux_my_ib = flux_dens_ib*vy[0];
        real flux_mz_ib = flux_dens_ib*lz[0];

        // apply the spherical-geometry low-order update to the innermost polar cell
        {
            real z_i = _get_zedge(0);
            real z_o = _get_zedge(1);
            real vol_z = _get_vol_z(0);

            dens[0] -= dt*(sin(z_o)*flux_dens[0] - sin(z_i)*flux_dens_ib) / (y*vol_z);
            mx[0] -= dt*(sin(z_o)*flux_mx[0] - sin(z_i)*flux_mx_ib) / (y*vol_z);
            my[0] -= dt*(sin(z_o)*flux_my[0] - sin(z_i)*flux_my_ib) / (y*vol_z);
            mz[0] -= dt*(sin(z_o)*flux_mz[0] - sin(z_i)*flux_mz_ib) / (y*vol_z);

            if (dens[0] < 0.0) dens[0] = mx[0] = my[0] = mz[0] = 0.0;
        }

        // apply the spherical-geometry low-order update to the remaining polar cells
        for (int iz = 1; iz < N_Z; iz++)
        {
            real z_i = _get_zedge(iz);
            real z_o = _get_zedge(iz + 1);
            real vol_z = _get_vol_z(iz);

            dens[iz] -= dt*(sin(z_o)*flux_dens[iz] - sin(z_i)*flux_dens[iz - 1]) / (y*vol_z);
            mx[iz] -= dt*(sin(z_o)*flux_mx[iz] - sin(z_i)*flux_mx[iz - 1]) / (y*vol_z);
            my[iz] -= dt*(sin(z_o)*flux_my[iz] - sin(z_i)*flux_my[iz - 1]) / (y*vol_z);
            mz[iz] -= dt*(sin(z_o)*flux_mz[iz] - sin(z_i)*flux_mz[iz - 1]) / (y*vol_z);

            if (dens[iz] < 0.0) dens[iz] = mx[iz] = my[iz] = mz[iz] = 0.0;
        }

        // apply volume-scaled antidiffusive transfers across interior polar faces
        for (int iz = 0; iz < N_Z - 1; iz++)
        {
            real z_face = _get_zedge(iz + 1);
            real area_f = sin(z_face);
            real vol_L = _get_vol_z(iz);
            real vol_R = _get_vol_z(iz + 1);

            real corr_dens_L = -dt*area_f*face_dens[iz] / (y*vol_L);
            real corr_mx_L = -dt*area_f*face_lx[iz] / (y*vol_L);
            real corr_my_L = -dt*area_f*face_vy[iz] / (y*vol_L);
            real corr_mz_L = -dt*area_f*face_lz[iz] / (y*vol_L);
            real corr_dens_R =  dt*area_f*face_dens[iz] / (y*vol_R);
            real corr_mx_R =  dt*area_f*face_lx[iz] / (y*vol_R);
            real corr_my_R =  dt*area_f*face_vy[iz] / (y*vol_R);
            real corr_mz_R =  dt*area_f*face_lz[iz] / (y*vol_R);

            real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
            real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;

            // bound each transported primitive quantity by neighboring stage values
            _local_bounds(lx, iz,     N_Z, lx_min_L, lx_max_L);
            _local_bounds(vy, iz,     N_Z, vy_min_L, vy_max_L);
            _local_bounds(lz, iz,     N_Z, lz_min_L, lz_max_L);
            _local_bounds(lx, iz + 1, N_Z, lx_min_R, lx_max_R);
            _local_bounds(vy, iz + 1, N_Z, vy_min_R, vy_max_R);
            _local_bounds(lz, iz + 1, N_Z, lz_min_R, lz_max_R);

            real scale_L = _invariant_scale(
                dens[iz], mx[iz], my[iz], mz[iz],
                corr_dens_L, corr_mx_L, corr_my_L, corr_mz_L,
                lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
            );
            real scale_R = _invariant_scale(
                dens[iz + 1], mx[iz + 1], my[iz + 1], mz[iz + 1],
                corr_dens_R, corr_mx_R, corr_my_R, corr_mz_R,
                lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
            );
            // limit both cell corrections by one shared scale to preserve conservation and the local invariant domain
            real scale = fmin(scale_L, scale_R);

            dens[iz] += scale*corr_dens_L;
            mx[iz] += scale*corr_mx_L;
            my[iz] += scale*corr_my_L;
            mz[iz] += scale*corr_mz_L;
            dens[iz + 1] += scale*corr_dens_R;
            mx[iz + 1] += scale*corr_mx_R;
            my[iz + 1] += scale*corr_my_R;
            mz[iz + 1] += scale*corr_mz_R;
        }

        // form the second SSPRK(3,3) convex combination after the second Euler evaluation
        if (stage == 1)
        {
            for (int iz = 0; iz < N_Z; iz++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

                dens[iz] = 0.75*dev_dustdens[idx_cell] + 0.25*dens[iz];
                mx[iz] = 0.75*dev_dustmomx[idx_cell] + 0.25*mx[iz];
                my[iz] = 0.75*dev_dustmomy[idx_cell] + 0.25*my[iz];
                mz[iz] = 0.75*dev_dustmomz[idx_cell] + 0.25*mz[iz];
            }
        }
    }

    // form the final SSPRK(3,3) combination and write the polar column to global memory
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[idx_cell] = (1.0/3.0)*dev_dustdens[idx_cell] + (2.0/3.0)*dens[iz];
        dev_dustmomx[idx_cell] = (1.0/3.0)*dev_dustmomx[idx_cell] + (2.0/3.0)*mx[iz];
        dev_dustmomy[idx_cell] = (1.0/3.0)*dev_dustmomy[idx_cell] + (2.0/3.0)*my[iz];
        dev_dustmomz[idx_cell] = (1.0/3.0)*dev_dustmomz[idx_cell] + (2.0/3.0)*mz[iz];
    }
}
