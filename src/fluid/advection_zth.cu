#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =====================================================================================================================
// kernel: advection_zth
// purpose: polar transport with nonuniform PPM, pressureless HLL fluxes, boundary fluxes, and invariant-domain limiting
//
// parallelization: one thread per azimuthal-radial column with a serial loop over N_Z polar cells
//
// per call:
//   1 three SSPRK(3,3) forward-Euler evaluations
//   2 PPM high-order and cell-centred low-order HLL flux construction
//   3 spherical-geometry low-order conservative update
//   4 invariant-domain-limited antidiffusive correction
// =====================================================================================================================

__global__
void advection_zth (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_z, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Y) return;
    if (N_Z == 1) return;

    int ix = idx_col % N_X;
    int iy = idx_col / N_X;

    real y = _get_ycent(iy);
    real geom_z = _get_area_z(iy) / _get_vol_y(iy);

    // load one polar column from global memory
    real rhod[N_Z], mx[N_Z], my[N_Z], mz[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        rhod[iz] = dev_dustdens[idx_cell];
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

            _recover_dust_state(rhod[iz], R, mx[iz], my[iz], mz[iz], lx[iz], vy[iz], lz[iz]);
        }

        // reconstruct PPM face values in the spherical polar finite-volume coordinate
        real face_work_rhod[N_Z + 1], face_work_x[N_Z + 1], face_work_y[N_Z + 1], face_work_z[N_Z + 1];

        _thread_ppm_faces(rhod, dev_ppm_weight_z, face_work_rhod, N_Z);
        _thread_ppm_faces(lx, dev_ppm_weight_z, face_work_x, N_Z);
        _thread_ppm_faces(vy, dev_ppm_weight_z, face_work_y, N_Z);
        _thread_ppm_faces(lz, dev_ppm_weight_z, face_work_z, N_Z);

        // compute interior face fluxes and the configured outer boundary flux
        real flux_rhod[N_Z], flux_mx[N_Z], flux_my[N_Z], flux_mz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                #ifdef HALF_DISK
                // impose zero flux at the reflecting midplane boundary
                flux_rhod[iz] = flux_mx[iz] = flux_my[iz] = flux_mz[iz] = 0.0;
                #else  // !HALF_DISK
                // permit outward transport and suppress inflow at the outer polar boundary
                real speed_o = lz[iz] / y;
                flux_rhod[iz] = (speed_o > 0.0) ? speed_o*fmax(rhod[iz], 0.0) : 0.0;
                flux_mx[iz] = flux_rhod[iz]*lx[iz];
                flux_my[iz] = flux_rhod[iz]*vy[iz];
                flux_mz[iz] = flux_rhod[iz]*lz[iz];
                #endif // HALF_DISK
                face_work_rhod[iz] = face_work_x[iz] = face_work_y[iz] = face_work_z[iz] = 0.0;

                continue;
            }

            // reconstruct high-order PPM states at the interior polar face
            // use zero PPM tracing fraction because SSPRK supplies temporal integration
            real rhod_L = fmax(_thread_ppm_state(face_work_rhod, rhod, iz,     iz + 1, true,  0.0), 0.0);
            real rhod_R = fmax(_thread_ppm_state(face_work_rhod, rhod, iz + 1, iz + 2, false, 0.0), 0.0);
            real lx_L =      _thread_ppm_state(face_work_x, lx, iz,     iz + 1, true,  0.0);
            real lx_R =      _thread_ppm_state(face_work_x, lx, iz + 1, iz + 2, false, 0.0);
            real vy_L =      _thread_ppm_state(face_work_y, vy, iz,     iz + 1, true,  0.0);
            real vy_R =      _thread_ppm_state(face_work_y, vy, iz + 1, iz + 2, false, 0.0);
            real lz_L =      _thread_ppm_state(face_work_z, lz, iz,     iz + 1, true,  0.0);
            real lz_R =      _thread_ppm_state(face_work_z, lz, iz + 1, iz + 2, false, 0.0);

            _pressureless_hll_flux(
                lz_L / y, lz_R / y,
                rhod_L, lx_L, vy_L, lz_L,
                rhod_R, lx_R, vy_R, lz_R,
                flux_rhod[iz], flux_mx[iz], flux_my[iz], flux_mz[iz]
            );

            // compute the low-order HLL flux from adjacent cell-centred states
            real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;
            _pressureless_hll_flux(
                lz[iz] / y, lz[iz + 1] / y,
                rhod[iz], lx[iz], vy[iz], lz[iz],
                rhod[iz + 1], lx[iz + 1], vy[iz + 1], lz[iz + 1],
                flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
            );

            // retain low-order fluxes and store high-minus-low differences for the antidiffusive correction
            face_work_rhod[iz] = flux_rhod[iz] - flux_rhod_low;
            face_work_x[iz] = flux_mx[iz] - flux_mx_low;
            face_work_y[iz] = flux_my[iz] - flux_my_low;
            face_work_z[iz] = flux_mz[iz] - flux_mz_low;

            flux_rhod[iz] = flux_rhod_low;
            flux_mx[iz] = flux_mx_low;
            flux_my[iz] = flux_my_low;
            flux_mz[iz] = flux_mz_low;
        }

        // permit outward transport and suppress inflow at the inner polar boundary
        real speed_i = lz[0] / y;
        real flux_rhod_i = (speed_i < 0.0) ? speed_i*fmax(rhod[0], 0.0) : 0.0;
        real flux_mx_i = flux_rhod_i*lx[0];
        real flux_my_i = flux_rhod_i*vy[0];
        real flux_mz_i = flux_rhod_i*lz[0];

        // apply the spherical-geometry low-order update to the innermost polar cell
        {
            real z_i = _get_zface(0);
            real z_o = _get_zface(1);
            real vol_z = _get_vol_z(0);

            rhod[0] -= dt*geom_z*(sin(z_o)*flux_rhod[0] - sin(z_i)*flux_rhod_i) / vol_z;
            mx[0] -= dt*geom_z*(sin(z_o)*flux_mx[0] - sin(z_i)*flux_mx_i) / vol_z;
            my[0] -= dt*geom_z*(sin(z_o)*flux_my[0] - sin(z_i)*flux_my_i) / vol_z;
            mz[0] -= dt*geom_z*(sin(z_o)*flux_mz[0] - sin(z_i)*flux_mz_i) / vol_z;

            if (rhod[0] < 0.0) rhod[0] = mx[0] = my[0] = mz[0] = 0.0;
        }

        // apply the spherical-geometry low-order update to the remaining polar cells
        for (int iz = 1; iz < N_Z; iz++)
        {
            real z_i = _get_zface(iz);
            real z_o = _get_zface(iz + 1);
            real vol_z = _get_vol_z(iz);

            rhod[iz] -= dt*geom_z*(sin(z_o)*flux_rhod[iz] - sin(z_i)*flux_rhod[iz - 1]) / vol_z;
            mx[iz] -= dt*geom_z*(sin(z_o)*flux_mx[iz] - sin(z_i)*flux_mx[iz - 1]) / vol_z;
            my[iz] -= dt*geom_z*(sin(z_o)*flux_my[iz] - sin(z_i)*flux_my[iz - 1]) / vol_z;
            mz[iz] -= dt*geom_z*(sin(z_o)*flux_mz[iz] - sin(z_i)*flux_mz[iz - 1]) / vol_z;

            if (rhod[iz] < 0.0) rhod[iz] = mx[iz] = my[iz] = mz[iz] = 0.0;
        }

        // apply volume-scaled antidiffusive transfers across interior polar faces
        for (int iz = 0; iz < N_Z - 1; iz++)
        {
            real z_face = _get_zface(iz + 1);
            real area_f = sin(z_face);
            real vol_L = _get_vol_z(iz);
            real vol_R = _get_vol_z(iz + 1);

            real corr_rhod_L = -dt*geom_z*area_f*face_work_rhod[iz] / vol_L;
            real corr_mx_L = -dt*geom_z*area_f*face_work_x[iz] / vol_L;
            real corr_my_L = -dt*geom_z*area_f*face_work_y[iz] / vol_L;
            real corr_mz_L = -dt*geom_z*area_f*face_work_z[iz] / vol_L;
            real corr_rhod_R =  dt*geom_z*area_f*face_work_rhod[iz] / vol_R;
            real corr_mx_R =  dt*geom_z*area_f*face_work_x[iz] / vol_R;
            real corr_my_R =  dt*geom_z*area_f*face_work_y[iz] / vol_R;
            real corr_mz_R =  dt*geom_z*area_f*face_work_z[iz] / vol_R;

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
                rhod[iz], mx[iz], my[iz], mz[iz],
                corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
                lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
            );
            real scale_R = _invariant_scale(
                rhod[iz + 1], mx[iz + 1], my[iz + 1], mz[iz + 1],
                corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
                lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
            );
            // limit both cell corrections by one shared scale to preserve conservation and the local invariant domain
            real scale = fmin(scale_L, scale_R);

            rhod[iz] += scale*corr_rhod_L;
            mx[iz] += scale*corr_mx_L;
            my[iz] += scale*corr_my_L;
            mz[iz] += scale*corr_mz_L;
            rhod[iz + 1] += scale*corr_rhod_R;
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

                rhod[iz] = 0.75*dev_dustdens[idx_cell] + 0.25*rhod[iz];
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

        dev_dustdens[idx_cell] = (1.0/3.0)*dev_dustdens[idx_cell] + (2.0/3.0)*rhod[iz];
        dev_dustmomx[idx_cell] = (1.0/3.0)*dev_dustmomx[idx_cell] + (2.0/3.0)*mx[iz];
        dev_dustmomy[idx_cell] = (1.0/3.0)*dev_dustmomy[idx_cell] + (2.0/3.0)*my[iz];
        dev_dustmomz[idx_cell] = (1.0/3.0)*dev_dustmomz[idx_cell] + (2.0/3.0)*mz[iz];
    }
}
