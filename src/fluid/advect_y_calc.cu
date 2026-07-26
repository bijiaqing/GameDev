#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advect_y_calc
// purpose: radial transport with nonuniform PPM, pressureless HLL fluxes, open boundaries, and invariant-domain limiting
//
// parallelization: one thread per azimuthal-polar column with a serial loop over N_Y radial cells
//
// per call:
//   1 three SSPRK(3,3) forward-Euler evaluations
//   2 PPM high-order and cell-centred low-order HLL flux construction
//   3 geometry-aware low-order conservative update
//   4 invariant-domain-limited antidiffusive correction
// =========================================================================================================================

__global__
void advect_y_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_y, real dt)
{
    int idx_col = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_col >= N_X*N_Z) return;

    int ix = idx_col % N_X;
    int iz = idx_col / N_X;

    real z = _get_zcent(iz);

    // load one radial column from global memory
    real rhod[N_Y], mx[N_Y], my[N_Y], mz[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        rhod[iy] = dev_dustdens[idx_cell];
        mx[iy] = dev_dustmomx[idx_cell];
        my[iy] = dev_dustmomy[idx_cell];
        mz[iy] = dev_dustmomz[idx_cell];
    }

    // advance three forward-Euler operator evaluations for SSPRK(3,3)
    for (int stage = 0; stage < 3; stage++)
    {
        // recover primitive quantities from the current stage state
        real lx[N_Y], vy[N_Y], lz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            real y = _get_ycent(iy);
            real R = y*sin(z);

            _recover_dust_state(rhod[iy], R, mx[iy], my[iy], mz[iy], lx[iy], vy[iy], lz[iy]);
        }

        // reconstruct PPM face values in the radial finite-volume coordinate
        real face_work_rhod[N_Y + 1], face_work_x[N_Y + 1], face_work_y[N_Y + 1], face_work_z[N_Y + 1];

        _ppm_faces_nonuniform(rhod, dev_ppm_weight_y, face_work_rhod, N_Y);
        _ppm_faces_nonuniform(lx, dev_ppm_weight_y, face_work_x, N_Y);
        _ppm_faces_nonuniform(vy, dev_ppm_weight_y, face_work_y, N_Y);
        _ppm_faces_nonuniform(lz, dev_ppm_weight_y, face_work_z, N_Y);

        // compute interior face fluxes and the outflow-only outer boundary flux
        real flux_rhod[N_Y], flux_mx[N_Y], flux_my[N_Y], flux_mz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y - 1)
            {
                // permit outward transport and suppress inflow at the outer radial boundary
                real speed_o = vy[iy];

                flux_rhod[iy] = (speed_o > 0.0) ? speed_o*fmax(rhod[iy], 0.0) : 0.0;
                flux_mx[iy] = flux_rhod[iy]*lx[iy];
                flux_my[iy] = flux_rhod[iy]*vy[iy];
                flux_mz[iy] = flux_rhod[iy]*lz[iy];
                face_work_rhod[iy] = face_work_x[iy] = face_work_y[iy] = face_work_z[iy] = 0.0;

                continue;
            }

            // reconstruct high-order PPM states at the interior radial face
            // use zero PPM tracing fraction because SSPRK supplies temporal integration
            real rhod_L = fmax(_ppm_face_value(face_work_rhod, rhod, iy,     iy + 1, true,  0.0), 0.0);
            real rhod_R = fmax(_ppm_face_value(face_work_rhod, rhod, iy + 1, iy + 2, false, 0.0), 0.0);
            real lx_L =      _ppm_face_value(face_work_x, lx, iy,     iy + 1, true,  0.0);
            real lx_R =      _ppm_face_value(face_work_x, lx, iy + 1, iy + 2, false, 0.0);
            real vy_L =      _ppm_face_value(face_work_y, vy, iy,     iy + 1, true,  0.0);
            real vy_R =      _ppm_face_value(face_work_y, vy, iy + 1, iy + 2, false, 0.0);
            real lz_L =      _ppm_face_value(face_work_z, lz, iy,     iy + 1, true,  0.0);
            real lz_R =      _ppm_face_value(face_work_z, lz, iy + 1, iy + 2, false, 0.0);

            _pressureless_hll_flux(
                vy_L, vy_R,
                rhod_L, lx_L, vy_L, lz_L,
                rhod_R, lx_R, vy_R, lz_R,
                flux_rhod[iy], flux_mx[iy], flux_my[iy], flux_mz[iy]
            );

            // compute the low-order HLL flux from adjacent cell-centred states
            real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;
            _pressureless_hll_flux(
                vy[iy], vy[iy + 1],
                rhod[iy], lx[iy], vy[iy], lz[iy],
                rhod[iy + 1], lx[iy + 1], vy[iy + 1], lz[iy + 1],
                flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
            );

            // retain low-order fluxes and store high-minus-low differences for the antidiffusive correction
            face_work_rhod[iy] = flux_rhod[iy] - flux_rhod_low;
            face_work_x[iy] = flux_mx[iy] - flux_mx_low;
            face_work_y[iy] = flux_my[iy] - flux_my_low;
            face_work_z[iy] = flux_mz[iy] - flux_mz_low;
            
            flux_rhod[iy] = flux_rhod_low;
            flux_mx[iy] = flux_mx_low;
            flux_my[iy] = flux_my_low;
            flux_mz[iy] = flux_mz_low;
        }

        // permit outward transport and suppress inflow at the inner radial boundary
        real speed_i = vy[0];
        real flux_rhod_i = (speed_i < 0.0) ? speed_i*fmax(rhod[0], 0.0) : 0.0;
        real flux_mx_i = flux_rhod_i*lx[0];
        real flux_my_i = flux_rhod_i*vy[0];
        real flux_mz_i = flux_rhod_i*lz[0];

        // apply the geometry-aware low-order update to the innermost radial cell
        {
            real vol_y = _get_vol_y(0);
            real area_i = _get_area_y(0);
            real area_o = _get_area_y(1);

            rhod[0] -= dt*(area_o*flux_rhod[0] - area_i*flux_rhod_i) / vol_y;
            mx[0] -= dt*(area_o*flux_mx[0] - area_i*flux_mx_i) / vol_y;
            my[0] -= dt*(area_o*flux_my[0] - area_i*flux_my_i) / vol_y;
            mz[0] -= dt*(area_o*flux_mz[0] - area_i*flux_mz_i) / vol_y;

            if (rhod[0] < 0.0) rhod[0] = mx[0] = my[0] = mz[0] = 0.0;
        }

        // apply the geometry-aware low-order update to the remaining radial cells
        for (int iy = 1; iy < N_Y; iy++)
        {
            real vol_y = _get_vol_y(iy);
            real area_i = _get_area_y(iy);
            real area_o = _get_area_y(iy + 1);

            rhod[iy] -= dt*(area_o*flux_rhod[iy] - area_i*flux_rhod[iy - 1]) / vol_y;
            mx[iy] -= dt*(area_o*flux_mx[iy] - area_i*flux_mx[iy - 1]) / vol_y;
            my[iy] -= dt*(area_o*flux_my[iy] - area_i*flux_my[iy - 1]) / vol_y;
            mz[iy] -= dt*(area_o*flux_mz[iy] - area_i*flux_mz[iy - 1]) / vol_y;

            if (rhod[iy] < 0.0) rhod[iy] = mx[iy] = my[iy] = mz[iy] = 0.0;
        }

        // apply volume-scaled antidiffusive transfers across interior radial faces
        for (int iy = 0; iy < N_Y - 1; iy++)
        {
            real area_f = _get_area_y(iy + 1);
            real vol_L = _get_vol_y(iy);
            real vol_R = _get_vol_y(iy + 1);

            real corr_rhod_L = -dt*area_f*face_work_rhod[iy] / vol_L;
            real corr_mx_L = -dt*area_f*face_work_x[iy] / vol_L;
            real corr_my_L = -dt*area_f*face_work_y[iy] / vol_L;
            real corr_mz_L = -dt*area_f*face_work_z[iy] / vol_L;
            real corr_rhod_R =  dt*area_f*face_work_rhod[iy] / vol_R;
            real corr_mx_R =  dt*area_f*face_work_x[iy] / vol_R;
            real corr_my_R =  dt*area_f*face_work_y[iy] / vol_R;
            real corr_mz_R =  dt*area_f*face_work_z[iy] / vol_R;

            real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
            real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;

            // bound each transported primitive quantity by neighboring stage values
            _local_bounds(lx, iy,     N_Y, lx_min_L, lx_max_L);
            _local_bounds(vy, iy,     N_Y, vy_min_L, vy_max_L);
            _local_bounds(lz, iy,     N_Y, lz_min_L, lz_max_L);
            _local_bounds(lx, iy + 1, N_Y, lx_min_R, lx_max_R);
            _local_bounds(vy, iy + 1, N_Y, vy_min_R, vy_max_R);
            _local_bounds(lz, iy + 1, N_Y, lz_min_R, lz_max_R);

            real scale_L = _invariant_scale(
                rhod[iy], mx[iy], my[iy], mz[iy],
                corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
                lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
            );
            real scale_R = _invariant_scale(
                rhod[iy + 1], mx[iy + 1], my[iy + 1], mz[iy + 1],
                corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
                lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
            );
            // limit both cell corrections by one shared scale to preserve conservation and the local invariant domain
            real scale = fmin(scale_L, scale_R);

            rhod[iy] += scale*corr_rhod_L;
            mx[iy] += scale*corr_mx_L;
            my[iy] += scale*corr_my_L;
            mz[iy] += scale*corr_mz_L;
            rhod[iy + 1] += scale*corr_rhod_R;
            mx[iy + 1] += scale*corr_mx_R;
            my[iy + 1] += scale*corr_my_R;
            mz[iy + 1] += scale*corr_mz_R;
        }

        // form the second SSPRK(3,3) convex combination after the second Euler evaluation
        if (stage == 1)
        {
            for (int iy = 0; iy < N_Y; iy++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

                rhod[iy] = 0.75*dev_dustdens[idx_cell] + 0.25*rhod[iy];
                mx[iy] = 0.75*dev_dustmomx[idx_cell] + 0.25*mx[iy];
                my[iy] = 0.75*dev_dustmomy[idx_cell] + 0.25*my[iy];
                mz[iy] = 0.75*dev_dustmomz[idx_cell] + 0.25*mz[iy];
            }
        }
    }

    // form the final SSPRK(3,3) combination and write the radial column to global memory
    for (int iy = 0; iy < N_Y; iy++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[idx_cell] = (1.0/3.0)*dev_dustdens[idx_cell] + (2.0/3.0)*rhod[iy];
        dev_dustmomx[idx_cell] = (1.0/3.0)*dev_dustmomx[idx_cell] + (2.0/3.0)*mx[iy];
        dev_dustmomy[idx_cell] = (1.0/3.0)*dev_dustmomy[idx_cell] + (2.0/3.0)*my[iy];
        dev_dustmomz[idx_cell] = (1.0/3.0)*dev_dustmomz[idx_cell] + (2.0/3.0)*mz[iy];
    }
}
