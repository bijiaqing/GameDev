#include <fluid_kern.cuh>
#include <advection.cuh>
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
    const real *dev_weight_y, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Z) return;

    int ix = idx % N_X;
    int iz = idx / N_X;

    real dy = _get_dy();
    real dz = _get_dz();

    real zc = Z_MIN + (iz + 0.5)*dz;

    // select cylindrical radial geometry in 2D and spherical radial geometry in 3D
    real pow_y = _get_powy();

    // load one radial column from global memory
    real dens[N_Y], momx[N_Y], momy[N_Y], momz[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iy] = dev_dustdens[ic];
        momx[iy] = dev_dustmomx[ic];
        momy[iy] = dev_dustmomy[ic];
        momz[iy] = dev_dustmomz[ic];
    }

    // advance three forward-Euler operator evaluations for SSPRK(3,3)
    for (int stage = 0; stage < 3; stage++)
    {
        // recover primitive quantities from the current stage state
        real velx[N_Y], vely[N_Y], velz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            real yc = Y_MIN*pow(dy, iy + 0.5);
            real Rc = yc*sin(zc);

            _recover_dust_state(dens[iy], Rc, momx[iy], momy[iy], momz[iy], velx[iy], vely[iy], velz[iy]);
        }

        // reconstruct PPM face values in the radial finite-volume coordinate
        real edge_dens[N_Y + 1], edge_velx[N_Y + 1], edge_vely[N_Y + 1], edge_velz[N_Y + 1];

        _ppm_edges_nonuniform(dens, dev_weight_y, edge_dens, N_Y);
        _ppm_edges_nonuniform(velx, dev_weight_y, edge_velx, N_Y);
        _ppm_edges_nonuniform(vely, dev_weight_y, edge_vely, N_Y);
        _ppm_edges_nonuniform(velz, dev_weight_y, edge_velz, N_Y);

        // compute interior face fluxes and the outflow-only outer boundary flux
        real flux_dens[N_Y], flux_momx[N_Y], flux_momy[N_Y], flux_momz[N_Y];
        for (int iy = 0; iy < N_Y; iy++)
        {
            if (iy == N_Y - 1)
            {
                // permit outward transport and suppress inflow at the outer radial boundary
                real speed_ob = vely[iy];
                real outflow = (speed_ob > 0.0) ? 1.0 : 0.0;

                flux_dens[iy] = outflow*speed_ob*fmax(dens[iy], 0.0);
                flux_momx[iy] = flux_dens[iy]*velx[iy];
                flux_momy[iy] = flux_dens[iy]*vely[iy];
                flux_momz[iy] = flux_dens[iy]*velz[iy];
                edge_dens[iy] = edge_velx[iy] = edge_vely[iy] = edge_velz[iy] = 0.0;

                continue;
            }

            // reconstruct high-order PPM states at the interior radial face
            // use zero PPM tracing fraction because SSPRK supplies temporal integration
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

            // compute the low-order HLL flux from adjacent cell-centred states
            real flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low;
            _pressureless_hll_flux(
                vely[iy], vely[iy + 1],
                dens[iy], velx[iy], vely[iy], velz[iy],
                dens[iy + 1], velx[iy + 1], vely[iy + 1], velz[iy + 1],
                flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low
            );

            // retain low-order fluxes and store high-minus-low differences for the antidiffusive correction
            edge_dens[iy] = flux_dens[iy] - flux_dens_low;
            edge_velx[iy] = flux_momx[iy] - flux_momx_low;
            edge_vely[iy] = flux_momy[iy] - flux_momy_low;
            edge_velz[iy] = flux_momz[iy] - flux_momz_low;
            
            flux_dens[iy] = flux_dens_low;
            flux_momx[iy] = flux_momx_low;
            flux_momy[iy] = flux_momy_low;
            flux_momz[iy] = flux_momz_low;
        }

        // permit outward transport and suppress inflow at the inner radial boundary
        real speed_ib = vely[0];
        real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
        real flux_momx_ib = flux_dens_ib*velx[0];
        real flux_momy_ib = flux_dens_ib*vely[0];
        real flux_momz_ib = flux_dens_ib*velz[0];

        // precompute the outer-to-inner face area ratio for logarithmic radial cells
        real area_ratio = pow(dy, pow_y - 1.0);

        // apply the geometry-aware low-order update to the innermost radial cell
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

        // apply the geometry-aware low-order update to the remaining radial cells
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

        // apply volume-scaled antidiffusive transfers across interior radial faces
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

            // bound each transported primitive quantity by neighboring stage values
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
            // limit both cell corrections by one shared scale to preserve conservation and the local invariant domain
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

        // form the second SSPRK(3,3) convex combination after the second Euler evaluation
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

    // form the final SSPRK(3,3) combination and write the radial column to global memory
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = (1.0/3.0)*dev_dustdens[ic] + (2.0/3.0)*dens[iy];
        dev_dustmomx[ic] = (1.0/3.0)*dev_dustmomx[ic] + (2.0/3.0)*momx[iy];
        dev_dustmomy[ic] = (1.0/3.0)*dev_dustmomy[ic] + (2.0/3.0)*momy[iy];
        dev_dustmomz[ic] = (1.0/3.0)*dev_dustmomz[ic] + (2.0/3.0)*momz[iy];
    }
}
