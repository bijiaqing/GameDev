#include <fluid_kern.cuh>
#include <advection.cuh>
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
    const real *dev_weight_z, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_X*N_Y) return;
    if (N_Z == 1) return;

    int ix = idx % N_X;
    int iy = idx / N_X;

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);

    // load one polar column from global memory
    real dens[N_Z], momx[N_Z], momy[N_Z], momz[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iz] = dev_dustdens[ic];
        momx[iz] = dev_dustmomx[ic];
        momy[iz] = dev_dustmomy[ic];
        momz[iz] = dev_dustmomz[ic];
    }

    // advance three forward-Euler operator evaluations for SSPRK(3,3)
    for (int stage = 0; stage < 3; stage++)
    {
        // recover primitive quantities from the current stage state
        real velx[N_Z], vely[N_Z], velz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            real zc = Z_MIN + (iz + 0.5)*dz;
            real Rc = yc*sin(zc);

            _recover_dust_state(dens[iz], Rc, momx[iz], momy[iz], momz[iz], velx[iz], vely[iz], velz[iz]);
        }

        // reconstruct PPM face values in the spherical polar finite-volume coordinate
        real edge_dens[N_Z + 1], edge_velx[N_Z + 1], edge_vely[N_Z + 1], edge_velz[N_Z + 1];

        _ppm_edges_nonuniform(dens, dev_weight_z, edge_dens, N_Z);
        _ppm_edges_nonuniform(velx, dev_weight_z, edge_velx, N_Z);
        _ppm_edges_nonuniform(vely, dev_weight_z, edge_vely, N_Z);
        _ppm_edges_nonuniform(velz, dev_weight_z, edge_velz, N_Z);

        // compute interior face fluxes and the configured outer boundary flux
        real flux_dens[N_Z], flux_momx[N_Z], flux_momy[N_Z], flux_momz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                #ifdef HALFDISK
                // impose zero flux at the reflecting midplane boundary
                flux_dens[iz] = flux_momx[iz] = flux_momy[iz] = flux_momz[iz] = 0.0;
                #else
                // permit outward transport and suppress inflow at the outer polar boundary
                real speed_ob = velz[iz] / yc;
                flux_dens[iz] = (speed_ob > 0.0) ? speed_ob*fmax(dens[iz], 0.0) : 0.0;
                flux_momx[iz] = flux_dens[iz]*velx[iz];
                flux_momy[iz] = flux_dens[iz]*vely[iz];
                flux_momz[iz] = flux_dens[iz]*velz[iz];
                #endif
                edge_dens[iz] = edge_velx[iz] = edge_vely[iz] = edge_velz[iz] = 0.0;

                continue;
            }

            // reconstruct high-order PPM states at the interior polar face
            // use zero PPM tracing fraction because SSPRK supplies temporal integration
            real dens_L = fmax(_ppm_face_value(edge_dens, dens, iz,     iz + 1, true,  0.0), 0.0);
            real dens_R = fmax(_ppm_face_value(edge_dens, dens, iz + 1, iz + 2, false, 0.0), 0.0);
            real velx_L =      _ppm_face_value(edge_velx, velx, iz,     iz + 1, true,  0.0);
            real velx_R =      _ppm_face_value(edge_velx, velx, iz + 1, iz + 2, false, 0.0);
            real vely_L =      _ppm_face_value(edge_vely, vely, iz,     iz + 1, true,  0.0);
            real vely_R =      _ppm_face_value(edge_vely, vely, iz + 1, iz + 2, false, 0.0);
            real velz_L =      _ppm_face_value(edge_velz, velz, iz,     iz + 1, true,  0.0);
            real velz_R =      _ppm_face_value(edge_velz, velz, iz + 1, iz + 2, false, 0.0);

            _pressureless_hll_flux(
                velz_L / yc, velz_R / yc,
                dens_L, velx_L, vely_L, velz_L,
                dens_R, velx_R, vely_R, velz_R,
                flux_dens[iz], flux_momx[iz], flux_momy[iz], flux_momz[iz]
            );

            // compute the low-order HLL flux from adjacent cell-centred states
            real flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low;
            _pressureless_hll_flux(
                velz[iz] / yc, velz[iz + 1] / yc,
                dens[iz], velx[iz], vely[iz], velz[iz],
                dens[iz + 1], velx[iz + 1], vely[iz + 1], velz[iz + 1],
                flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low
            );

            // retain low-order fluxes and store high-minus-low differences for the antidiffusive correction
            edge_dens[iz] = flux_dens[iz] - flux_dens_low;
            edge_velx[iz] = flux_momx[iz] - flux_momx_low;
            edge_vely[iz] = flux_momy[iz] - flux_momy_low;
            edge_velz[iz] = flux_momz[iz] - flux_momz_low;
            
            flux_dens[iz] = flux_dens_low;
            flux_momx[iz] = flux_momx_low;
            flux_momy[iz] = flux_momy_low;
            flux_momz[iz] = flux_momz_low;
        }

        // permit outward transport and suppress inflow at the inner polar boundary
        real speed_ib = velz[0] / yc;
        real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
        real flux_momx_ib = flux_dens_ib*velx[0];
        real flux_momy_ib = flux_dens_ib*vely[0];
        real flux_momz_ib = flux_dens_ib*velz[0];

        // apply the spherical-geometry low-order update to the innermost polar cell
        {
            real z0 = Z_MIN;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);

            dens[0] -= dt*(sin(z1)*flux_dens[0] - sin(z0)*flux_dens_ib) / (yc*vol_z);
            momx[0] -= dt*(sin(z1)*flux_momx[0] - sin(z0)*flux_momx_ib) / (yc*vol_z);
            momy[0] -= dt*(sin(z1)*flux_momy[0] - sin(z0)*flux_momy_ib) / (yc*vol_z);
            momz[0] -= dt*(sin(z1)*flux_momz[0] - sin(z0)*flux_momz_ib) / (yc*vol_z);

            if (dens[0] < 0.0) dens[0] = momx[0] = momy[0] = momz[0] = 0.0;
        }

        // apply the spherical-geometry low-order update to the remaining polar cells
        for (int iz = 1; iz < N_Z; iz++)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);

            dens[iz] -= dt*(sin(z1)*flux_dens[iz] - sin(z0)*flux_dens[iz - 1]) / (yc*vol_z);
            momx[iz] -= dt*(sin(z1)*flux_momx[iz] - sin(z0)*flux_momx[iz - 1]) / (yc*vol_z);
            momy[iz] -= dt*(sin(z1)*flux_momy[iz] - sin(z0)*flux_momy[iz - 1]) / (yc*vol_z);
            momz[iz] -= dt*(sin(z1)*flux_momz[iz] - sin(z0)*flux_momz[iz - 1]) / (yc*vol_z);

            if (dens[iz] < 0.0) dens[iz] = momx[iz] = momy[iz] = momz[iz] = 0.0;
        }

        // apply volume-scaled antidiffusive transfers across interior polar faces
        for (int iz = 0; iz < N_Z - 1; iz++)
        {
            real z_face = Z_MIN + static_cast<real>(iz + 1)*dz;
            real z0_L = z_face - dz;
            real z1_R = z_face + dz;
            real area_f = sin(z_face);
            real vol_L = cos(z0_L) - cos(z_face);
            real vol_R = cos(z_face) - cos(z1_R);

            real corr_dens_L = -dt*area_f*edge_dens[iz] / (yc*vol_L);
            real corr_momx_L = -dt*area_f*edge_velx[iz] / (yc*vol_L);
            real corr_momy_L = -dt*area_f*edge_vely[iz] / (yc*vol_L);
            real corr_momz_L = -dt*area_f*edge_velz[iz] / (yc*vol_L);
            real corr_dens_R =  dt*area_f*edge_dens[iz] / (yc*vol_R);
            real corr_momx_R =  dt*area_f*edge_velx[iz] / (yc*vol_R);
            real corr_momy_R =  dt*area_f*edge_vely[iz] / (yc*vol_R);
            real corr_momz_R =  dt*area_f*edge_velz[iz] / (yc*vol_R);

            real velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L;
            real velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R;

            // bound each transported primitive quantity by neighboring stage values
            _local_bounds(velx, iz,     N_Z, velx_min_L, velx_max_L);
            _local_bounds(vely, iz,     N_Z, vely_min_L, vely_max_L);
            _local_bounds(velz, iz,     N_Z, velz_min_L, velz_max_L);
            _local_bounds(velx, iz + 1, N_Z, velx_min_R, velx_max_R);
            _local_bounds(vely, iz + 1, N_Z, vely_min_R, vely_max_R);
            _local_bounds(velz, iz + 1, N_Z, velz_min_R, velz_max_R);

            real scale_L = _invariant_scale(
                dens[iz], momx[iz], momy[iz], momz[iz],
                corr_dens_L, corr_momx_L, corr_momy_L, corr_momz_L,
                velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L
            );
            real scale_R = _invariant_scale(
                dens[iz + 1], momx[iz + 1], momy[iz + 1], momz[iz + 1],
                corr_dens_R, corr_momx_R, corr_momy_R, corr_momz_R,
                velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R
            );
            // limit both cell corrections by one shared scale to preserve conservation and the local invariant domain
            real scale = fmin(scale_L, scale_R);

            dens[iz] += scale*corr_dens_L;
            momx[iz] += scale*corr_momx_L;
            momy[iz] += scale*corr_momy_L;
            momz[iz] += scale*corr_momz_L;
            dens[iz + 1] += scale*corr_dens_R;
            momx[iz + 1] += scale*corr_momx_R;
            momy[iz + 1] += scale*corr_momy_R;
            momz[iz + 1] += scale*corr_momz_R;
        }

        // form the second SSPRK(3,3) convex combination after the second Euler evaluation
        if (stage == 1)
        {
            for (int iz = 0; iz < N_Z; iz++)
            {
                int ic = ix + iy*N_X + iz*N_X*N_Y;

                dens[iz] = 0.75*dev_dustdens[ic] + 0.25*dens[iz];
                momx[iz] = 0.75*dev_dustmomx[ic] + 0.25*momx[iz];
                momy[iz] = 0.75*dev_dustmomy[ic] + 0.25*momy[iz];
                momz[iz] = 0.75*dev_dustmomz[ic] + 0.25*momz[iz];
            }
        }
    }

    // form the final SSPRK(3,3) combination and write the polar column to global memory
    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = (1.0/3.0)*dev_dustdens[ic] + (2.0/3.0)*dens[iz];
        dev_dustmomx[ic] = (1.0/3.0)*dev_dustmomx[ic] + (2.0/3.0)*momx[iz];
        dev_dustmomy[ic] = (1.0/3.0)*dev_dustmomy[ic] + (2.0/3.0)*momy[iz];
        dev_dustmomz[ic] = (1.0/3.0)*dev_dustmomz[ic] + (2.0/3.0)*momz[iz];
    }
}
