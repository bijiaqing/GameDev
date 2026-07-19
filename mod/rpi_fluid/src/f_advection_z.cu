#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_advection_z
// Purpose: PPM polar advection of dens, momx, momy, and momz
//          Primitive PPM reconstruction with a shared conservative HLL flux
//          Reconstruction uses polar finite-volume coordinate s = -cos(z)
//          sin(z)-weighted FV geometry
//          The upper-colatitude boundary (iz=0) is outflow
//          With HALFDISK, the last face is a reflecting midplane
//          otherwise it is the lower full-disk boundary and is also outflow
//          Skipped when no polar coordinate is active
//
// Parallelisation: NB_Z blocks, 1 thread per (ix, iy) column
//
// PPM edge indexing: edge[i] is the face between cells i-1 and i
// Interior faces use conservative nonuniform volume-average weights; physical boundary states remain first order
// SSPRK2 evaluates the instantaneous PPM/HLL spatial operator twice, time-centring both normal
// velocity divergence and sin(z) geometric dilution without a separate characteristic trace correction
// =========================================================================================================================

__global__
void f_advection_z (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
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

    real dens[N_Z], momx[N_Z], momy[N_Z], momz[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iz] = dev_dustdens[ic];
        momx[iz] = dev_dustmomx[ic];
        momy[iz] = dev_dustmomy[ic];
        momz[iz] = dev_dustmomz[ic];
    }

    for (int stage = 0; stage < 2; stage++)
    {
        real velx[N_Z], vely[N_Z], velz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            real zc = Z_MIN + (iz + 0.5)*dz;
            real Rc = yc*sin(zc);

            _recover_dust_state(dens[iz], Rc, momx[iz], momy[iz], momz[iz], velx[iz], vely[iz], velz[iz]);
        }

        // pass 1: PPM edge values (primitive)
        real edge_dens[N_Z + 1], edge_velx[N_Z + 1], edge_vely[N_Z + 1], edge_velz[N_Z + 1];

        _ppm_edges_nonuniform(dens, dev_weight_z, edge_dens, N_Z);
        _ppm_edges_nonuniform(velx, dev_weight_z, edge_velx, N_Z);
        _ppm_edges_nonuniform(vely, dev_weight_z, edge_vely, N_Z);
        _ppm_edges_nonuniform(velz, dev_weight_z, edge_velz, N_Z);

        // pass 2: shared conservative face fluxes
        real flux_dens[N_Z], flux_momx[N_Z], flux_momy[N_Z], flux_momz[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            if (iz == N_Z - 1)
            {
                #ifdef HALFDISK
                // the last face is the symmetry midplane: no transport across it
                flux_dens[iz] = flux_momx[iz] = flux_momy[iz] = flux_momz[iz] = 0.0;
                #else
                // permit increasing-Z outflow and prohibit external inflow, using the first-order boundary state
                real speed_ob = velz[iz] / yc;
                flux_dens[iz] = (speed_ob > 0.0) ? speed_ob*fmax(dens[iz], 0.0) : 0.0;
                flux_momx[iz] = flux_dens[iz]*velx[iz];
                flux_momy[iz] = flux_dens[iz]*vely[iz];
                flux_momz[iz] = flux_dens[iz]*velz[iz];
                #endif

                continue;
            }

            // Instantaneous limited PPM states. SSPRK2 supplies the temporal centring.
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
        }

        // upper-colatitude domain boundary: permit decreasing-Z outflow only
        real speed_ib = velz[0] / yc;
        real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
        real flux_momx_ib = flux_dens_ib*velx[0];
        real flux_momy_ib = flux_dens_ib*vely[0];
        real flux_momz_ib = flux_dens_ib*velz[0];

        // positivity-preserving conservative face scaling
        real flux_scale[N_Z];
        for (int iz = 0; iz < N_Z; iz++)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real z1 = z0 + dz;
            real vol_z = cos(z0) - cos(z1);
            real flux_i = (iz == 0) ? flux_dens_ib : flux_dens[iz - 1];

            real mass_leave = dt*(sin(z1)*fmax(flux_dens[iz], 0.0) + sin(z0)*fmax(-flux_i, 0.0));
            real mass_avail = fmax(dens[iz], 0.0)*yc*vol_z;
            real mass_allow = (1.0 - 1.0e-12)*mass_avail;

            flux_scale[iz] = (mass_leave > mass_allow && mass_leave > 0.0) ? mass_allow / mass_leave : 1.0;
        }

        flux_dens_ib *= flux_scale[0];
        flux_momx_ib *= flux_scale[0];
        flux_momy_ib *= flux_scale[0];
        flux_momz_ib *= flux_scale[0];

        for (int iz = 0; iz < N_Z; iz++)
        {
            int iz_up = (flux_dens[iz] >= 0.0 || iz == N_Z - 1) ? iz : iz + 1;
            real scale = flux_scale[iz_up];

            flux_dens[iz] *= scale;
            flux_momx[iz] *= scale;
            flux_momy[iz] *= scale;
            flux_momz[iz] *= scale;
        }

        // conservative update with sin(z) geometry
        // divergence theorem for the Z-direction flux at fixed y = yc requires an explicit 1/y factor:
        // (1/(y*sin(z))) d(sin(z)*dens*v_z)/dz, discretized as
        // dt*(sin(z_out)*F_out - sin(z_in)*F_in) / (yc*vol_z)
        // Unlike the X-direction sweep, where R cancels because velx = v_x*R absorbs it exactly,
        // this y factor does NOT cancel here, since velz = v_z*y supplies only the transport conversion
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

    }

    // The device arrays still contain U^n, so no additional local copy or global work array is needed.
    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = 0.5*(dev_dustdens[ic] + dens[iz]);
        dev_dustmomx[ic] = 0.5*(dev_dustmomx[ic] + momx[iz]);
        dev_dustmomy[ic] = 0.5*(dev_dustmomy[ic] + momy[iz]);
        dev_dustmomz[ic] = 0.5*(dev_dustmomz[ic] + momz[iz]);
    }
}

// =========================================================================================================================
