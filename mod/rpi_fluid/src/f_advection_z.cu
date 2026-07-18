#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_advection_z
// Purpose: PPM polar advection of rho_d and all three momentum densities.
//          Primitive PPM reconstruction with a shared conservative HLL flux.  Reconstruction
//          and characteristic tracing use polar volume coordinate s=-cos(theta).
//          sin(theta)-weighted FV geometry.  The upper-colatitude boundary (iz=0) is outflow.
//          With HALFDISK, the last face is a reflecting midplane; otherwise it is the lower
//          full-disk boundary and is also outflow. Skipped when no polar coordinate is active.
//
// Parallelisation: NB_Z blocks, 1 thread per (ix, iy) column.
//
// PPM edge indexing: edge[i] is the face between cells i-1 and i.  Interior faces use
// conservative nonuniform volume-average weights; physical boundary states remain first order.
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

    real yc = Y_MIN*pow(_get_dy(), iy + 0.5);
    real dz = _get_dz();

    // ---- Load column (primitive) ----
    real dens[N_Z], momx[N_Z], momy[N_Z], momz[N_Z], velz[N_Z], velx[N_Z], vely[N_Z];

    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iz] = dev_dustdens[ic];
        momx[iz] = dev_dustmomx[ic];
        momy[iz] = dev_dustmomy[ic];
        momz[iz] = dev_dustmomz[ic];

        real zc = Z_MIN + (iz + 0.5)*dz;
        real Rc = yc*sin(zc);
        
        _recover_dust_state(dens[iz], Rc, momx[iz], momy[iz], momz[iz], velx[iz], vely[iz], velz[iz]);
    }

    // ---- Pass 1: PPM edge values (primitive) ----
    real edge_dens[N_Z + 1], edge_velx[N_Z + 1], edge_vely[N_Z + 1], edge_velz[N_Z + 1];

    _ppm_edges_nonuniform(dens, dev_weight_z, edge_dens, N_Z);
    _ppm_edges_nonuniform(velx, dev_weight_z, edge_velx, N_Z);
    _ppm_edges_nonuniform(vely, dev_weight_z, edge_vely, N_Z);
    _ppm_edges_nonuniform(velz, dev_weight_z, edge_velz, N_Z);

    // ---- Pass 2: shared conservative face fluxes ----
    real flux_dens[N_Z], flux_momx[N_Z], flux_momy[N_Z], flux_momz[N_Z];

    for (int iz = 0; iz < N_Z; iz++)
    {
        if (iz == N_Z - 1)
        {
            #ifdef HALFDISK
            // The last face is the symmetry midplane: no transport across it.
            flux_dens[iz] = 0.0;
            flux_momx[iz] = 0.0;
            flux_momy[iz] = 0.0;
            flux_momz[iz] = 0.0;
            #else
            // In a full disk the midplane is an interior face.  The last face is instead the
            // lower-colatitude domain boundary: permit increasing-theta outflow and prohibit
            // external inflow, using the first-order boundary state.
            real speed_ob = velz[iz] / yc;
            flux_dens[iz] = (speed_ob > 0.0) ? speed_ob*fmax(dens[iz], 0.0) : 0.0;
            flux_momx[iz] = flux_dens[iz]*velx[iz];
            flux_momy[iz] = flux_dens[iz]*vely[iz];
            flux_momz[iz] = flux_dens[iz]*velz[iz];
            #endif

            continue;
        }

        // Volume-swept PPM fractions.  Since dtheta/dt=v_theta/r=l_theta/r^2,
        // trace the two footpoints in theta and measure their swept -cos(theta) volume.
        real z_face = Z_MIN + (iz + 1)*dz;
        real z_foot_L = z_face - fabs(velz[iz])*dt / (yc*yc);
        real z_foot_R = z_face + fabs(velz[iz+1])*dt / (yc*yc);
        real s_face = -cos(z_face);
        real s_in_L = -cos(z_face - dz);
        real s_out_R = -cos(z_face + dz);
        real s_foot_L = -cos(z_foot_L);
        real s_foot_R = -cos(z_foot_R);
        real cfl_L = (s_face - s_foot_L) / (s_face - s_in_L);
        real cfl_R = (s_foot_R - s_face) / (s_out_R - s_face);

        real dens_L = fmax(_ppm_face_value(edge_dens, dens, iz,     iz + 1, true,  cfl_L), 0.0);
        real dens_R = fmax(_ppm_face_value(edge_dens, dens, iz + 1, iz + 2, false, cfl_R), 0.0);
        real velx_L =      _ppm_face_value(edge_velx, velx, iz,     iz + 1, true,  cfl_L);
        real velx_R =      _ppm_face_value(edge_velx, velx, iz + 1, iz + 2, false, cfl_R);
        real vely_L =      _ppm_face_value(edge_vely, vely, iz,     iz + 1, true,  cfl_L);
        real vely_R =      _ppm_face_value(edge_vely, vely, iz + 1, iz + 2, false, cfl_R);
        real velz_L =      _ppm_face_value(edge_velz, velz, iz,     iz + 1, true,  cfl_L);
        real velz_R =      _ppm_face_value(edge_velz, velz, iz + 1, iz + 2, false, cfl_R);

        _pressureless_hll_flux(velz_L / yc, velz_R / yc,
            dens_L, velx_L, vely_L, velz_L,
            dens_R, velx_R, vely_R, velz_R,
            flux_dens[iz], flux_momx[iz], flux_momy[iz], flux_momz[iz]
        );
    }

    // Upper-colatitude domain boundary: permit decreasing-theta outflow only.
    real speed_ib = velz[0] / yc;
    real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
    real flux_momx_ib = flux_dens_ib*velx[0];
    real flux_momy_ib = flux_dens_ib*vely[0];
    real flux_momz_ib = flux_dens_ib*velz[0];

    // Positivity-preserving conservative face scaling.
    real flux_scale[N_Z];
    for (int iz = 0; iz < N_Z; iz++)
    {
        real z_in = Z_MIN + iz*dz;
        real z_out = z_in + dz;
        real dcos = cos(z_in) - cos(z_out);
        real flux_in = (iz == 0) ? flux_dens_ib : flux_dens[iz-1];
        real outgoing = dt*(sin(z_out)*fmax(flux_dens[iz], 0.0) + sin(z_in)*fmax(-flux_in, 0.0));
        real available = fmax(dens[iz], 0.0)*yc*dcos;
        real allowed = (1.0 - 1.0e-12)*available;
        
        flux_scale[iz] = (outgoing > allowed && outgoing > 0.0) ? allowed / outgoing : 1.0;
    }

    flux_dens_ib *= flux_scale[0];
    flux_momx_ib *= flux_scale[0];
    flux_momy_ib *= flux_scale[0];
    flux_momz_ib *= flux_scale[0];

    for (int iz = 0; iz < N_Z; iz++)
    {
        int donor  = (flux_dens[iz] >= 0.0 || iz == N_Z-1) ? iz : iz+1;
        real scale = flux_scale[donor];

        flux_dens[iz] *= scale;
        flux_momx[iz] *= scale;
        flux_momy[iz] *= scale;
        flux_momz[iz] *= scale;
    }

    // ---- Conservative update with sin(theta) geometry ----
    // Divergence theorem for the theta-direction flux at fixed r=yc requires an explicit 1/r
    // factor: (1/(r*sin(theta))) d(sin(theta)*rho*v_theta)/d(theta), discretized as
    // dt*(sin(theta_out)*F_out - sin(theta_in)*F_in) / (yc*d_cosz).  Unlike the X-direction
    // (azimuthal) sweep -- where R cancels because l_phi = v_phi*R absorbs it exactly -- this
    // r factor does NOT cancel here, since velz (l_theta) carries no compensating structure.
    // iz = 0 peeled out explicitly: computes the upper colatitude boundary flux here (outflow
    // only) and uses it directly, avoiding a redundant iz==0 branch on every iteration below.
    {
        real z0 = Z_MIN;
        real d_cosz = cos(z0) - cos(z0 + dz);

        dens[0] -= dt*(sin(z0 + dz)*flux_dens[0] - sin(z0)*flux_dens_ib) / (yc*d_cosz);
        momx[0] -= dt*(sin(z0 + dz)*flux_momx[0] - sin(z0)*flux_momx_ib) / (yc*d_cosz);
        momy[0] -= dt*(sin(z0 + dz)*flux_momy[0] - sin(z0)*flux_momy_ib) / (yc*d_cosz);
        momz[0] -= dt*(sin(z0 + dz)*flux_momz[0] - sin(z0)*flux_momz_ib) / (yc*d_cosz);
        
        if (dens[0] < 0.0)
        {
            dens[0] = 0.0;
            momx[0] = 0.0;
            momy[0] = 0.0;
            momz[0] = 0.0;
        }
    }

    for (int iz = 1; iz < N_Z; iz++)
    {
        real z0 = Z_MIN + dz*static_cast<real>(iz);
        real d_cosz = cos(z0) - cos(z0 + dz);

        dens[iz] -= dt*(sin(z0 + dz)*flux_dens[iz] - sin(z0)*flux_dens[iz - 1]) / (yc*d_cosz);
        momx[iz] -= dt*(sin(z0 + dz)*flux_momx[iz] - sin(z0)*flux_momx[iz - 1]) / (yc*d_cosz);
        momy[iz] -= dt*(sin(z0 + dz)*flux_momy[iz] - sin(z0)*flux_momy[iz - 1]) / (yc*d_cosz);
        momz[iz] -= dt*(sin(z0 + dz)*flux_momz[iz] - sin(z0)*flux_momz[iz - 1]) / (yc*d_cosz);
        
        if (dens[iz] < 0.0)
        {
            dens[iz] = 0.0;
            momx[iz] = 0.0;
            momy[iz] = 0.0;
            momz[iz] = 0.0;
        }
    }

    // ---- Write back ----
    for (int iz = 0; iz < N_Z; iz++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = dens[iz];
        dev_dustmomx[ic] = momx[iz];
        dev_dustmomy[ic] = momy[iz];
        dev_dustmomz[ic] = momz[iz];
    }
}

// =========================================================================================================================
