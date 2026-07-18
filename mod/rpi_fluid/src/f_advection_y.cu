#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_advection_y
// Purpose: PPM (Colella-Woodward 1984) radial advection of rho_d and all three momentum densities.
//          Primitive PPM reconstruction with a shared conservative HLL flux.  Logarithmic
//          grid.  Outflow BC at both radial boundaries.
//
// Parallelisation: NB_Y blocks, 1 thread per (ix, iz) column, sequential loop over N_Y cells.
//
// Density and velocity are reconstructed as primitive interface states.  The face solver then
// converts each left/right pair into one conservative HLL flux for density and all momenta.
//
// Two-pass PPM in radial volume coordinate s=r^pow_y/pow_y:
//   Pass 1: reconstruct conservative face values from nonuniform volume averages using
//           mesh-dependent cubic weights (linear at the first/last interior face).
//   Pass 2: Colella-Woodward limiting and tracing with the fraction of cell volume swept.
//
// FV update (spherical radial geometry):
//   dens_iy^{n+1} = dens_iy^n
//                 - dt*(r_out^(pow_y-1)*J_out - r_in^(pow_y-1)*J_in) / vol_y
//   pow_y=2 for radial-only and azimuthal-radial disks; pow_y=3 when the polar
//   coordinate is active (radial-colatitude or full 3D spherical geometry).
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
    real pow_y = _get_powy();
    real zc = (N_Z > 1) ? Z_MIN + (iz + 0.5)*_get_dz() : 0.5*(Z_MIN + Z_MAX);

    // ---- Load column ----
    // Recover all primitive velocities from the current conserved state.  This makes the
    // normal transport velocity include changes made by any preceding directional sweep.
    // The HLL solve below converts these primitives back to a common conservative face flux.
    real dens[N_Y], momx[N_Y], momy[N_Y], momz[N_Y], vely[N_Y], velx[N_Y], velz[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iy] = dev_dustdens[ic];
        momx[iy] = dev_dustmomx[ic];
        momy[iy] = dev_dustmomy[ic];
        momz[iy] = dev_dustmomz[ic];

        real r_c = Y_MIN*pow(dy, iy + 0.5);
        real Rc = r_c*sin(zc);
        
        _recover_dust_state(dens[iy], Rc, momx[iy], momy[iy], momz[iy], velx[iy], vely[iy], velz[iy]);
    }

    // ---- Pass 1: PPM edge values ----
    real edge_dens[N_Y + 1], edge_velx[N_Y + 1], edge_vely[N_Y + 1], edge_velz[N_Y + 1];
    
    _ppm_edges_nonuniform(dens, dev_weight_y, edge_dens, N_Y);
    _ppm_edges_nonuniform(velx, dev_weight_y, edge_velx, N_Y);
    _ppm_edges_nonuniform(vely, dev_weight_y, edge_vely, N_Y);
    _ppm_edges_nonuniform(velz, dev_weight_y, edge_velz, N_Y);

    // ---- Pass 2: face fluxes ----
    real flux_dens[N_Y], flux_momx[N_Y], flux_momy[N_Y], flux_momz[N_Y];

    for (int iy = 0; iy < N_Y; iy++)
    {
        if (iy == N_Y - 1) // outer boundary: outflow only
        {
            real v_face = vely[iy];
            real outflow = (v_face > 0.0) ? 1.0 : 0.0;

            flux_dens[iy] = outflow*v_face*fmax(dens[iy], 0.0);
            flux_momx[iy] = flux_dens[iy]*velx[iy];
            flux_momy[iy] = flux_dens[iy]*vely[iy];
            flux_momz[iy] = flux_dens[iy]*velz[iy];
            
            continue;
        }

        // Trace PPM states by the fraction of radial cell volume swept to this face.  In the
        // volume coordinate s=r^d/d the existing PPM integral formulas remain conservative.
        real r_face = Y_MIN*pow(dy, static_cast<real>(iy + 1));
        real r_in_L = r_face / dy;
        real r_out_R = r_face*dy;
        real s_face = pow(r_face, pow_y) / pow_y;
        real s_in_L = pow(r_in_L, pow_y) / pow_y;
        real s_out_R = pow(r_out_R, pow_y) / pow_y;
        real r_foot_L = r_face - fabs(vely[iy])*dt;
        real r_foot_R = r_face + fabs(vely[iy+1])*dt;
        real s_foot_L = pow(r_foot_L, pow_y) / pow_y;
        real s_foot_R = pow(r_foot_R, pow_y) / pow_y;
        real cfl_L = (s_face - s_foot_L) / (s_face - s_in_L);
        real cfl_R = (s_foot_R - s_face) / (s_out_R - s_face);

        real dens_L = fmax(_ppm_face_value(edge_dens, dens, iy,     iy + 1, true,  cfl_L), 0.0);
        real dens_R = fmax(_ppm_face_value(edge_dens, dens, iy + 1, iy + 2, false, cfl_R), 0.0);
        real velx_L =      _ppm_face_value(edge_velx, velx, iy,     iy + 1, true,  cfl_L);
        real velx_R =      _ppm_face_value(edge_velx, velx, iy + 1, iy + 2, false, cfl_R);
        real vely_L =      _ppm_face_value(edge_vely, vely, iy,     iy + 1, true,  cfl_L);
        real vely_R =      _ppm_face_value(edge_vely, vely, iy + 1, iy + 2, false, cfl_R);
        real velz_L =      _ppm_face_value(edge_velz, velz, iy,     iy + 1, true,  cfl_L);
        real velz_R =      _ppm_face_value(edge_velz, velz, iy + 1, iy + 2, false, cfl_R);

        _pressureless_hll_flux(vely_L, vely_R,
            dens_L, velx_L, vely_L, velz_L,
            dens_R, velx_R, vely_R, velz_R,
            flux_dens[iy], flux_momx[iy], flux_momy[iy], flux_momz[iy]
        );
    }

    // Inner radial boundary: outflow only.
    real v_face_ib = vely[0];
    real flux_dens_ib = (v_face_ib < 0.0) ? v_face_ib*fmax(dens[0], 0.0) : 0.0;
    real flux_momx_ib = flux_dens_ib*velx[0];
    real flux_momy_ib = flux_dens_ib*vely[0];
    real flux_momz_ib = flux_dens_ib*velz[0];

    // Limit the total outward mass from each cell, then apply the same donor factor to every
    // conserved component at a face.
    real flux_scale[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
        real flux_in = (iy == 0) ? flux_dens_ib : flux_dens[iy-1];
        real outgoing = dt*(pow(y0*dy, pow_y - 1.0)*fmax(flux_dens[iy], 0.0) + pow(y0,  pow_y - 1.0)*fmax(-flux_in, 0.0));
        real available = fmax(dens[iy], 0.0)*vol_y;
        real allowed = (1.0 - 1.0e-12)*available;
        
        flux_scale[iy] = (outgoing > allowed && outgoing > 0.0) ? allowed / outgoing : 1.0;
    }

    flux_dens_ib *= flux_scale[0];
    flux_momx_ib *= flux_scale[0];
    flux_momy_ib *= flux_scale[0];
    flux_momz_ib *= flux_scale[0];

    for (int iy = 0; iy < N_Y; iy++)
    {
        int  donor = (flux_dens[iy] >= 0.0 || iy == N_Y-1) ? iy : iy+1;
        real scale = flux_scale[donor];
        
        flux_dens[iy] *= scale;
        flux_momx[iy] *= scale;
        flux_momy[iy] *= scale;
        flux_momz[iy] *= scale;
    }

    // ---- Conservative update with spherical radial geometry ----
    // Generalized surface weighting r^(pow_y-1) matches the radial cell volume
    // vol_y = integral r^(pow_y-1) dr.  _get_powy() selects cylindrical disk
    // geometry when N_Z=1 and spherical geometry when the polar coordinate is active.
    // iy = 0 peeled out explicitly: computes the inner boundary flux here (outflow only) and
    // uses it directly, avoiding a redundant iy==0 branch on every iteration of the loop below.

    real dy_dim  = pow(dy, pow_y - 1.0);  // ratio of outer to inner radial face areas

    {
        real y0 = Y_MIN;
        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
        real y_dim = pow(y0, pow_y - 1.0);

        dens[0] -= dt*y_dim*(dy_dim*flux_dens[0] - flux_dens_ib) / vol_y;
        momx[0] -= dt*y_dim*(dy_dim*flux_momx[0] - flux_momx_ib) / vol_y;
        momy[0] -= dt*y_dim*(dy_dim*flux_momy[0] - flux_momy_ib) / vol_y;
        momz[0] -= dt*y_dim*(dy_dim*flux_momz[0] - flux_momz_ib) / vol_y;
        
        if (dens[0] < 0.0)
        {
            dens[0] = 0.0;
            momx[0] = 0.0;
            momy[0] = 0.0;
            momz[0] = 0.0;
        }
    }

    for (int iy = 1; iy < N_Y; iy++)
    {
        real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
        real y_dim = pow(y0, pow_y - 1.0);

        dens[iy] -= dt*y_dim*(dy_dim*flux_dens[iy] - flux_dens[iy-1]) / vol_y;
        momx[iy] -= dt*y_dim*(dy_dim*flux_momx[iy] - flux_momx[iy-1]) / vol_y;
        momy[iy] -= dt*y_dim*(dy_dim*flux_momy[iy] - flux_momy[iy-1]) / vol_y;
        momz[iy] -= dt*y_dim*(dy_dim*flux_momz[iy] - flux_momz[iy-1]) / vol_y;
        
        if (dens[iy] < 0.0)
        {
            dens[iy] = 0.0;
            momx[iy] = 0.0;
            momy[iy] = 0.0;
            momz[iy] = 0.0;
        }
    }

    // ---- Write back ----
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = dens[iy];
        dev_dustmomx[ic] = momx[iy];
        dev_dustmomy[ic] = momy[iy];
        dev_dustmomz[ic] = momz[iy];
    }
}

// =========================================================================================================================
