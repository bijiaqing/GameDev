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
//
// Two-pass PPM in radial volume coordinate s=y^pow_y/pow_y:
//   Pass 1: reconstruct conservative face values from nonuniform volume averages using mesh-dependent cubic weights
//           (linear at the first/last interior face)
//   Pass 2: Colella-Woodward limiting and tracing with the fraction of cell volume swept
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

    // recover all primitive variables from the current conserved state
    // the HLL solve below converts these primitives back to a common conservative face flux
    real dens[N_Y], momx[N_Y], momy[N_Y], momz[N_Y], velx[N_Y], vely[N_Y], velz[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dens[iy] = dev_dustdens[ic];
        momx[iy] = dev_dustmomx[ic];
        momy[iy] = dev_dustmomy[ic];
        momz[iy] = dev_dustmomz[ic];

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

            continue;
        }

        // trace PPM states by the fraction of radial cell volume swept to this face
        // in the volume coordinate s = y^d/d the existing PPM integral formulas remain conservative
        // y_inner_L | left cell iy | y_face | right cell iy+1 | y_outer_R
        real y_face = Y_MIN*pow(dy, static_cast<real>(iy + 1));
        real y_inner_L = y_face / dy; // inner face of the left cell
        real y_outer_R = y_face*dy;   // outer face of the right cell

        // s = y^pow_y / pow_y is the volume coordinate for the radial sweep
        real s_face = pow(y_face, pow_y) / pow_y;
        real s_inner_L = pow(y_inner_L, pow_y) / pow_y;
        real s_outer_R = pow(y_outer_R, pow_y) / pow_y;

        // trace the left and right states to the face by the fraction of cell volume swept
        real y_trace_L = y_face - fabs(vely[iy])    *dt;
        real y_trace_R = y_face + fabs(vely[iy + 1])*dt;
        real s_trace_L = pow(y_trace_L, pow_y) / pow_y;
        real s_trace_R = pow(y_trace_R, pow_y) / pow_y;

        // compute the CFL fraction of the left/right cell volume swept to the face
        real cfl_L = (s_face - s_trace_L) / (s_face - s_inner_L);
        real cfl_R = (s_trace_R - s_face) / (s_outer_R - s_face);

        // compute the left/right face states from the PPM edge values and the CFL fraction swept
        real dens_L = fmax(_ppm_face_value(edge_dens, dens, iy,     iy + 1, true,  cfl_L), 0.0);
        real dens_R = fmax(_ppm_face_value(edge_dens, dens, iy + 1, iy + 2, false, cfl_R), 0.0);
        real velx_L =      _ppm_face_value(edge_velx, velx, iy,     iy + 1, true,  cfl_L);
        real velx_R =      _ppm_face_value(edge_velx, velx, iy + 1, iy + 2, false, cfl_R);
        real vely_L =      _ppm_face_value(edge_vely, vely, iy,     iy + 1, true,  cfl_L);
        real vely_R =      _ppm_face_value(edge_vely, vely, iy + 1, iy + 2, false, cfl_R);
        real velz_L =      _ppm_face_value(edge_velz, velz, iy,     iy + 1, true,  cfl_L);
        real velz_R =      _ppm_face_value(edge_velz, velz, iy + 1, iy + 2, false, cfl_R);

        _pressureless_hll_flux(
            vely_L, vely_R,
            dens_L, velx_L, vely_L, velz_L,
            dens_R, velx_R, vely_R, velz_R,
            flux_dens[iy], flux_momx[iy], flux_momy[iy], flux_momz[iy]
        );
    }

    // inner radial boundary (ib): outflow only
    real speed_ib = vely[0];
    real flux_dens_ib = (speed_ib < 0.0) ? speed_ib*fmax(dens[0], 0.0) : 0.0;
    real flux_momx_ib = flux_dens_ib*velx[0];
    real flux_momy_ib = flux_dens_ib*vely[0];
    real flux_momz_ib = flux_dens_ib*velz[0];

    // limit the total outward mass from each cell
    // then apply the same upwind factor to every conserved component at a face
    real flux_scale[N_Y];
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
        real vol_y = pow(y0, pow_y)*(pow(dy, pow_y) - 1.0) / pow_y;
        real flux_i = (iy == 0) ? flux_dens_ib : flux_dens[iy - 1];

        real mass_leave = dt*(pow(y0*dy, pow_y - 1.0)*fmax(flux_dens[iy], 0.0) + pow(y0, pow_y - 1.0)*fmax(-flux_i, 0.0));
        real mass_avail = fmax(dens[iy], 0.0)*vol_y;
        real mass_allow = (1.0 - 1.0e-12)*mass_avail;

        flux_scale[iy] = (mass_leave > mass_allow && mass_leave > 0.0) ? mass_allow / mass_leave : 1.0;
    }

    flux_dens_ib *= flux_scale[0];
    flux_momx_ib *= flux_scale[0];
    flux_momy_ib *= flux_scale[0];
    flux_momz_ib *= flux_scale[0];

    for (int iy = 0; iy < N_Y; iy++)
    {
        int iy_up = (flux_dens[iy] >= 0.0 || iy == N_Y - 1) ? iy : iy + 1;
        real scale = flux_scale[iy_up];

        flux_dens[iy] *= scale;
        flux_momx[iy] *= scale;
        flux_momy[iy] *= scale;
        flux_momz[iy] *= scale;
    }

    // conservative update with spherical radial geometry
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
