#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_advection_x
// Purpose: FARGO + PPM azimuthal advection of rho_d and all three momentum densities.
//          Periodic BC (phi wraps around).  Primitive PPM reconstruction followed by a
//          shared conservative HLL flux for density and all momentum components.
//
// Parallelisation: NB_X blocks, 1 thread per (iy, iz) ring, loop over N_X cells.
//
// FARGO step: compute ring-mean angular momentum velx_avg; integer-shift all 4 fields by
//   the nearest integer n_shift = round(velx_avg * dt / (Rc^2 * dphi)) cells.  The PPM
//   residual is measured relative to the angular speed actually represented by that integer
//   shift, Omega_shift = n_shift*dphi/dt.  This retains the fractional mean-ring displacement.
//   All primitive velocities are recovered from the current density and momenta at entry.
//   Vacuum cells use lx_K, preventing a retrograde FARGO residual of -(lx_K/Rc) from
//   draining cells at the inner boundary.
//
// Left and right density/velocity states are reconstructed with PPM.  The pressureless HLL
// solve converts them to one interface flux, so mass and momentum cannot select different
// upwind states.  All shifted primitive velocities correspond to the current conserved state.
//
// PPM edges: PERIODIC, all faces use the full 4-point stencil (modular wrapping).
//   qedge[i] = face between cells i-1 and i.
// =========================================================================================================================

__global__
void f_advection_x (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_Y*N_Z) return;

    int iy = idx % N_Y;
    int iz = idx / N_Y;

    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = (N_Z > 1) ? (Z_MIN + (iz + 0.5)*dz) : 0.5*(Z_MIN + Z_MAX);
    real Rc = yc*sin(zc);

    // ---- Load ring ----
    real dens[N_X], momx[N_X], momy[N_X], momz[N_X], velx[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        
        dens[ix] = dev_dustdens[ic];
        momx[ix] = dev_dustmomx[ic];
        momy[ix] = dev_dustmomy[ic];
        momz[ix] = dev_dustmomz[ic];

        real vely_tmp, velz_tmp;
        _recover_dust_state(dens[ix], Rc, momx[ix], momy[ix], momz[ix], velx[ix], vely_tmp, velz_tmp);
    }

    // ---- FARGO: ring-mean angular momentum and integer shift ----
    real velx_avg = 0.0;
    for (int ix = 0; ix < N_X; ix++)
    {
        velx_avg += velx[ix];
    }
    velx_avg /= static_cast<real>(N_X);
    
    real shift_cells = velx_avg*dt / (Rc*Rc*dx);
    int n_shift = __double2int_rn(shift_cells);
    real omega_shift = static_cast<real>(n_shift)*dx / dt;
    real velx_shift_frame = Rc*Rc*omega_shift;

    // Integer-shift all 4 fields (circular permutation)
    real dens_shift[N_X], momx_shift[N_X], momy_shift[N_X], momz_shift[N_X], velx_shift[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ix_old = ((ix - n_shift) % N_X + N_X) % N_X;
        
        dens_shift[ix] = dens[ix_old];
        momx_shift[ix] = momx[ix_old];
        momy_shift[ix] = momy[ix_old];
        momz_shift[ix] = momz[ix_old];
        velx_shift[ix] = velx[ix_old];
    }

    // Recover shifted primitive velocities from the shifted conserved state
    real vely_shift[N_X], velz_shift[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        _recover_dust_state(dens_shift[ix], Rc,
            momx_shift[ix], momy_shift[ix], momz_shift[ix],
            velx_shift[ix], vely_shift[ix], velz_shift[ix]
        );
    }

    // Residual angular momentum after the integer shift
    // Subtracting velx_avg here would discard the fractional part of the mean orbital displacement
    real velx_res[N_X];
    for (int ix = 0; ix < N_X; ix++) 
    {
        velx_res[ix] = velx_shift[ix] - velx_shift_frame;
    }

    // ---- Pass 1: PPM edge values (fully periodic, all faces use 4-point stencil) ----
    real edge_dens[N_X], edge_velx[N_X], edge_vely[N_X], edge_velz[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm2 = (ix - 2 + N_X) % N_X;
        int ixm1 = (ix - 1 + N_X) % N_X;
        int ixp1 = (ix + 1) % N_X;

        edge_dens[ix] = _ppm_edge(dens_shift[ixm2], dens_shift[ixm1], dens_shift[ix], dens_shift[ixp1]);
        edge_velx[ix] = _ppm_edge(velx_shift[ixm2], velx_shift[ixm1], velx_shift[ix], velx_shift[ixp1]);
        edge_vely[ix] = _ppm_edge(vely_shift[ixm2], vely_shift[ixm1], vely_shift[ix], vely_shift[ixp1]);
        edge_velz[ix] = _ppm_edge(velz_shift[ixm2], velz_shift[ixm1], velz_shift[ix], velz_shift[ixp1]);
    }

    // ---- Pass 2: shared conservative flux at face ix+1/2 (between cells ix and ix+1) ----
    real flux_dens[N_X], flux_momx[N_X], flux_momy[N_X], flux_momz[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        int ixp2 = (ix + 2) % N_X;

        real cfl_L = fabs(velx_res[ix]  /(Rc*Rc))*dt / dx;
        real cfl_R = fabs(velx_res[ixp1]/(Rc*Rc))*dt / dx;

        real dens_L = fmax(_ppm_face_value(edge_dens, dens_shift, ix,   ixp1, true,  cfl_L), 0.0);
        real dens_R = fmax(_ppm_face_value(edge_dens, dens_shift, ixp1, ixp2, false, cfl_R), 0.0);
        real velx_L =      _ppm_face_value(edge_velx, velx_shift, ix,   ixp1, true,  cfl_L);
        real velx_R =      _ppm_face_value(edge_velx, velx_shift, ixp1, ixp2, false, cfl_R);
        real vely_L =      _ppm_face_value(edge_vely, vely_shift, ix,   ixp1, true,  cfl_L);
        real vely_R =      _ppm_face_value(edge_vely, vely_shift, ixp1, ixp2, false, cfl_R);
        real velz_L =      _ppm_face_value(edge_velz, velz_shift, ix,   ixp1, true,  cfl_L);
        real velz_R =      _ppm_face_value(edge_velz, velz_shift, ixp1, ixp2, false, cfl_R);

        real omega_L = (velx_L - velx_shift_frame) / (Rc*Rc);
        real omega_R = (velx_R - velx_shift_frame) / (Rc*Rc);
        
        _pressureless_hll_flux(omega_L, omega_R, 
            dens_L, velx_L, vely_L, velz_L, 
            dens_R, velx_R, vely_R, velz_R,
            flux_dens[ix], flux_momx[ix], flux_momy[ix], flux_momz[ix]
        );
    }

    // Positivity-preserving conservative flux scaling
    // A face and all of its momentum components receive the same donor-cell factor
    real flux_scale[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm1 = (ix - 1 + N_X) % N_X;
        
        real outgoing = dt*(fmax(flux_dens[ix], 0.0) + fmax(-flux_dens[ixm1], 0.0));
        real available = fmax(dens_shift[ix], 0.0)*dx;
        real allowed = (1.0 - 1.0e-12)*available;
        
        flux_scale[ix] = (outgoing > allowed && outgoing > 0.0) ? allowed / outgoing : 1.0;
    }

    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        int donor = (flux_dens[ix] >= 0.0) ? ix : ixp1;
        
        real scale = flux_scale[donor];

        flux_dens[ix] *= scale;
        flux_momx[ix] *= scale;
        flux_momy[ix] *= scale;
        flux_momz[ix] *= scale;
    }

    // ---- Conservative update: dens^{n+1} = dens_shift - dt*(flux[ix] - flux[ix-1]) / dphi ----
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm1 = (ix - 1 + N_X) % N_X;

        dens_shift[ix] -= dt*(flux_dens[ix] - flux_dens[ixm1]) / dx;
        momx_shift[ix] -= dt*(flux_momx[ix] - flux_momx[ixm1]) / dx;
        momy_shift[ix] -= dt*(flux_momy[ix] - flux_momy[ixm1]) / dx;
        momz_shift[ix] -= dt*(flux_momz[ix] - flux_momz[ixm1]) / dx;
        
        if (dens_shift[ix] < 0.0)
        {
            // Only roundoff can remain after conservative flux limiting.
            dens_shift[ix] = 0.0;
            momx_shift[ix] = 0.0;
            momy_shift[ix] = 0.0;
            momz_shift[ix] = 0.0;
        }
    }

    // ---- Write back ----
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = dens_shift[ix];
        dev_dustmomx[ic] = momx_shift[ix];
        dev_dustmomy[ic] = momy_shift[ix];
        dev_dustmomz[ic] = momz_shift[ix];
    }
}

// =========================================================================================================================
