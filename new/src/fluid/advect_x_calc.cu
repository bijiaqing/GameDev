#include <advection.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advect_x_calc
// purpose: periodic azimuthal transport with FARGO, PPM, pressureless HLL fluxes, and invariant-domain limiting
//
// parallelization: one thread per radial-polar ring with a serial loop over N_X azimuthal cells
//
// per call:
//   1 FARGO integer shift and residual-frame construction
//   2 PPM high-order and cell-centred low-order HLL flux construction
//   3 low-order conservative update
//   4 invariant-domain-limited antidiffusive correction
// =========================================================================================================================

__global__
void advect_x_calc (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_Y*N_Z) return;

    int iy = idx % N_Y;
    int iz = idx / N_Y;

    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;
    real Rc = yc*sin(zc);

    // load one ring and recover primitive quantities with a Keplerian azimuthal fallback in near-vacuum cells
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

    // average the specific angular momentum used to choose the FARGO integer shift
    real velx_avg = 0.0;
    for (int ix = 0; ix < N_X; ix++)
    {
        velx_avg += velx[ix];
    }
    velx_avg /= static_cast<real>(N_X);

    // represent the ring-mean displacement by the nearest integer shift and leave its fractional remainder for PPM
    real shift_cells = velx_avg*dt / (Rc*Rc*dx);
    int n_shift = __double2int_rn(shift_cells);
    real omega_shift = static_cast<real>(n_shift)*dx / dt;
    real velx_frame = Rc*Rc*omega_shift;

    // circularly shift each conserved state from cell ix minus n_shift
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

    // recover primitives from the shifted conserved state for consistent interface reconstruction
    real vely_shift[N_X], velz_shift[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        _recover_dust_state(dens_shift[ix], Rc,
            momx_shift[ix], momy_shift[ix], momz_shift[ix],
            velx_shift[ix], vely_shift[ix], velz_shift[ix]
        );
    }

    // subtract the integer-shift frame from specific angular momentum to obtain the PPM transport residual
    real velx_res[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        velx_res[ix] = velx_shift[ix] - velx_frame;
    }

    // reconstruct periodic PPM face values with edge ix between cells ix minus one and ix
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

    // compute high-order fluxes from PPM-traced interface states with the pressureless HLL solver
    // use the residual angular displacement over one cell as each one-sided PPM tracing fraction
    real flux_dens[N_X], flux_momx[N_X], flux_momy[N_X], flux_momz[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        int ixp2 = (ix + 2) % N_X;

        real cfl_L = fabs(velx_res[ix]  /(Rc*Rc))*dt / dx;
        real cfl_R = fabs(velx_res[ixp1]/(Rc*Rc))*dt / dx;

        // clamp reconstructed density nonnegative while preserving the signs of the reconstructed primitive quantities
        real dens_L = fmax(_ppm_face_value(edge_dens, dens_shift, ix,   ixp1, true,  cfl_L), 0.0);
        real dens_R = fmax(_ppm_face_value(edge_dens, dens_shift, ixp1, ixp2, false, cfl_R), 0.0);
        real velx_L =      _ppm_face_value(edge_velx, velx_shift, ix,   ixp1, true,  cfl_L);
        real velx_R =      _ppm_face_value(edge_velx, velx_shift, ixp1, ixp2, false, cfl_R);
        real vely_L =      _ppm_face_value(edge_vely, vely_shift, ix,   ixp1, true,  cfl_L);
        real vely_R =      _ppm_face_value(edge_vely, vely_shift, ixp1, ixp2, false, cfl_R);
        real velz_L =      _ppm_face_value(edge_velz, velz_shift, ix,   ixp1, true,  cfl_L);
        real velz_R =      _ppm_face_value(edge_velz, velz_shift, ixp1, ixp2, false, cfl_R);

        // convert reconstructed specific angular momentum to residual angular transport speed
        real omega_L = (velx_L - velx_frame) / (Rc*Rc);
        real omega_R = (velx_R - velx_frame) / (Rc*Rc);

        _pressureless_hll_flux(
            omega_L, omega_R,
            dens_L, velx_L, vely_L, velz_L,
            dens_R, velx_R, vely_R, velz_R,
            flux_dens[ix], flux_momx[ix], flux_momy[ix], flux_momz[ix]
        );
    }

    // compute low-order HLL fluxes and the antidiffusive difference between high- and low-order fluxes
    // reuse density and momentum arrays for flux differences and flux arrays for low-order fluxes
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        real omega_L = velx_res[ix] / (Rc*Rc);
        real omega_R = velx_res[ixp1] / (Rc*Rc);
        real flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low;

        _pressureless_hll_flux(
            omega_L, omega_R,
            dens_shift[ix], velx_shift[ix], vely_shift[ix], velz_shift[ix],
            dens_shift[ixp1], velx_shift[ixp1], vely_shift[ixp1], velz_shift[ixp1],
            flux_dens_low, flux_momx_low, flux_momy_low, flux_momz_low
        );

        dens[ix] = flux_dens[ix] - flux_dens_low;
        momx[ix] = flux_momx[ix] - flux_momx_low;
        momy[ix] = flux_momy[ix] - flux_momy_low;
        momz[ix] = flux_momz[ix] - flux_momz_low;
        
        flux_dens[ix] = flux_dens_low;
        flux_momx[ix] = flux_momx_low;
        flux_momy[ix] = flux_momy_low;
        flux_momz[ix] = flux_momz_low;
    }

    // apply the low-order conservative update with a zero-state backstop for any negative density result
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm1 = (ix - 1 + N_X) % N_X;

        dens_shift[ix] -= dt*(flux_dens[ix] - flux_dens[ixm1]) / dx;
        momx_shift[ix] -= dt*(flux_momx[ix] - flux_momx[ixm1]) / dx;
        momy_shift[ix] -= dt*(flux_momy[ix] - flux_momy[ixm1]) / dx;
        momz_shift[ix] -= dt*(flux_momz[ix] - flux_momz[ixm1]) / dx;

        if (dens_shift[ix] < 0.0) dens_shift[ix] = momx_shift[ix] = momy_shift[ix] = momz_shift[ix] = 0.0;
    }

    // express each antidiffusive face flux as equal-and-opposite corrections to its adjacent cells
    // limit both corrections by one shared scale to preserve conservation and the local invariant domain
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;

        real corr_dens_L = -dt*dens[ix] / dx;
        real corr_momx_L = -dt*momx[ix] / dx;
        real corr_momy_L = -dt*momy[ix] / dx;
        real corr_momz_L = -dt*momz[ix] / dx;

        real corr_dens_R =  dt*dens[ix] / dx;
        real corr_momx_R =  dt*momx[ix] / dx;
        real corr_momy_R =  dt*momy[ix] / dx;
        real corr_momz_R =  dt*momz[ix] / dx;

        real velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L;
        real velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R;

        // bound each transported primitive quantity by neighboring shifted-cell values
        _local_bounds_periodic(velx_shift, ix,   N_X, velx_min_L, velx_max_L);
        _local_bounds_periodic(vely_shift, ix,   N_X, vely_min_L, vely_max_L);
        _local_bounds_periodic(velz_shift, ix,   N_X, velz_min_L, velz_max_L);
        _local_bounds_periodic(velx_shift, ixp1, N_X, velx_min_R, velx_max_R);
        _local_bounds_periodic(vely_shift, ixp1, N_X, vely_min_R, vely_max_R);
        _local_bounds_periodic(velz_shift, ixp1, N_X, velz_min_R, velz_max_R);

        real scale_L = _invariant_scale(
            dens_shift[ix], momx_shift[ix], momy_shift[ix], momz_shift[ix],
            corr_dens_L, corr_momx_L, corr_momy_L, corr_momz_L,
            velx_min_L, velx_max_L, vely_min_L, vely_max_L, velz_min_L, velz_max_L
        );
        real scale_R = _invariant_scale(
            dens_shift[ixp1], momx_shift[ixp1], momy_shift[ixp1], momz_shift[ixp1],
            corr_dens_R, corr_momx_R, corr_momy_R, corr_momz_R,
            velx_min_R, velx_max_R, vely_min_R, vely_max_R, velz_min_R, velz_max_R
        );
        real scale = fmin(scale_L, scale_R);

        dens_shift[ix] += scale*corr_dens_L;
        momx_shift[ix] += scale*corr_momx_L;
        momy_shift[ix] += scale*corr_momy_L;
        momz_shift[ix] += scale*corr_momz_L;

        dens_shift[ixp1] += scale*corr_dens_R;
        momx_shift[ixp1] += scale*corr_momx_R;
        momy_shift[ixp1] += scale*corr_momy_R;
        momz_shift[ixp1] += scale*corr_momz_R;
    }

    // write the updated ring to global memory
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[ic] = dens_shift[ix];
        dev_dustmomx[ic] = momx_shift[ix];
        dev_dustmomy[ic] = momy_shift[ix];
        dev_dustmomz[ic] = momz_shift[ix];
    }
}
