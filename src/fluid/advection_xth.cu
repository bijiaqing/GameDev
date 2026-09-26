#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =====================================================================================================================
// kernel: advection_xth
// purpose: periodic azimuthal transport with FARGO, PPM, pressureless HLL fluxes, and invariant-domain limiting
//
// parallelization: one thread per radial-polar ring with a serial loop over N_X azimuthal cells
//
// per call:
//   1 FARGO integer shift and residual-frame construction
//   2 PPM high-order and cell-centred low-order HLL flux construction
//   3 low-order conservative update
//   4 invariant-domain-limited antidiffusive correction
// =====================================================================================================================

__global__
void advection_xth (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt)
{
    int idx_ring = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;

    real dx = _get_dx();

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);

    // load one ring and recover primitive quantities with a Keplerian azimuthal fallback in near-vacuum cells
    real work_rhod[N_X], work_x[N_X], work_y[N_X], work_z[N_X], lx[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        work_rhod[ix] = dev_dustdens[idx_cell];
        work_x[ix] = dev_dustmomx[idx_cell];
        work_y[ix] = dev_dustmomy[idx_cell];
        work_z[ix] = dev_dustmomz[idx_cell];

        real vy_tmp, lz_tmp;
        _recover_dust_state(work_rhod[ix], R, work_x[ix], work_y[ix], work_z[ix], lx[ix], vy_tmp, lz_tmp);
    }

    // average the specific angular momentum used to choose the FARGO integer shift
    real lx_avg = 0.0;
    for (int ix = 0; ix < N_X; ix++)
    {
        lx_avg += lx[ix];
    }
    lx_avg /= static_cast<real>(N_X);

    // represent the ring-mean displacement by the nearest integer shift and leave its fractional remainder for PPM
    real shift_cells = lx_avg*dt / (R*R*dx);
    int shift_count = __double2int_rn(shift_cells);
    real omega_shift = static_cast<real>(shift_count)*dx / dt;
    real lx_frame = R*R*omega_shift;

    // circularly shift each conserved state from cell ix minus shift_count
    real rhod_shift[N_X], mx_shift[N_X], my_shift[N_X], mz_shift[N_X], lx_shift[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ix_old = ((ix - shift_count) % N_X + N_X) % N_X;

        rhod_shift[ix] = work_rhod[ix_old];
        mx_shift[ix] = work_x[ix_old];
        my_shift[ix] = work_y[ix_old];
        mz_shift[ix] = work_z[ix_old];
        lx_shift[ix] = lx[ix_old];
    }

    // recover primitives from the shifted conserved state for consistent interface reconstruction
    real vy_shift[N_X], lz_shift[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        _recover_dust_state(rhod_shift[ix], R,
            mx_shift[ix], my_shift[ix], mz_shift[ix],
            lx_shift[ix], vy_shift[ix], lz_shift[ix]
        );
    }

    // subtract the integer-shift frame from specific angular momentum to obtain the PPM transport residual
    real lx_res[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        lx_res[ix] = lx_shift[ix] - lx_frame;
    }

    // reconstruct periodic PPM face values with face ix between cells ix minus one and ix
    real face_rhod[N_X], face_lx[N_X], face_vy[N_X], face_lz[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm2 = (ix - 2 + N_X) % N_X;
        int ixm1 = (ix - 1 + N_X) % N_X;
        int ixp1 = (ix + 1) % N_X;

        face_rhod[ix] = _ppm_face_uniform(rhod_shift[ixm2], rhod_shift[ixm1], rhod_shift[ix], rhod_shift[ixp1]);
        face_lx[ix] = _ppm_face_uniform(lx_shift[ixm2], lx_shift[ixm1], lx_shift[ix], lx_shift[ixp1]);
        face_vy[ix] = _ppm_face_uniform(vy_shift[ixm2], vy_shift[ixm1], vy_shift[ix], vy_shift[ixp1]);
        face_lz[ix] = _ppm_face_uniform(lz_shift[ixm2], lz_shift[ixm1], lz_shift[ix], lz_shift[ixp1]);
    }

    // compute high-order fluxes from PPM-traced interface states with the pressureless HLL solver
    // use the residual angular displacement over one cell as each one-sided PPM tracing fraction
    real flux_rhod[N_X], flux_mx[N_X], flux_my[N_X], flux_mz[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        int ixp2 = (ix + 2) % N_X;

        real cfl_L = fabs(lx_res[ix]  /(R*R))*dt / dx;
        real cfl_R = fabs(lx_res[ixp1] / (R*R))*dt / dx;

        // clamp reconstructed density nonnegative while preserving the signs of the reconstructed primitive quantities
        real rhod_L = fmax(_thread_ppm_state(face_rhod, rhod_shift, ix,   ixp1, true,  cfl_L), 0.0);
        real rhod_R = fmax(_thread_ppm_state(face_rhod, rhod_shift, ixp1, ixp2, false, cfl_R), 0.0);
        real lx_L =      _thread_ppm_state(face_lx, lx_shift, ix,   ixp1, true,  cfl_L);
        real lx_R =      _thread_ppm_state(face_lx, lx_shift, ixp1, ixp2, false, cfl_R);
        real vy_L =      _thread_ppm_state(face_vy, vy_shift, ix,   ixp1, true,  cfl_L);
        real vy_R =      _thread_ppm_state(face_vy, vy_shift, ixp1, ixp2, false, cfl_R);
        real lz_L =      _thread_ppm_state(face_lz, lz_shift, ix,   ixp1, true,  cfl_L);
        real lz_R =      _thread_ppm_state(face_lz, lz_shift, ixp1, ixp2, false, cfl_R);

        // convert reconstructed specific angular momentum to residual angular transport speed
        real omega_L = (lx_L - lx_frame) / (R*R);
        real omega_R = (lx_R - lx_frame) / (R*R);

        _pressureless_hll_flux(
            omega_L, omega_R,
            rhod_L, lx_L, vy_L, lz_L,
            rhod_R, lx_R, vy_R, lz_R,
            flux_rhod[ix], flux_mx[ix], flux_my[ix], flux_mz[ix]
        );
    }

    // compute low-order HLL fluxes and the antidiffusive difference between high- and low-order fluxes
    // reuse density and momentum arrays for flux differences and flux arrays for low-order fluxes
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;
        real omega_L = lx_res[ix] / (R*R);
        real omega_R = lx_res[ixp1] / (R*R);
        real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;

        _pressureless_hll_flux(
            omega_L, omega_R,
            rhod_shift[ix], lx_shift[ix], vy_shift[ix], lz_shift[ix],
            rhod_shift[ixp1], lx_shift[ixp1], vy_shift[ixp1], lz_shift[ixp1],
            flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
        );

        work_rhod[ix] = flux_rhod[ix] - flux_rhod_low;
        work_x[ix] = flux_mx[ix] - flux_mx_low;
        work_y[ix] = flux_my[ix] - flux_my_low;
        work_z[ix] = flux_mz[ix] - flux_mz_low;

        flux_rhod[ix] = flux_rhod_low;
        flux_mx[ix] = flux_mx_low;
        flux_my[ix] = flux_my_low;
        flux_mz[ix] = flux_mz_low;
    }

    // apply the low-order conservative update with a zero-state backstop for any negative density result
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixm1 = (ix - 1 + N_X) % N_X;

        rhod_shift[ix] -= dt*(flux_rhod[ix] - flux_rhod[ixm1]) / dx;
        mx_shift[ix] -= dt*(flux_mx[ix] - flux_mx[ixm1]) / dx;
        my_shift[ix] -= dt*(flux_my[ix] - flux_my[ixm1]) / dx;
        mz_shift[ix] -= dt*(flux_mz[ix] - flux_mz[ixm1]) / dx;

        if (rhod_shift[ix] < 0.0) rhod_shift[ix] = mx_shift[ix] = my_shift[ix] = mz_shift[ix] = 0.0;
    }

    // express each antidiffusive face flux as equal-and-opposite corrections to its adjacent cells
    // limit both corrections by one shared scale to preserve conservation and the local invariant domain
    for (int ix = 0; ix < N_X; ix++)
    {
        int ixp1 = (ix + 1) % N_X;

        real corr_rhod_L = -dt*work_rhod[ix] / dx;
        real corr_mx_L = -dt*work_x[ix] / dx;
        real corr_my_L = -dt*work_y[ix] / dx;
        real corr_mz_L = -dt*work_z[ix] / dx;

        real corr_rhod_R =  dt*work_rhod[ix] / dx;
        real corr_mx_R =  dt*work_x[ix] / dx;
        real corr_my_R =  dt*work_y[ix] / dx;
        real corr_mz_R =  dt*work_z[ix] / dx;

        real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
        real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;

        // bound each transported primitive quantity by neighboring shifted-cell values
        _local_bounds_periodic(lx_shift, ix,   N_X, lx_min_L, lx_max_L);
        _local_bounds_periodic(vy_shift, ix,   N_X, vy_min_L, vy_max_L);
        _local_bounds_periodic(lz_shift, ix,   N_X, lz_min_L, lz_max_L);
        _local_bounds_periodic(lx_shift, ixp1, N_X, lx_min_R, lx_max_R);
        _local_bounds_periodic(vy_shift, ixp1, N_X, vy_min_R, vy_max_R);
        _local_bounds_periodic(lz_shift, ixp1, N_X, lz_min_R, lz_max_R);

        real scale_L = _invariant_scale(
            rhod_shift[ix], mx_shift[ix], my_shift[ix], mz_shift[ix],
            corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
            lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
        );
        real scale_R = _invariant_scale(
            rhod_shift[ixp1], mx_shift[ixp1], my_shift[ixp1], mz_shift[ixp1],
            corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
            lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
        );
        real scale = fmin(scale_L, scale_R);

        rhod_shift[ix] += scale*corr_rhod_L;
        mx_shift[ix] += scale*corr_mx_L;
        my_shift[ix] += scale*corr_my_L;
        mz_shift[ix] += scale*corr_mz_L;

        rhod_shift[ixp1] += scale*corr_rhod_R;
        mx_shift[ixp1] += scale*corr_mx_R;
        my_shift[ixp1] += scale*corr_my_R;
        mz_shift[ixp1] += scale*corr_mz_R;
    }

    // write the updated ring to global memory
    for (int ix = 0; ix < N_X; ix++)
    {
        int idx_cell = ix + iy*N_X + iz*N_X*N_Y;

        dev_dustdens[idx_cell] = rhod_shift[ix];
        dev_dustmomx[idx_cell] = mx_shift[ix];
        dev_dustmomy[idx_cell] = my_shift[ix];
        dev_dustmomz[idx_cell] = mz_shift[ix];
    }
}
