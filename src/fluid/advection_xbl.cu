#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advection_xbl
// purpose: reproduce the thread-sweep FARGO and PPM update with one cooperative block per azimuthal ring
// workspace: reuse 11 explicit full-grid fields and retain serial invariant-domain correction order within each ring
// =========================================================================================================================

__device__ __forceinline__
real _block_ppm_xstate (const real *value, int ix, bool upwind_on_left, real cfl)
{
    int ixm2 = (ix - 2 + N_X) % N_X;
    int ixm1 = (ix - 1 + N_X) % N_X;
    int ixp1 = (ix + 1) % N_X;
    int ixp2 = (ix + 2) % N_X;

    real value_L = _ppm_face_uniform(value[ixm2], value[ixm1], value[ix],   value[ixp1]);
    real value_R = _ppm_face_uniform(value[ixm1], value[ix],   value[ixp1], value[ixp2]);
    real dvalue, coeff_curv;

    _ppm_limit(value[ix], value_L, value_R, dvalue, coeff_curv);
    return upwind_on_left ?
        _ppm_state_R(value_R, dvalue, coeff_curv, cfl) :
        _ppm_state_L(value_L, dvalue, coeff_curv, cfl);
}

__device__ __forceinline__
void _block_x_lowflux (
    int ix, real R, real lx_frame,
    const real *rhod, const real *lx, const real *vy, const real *lz,
    real &flux_rhod, real &flux_mx, real &flux_my, real &flux_mz)
{
    int ixp1 = (ix + 1) % N_X;
    real omega_L = (lx[ix]   - lx_frame) / (R*R);
    real omega_R = (lx[ixp1] - lx_frame) / (R*R);

    _pressureless_hll_flux(
        omega_L, omega_R,
        rhod[ix],   lx[ix],   vy[ix],   lz[ix],
        rhod[ixp1], lx[ixp1], vy[ixp1], lz[ixp1],
        flux_rhod, flux_mx, flux_my, flux_mz
    );
}

__global__
void advection_xbl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_adv_work, real dt)
{
    int idx_ring = blockIdx.x;
    if (idx_ring >= N_Y*N_Z) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;
    int idx_base = iy*N_X + iz*N_X*N_Y;

    real *rhod = _block_field(dev_adv_work, BLOCK_RHOD, idx_ring, N_X);
    real *mx = _block_field(dev_adv_work, BLOCK_MX, idx_ring, N_X);
    real *my = _block_field(dev_adv_work, BLOCK_MY, idx_ring, N_X);
    real *mz = _block_field(dev_adv_work, BLOCK_MZ, idx_ring, N_X);
    real *lx = _block_field(dev_adv_work, BLOCK_LX, idx_ring, N_X);
    real *vy = _block_field(dev_adv_work, BLOCK_VY, idx_ring, N_X);
    real *lz = _block_field(dev_adv_work, BLOCK_LZ, idx_ring, N_X);
    real *anti_rhod = _block_field(dev_adv_work, BLOCK_ANTI_RHOD, idx_ring, N_X);
    real *anti_mx = _block_field(dev_adv_work, BLOCK_ANTI_MX, idx_ring, N_X);
    real *anti_my = _block_field(dev_adv_work, BLOCK_ANTI_MY, idx_ring, N_X);
    real *anti_mz = _block_field(dev_adv_work, BLOCK_ANTI_MZ, idx_ring, N_X);

    real dx = _get_dx();
    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);

    __shared__ real lx_frame;
    __shared__ int shift_count;

    // choose one ring-mean integer FARGO shift before loading the shifted state
    if (threadIdx.x == 0)
    {
        real lx_avg = 0.0;
        for (int ix = 0; ix < N_X; ix++)
        {
            int idx_cell = idx_base + ix;
            real lx_cell, vy_cell, lz_cell;
            _recover_dust_state(
                dev_dustdens[idx_cell], R,
                dev_dustmomx[idx_cell], dev_dustmomy[idx_cell], dev_dustmomz[idx_cell],
                lx_cell, vy_cell, lz_cell
            );
            lx_avg += lx_cell;
        }
        lx_avg /= static_cast<real>(N_X);

        real shift_cells = lx_avg*dt / (R*R*dx);
        shift_count = __double2int_rn(shift_cells);
        lx_frame = R*R*static_cast<real>(shift_count)*dx / dt;
    }
    __syncthreads();

    // load the circularly shifted conserved state and synchronized primitives
    for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
    {
        int ix_old = ((ix - shift_count) % N_X + N_X) % N_X;
        int idx_cell_old = idx_base + ix_old;

        rhod[ix] = dev_dustdens[idx_cell_old];
        mx[ix] = dev_dustmomx[idx_cell_old];
        my[ix] = dev_dustmomy[idx_cell_old];
        mz[ix] = dev_dustmomz[idx_cell_old];
        _recover_dust_state(rhod[ix], R, mx[ix], my[ix], mz[ix], lx[ix], vy[ix], lz[ix]);
    }
    __syncthreads();

    // construct PPM and low-order HLL fluxes cooperatively at every periodic face
    for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
    {
        int ixp1 = (ix + 1) % N_X;
        real cfl_L = fabs((lx[ix]   - lx_frame) / (R*R))*dt / dx;
        real cfl_R = fabs((lx[ixp1] - lx_frame) / (R*R))*dt / dx;

        real rhod_L = fmax(_block_ppm_xstate(rhod, ix,   true,  cfl_L), 0.0);
        real rhod_R = fmax(_block_ppm_xstate(rhod, ixp1, false, cfl_R), 0.0);
        real lx_L = _block_ppm_xstate(lx, ix,   true,  cfl_L);
        real lx_R = _block_ppm_xstate(lx, ixp1, false, cfl_R);
        real vy_L = _block_ppm_xstate(vy, ix,   true,  cfl_L);
        real vy_R = _block_ppm_xstate(vy, ixp1, false, cfl_R);
        real lz_L = _block_ppm_xstate(lz, ix,   true,  cfl_L);
        real lz_R = _block_ppm_xstate(lz, ixp1, false, cfl_R);

        real flux_rhod, flux_mx, flux_my, flux_mz;
        _pressureless_hll_flux(
            (lx_L - lx_frame) / (R*R), (lx_R - lx_frame) / (R*R),
            rhod_L, lx_L, vy_L, lz_L,
            rhod_R, lx_R, vy_R, lz_R,
            flux_rhod, flux_mx, flux_my, flux_mz
        );

        real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;
        _block_x_lowflux(
            ix, R, lx_frame, rhod, lx, vy, lz,
            flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
        );

        anti_rhod[ix] = flux_rhod - flux_rhod_low;
        anti_mx[ix] = flux_mx - flux_mx_low;
        anti_my[ix] = flux_my - flux_my_low;
        anti_mz[ix] = flux_mz - flux_mz_low;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // apply the low-order update in deterministic face order
        real flux_rhod_wrap, flux_mx_wrap, flux_my_wrap, flux_mz_wrap;
        _block_x_lowflux(
            N_X - 1, R, lx_frame, rhod, lx, vy, lz,
            flux_rhod_wrap, flux_mx_wrap, flux_my_wrap, flux_mz_wrap
        );
        real flux_rhod_i = flux_rhod_wrap;
        real flux_mx_i = flux_mx_wrap;
        real flux_my_i = flux_my_wrap;
        real flux_mz_i = flux_mz_wrap;

        // advance the low-order conservative state through the periodic flux divergence
        for (int ix = 0; ix < N_X; ix++)
        {
            real flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o;
            if (ix < N_X - 1)
            {
                _block_x_lowflux(
                    ix, R, lx_frame, rhod, lx, vy, lz,
                    flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o
                );
            }
            else
            {
                flux_rhod_o = flux_rhod_wrap;
                flux_mx_o = flux_mx_wrap;
                flux_my_o = flux_my_wrap;
                flux_mz_o = flux_mz_wrap;
            }

            rhod[ix] -= dt*(flux_rhod_o - flux_rhod_i) / dx;
            mx[ix] -= dt*(flux_mx_o - flux_mx_i) / dx;
            my[ix] -= dt*(flux_my_o - flux_my_i) / dx;
            mz[ix] -= dt*(flux_mz_o - flux_mz_i) / dx;
            if (rhod[ix] < 0.0) rhod[ix] = mx[ix] = my[ix] = mz[ix] = 0.0;

            flux_rhod_i = flux_rhod_o;
            flux_mx_i = flux_mx_o;
            flux_my_i = flux_my_o;
            flux_mz_i = flux_mz_o;
        }

        // restore antidiffusive fluxes with one conservative invariant-domain scale per face
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixp1 = (ix + 1) % N_X;
            real corr_rhod_L = -dt*anti_rhod[ix] / dx;
            real corr_mx_L = -dt*anti_mx[ix] / dx;
            real corr_my_L = -dt*anti_my[ix] / dx;
            real corr_mz_L = -dt*anti_mz[ix] / dx;
            real corr_rhod_R = -corr_rhod_L;
            real corr_mx_R = -corr_mx_L;
            real corr_my_R = -corr_my_L;
            real corr_mz_R = -corr_mz_L;

            real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
            real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;
            _local_bounds_periodic(lx, ix,   N_X, lx_min_L, lx_max_L);
            _local_bounds_periodic(vy, ix,   N_X, vy_min_L, vy_max_L);
            _local_bounds_periodic(lz, ix,   N_X, lz_min_L, lz_max_L);
            _local_bounds_periodic(lx, ixp1, N_X, lx_min_R, lx_max_R);
            _local_bounds_periodic(vy, ixp1, N_X, vy_min_R, vy_max_R);
            _local_bounds_periodic(lz, ixp1, N_X, lz_min_R, lz_max_R);

            real scale_L = _invariant_scale(
                rhod[ix], mx[ix], my[ix], mz[ix],
                corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
                lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
            );
            real scale_R = _invariant_scale(
                rhod[ixp1], mx[ixp1], my[ixp1], mz[ixp1],
                corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
                lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
            );
            real scale = fmin(scale_L, scale_R);

            rhod[ix] += scale*corr_rhod_L;
            mx[ix] += scale*corr_mx_L;
            my[ix] += scale*corr_my_L;
            mz[ix] += scale*corr_mz_L;
            rhod[ixp1] += scale*corr_rhod_R;
            mx[ixp1] += scale*corr_mx_R;
            my[ixp1] += scale*corr_my_R;
            mz[ixp1] += scale*corr_mz_R;
        }
    }
    __syncthreads();

    // store the corrected conserved ring
    for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
    {
        int idx_cell = idx_base + ix;
        dev_dustdens[idx_cell] = rhod[ix];
        dev_dustmomx[idx_cell] = mx[ix];
        dev_dustmomy[idx_cell] = my[ix];
        dev_dustmomz[idx_cell] = mz[ix];
    }
}
