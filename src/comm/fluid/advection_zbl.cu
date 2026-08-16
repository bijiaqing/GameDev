#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advection_zbl
// purpose: reproduce the thread-sweep polar SSPRK and PPM update with one cooperative block per column
// workspace: reuse 12 explicit full-grid fields and retain serial invariant-domain correction order within each column
// =========================================================================================================================

__device__ __forceinline__
void _block_z_lowflux (
    int iz, real y,
    const real *rhod, const real *lx, const real *vy, const real *lz,
    real &flux_rhod, real &flux_mx, real &flux_my, real &flux_mz)
{
    if (iz == N_Z - 1)
    {
        #ifdef HALF_DISK
        flux_rhod = flux_mx = flux_my = flux_mz = 0.0;
        #else
        real speed_o = lz[iz] / y;
        flux_rhod = (speed_o > 0.0) ? speed_o*fmax(rhod[iz], 0.0) : 0.0;
        flux_mx = flux_rhod*lx[iz];
        flux_my = flux_rhod*vy[iz];
        flux_mz = flux_rhod*lz[iz];
        #endif
        return;
    }

    _pressureless_hll_flux(
        lz[iz] / y, lz[iz + 1] / y,
        rhod[iz],     lx[iz],     vy[iz],     lz[iz],
        rhod[iz + 1], lx[iz + 1], vy[iz + 1], lz[iz + 1],
        flux_rhod, flux_mx, flux_my, flux_mz
    );
}

__global__
void advection_zbl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_z, real *dev_adv_work, real dt)
{
    int idx_col = blockIdx.x;
    if (idx_col >= N_X*N_Y || N_Z == 1) return;

    int ix = idx_col % N_X;
    int iy = idx_col / N_X;
    int idx_base = ix + iy*N_X;

    real *rhod = _block_field(dev_adv_work, BLOCK_RHOD, idx_col, N_Z);
    real *mx = _block_field(dev_adv_work, BLOCK_MX, idx_col, N_Z);
    real *my = _block_field(dev_adv_work, BLOCK_MY, idx_col, N_Z);
    real *mz = _block_field(dev_adv_work, BLOCK_MZ, idx_col, N_Z);
    real *lx = _block_field(dev_adv_work, BLOCK_LX, idx_col, N_Z);
    real *vy = _block_field(dev_adv_work, BLOCK_VY, idx_col, N_Z);
    real *lz = _block_field(dev_adv_work, BLOCK_LZ, idx_col, N_Z);
    real *anti_rhod = _block_field(dev_adv_work, BLOCK_ANTI_RHOD, idx_col, N_Z);
    real *anti_mx = _block_field(dev_adv_work, BLOCK_ANTI_MX, idx_col, N_Z);
    real *anti_my = _block_field(dev_adv_work, BLOCK_ANTI_MY, idx_col, N_Z);
    real *anti_mz = _block_field(dev_adv_work, BLOCK_ANTI_MZ, idx_col, N_Z);
    real *rhod_low = _block_field(dev_adv_work, BLOCK_RHOD_LOW, idx_col, N_Z);

    real y = _get_ycent(iy);
    real geom_z = _get_area_z(iy) / _get_vol_y(iy);

    // load one conserved polar column into the explicit workspace
    for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
    {
        int idx_cell = idx_base + iz*N_X*N_Y;
        rhod[iz] = dev_dustdens[idx_cell];
        mx[iz] = dev_dustmomx[idx_cell];
        my[iz] = dev_dustmomy[idx_cell];
        mz[iz] = dev_dustmomz[idx_cell];
    }
    __syncthreads();

    // advance three forward-Euler evaluations and SSPRK(3,3) convex combinations
    for (int stage = 0; stage < 3; stage++)
    {
        // recover primitives from the current SSPRK stage state
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real z = _get_zcent(iz);
            real R = y*sin(z);
            _recover_dust_state(rhod[iz], R, mx[iz], my[iz], mz[iz], lx[iz], vy[iz], lz[iz]);
        }
        __syncthreads();

        // construct PPM and low-order HLL fluxes cooperatively at polar faces
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            if (iz == N_Z - 1)
            {
                anti_rhod[iz] = anti_mx[iz] = anti_my[iz] = anti_mz[iz] = 0.0;
                continue;
            }

            real rhod_L = fmax(_block_ppm_state(rhod, dev_ppm_weight_z, iz,     N_Z, true),  0.0);
            real rhod_R = fmax(_block_ppm_state(rhod, dev_ppm_weight_z, iz + 1, N_Z, false), 0.0);
            real lx_L = _block_ppm_state(lx, dev_ppm_weight_z, iz,     N_Z, true);
            real lx_R = _block_ppm_state(lx, dev_ppm_weight_z, iz + 1, N_Z, false);
            real vy_L = _block_ppm_state(vy, dev_ppm_weight_z, iz,     N_Z, true);
            real vy_R = _block_ppm_state(vy, dev_ppm_weight_z, iz + 1, N_Z, false);
            real lz_L = _block_ppm_state(lz, dev_ppm_weight_z, iz,     N_Z, true);
            real lz_R = _block_ppm_state(lz, dev_ppm_weight_z, iz + 1, N_Z, false);

            real flux_rhod, flux_mx, flux_my, flux_mz;
            _pressureless_hll_flux(
                lz_L / y, lz_R / y,
                rhod_L, lx_L, vy_L, lz_L,
                rhod_R, lx_R, vy_R, lz_R,
                flux_rhod, flux_mx, flux_my, flux_mz
            );

            real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;
            _block_z_lowflux(
                iz, y, rhod, lx, vy, lz,
                flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
            );
            anti_rhod[iz] = flux_rhod - flux_rhod_low;
            anti_mx[iz] = flux_mx - flux_mx_low;
            anti_my[iz] = flux_my - flux_my_low;
            anti_mz[iz] = flux_mz - flux_mz_low;
        }
        __syncthreads();

        // apply the geometry-aware low-order update independently in every polar cell
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_rhod_i, flux_mx_i, flux_my_i, flux_mz_i;
            if (iz == 0)
            {
                real speed_i = lz[0] / y;
                flux_rhod_i = (speed_i < 0.0) ? speed_i*fmax(rhod[0], 0.0) : 0.0;
                flux_mx_i = flux_rhod_i*lx[0];
                flux_my_i = flux_rhod_i*vy[0];
                flux_mz_i = flux_rhod_i*lz[0];
            }
            else
            {
                _block_z_lowflux(
                    iz - 1, y, rhod, lx, vy, lz,
                    flux_rhod_i, flux_mx_i, flux_my_i, flux_mz_i
                );
            }

            real flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o;
            _block_z_lowflux(
                iz, y, rhod, lx, vy, lz,
                flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o
            );

            real z_i = _get_zface(iz);
            real z_o = _get_zface(iz + 1);
            real vol_z = _get_vol_z(iz);
            rhod_low[iz] = rhod[iz] - dt*geom_z*(sin(z_o)*flux_rhod_o - sin(z_i)*flux_rhod_i) / vol_z;
            mx[iz] -= dt*geom_z*(sin(z_o)*flux_mx_o - sin(z_i)*flux_mx_i) / vol_z;
            my[iz] -= dt*geom_z*(sin(z_o)*flux_my_o - sin(z_i)*flux_my_i) / vol_z;
            mz[iz] -= dt*geom_z*(sin(z_o)*flux_mz_o - sin(z_i)*flux_mz_i) / vol_z;
            if (rhod_low[iz] < 0.0) rhod_low[iz] = mx[iz] = my[iz] = mz[iz] = 0.0;
        }
        __syncthreads();

        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            rhod[iz] = rhod_low[iz];
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            // restore antidiffusive transfers with one invariant-domain scale per interior face
            for (int iz = 0; iz < N_Z - 1; iz++)
            {
                real area_f = sin(_get_zface(iz + 1));
                real vol_L = _get_vol_z(iz);
                real vol_R = _get_vol_z(iz + 1);
                real corr_rhod_L = -dt*geom_z*area_f*anti_rhod[iz] / vol_L;
                real corr_mx_L = -dt*geom_z*area_f*anti_mx[iz] / vol_L;
                real corr_my_L = -dt*geom_z*area_f*anti_my[iz] / vol_L;
                real corr_mz_L = -dt*geom_z*area_f*anti_mz[iz] / vol_L;
                real corr_rhod_R = dt*geom_z*area_f*anti_rhod[iz] / vol_R;
                real corr_mx_R = dt*geom_z*area_f*anti_mx[iz] / vol_R;
                real corr_my_R = dt*geom_z*area_f*anti_my[iz] / vol_R;
                real corr_mz_R = dt*geom_z*area_f*anti_mz[iz] / vol_R;

                real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
                real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;
                _local_bounds(lx, iz,     N_Z, lx_min_L, lx_max_L);
                _local_bounds(vy, iz,     N_Z, vy_min_L, vy_max_L);
                _local_bounds(lz, iz,     N_Z, lz_min_L, lz_max_L);
                _local_bounds(lx, iz + 1, N_Z, lx_min_R, lx_max_R);
                _local_bounds(vy, iz + 1, N_Z, vy_min_R, vy_max_R);
                _local_bounds(lz, iz + 1, N_Z, lz_min_R, lz_max_R);

                real scale_L = _invariant_scale(
                    rhod[iz], mx[iz], my[iz], mz[iz],
                    corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
                    lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
                );
                real scale_R = _invariant_scale(
                    rhod[iz + 1], mx[iz + 1], my[iz + 1], mz[iz + 1],
                    corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
                    lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
                );
                real scale = fmin(scale_L, scale_R);

                rhod[iz] += scale*corr_rhod_L;
                mx[iz] += scale*corr_mx_L;
                my[iz] += scale*corr_my_L;
                mz[iz] += scale*corr_mz_L;
                rhod[iz + 1] += scale*corr_rhod_R;
                mx[iz + 1] += scale*corr_mx_R;
                my[iz + 1] += scale*corr_my_R;
                mz[iz + 1] += scale*corr_mz_R;
            }
        }
        __syncthreads();

        if (stage == 1)
        {
            // form the second SSPRK stage from the original and twice-Euler-updated states
            for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
            {
                int idx_cell = idx_base + iz*N_X*N_Y;
                rhod[iz] = 0.75*dev_dustdens[idx_cell] + 0.25*rhod[iz];
                mx[iz] = 0.75*dev_dustmomx[idx_cell] + 0.25*mx[iz];
                my[iz] = 0.75*dev_dustmomy[idx_cell] + 0.25*my[iz];
                mz[iz] = 0.75*dev_dustmomz[idx_cell] + 0.25*mz[iz];
            }
        }
        __syncthreads();
    }

    // form the final SSPRK combination while storing the conserved column
    for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
    {
        int idx_cell = idx_base + iz*N_X*N_Y;
        dev_dustdens[idx_cell] = (1.0/3.0)*dev_dustdens[idx_cell] + (2.0/3.0)*rhod[iz];
        dev_dustmomx[idx_cell] = (1.0/3.0)*dev_dustmomx[idx_cell] + (2.0/3.0)*mx[iz];
        dev_dustmomy[idx_cell] = (1.0/3.0)*dev_dustmomy[idx_cell] + (2.0/3.0)*my[iz];
        dev_dustmomz[idx_cell] = (1.0/3.0)*dev_dustmomz[idx_cell] + (2.0/3.0)*mz[iz];
    }
}
