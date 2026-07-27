#include <_transport.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// kernel: advection_ybl
// purpose: preserve the fiducial radial SSPRK and PPM update while assigning one radial column to one block
// workspace: reuse 11 explicit full-grid fields and retain serial invariant-domain correction order within each column
// =========================================================================================================================

__device__ __forceinline__
void _block_y_lowflux (
    int iy,
    const real *rhod, const real *lx, const real *vy, const real *lz,
    real &flux_rhod, real &flux_mx, real &flux_my, real &flux_mz)
{
    if (iy == N_Y - 1)
    {
        real speed_o = vy[iy];
        flux_rhod = (speed_o > 0.0) ? speed_o*fmax(rhod[iy], 0.0) : 0.0;
        flux_mx = flux_rhod*lx[iy];
        flux_my = flux_rhod*vy[iy];
        flux_mz = flux_rhod*lz[iy];
        return;
    }

    _pressureless_hll_flux(
        vy[iy], vy[iy + 1],
        rhod[iy],     lx[iy],     vy[iy],     lz[iy],
        rhod[iy + 1], lx[iy + 1], vy[iy + 1], lz[iy + 1],
        flux_rhod, flux_mx, flux_my, flux_mz
    );
}

__global__
void advection_ybl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_y, real *dev_adv_work, real dt)
{
    int idx_col = blockIdx.x;
    if (idx_col >= N_X*N_Z) return;

    int ix = idx_col % N_X;
    int iz = idx_col / N_X;
    int idx_base = ix + iz*N_X*N_Y;

    real *rhod = _block_field(dev_adv_work, BLOCK_RHOD, idx_col, N_Y);
    real *mx = _block_field(dev_adv_work, BLOCK_MX, idx_col, N_Y);
    real *my = _block_field(dev_adv_work, BLOCK_MY, idx_col, N_Y);
    real *mz = _block_field(dev_adv_work, BLOCK_MZ, idx_col, N_Y);
    real *lx = _block_field(dev_adv_work, BLOCK_LX, idx_col, N_Y);
    real *vy = _block_field(dev_adv_work, BLOCK_VY, idx_col, N_Y);
    real *lz = _block_field(dev_adv_work, BLOCK_LZ, idx_col, N_Y);
    real *anti_rhod = _block_field(dev_adv_work, BLOCK_ANTI_RHOD, idx_col, N_Y);
    real *anti_mx = _block_field(dev_adv_work, BLOCK_ANTI_MX, idx_col, N_Y);
    real *anti_my = _block_field(dev_adv_work, BLOCK_ANTI_MY, idx_col, N_Y);
    real *anti_mz = _block_field(dev_adv_work, BLOCK_ANTI_MZ, idx_col, N_Y);

    real z = _get_zcent(iz);

    for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
    {
        int idx_cell = idx_base + iy*N_X;
        rhod[iy] = dev_dustdens[idx_cell];
        mx[iy] = dev_dustmomx[idx_cell];
        my[iy] = dev_dustmomy[idx_cell];
        mz[iy] = dev_dustmomz[idx_cell];
    }
    __syncthreads();

    for (int stage = 0; stage < 3; stage++)
    {
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real y = _get_ycent(iy);
            real R = y*sin(z);
            _recover_dust_state(rhod[iy], R, mx[iy], my[iy], mz[iy], lx[iy], vy[iy], lz[iy]);
        }
        __syncthreads();

        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            if (iy == N_Y - 1)
            {
                anti_rhod[iy] = anti_mx[iy] = anti_my[iy] = anti_mz[iy] = 0.0;
                continue;
            }

            real rhod_L = fmax(_block_ppm_state(rhod, dev_ppm_weight_y, iy,     N_Y, true),  0.0);
            real rhod_R = fmax(_block_ppm_state(rhod, dev_ppm_weight_y, iy + 1, N_Y, false), 0.0);
            real lx_L = _block_ppm_state(lx, dev_ppm_weight_y, iy,     N_Y, true);
            real lx_R = _block_ppm_state(lx, dev_ppm_weight_y, iy + 1, N_Y, false);
            real vy_L = _block_ppm_state(vy, dev_ppm_weight_y, iy,     N_Y, true);
            real vy_R = _block_ppm_state(vy, dev_ppm_weight_y, iy + 1, N_Y, false);
            real lz_L = _block_ppm_state(lz, dev_ppm_weight_y, iy,     N_Y, true);
            real lz_R = _block_ppm_state(lz, dev_ppm_weight_y, iy + 1, N_Y, false);

            real flux_rhod, flux_mx, flux_my, flux_mz;
            _pressureless_hll_flux(
                vy_L, vy_R,
                rhod_L, lx_L, vy_L, lz_L,
                rhod_R, lx_R, vy_R, lz_R,
                flux_rhod, flux_mx, flux_my, flux_mz
            );

            real flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low;
            _block_y_lowflux(
                iy, rhod, lx, vy, lz,
                flux_rhod_low, flux_mx_low, flux_my_low, flux_mz_low
            );

            anti_rhod[iy] = flux_rhod - flux_rhod_low;
            anti_mx[iy] = flux_mx - flux_mx_low;
            anti_my[iy] = flux_my - flux_my_low;
            anti_mz[iy] = flux_mz - flux_mz_low;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            real speed_i = vy[0];
            real flux_rhod_i = (speed_i < 0.0) ? speed_i*fmax(rhod[0], 0.0) : 0.0;
            real flux_mx_i = flux_rhod_i*lx[0];
            real flux_my_i = flux_rhod_i*vy[0];
            real flux_mz_i = flux_rhod_i*lz[0];

            for (int iy = 0; iy < N_Y; iy++)
            {
                real flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o;
                _block_y_lowflux(
                    iy, rhod, lx, vy, lz,
                    flux_rhod_o, flux_mx_o, flux_my_o, flux_mz_o
                );

                real vol_y = _get_vol_y(iy);
                real area_i = _get_area_y(iy);
                real area_o = _get_area_y(iy + 1);

                rhod[iy] -= dt*(area_o*flux_rhod_o - area_i*flux_rhod_i) / vol_y;
                mx[iy] -= dt*(area_o*flux_mx_o - area_i*flux_mx_i) / vol_y;
                my[iy] -= dt*(area_o*flux_my_o - area_i*flux_my_i) / vol_y;
                mz[iy] -= dt*(area_o*flux_mz_o - area_i*flux_mz_i) / vol_y;
                if (rhod[iy] < 0.0) rhod[iy] = mx[iy] = my[iy] = mz[iy] = 0.0;

                flux_rhod_i = flux_rhod_o;
                flux_mx_i = flux_mx_o;
                flux_my_i = flux_my_o;
                flux_mz_i = flux_mz_o;
            }

            for (int iy = 0; iy < N_Y - 1; iy++)
            {
                real area_f = _get_area_y(iy + 1);
                real vol_L = _get_vol_y(iy);
                real vol_R = _get_vol_y(iy + 1);

                real corr_rhod_L = -dt*area_f*anti_rhod[iy] / vol_L;
                real corr_mx_L = -dt*area_f*anti_mx[iy] / vol_L;
                real corr_my_L = -dt*area_f*anti_my[iy] / vol_L;
                real corr_mz_L = -dt*area_f*anti_mz[iy] / vol_L;
                real corr_rhod_R = dt*area_f*anti_rhod[iy] / vol_R;
                real corr_mx_R = dt*area_f*anti_mx[iy] / vol_R;
                real corr_my_R = dt*area_f*anti_my[iy] / vol_R;
                real corr_mz_R = dt*area_f*anti_mz[iy] / vol_R;

                real lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L;
                real lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R;
                _local_bounds(lx, iy,     N_Y, lx_min_L, lx_max_L);
                _local_bounds(vy, iy,     N_Y, vy_min_L, vy_max_L);
                _local_bounds(lz, iy,     N_Y, lz_min_L, lz_max_L);
                _local_bounds(lx, iy + 1, N_Y, lx_min_R, lx_max_R);
                _local_bounds(vy, iy + 1, N_Y, vy_min_R, vy_max_R);
                _local_bounds(lz, iy + 1, N_Y, lz_min_R, lz_max_R);

                real scale_L = _invariant_scale(
                    rhod[iy], mx[iy], my[iy], mz[iy],
                    corr_rhod_L, corr_mx_L, corr_my_L, corr_mz_L,
                    lx_min_L, lx_max_L, vy_min_L, vy_max_L, lz_min_L, lz_max_L
                );
                real scale_R = _invariant_scale(
                    rhod[iy + 1], mx[iy + 1], my[iy + 1], mz[iy + 1],
                    corr_rhod_R, corr_mx_R, corr_my_R, corr_mz_R,
                    lx_min_R, lx_max_R, vy_min_R, vy_max_R, lz_min_R, lz_max_R
                );
                real scale = fmin(scale_L, scale_R);

                rhod[iy] += scale*corr_rhod_L;
                mx[iy] += scale*corr_mx_L;
                my[iy] += scale*corr_my_L;
                mz[iy] += scale*corr_mz_L;
                rhod[iy + 1] += scale*corr_rhod_R;
                mx[iy + 1] += scale*corr_mx_R;
                my[iy + 1] += scale*corr_my_R;
                mz[iy + 1] += scale*corr_mz_R;
            }

            if (stage == 1)
            {
                for (int iy = 0; iy < N_Y; iy++)
                {
                    int idx_cell = idx_base + iy*N_X;
                    rhod[iy] = 0.75*dev_dustdens[idx_cell] + 0.25*rhod[iy];
                    mx[iy] = 0.75*dev_dustmomx[idx_cell] + 0.25*mx[iy];
                    my[iy] = 0.75*dev_dustmomy[idx_cell] + 0.25*my[iy];
                    mz[iy] = 0.75*dev_dustmomz[idx_cell] + 0.25*mz[iy];
                }
            }
        }
        __syncthreads();
    }

    for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
    {
        int idx_cell = idx_base + iy*N_X;
        dev_dustdens[idx_cell] = (1.0/3.0)*dev_dustdens[idx_cell] + (2.0/3.0)*rhod[iy];
        dev_dustmomx[idx_cell] = (1.0/3.0)*dev_dustmomx[idx_cell] + (2.0/3.0)*mx[iy];
        dev_dustmomy[idx_cell] = (1.0/3.0)*dev_dustmomy[idx_cell] + (2.0/3.0)*my[iy];
        dev_dustmomz[idx_cell] = (1.0/3.0)*dev_dustmomz[idx_cell] + (2.0/3.0)*mz[iy];
    }
}
