#ifdef DIFFUSION

#include <param_grid.cuh>
#include <param_phys.cuh>
#include <fluid_kern.cuh>

// =====================================================================================================================
// kernel: diffusion_ybl
// purpose: solve geometry-aware radial Crank-Nicolson diffusion cooperatively in block-shared memory
// =====================================================================================================================

__device__ __forceinline__
real _block_dr_cent_i (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0) / _get_dy(); }

__device__ __forceinline__
real _block_dr_cent_o (int iy) { return _get_ycent(iy)*(_get_dy() - 1.0); }

__global__
void diffusion_ybl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_col = blockIdx.x;
    if (idx_col >= N_X*N_Z || dt <= 0.0) return;

    int ix = idx_col % N_X;
    int iz = idx_col / N_X;
    int idx_base = ix + iz*N_X*N_Y;
    real z = _get_zcent(iz);

    // partition dynamic shared memory among density, CN coefficients, and reusable solve work arrays
    extern __shared__ real shared_work[];
    real *rhod = shared_work;
    real *cn_lower = rhod + N_Y;
    real *cn_diag = cn_lower + N_Y;
    real *cn_upper = cn_diag + N_Y;
    real *rhod_work = cn_upper + N_Y;
    real *temp_work = rhod_work + N_Y;

    __shared__ int sub_count;
    __shared__ real dt_sub;

    // load one density column and assemble full-step zero-flux CN coefficients
    for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
    {
        int idx_cell = idx_base + iy*N_X;
        rhod[iy] = dev_dustdens[idx_cell];

        real y_i = _get_yface(iy);
        real y_o = _get_yface(iy + 1);
        real vol_y = _get_vol_y(iy);
        real R_i = y_i*sin(z);
        real R_o = y_o*sin(z);
        real diff_yi = _get_diffusivity(R_i, y_i*cos(z), _get_hg(R_i), SCHMIDT_Y);
        real diff_yo = _get_diffusivity(R_o, y_o*cos(z), _get_hg(R_o), SCHMIDT_Y);
        real cn_i = (iy > 0) ?
            0.5*dt*_get_area_y(iy)*diff_yi / (_block_dr_cent_i(iy)*vol_y) : 0.0;
        real cn_o = (iy < N_Y - 1) ?
            0.5*dt*_get_area_y(iy + 1)*diff_yo / (_block_dr_cent_o(iy)*vol_y) : 0.0;
        // weight each face conductance by the gas weight so the solve advances rho_d/w
        // the resulting diagonal is also the outgoing coefficient sum used by positivity subcycling
        real gas = _get_diffusion_weight(_get_ycent(iy), z);
        cn_i *= _get_diffusion_weight(y_i, z) / gas;
        cn_o *= _get_diffusion_weight(y_o, z) / gas;
        cn_lower[iy] = -cn_i;
        cn_upper[iy] = -cn_o;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        real max_cn_sum = 0.0;
        for (int iy = 0; iy < N_Y; iy++)
        {
            max_cn_sum = fmax(max_cn_sum, -cn_lower[iy] - cn_upper[iy]);
        }
        sub_count = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
        if (sub_count < 1) sub_count = 1;
        dt_sub = dt / static_cast<real>(sub_count);
    }
    __syncthreads();

    real inv_sub_count = 1.0 / static_cast<real>(sub_count);
    for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
    {
        cn_lower[iy] *= inv_sub_count;
        cn_upper[iy] *= inv_sub_count;
        cn_diag[iy] = 1.0 - cn_lower[iy] - cn_upper[iy];
    }
    __syncthreads();

    // advance density and donor momentum through every positivity-controlled CN substep
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real cn_i = -cn_lower[iy];
            real cn_o = -cn_upper[iy];
            real rhod_prev = (iy > 0) ? rhod[iy - 1] / _get_diffusion_weight(_get_ycent(iy - 1), z) : rhod[iy]
                / _get_diffusion_weight(_get_ycent(iy), z);
            real rhod_next = (iy < N_Y - 1) ? rhod[iy + 1] / _get_diffusion_weight(_get_ycent(iy + 1), z) : rhod[iy]
                / _get_diffusion_weight(_get_ycent(iy), z);
            rhod_work[iy] = cn_i*rhod_prev + (1.0 - cn_i - cn_o)*rhod[iy] / _get_diffusion_weight(_get_ycent(iy), z)
                + cn_o*rhod_next;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            // solve density (w = 1) or concentration with serial Thomas elimination
            temp_work[0] = cn_upper[0] / cn_diag[0];
            rhod_work[0] /= cn_diag[0];
            for (int iy = 1; iy < N_Y; iy++)
            {
                real pivot = cn_diag[iy] - cn_lower[iy]*temp_work[iy - 1];
                temp_work[iy] = (iy < N_Y - 1) ? cn_upper[iy] / pivot : 0.0;
                rhod_work[iy] = (rhod_work[iy] - cn_lower[iy]*rhod_work[iy - 1]) / pivot;
            }
            for (int iy = N_Y - 2; iy >= 0; iy--)
            {
                rhod_work[iy] -= temp_work[iy]*rhod_work[iy + 1];
            }
        }
        __syncthreads();

        // reconstruct the time-centered mass flux from the old and solved diffused variables, with zero boundary flux
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            if (iy == N_Y - 1)
            {
                temp_work[iy] = 0.0;
            }
            else
            {
                real cn_o = -cn_upper[iy];
                temp_work[iy] = -(_get_diffusion_weight(_get_ycent(iy), z)*cn_o*_get_vol_y(iy) / dt_sub)*
                    ((rhod[iy + 1] / _get_diffusion_weight(_get_ycent(iy + 1), z) - rhod[iy]
                    / _get_diffusion_weight(_get_ycent(iy), z)) + (rhod_work[iy + 1] - rhod_work[iy]));
            }
        }
        __syncthreads();

        // compute donor factors from all raw radial face transfers before scaling any face
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real flux_i = (iy > 0) ? temp_work[iy - 1] : 0.0;
            real out_rate = fmax(temp_work[iy], 0.0) + fmax(-flux_i, 0.0);
            real mass = fmax(rhod[iy], 0.0)*_get_vol_y(iy);
            rhod_work[iy] = (out_rate > 0.0)
                ? fmin(1.0, POS_LIMIT*mass / (dt_sub*out_rate)) : 1.0;
        }
        __syncthreads();

        for (int iy = threadIdx.x; iy < N_Y - 1; iy += blockDim.x)
        {
            int iy_up = (temp_work[iy] >= 0.0) ? iy : iy + 1;
            temp_work[iy] *= rhod_work[iy_up];
        }
        __syncthreads();

        // accept density from the limited conservative divergence
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real flux_i = (iy > 0) ? temp_work[iy - 1] : 0.0;
            rhod_work[iy] = rhod[iy]
                - dt_sub*(temp_work[iy] - flux_i) / _get_vol_y(iy);
        }
        __syncthreads();

        // transport every momentum component with the same mass flux and its donor primitive
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            int iy_up = (temp_work[iy] >= 0.0 || iy == N_Y - 1) ? iy : iy + 1;
            real y_up = _get_ycent(iy_up);
            real R_up = y_up*sin(z);
            real rhod_up = rhod[iy_up];
            int idx_cell_up = idx_base + iy_up*N_X;
            real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / rhod_up : sqrt(G*M_S*fmax(R_up, 0.0));
            cn_diag[iy] = temp_work[iy]*lx_up;
        }
        __syncthreads();
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real flux_i = (iy > 0) ? cn_diag[iy - 1] : 0.0;
            dev_dustmomx[idx_base + iy*N_X] -= dt_sub*(cn_diag[iy] - flux_i) / _get_vol_y(iy);
        }
        __syncthreads();

        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            int iy_up = (temp_work[iy] >= 0.0 || iy == N_Y - 1) ? iy : iy + 1;
            real rhod_up = rhod[iy_up];
            int idx_cell_up = idx_base + iy_up*N_X;
            real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / rhod_up : 0.0;
            cn_diag[iy] = temp_work[iy]*vy_up;
        }
        __syncthreads();
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real flux_i = (iy > 0) ? cn_diag[iy - 1] : 0.0;
            dev_dustmomy[idx_base + iy*N_X] -= dt_sub*(cn_diag[iy] - flux_i) / _get_vol_y(iy);
        }
        __syncthreads();

        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            int iy_up = (temp_work[iy] >= 0.0 || iy == N_Y - 1) ? iy : iy + 1;
            real rhod_up = rhod[iy_up];
            int idx_cell_up = idx_base + iy_up*N_X;
            real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / rhod_up : 0.0;
            cn_diag[iy] = temp_work[iy]*lz_up;
        }
        __syncthreads();
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            real flux_i = (iy > 0) ? cn_diag[iy - 1] : 0.0;
            dev_dustmomz[idx_base + iy*N_X] -= dt_sub*(cn_diag[iy] - flux_i) / _get_vol_y(iy);
            rhod[iy] = rhod_work[iy];
        }
        __syncthreads();

        // restore the CN diagonal after all momentum components consume the reused workspace
        for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
        {
            cn_diag[iy] = 1.0 - cn_lower[iy] - cn_upper[iy];
        }
        __syncthreads();
    }

    // store density after all substeps
    for (int iy = threadIdx.x; iy < N_Y; iy += blockDim.x)
    {
        dev_dustdens[idx_base + iy*N_X] = rhod[iy];
    }
}

#endif // DIFFUSION
