#ifdef DIFFUSION

#include <param_grid.cuh>
#include <param_phys.cuh>
#include <fluid_kern.cuh>

// =========================================================================================================================
// kernel: diffusion_zbl
// purpose: solve spherical polar Crank-Nicolson diffusion cooperatively in block-shared memory
// =========================================================================================================================

__global__
void diffusion_zbl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_col = blockIdx.x;
    if (idx_col >= N_X*N_Y || N_Z == 1 || dt <= 0.0) return;

    int ix = idx_col % N_X;
    int iy = idx_col / N_X;
    int idx_base = ix + iy*N_X;
    real y = _get_ycent(iy);
    real dz = _get_dz();

    // partition dynamic shared memory among density, CN coefficients, and reusable solve work arrays
    extern __shared__ real shared_work[];
    real *rhod = shared_work;
    real *cn_lower = rhod + N_Z;
    real *cn_diag = cn_lower + N_Z;
    real *cn_upper = cn_diag + N_Z;
    real *rhod_work = cn_upper + N_Z;
    real *temp_work = rhod_work + N_Z;

    __shared__ int sub_count;
    __shared__ real dt_sub;

    // load one density column and assemble full-step zero-flux polar CN coefficients
    for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
    {
        int idx_cell = idx_base + iz*N_X*N_Y;
        rhod[iz] = dev_dustdens[idx_cell];

        real z_i = _get_zface(iz);
        real z_o = _get_zface(iz + 1);
        real vol_z = _get_vol_z(iz);
        real dz_len = y*dz;

        real diff_zi = 0.0;
        if (iz > 0)
        {
            real R_i = y*sin(z_i);
            diff_zi = _get_diffusivity(R_i, y*cos(z_i), _get_hg(R_i), SCHMIDT_Z);
        }
        real diff_zo = 0.0;
        if (iz < N_Z - 1)
        {
            real R_o = y*sin(z_o);
            diff_zo = _get_diffusivity(R_o, y*cos(z_o), _get_hg(R_o), SCHMIDT_Z);
        }

        real cn_i = (iz > 0) ? 0.5*dt*sin(z_i)*diff_zi / (y*dz_len*vol_z) : 0.0;
        real cn_o = (iz < N_Z - 1) ? 0.5*dt*sin(z_o)*diff_zo / (y*dz_len*vol_z) : 0.0;
        // Solve concentration with gas-weighted face conductances; the diagonal
        // is also the density outgoing sum used by positivity subcycling.
        real gas = _get_diffusion_weight(y, _get_zcent(iz));
        cn_i *= _get_diffusion_weight(y, z_i) / gas;
        cn_o *= _get_diffusion_weight(y, z_o) / gas;
        cn_lower[iz] = -cn_i;
        cn_upper[iz] = -cn_o;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        real max_cn_sum = 0.0;
        for (int iz = 0; iz < N_Z; iz++)
        {
            max_cn_sum = fmax(max_cn_sum, -cn_lower[iz] - cn_upper[iz]);
        }
        sub_count = static_cast<int>(ceil(max_cn_sum / POS_LIMIT));
        if (sub_count < 1) sub_count = 1;
        dt_sub = dt / static_cast<real>(sub_count);
    }
    __syncthreads();

    real inv_sub_count = 1.0 / static_cast<real>(sub_count);
    for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
    {
        cn_lower[iz] *= inv_sub_count;
        cn_upper[iz] *= inv_sub_count;
        cn_diag[iz] = 1.0 - cn_lower[iz] - cn_upper[iz];
    }
    __syncthreads();

    // advance density and donor momentum through every positivity-controlled CN substep
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real cn_i = -cn_lower[iz];
            real cn_o = -cn_upper[iz];
            real rhod_prev = (iz > 0) ? rhod[iz - 1] / _get_diffusion_weight(y, _get_zcent(iz - 1)) : rhod[iz] / _get_diffusion_weight(y, _get_zcent(iz));
            real rhod_next = (iz < N_Z - 1) ? rhod[iz + 1] / _get_diffusion_weight(y, _get_zcent(iz + 1)) : rhod[iz] / _get_diffusion_weight(y, _get_zcent(iz));
            rhod_work[iz] = cn_i*rhod_prev + (1.0 - cn_i - cn_o)*rhod[iz] / _get_diffusion_weight(y, _get_zcent(iz)) + cn_o*rhod_next;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            // Solve density (weight=1) or concentration with Thomas elimination.
            temp_work[0] = cn_upper[0] / cn_diag[0];
            rhod_work[0] /= cn_diag[0];
            for (int iz = 1; iz < N_Z; iz++)
            {
                real pivot = cn_diag[iz] - cn_lower[iz]*temp_work[iz - 1];
                temp_work[iz] = (iz < N_Z - 1) ? cn_upper[iz] / pivot : 0.0;
                rhod_work[iz] = (rhod_work[iz] - cn_lower[iz]*rhod_work[iz - 1]) / pivot;
            }
            for (int iz = N_Z - 2; iz >= 0; iz--)
            {
                rhod_work[iz] -= temp_work[iz]*rhod_work[iz + 1];
            }
        }
        __syncthreads();

        // Reconstruct mass flux from old density/weight and the solved diffused variable, with zero boundary flux
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            if (iz == N_Z - 1)
            {
                temp_work[iz] = 0.0;
            }
            else
            {
                real cn_o = -cn_upper[iz];
                temp_work[iz] = -(_get_diffusion_weight(y, _get_zcent(iz))*cn_o*y*_get_vol_z(iz) / dt_sub)*
                    ((rhod[iz + 1] / _get_diffusion_weight(y, _get_zcent(iz + 1)) - rhod[iz] / _get_diffusion_weight(y, _get_zcent(iz))) + (rhod_work[iz + 1] - rhod_work[iz]));
            }
        }
        __syncthreads();

        // compute donor factors from all raw polar face transfers before scaling any face
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_i = (iz > 0) ? temp_work[iz - 1] : 0.0;
            real out_rate = fmax(temp_work[iz], 0.0) + fmax(-flux_i, 0.0);
            real mass = fmax(rhod[iz], 0.0)*y*_get_vol_z(iz);
            rhod_work[iz] = (out_rate > 0.0)
                ? fmin(1.0, POS_LIMIT*mass / (dt_sub*out_rate)) : 1.0;
        }
        __syncthreads();

        for (int iz = threadIdx.x; iz < N_Z - 1; iz += blockDim.x)
        {
            int iz_up = (temp_work[iz] >= 0.0) ? iz : iz + 1;
            temp_work[iz] *= rhod_work[iz_up];
        }
        __syncthreads();

        // accept density from the limited conservative divergence
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_i = (iz > 0) ? temp_work[iz - 1] : 0.0;
            rhod_work[iz] = rhod[iz]
                - dt_sub*(temp_work[iz] - flux_i) / (y*_get_vol_z(iz));
        }
        __syncthreads();

        // transport every momentum component with the same mass flux and its donor primitive
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            int iz_up = (temp_work[iz] >= 0.0 || iz == N_Z - 1) ? iz : iz + 1;
            real z_up = _get_zcent(iz_up);
            real R_up = y*sin(z_up);
            real rhod_up = rhod[iz_up];
            int idx_cell_up = idx_base + iz_up*N_X*N_Y;
            real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_cell_up] / rhod_up : sqrt(G*M_S*fmax(R_up, 0.0));
            cn_diag[iz] = temp_work[iz]*lx_up;
        }
        __syncthreads();
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_i = (iz > 0) ? cn_diag[iz - 1] : 0.0;
            dev_dustmomx[idx_base + iz*N_X*N_Y] -= dt_sub*(cn_diag[iz] - flux_i) / (y*_get_vol_z(iz));
        }
        __syncthreads();

        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            int iz_up = (temp_work[iz] >= 0.0 || iz == N_Z - 1) ? iz : iz + 1;
            real rhod_up = rhod[iz_up];
            int idx_cell_up = idx_base + iz_up*N_X*N_Y;
            real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_cell_up] / rhod_up : 0.0;
            cn_diag[iz] = temp_work[iz]*vy_up;
        }
        __syncthreads();
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_i = (iz > 0) ? cn_diag[iz - 1] : 0.0;
            dev_dustmomy[idx_base + iz*N_X*N_Y] -= dt_sub*(cn_diag[iz] - flux_i) / (y*_get_vol_z(iz));
        }
        __syncthreads();

        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            int iz_up = (temp_work[iz] >= 0.0 || iz == N_Z - 1) ? iz : iz + 1;
            real rhod_up = rhod[iz_up];
            int idx_cell_up = idx_base + iz_up*N_X*N_Y;
            real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_cell_up] / rhod_up : 0.0;
            cn_diag[iz] = temp_work[iz]*lz_up;
        }
        __syncthreads();
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            real flux_i = (iz > 0) ? cn_diag[iz - 1] : 0.0;
            dev_dustmomz[idx_base + iz*N_X*N_Y] -= dt_sub*(cn_diag[iz] - flux_i) / (y*_get_vol_z(iz));
            rhod[iz] = rhod_work[iz];
        }
        __syncthreads();

        // restore the CN diagonal after all momentum components consume the reused workspace
        for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
        {
            cn_diag[iz] = 1.0 - cn_lower[iz] - cn_upper[iz];
        }
        __syncthreads();
    }

    // store density after all substeps
    for (int iz = threadIdx.x; iz < N_Z; iz += blockDim.x)
    {
        dev_dustdens[idx_base + iz*N_X*N_Y] = rhod[iz];
    }
}

#endif // DIFFUSION
