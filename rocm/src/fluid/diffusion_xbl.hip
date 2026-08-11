#ifdef DIFFUSION

#include <param_grid.cuh>
#include <param_phys.cuh>
#include <fluid_kern.cuh>

// =========================================================================================================================
// kernel: diffusion_xbl
// purpose: solve periodic Crank-Nicolson diffusion cooperatively in block-shared memory
// =========================================================================================================================

__global__
void diffusion_xbl (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx_ring = blockIdx.x;
    if (idx_ring >= N_Y*N_Z || dt <= 0.0) return;

    int iy = idx_ring % N_Y;
    int iz = idx_ring / N_Y;
    int idx_base = iy*N_X + iz*N_X*N_Y;

    // partition dynamic shared memory among density, cyclic solve, and reusable flux work arrays
    extern __shared__ real shared_work[];
    real *rhod = shared_work;
    real *rhod_work = rhod + N_X;
    real *cycle_work = rhod_work + N_X;
    real *upper_work = cycle_work + N_X;

    real dx = _get_dx();
    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);
    real dx_len = R*dx;
    real h_g = _get_hg(R);
    real diff_x = _get_nu(R, h_g) / SCHMIDT_X;

    __shared__ int sub_count;
    __shared__ real dt_sub;
    __shared__ real cn_coeff;
    __shared__ real cn_diag;
    __shared__ real cn_wrap;
    __shared__ real sm_gamma;
    __shared__ real sm_vlast;
    __shared__ real sm_diag0;
    __shared__ real sm_diagN;
    __shared__ real corr_scale;

    // load one density ring and construct positivity-controlled cyclic CN coefficients
    for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
    {
        rhod[ix] = dev_dustdens[idx_base + ix];
    }

    if (threadIdx.x == 0)
    {
        sub_count = static_cast<int>(ceil(dt*diff_x / (dx_len*dx_len) / POS_LIMIT));
        if (sub_count < 1) sub_count = 1;

        dt_sub = dt / static_cast<real>(sub_count);
        cn_coeff = 0.5*dt_sub*diff_x / (dx_len*dx_len);
        cn_diag = 1.0 + 2.0*cn_coeff;
        cn_wrap = -cn_coeff;
        sm_gamma = -cn_diag;
        sm_vlast = cn_wrap / sm_gamma;
        sm_diag0 = cn_diag - sm_gamma;
        sm_diagN = cn_diag - cn_wrap*sm_vlast;
    }
    __syncthreads();

    // advance density and donor momentum through every accepted CN substep
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ixp1 = (ix + 1) % N_X;
            rhod_work[ix] = cn_coeff*rhod[ixm1] + (1.0 - 2.0*cn_coeff)*rhod[ix] + cn_coeff*rhod[ixp1];
            cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            // solve the cyclic tridiagonal system through a Sherman-Morrison correction
            real diag_cur = sm_diag0;
            upper_work[0] = -cn_coeff / diag_cur;
            rhod_work[0] /= diag_cur;
            cycle_work[0] /= diag_cur;

            for (int ix = 1; ix < N_X; ix++)
            {
                diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
                real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
                upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
                rhod_work[ix] = (rhod_work[ix] + cn_coeff*rhod_work[ix - 1]) / pivot;
                cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
            }

            for (int ix = N_X - 2; ix >= 0; ix--)
            {
                rhod_work[ix] -= upper_work[ix]*rhod_work[ix + 1];
                cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
            }

            real base_proj = rhod_work[0] + sm_vlast*rhod_work[N_X - 1];
            real corr_proj = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
            corr_scale = base_proj / (1.0 + corr_proj);
        }
        __syncthreads();

        // apply the Sherman-Morrison correction to the provisional density solution
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            rhod_work[ix] -= corr_scale*cycle_work[ix];
        }
        __syncthreads();

        // reconstruct the time-centred periodic diffusive mass flux
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ixp1 = (ix + 1) % N_X;
            cycle_work[ix] = -0.5*diff_x*
                ((rhod[ixp1] - rhod[ix]) + (rhod_work[ixp1] - rhod_work[ix])) / dx_len;
        }
        __syncthreads();

        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ix_up = (cycle_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real rhod_up = rhod[ix_up];
            real lx_up = (rhod_up >= RHO_VAC) ? dev_dustmomx[idx_base + ix_up] / rhod_up : sqrt(G*M_S*fmax(R, 0.0));
            upper_work[ix] = cycle_work[ix]*lx_up;
        }
        __syncthreads();
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            dev_dustmomx[idx_base + ix] -= dt_sub*(upper_work[ix] - upper_work[ixm1]) / dx_len;
        }
        __syncthreads();

        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ix_up = (cycle_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real rhod_up = rhod[ix_up];
            real vy_up = (rhod_up >= RHO_VAC) ? dev_dustmomy[idx_base + ix_up] / rhod_up : 0.0;
            upper_work[ix] = cycle_work[ix]*vy_up;
        }
        __syncthreads();
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            dev_dustmomy[idx_base + ix] -= dt_sub*(upper_work[ix] - upper_work[ixm1]) / dx_len;
        }
        __syncthreads();

        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ix_up = (cycle_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real rhod_up = rhod[ix_up];
            real lz_up = (rhod_up >= RHO_VAC) ? dev_dustmomz[idx_base + ix_up] / rhod_up : 0.0;
            upper_work[ix] = cycle_work[ix]*lz_up;
        }
        __syncthreads();
        for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            dev_dustmomz[idx_base + ix] -= dt_sub*(upper_work[ix] - upper_work[ixm1]) / dx_len;
            rhod[ix] = rhod_work[ix];
        }
        __syncthreads();
    }

    // store density after all substeps
    for (int ix = threadIdx.x; ix < N_X; ix += blockDim.x)
    {
        dev_dustdens[idx_base + ix] = rhod[ix];
    }
}

#endif
