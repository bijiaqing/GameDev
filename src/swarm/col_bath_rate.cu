#include <gpu.cuh>

#ifdef COLLISION
#include <_col_event.cuh>
#include <_col_rates.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_bath_rate
// calculate bath-start rates used by the pre-bath duration controller
// =====================================================================================================================

__global__
void col_bath_rate (const int *owner_ids, int owner_count, real *dev_col_rate, real *change_rate, real *second_rate,
    const swarm *dev_particle, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const query_environment *environment, cached_rate_moments *cached)
{
    int slot = blockIdx.x;
    if (slot >= owner_count) return;
    int idx_old_i = owner_ids[slot];

    // evaluate every cached pair against the frozen reservoir, one neighbor slot per thread
    __shared__ real rate_work[N_K];
    __shared__ real change_work[N_K], second_work[N_K], maximum_work[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        rate_work[idx_neighbor] = 0.0;
        change_work[idx_neighbor] = second_work[idx_neighbor] = maximum_work[idx_neighbor] = 0.0;
        int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
        if (dev_col_active[idx_old_i] == 0 || neighbor < 0
            || !(dev_col_measure[idx_old_i] > 0.0)) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);

        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];
        real vrel = 0.0;
        real pair_rate = _get_col_chain_rate <static_cast<kernel_type>(COAG_KERNEL)> (
            dev_particle, size_i, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment
        ) / dev_col_measure[idx_old_i];
        real mean = 0.0;
        real second = 0.0;
        real maximum = 0.0;
        pair_rate *= _erosion_outcome_moments(size_i, size_j, vrel >= V_FRAG, mean, second, maximum);
        maximum_work[idx_neighbor] = pair_rate > 0.0 ? maximum : 0.0;
        change_work[idx_neighbor] = pair_rate*mean;
        second_work[idx_neighbor] = pair_rate*second;
        rate_work[idx_neighbor] = pair_rate;
    }
    #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
    // MI300A fast path: fold 256 slots to 64 in shared memory, then reduce one 64-lane wavefront by shuffles
    if constexpr(N_K == 256 && COL_BATH_TPB == 128)
    {
        const int t = threadIdx.x;
        __shared__ unsigned char validity[128];
        __syncthreads();
        validity[t] = isfinite(rate_work[t]) && rate_work[t] >= 0.0 && isfinite(rate_work[t + 128]) && rate_work[t
            + 128] >= 0.0;
        rate_work[t] += rate_work[t + 128];
        change_work[t] += change_work[t + 128];
        second_work[t] += second_work[t + 128];
        maximum_work[t] = fmax(maximum_work[t], maximum_work[t + 128]);
        __syncthreads();
        if (t < 64)
        {
          real r = rate_work[t] + rate_work[t + 64];
          real a = change_work[t] + change_work[t + 64];
          real b = second_work[t] + second_work[t + 64];
          real m = fmax(maximum_work[t], maximum_work[t + 64]);
          int valid = validity[t] && validity[t + 64];
          for (int delta = 32; delta; delta /= 2)
          {
            r += __shfl_down(r, delta, 64);
            a += __shfl_down(a, delta, 64);
            b += __shfl_down(b, delta, 64);
            m = fmax(m, __shfl_down(m, delta, 64));
            int other = __shfl_down(valid, delta, 64);
            valid = valid && other;
          }
          if (t == 0){cached[idx_old_i] = {valid ? r : -1.0, a, b,
              m};dev_col_rate[idx_old_i] = r;change_rate[idx_old_i] = a;second_rate[idx_old_i] = b;}
        }
        return;
    }
    else
    #endif // GAMEDEV_ROCM && __gfx942__
    {
    // reduce the rate moments cooperatively for general N_K and block widths
    __shared__ unsigned char valid_work[N_K];
    for (int j = threadIdx.x; j < N_K; j += blockDim.x)
    {
        valid_work[j] = isfinite(rate_work[j]) && rate_work[j] >= 0.0;
    }
    __syncthreads();

    // fold the upper half into the lower half, including odd active counts
    for (int count = N_K; count > 1; count = (count + 1) / 2)
    {
        int half = (count + 1) / 2;
        for (int j = threadIdx.x; j < count / 2; j += blockDim.x)
        {
            rate_work[j] += rate_work[j + half];
            change_work[j] += change_work[j + half];
            second_work[j] += second_work[j + half];
            maximum_work[j] = fmax(maximum_work[j], maximum_work[j + half]);
            valid_work[j] = valid_work[j] && valid_work[j + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        real rate = rate_work[0];
        real change = change_work[0];
        real second = second_work[0];
        cached[idx_old_i] = {valid_work[0] ? rate : -1.0, change, second, maximum_work[0]};
        dev_col_rate[idx_old_i] = rate;
        change_rate[idx_old_i] = change;
        second_rate[idx_old_i] = second;
    }
    }
}

#endif // COLLISION
