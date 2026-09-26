#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_sizes.cuh>
#include <_collision.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_rate_bins
// accumulate mass-weighted rates for the pre-bath duration bound
// =====================================================================================================================

__global__
void col_rate_bins (const int *owner_ids, int owner_count, col_rate_bin *dev_col_bin, const swarm *dev_particle,
    const real *dev_col_rate, const real *change_rate, const real *second_rate, const int *dev_col_spatial,
    const int *dev_col_binmap,
    const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    int idx_bin = dev_col_binmap[idx_raw];
    real weight = dev_particle[idx].par_numr*_get_grain_mass(dev_particle[idx].par_size);
    atomicAdd(&dev_col_bin[idx_bin].owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0)
        || !isfinite(dev_col_rate[idx]) || dev_col_rate[idx] < 0.0
        || !isfinite(change_rate[idx]) || change_rate[idx] < 0
        || !isfinite(second_rate[idx]) || second_rate[idx] < 0)
    {
        atomicAdd(&dev_col_bin[idx_bin].invalid_count, 1);
        return;
    }
    atomicAdd(&dev_col_bin[idx_bin].mass, weight);
    atomicAdd(&dev_col_bin[idx_bin].weighted_rate, weight*dev_col_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_change, weight*change_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_second, weight*second_rate[idx]);
}

#endif // COLLISION && !BERNOULLI
