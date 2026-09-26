#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_sizes.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_count_bin
// count occupied size bins before merging statistically undersampled tails
// =====================================================================================================================

__global__
void col_count_bin (const int *owner_ids, int owner_count, int *dev_col_count, const swarm *dev_particle,
    const int *dev_col_spatial, const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    atomicAdd(dev_col_count + idx_raw, 1);
}

#endif // COLLISION && !BERNOULLI
