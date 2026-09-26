#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_sizes.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_size_scan
// record the smallest and largest active grain size in each refreshed owner's spatial group
// =====================================================================================================================

__global__
void col_size_scan (const int *ids, int count, const swarm *particles,
    const int *spatial, const unsigned char *active)
{
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if (slot >= count) return;
    int i = ids[slot];
    real size = particles[i].par_size;
    if (!active[i] || !(size > 0) || !isfinite(size)) return;
    // positive doubles have the same ordering as their unsigned bit patterns
    auto bits = static_cast<unsigned long long>(__double_as_longlong(size));
    atomicMin(&moving_min[spatial[i]], bits);
    atomicMax(&moving_max[spatial[i]], bits);
}

#endif // COLLISION && !BERNOULLI
