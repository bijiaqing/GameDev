#include <gpu.cuh>

#ifdef COLLISION
#include <_col_sizes.cuh>
#include <swarm_kern.cuh>

// retained and scratch size bounds shared through _col_sizes.cuh
__device__ unsigned long long moving_min[moving_groups], moving_max[moving_groups];
__device__ real moving_lower[moving_groups], moving_upper[moving_groups];

// =====================================================================================================================
// kernel: col_size_zero
// reset the scratch size extrema of every spatial group before the refreshed owners are scanned
// =====================================================================================================================

__global__
void col_size_zero ()
{
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g < moving_groups)
    {
        moving_min[g] = __double_as_longlong(INFINITY);
        moving_max[g] = 0;
    }
}

#endif // COLLISION
