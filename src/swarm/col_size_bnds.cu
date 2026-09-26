#include <gpu.cuh>

#ifdef COLLISION
#include <_col_sizes.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_size_bnds
// widen the observed size range to [0.5*s_min, 8*s_max] so the next interval's growth stays inside the bins
// =====================================================================================================================

__global__
void col_size_bnds ()
{
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g < moving_groups && moving_max[g] != 0)
    {
        moving_lower[g] = 0.5*__longlong_as_double(moving_min[g]);
        moving_upper[g] = 8.0*__longlong_as_double(moving_max[g]);
    }
}

#endif // COLLISION
