#ifndef GAMEDEV_SWARM_COL_SIZES_CUH
#define GAMEDEV_SWARM_COL_SIZES_CUH

// moving logarithmic size-bin bounds per spatial controller group

#ifdef COLLISION
#include <_col_types.cuh>

// moving log-size bin bounds per spatial group
// scratch extrema are reset globally; retained bounds change only for refreshed groups
// defined in col_size_zero.cu and shared by every kernel that bins sizes
extern __device__ unsigned long long moving_min[moving_groups], moving_max[moving_groups];
extern __device__ real moving_lower[moving_groups], moving_upper[moving_groups];

// map a grain diameter to its logarithmic controller bin inside one spatial group, clamping outliers
__device__ __forceinline__ int _get_col_sizebin (real size, int group)
{
    if (!isfinite(size) || !(size > 0)) return 0;
    real fraction = log(size / moving_lower[group]) / log(moving_upper[group] / moving_lower[group]);
    return max(0, min(COL_BIN_S - 1, static_cast<int>(floor(fraction*COL_BIN_S))));
}

#endif // COLLISION

#endif // GAMEDEV_SWARM_COL_SIZES_CUH
