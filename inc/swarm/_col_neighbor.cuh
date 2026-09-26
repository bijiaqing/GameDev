#ifndef GAMEDEV_SWARM_COL_NEIGHBOR_CUH
#define GAMEDEV_SWARM_COL_NEIGHBOR_CUH

#include <climits> // INT_MAX

#include <const_defs.cuh>

// packed collision-neighbor identifiers shared by the collision operator and the KD-tree heap

constexpr int COL_IMAGE_COUNT = 3;
static_assert(N_P <= INT_MAX / COL_IMAGE_COUNT,
    "N_P is too large for packed collision-neighbor identifiers");

// retain a physical particle index and its selected periodic image in one cached integer
__host__ __device__ __forceinline__
int _encode_col_neighbor (int idx_old, int image) { return COL_IMAGE_COUNT*idx_old + image; }

__host__ __device__ __forceinline__
int _get_col_idx_old (int neighbor) { return neighbor / COL_IMAGE_COUNT; }

__host__ __device__ __forceinline__
int _get_col_image (int neighbor) { return neighbor % COL_IMAGE_COUNT; }

__host__ __device__ __forceinline__
int _get_col_image_shift (int image) { return (image == 1) ? -1 : ((image == 2) ? 1 : 0); }

#endif // GAMEDEV_SWARM_COL_NEIGHBOR_CUH
