#ifndef GAMEDEV_SWARM_COL_IMAGE_CUH
#define GAMEDEV_SWARM_COL_IMAGE_CUH

#include <climits> // INT_MAX
#include <cstddef> // std::size_t

#include <const_defs.cuh>

// packed collision-neighbor identifiers and neighbor-slot offsets for the collision operator and KD-tree heap

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

#ifdef COLLISION
// address the neighbor slot of one owner in the flat N_P*N_K cache
__host__ __device__ __forceinline__
std::size_t _get_col_offset (int idx_owner, int idx_neighbor)
{
    return static_cast<std::size_t>(idx_owner)*static_cast<std::size_t>(N_K)
        + static_cast<std::size_t>(idx_neighbor);
}
#endif // COLLISION

#endif // GAMEDEV_SWARM_COL_IMAGE_CUH
