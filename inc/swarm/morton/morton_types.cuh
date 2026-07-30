#ifndef GAMEDEV_MORTON_TYPES_CUH
#define GAMEDEV_MORTON_TYPES_CUH

#include <cstdint>       // std::uint32_t, std::uint64_t

#include <cuda_runtime.h>  // CUDA vector types and device qualifiers

struct morton_point
{
    float3 cartesian;
    int index_old;
};

struct morton_node
{
    float3 lower;
    float width;
    int begin;
    int count;
    int child[8];
    int child_number;
};

struct morton_view
{
    const morton_point *points;
    const morton_node *nodes;
    int point_count;
    int node_count;
    int dimension;
};

__host__ __device__ inline
std::uint64_t _expand_morton_3d (std::uint32_t value)
{
    std::uint64_t bits = value & 0x1fffffU;
    bits = (bits | bits << 32) & 0x001f00000000ffffULL;
    bits = (bits | bits << 16) & 0x001f0000ff0000ffULL;
    bits = (bits | bits << 8)  & 0x100f00f00f00f00fULL;
    bits = (bits | bits << 4)  & 0x10c30c30c30c30c3ULL;
    bits = (bits | bits << 2)  & 0x1249249249249249ULL;
    return bits;
}

__host__ __device__ inline
std::uint64_t _get_morton_key (int ix, int iy, int iz)
{
    return _expand_morton_3d(static_cast<std::uint32_t>(ix))
        | (_expand_morton_3d(static_cast<std::uint32_t>(iy)) << 1)
        | (_expand_morton_3d(static_cast<std::uint32_t>(iz)) << 2);
}

__device__ __forceinline__
bool _morton_neighbor_less (float dist_a, int idx_a, float dist_b, int idx_b)
{
    return dist_a < dist_b || (dist_a == dist_b && idx_a < idx_b);
}

__device__ __forceinline__
float _get_morton_dist_sq (const float3 &a, const float3 &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

#endif // GAMEDEV_MORTON_TYPES_CUH
