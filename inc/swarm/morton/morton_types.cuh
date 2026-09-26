#ifndef GAMEDEV_SWARM_MORTON_TYPES_CUH
#define GAMEDEV_SWARM_MORTON_TYPES_CUH

#include <cstdint>       // std::uint32_t, std::uint64_t
#ifdef GAMEDEV_ROCM
#include <limits>        // std::numeric_limits
#endif // GAMEDEV_ROCM

#include <gpu.cuh>  // GPU vector types and device qualifiers
#ifdef GAMEDEV_ROCM

inline constexpr float MORTON_INF_F = std::numeric_limits<float>::infinity();
inline constexpr float MORTON_PI_F = 3.14159265358979323846f;
#endif // GAMEDEV_ROCM

// store one Morton-sorted search record and its original particle identifier
struct morton_point
{
    float3 cartesian;
    int idx_old;
};

// store one adaptive quadtree or octree node over a contiguous Morton-key range
struct morton_node
{
    float3 lower;
    float width;
    int idx_begin;
    int count;
    int idx_child[8];
    int child_count;
};

// expose the device-resident index without transferring ownership
struct morton_view
{
    const morton_point *dev_point;
    const morton_node *dev_node;
    int point_count;
    int node_count;
    int dim;
    int max_level;
};

// separate one 21-bit coordinate into every third bit of a 64-bit Morton key
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

// interleave integer Cartesian coordinates into one deterministic spatial key
__host__ __device__ inline
std::uint64_t _get_morton_key (int ix, int iy, int iz)
{
    return _expand_morton_3d(static_cast<std::uint32_t>(ix))
        | (_expand_morton_3d(static_cast<std::uint32_t>(iy)) << 1)
        | (_expand_morton_3d(static_cast<std::uint32_t>(iz)) << 2);
}

// order equal-distance neighbors by original particle identifier
__device__ __forceinline__
bool _morton_neighbor_less (float dist_a_sq, int idx_old_a, float dist_b_sq, int idx_old_b)
{
    return dist_a_sq < dist_b_sq || (dist_a_sq == dist_b_sq && idx_old_a < idx_old_b);
}

// calculate Cartesian squared distance without a square root
__device__ __forceinline__
float _get_morton_point_dist_sq (const float3 &point_a, const float3 &point_b)
{
    float dx = point_a.x - point_b.x;
    float dy = point_a.y - point_b.y;
    float dz = point_a.z - point_b.z;
    return dx*dx + dy*dy + dz*dz;
}

#endif // GAMEDEV_SWARM_MORTON_TYPES_CUH
