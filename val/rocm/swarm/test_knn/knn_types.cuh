#ifndef VAL_KNN_TYPES_CUH
#define VAL_KNN_TYPES_CUH

#include <hip/hip_runtime.h> // float3 and host-device qualifiers

#include <kdtree/builder.h>
#include <kdtree/index_heap.cuh>

#include <morton/morton_types.cuh>

// mirror the production KD-tree record while retaining periodic-image identity
struct kdtree_point
{
    float3 cartesian;
    int idx_old;
    int split_dim;
    int image;
};

// expose the benchmark point record through the bundled KD-tree trait interface
struct kdtree_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };

    static inline __host__ __device__ const point_t &get_point (const kdtree_point &point)
    {
        return point.cartesian;
    }
    static inline __host__ __device__ float get_coord (const kdtree_point &point, int dim)
    {
        return kdtree::get_coord(point.cartesian, dim);
    }
    static inline __host__ __device__ int get_dim (const kdtree_point &point)
    {
        return point.split_dim;
    }
    static inline __host__ __device__ void set_dim (kdtree_point &point, int dim)
    {
        point.split_dim = dim;
    }
};

template<int K>
using kdtree_heap = idx_old_heap<K, kdtree_point>;

#endif // VAL_KNN_TYPES_CUH
