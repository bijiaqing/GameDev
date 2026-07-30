#ifndef QAV_KNN_TYPES_CUH
#define QAV_KNN_TYPES_CUH

#include <cuda_runtime.h>

#include <kdtree/builder.h>

#include <morton/morton_types.cuh>

struct kd_point
{
    float3 cartesian;
    int index_old;
    int split_dim;
    int image;
};

struct kd_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };

    static inline __host__ __device__ const point_t &get_point (const kd_point &point) { return point.cartesian; }
    static inline __host__ __device__ float get_coord (const kd_point &point, int dim)
    {
        return kdtree::get_coord(point.cartesian, dim);
    }
    static inline __host__ __device__ int get_dim (const kd_point &point) { return point.split_dim; }
    static inline __host__ __device__ void set_dim (kd_point &point, int dim) { point.split_dim = dim; }
};

template<int K, bool DEDUPLICATE = false>
struct kd_heap
{
    const kd_point *point;
    unsigned long long key[K];
    int idx_point[K];

    __device__ explicit kd_heap (float radius, const kd_point *tree_point)
        : point(tree_point)
    {
        unsigned long long empty = encode(radius*radius, 0xffffffffU);
        #pragma unroll
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            key[idx_neighbor] = empty;
            idx_point[idx_neighbor] = -1;
        }
    }

    __device__ __forceinline__
    unsigned long long encode (float dist_sq, unsigned int stable_id) const
    {
        return (static_cast<unsigned long long>(__float_as_uint(dist_sq)) << 32) | stable_id;
    }

    __device__ __forceinline__
    float decode_dist (unsigned long long value) const
    {
        return __uint_as_float(static_cast<unsigned int>(value >> 32));
    }

    __device__ __forceinline__ float returnValue () const { return maxRadius2(); }
    __device__ __forceinline__ float returnDist2 (int idx_neighbor) const { return decode_dist(key[idx_neighbor]); }
    __device__ __forceinline__ int returnIndex (int idx_neighbor) const { return idx_point[idx_neighbor]; }
    __device__ __forceinline__ float initialCullDist2 () const { return expandedCullDist2(); }
    __device__ __forceinline__ float maxRadius2 () const { return decode_dist(key[0]); }
    __device__ __forceinline__
    float expandedCullDist2 () const
    {
        return nextafterf(maxRadius2(), __uint_as_float(0x7f800000U));
    }

    __device__ __forceinline__
    float processCandidate (int idx_candidate, float dist_sq)
    {
        unsigned int stable_id = static_cast<unsigned int>(point[idx_candidate].index_old);
        unsigned long long candidate = encode(dist_sq, stable_id);
        int position = -1;
        if constexpr (DEDUPLICATE)
        {
            for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
            {
                if (idx_point[idx_neighbor] < 0) continue;
                if (static_cast<unsigned int>(point[idx_point[idx_neighbor]].index_old) != stable_id) continue;
                position = idx_neighbor;
                break;
            }
        }

        if (position >= 0)
        {
            if (candidate >= key[position]) return expandedCullDist2();
        }
        else
        {
            if (candidate >= key[0]) return expandedCullDist2();
            position = 0;
        }

        while (true)
        {
            int child_1 = 2*position + 1;
            int child_max = -1;
            if (child_1 < K) child_max = child_1;
            int child_2 = child_1 + 1;
            if (child_2 < K && key[child_2] > key[child_max]) child_max = child_2;
            if (child_max < 0 || key[child_max] < candidate)
            {
                key[position] = candidate;
                idx_point[position] = idx_candidate;
                break;
            }
            key[position] = key[child_max];
            idx_point[position] = idx_point[child_max];
            position = child_max;
        }
        return expandedCullDist2();
    }
};

#endif // QAV_KNN_TYPES_CUH
