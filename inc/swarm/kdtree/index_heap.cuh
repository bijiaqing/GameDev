#ifndef GAMEDEV_KDTREE_INDEX_HEAP_CUH
#define GAMEDEV_KDTREE_INDEX_HEAP_CUH

#include <cuda_runtime.h>                  // CUDA device qualifiers and bit conversions

template<int K, typename Node>
struct index_old_heap
{
    const Node *node;
    unsigned long long key[K];
    bool deduplicate;

    __device__ explicit index_old_heap (
        float cutoff_radius, const Node *tree_node, bool deduplicate_images = false)
        : node(tree_node), deduplicate(deduplicate_images)
    {
        unsigned long long empty = encode(cutoff_radius*cutoff_radius, 0xffffffffU);
        #pragma unroll
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++) key[idx_neighbor] = empty;
    }

    __device__ __forceinline__
    unsigned long long encode (float dist_sq, unsigned int idx_old) const
    {
        return (static_cast<unsigned long long>(__float_as_uint(dist_sq)) << 32) | idx_old;
    }

    __device__ __forceinline__
    float decode_dist (unsigned long long value) const
    {
        return __uint_as_float(static_cast<unsigned int>(value >> 32));
    }

    __device__ __forceinline__
    unsigned int decode_idx (unsigned long long value) const
    {
        return static_cast<unsigned int>(value);
    }

    __device__ __forceinline__ float returnValue () const { return maxRadius2(); }
    __device__ __forceinline__ float returnDist2 (int idx_neighbor) const { return decode_dist(key[idx_neighbor]); }
    __device__ __forceinline__ int returnIndex (int idx_neighbor) const
    {
        unsigned int idx_old = decode_idx(key[idx_neighbor]);
        return (idx_old == 0xffffffffU) ? -1 : static_cast<int>(idx_old);
    }
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
        unsigned int idx_old = static_cast<unsigned int>(node[idx_candidate].index_old);
        unsigned long long candidate = encode(dist_sq, idx_old);

        int position = -1;
        if (deduplicate)
        {
            for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
            {
                if (decode_idx(key[idx_neighbor]) != idx_old) continue;
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
                break;
            }

            key[position] = key[child_max];
            position = child_max;
        }
        return expandedCullDist2();
    }
};

#endif // GAMEDEV_KDTREE_INDEX_HEAP_CUH
