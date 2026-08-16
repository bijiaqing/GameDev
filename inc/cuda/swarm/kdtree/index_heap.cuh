#ifndef GAMEDEV_KDTREE_INDEX_HEAP_CUH
#define GAMEDEV_KDTREE_INDEX_HEAP_CUH

#include <cuda_runtime.h>                  // CUDA device qualifiers and bit conversions

template<int K, typename Node>
struct idx_old_heap
{
    const Node *kdtree_node;
    const unsigned char *dev_active;
    unsigned long long near_key[K];
    bool dedup_needed;

    __device__ explicit idx_old_heap (
        float search_dist, const Node *tree_node, bool dedup_needed = false,
        const unsigned char *dev_active = nullptr)
        : kdtree_node(tree_node), dev_active(dev_active), dedup_needed(dedup_needed)
    {
        unsigned long long empty = encode(search_dist*search_dist, 0xffffffffU);
        #pragma unroll
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++) near_key[idx_neighbor] = empty;
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
    __device__ __forceinline__ float returnDist2 (int idx_neighbor) const
    {
        return decode_dist(near_key[idx_neighbor]);
    }
    __device__ __forceinline__ int returnIndex (int idx_neighbor) const
    {
        unsigned int idx_old = decode_idx(near_key[idx_neighbor]);
        return (idx_old == 0xffffffffU) ? -1 : static_cast<int>(idx_old);
    }
    __device__ __forceinline__ float initialCullDist2 () const { return expandedCullDist2(); }
    __device__ __forceinline__ float maxRadius2 () const { return decode_dist(near_key[0]); }
    __device__ __forceinline__
    float expandedCullDist2 () const
    {
        return nextafterf(maxRadius2(), __uint_as_float(0x7f800000U));
    }

    __device__ __forceinline__
    float processCandidate (int idx_candidate, float dist_sq)
    {
        unsigned int idx_old = static_cast<unsigned int>(kdtree_node[idx_candidate].idx_old);
        if (dev_active && dev_active[idx_old] == 0) return expandedCullDist2();

        unsigned long long candidate = encode(dist_sq, idx_old);

        int idx_slot = -1;
        if (dedup_needed)
        {
            for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
            {
                if (decode_idx(near_key[idx_neighbor]) != idx_old) continue;
                idx_slot = idx_neighbor;
                break;
            }
        }

        if (idx_slot >= 0)
        {
            if (candidate >= near_key[idx_slot]) return expandedCullDist2();
        }
        else
        {
            if (candidate >= near_key[0]) return expandedCullDist2();
            idx_slot = 0;
        }

        while (true)
        {
            int idx_child1 = 2*idx_slot + 1;
            int idx_child_max = -1;
            if (idx_child1 < K) idx_child_max = idx_child1;
            int idx_child2 = idx_child1 + 1;
            if (idx_child2 < K && near_key[idx_child2] > near_key[idx_child_max]) idx_child_max = idx_child2;

            if (idx_child_max < 0 || near_key[idx_child_max] < candidate)
            {
                near_key[idx_slot] = candidate;
                break;
            }

            near_key[idx_slot] = near_key[idx_child_max];
            idx_slot = idx_child_max;
        }
        return expandedCullDist2();
    }
};

#endif // GAMEDEV_KDTREE_INDEX_HEAP_CUH
