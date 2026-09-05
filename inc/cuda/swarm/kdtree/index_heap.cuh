#ifndef GAMEDEV_KDTREE_INDEX_HEAP_CUH
#define GAMEDEV_KDTREE_INDEX_HEAP_CUH

#include <cuda_runtime.h>                  // CUDA device qualifiers and bit conversions

// retain the exact top-K physical neighbors while filtering inactive particles and periodic duplicate images
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
        // initialize a finite max-heap whose invalid identifiers sort after every physical candidate
        unsigned long long empty = encode(search_dist*search_dist, 0xffffffffU);
        #pragma unroll
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++) near_key[idx_neighbor] = empty;
    }

    __device__ __forceinline__
    unsigned long long encode (float dist_sq, unsigned int neighbor) const
    {
        // order packed keys by distance, physical identifier, and selected image
        return (static_cast<unsigned long long>(__float_as_uint(dist_sq)) << 32) | neighbor;
    }

    __device__ __forceinline__
    float decode_dist (unsigned long long value) const
    {
        return __uint_as_float(static_cast<unsigned int>(value >> 32));
    }

    __device__ __forceinline__
    unsigned int decode_neighbor (unsigned long long value) const
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
        unsigned int neighbor = decode_neighbor(near_key[idx_neighbor]);
        return (neighbor == 0xffffffffU) ? -1 : _get_col_idx_old(static_cast<int>(neighbor));
    }
    __device__ __forceinline__ int returnImage (int idx_neighbor) const
    {
        unsigned int neighbor = decode_neighbor(near_key[idx_neighbor]);
        return (neighbor == 0xffffffffU) ? 0 : _get_col_image(static_cast<int>(neighbor));
    }
    __device__ __forceinline__ int returnNeighbor (int idx_neighbor) const
    {
        unsigned int neighbor = decode_neighbor(near_key[idx_neighbor]);
        return (neighbor == 0xffffffffU) ? -1 : static_cast<int>(neighbor);
    }
    __device__ __forceinline__ float initialCullDist2 () const { return expandedCullDist2(); }
    __device__ __forceinline__ float maxRadius2 () const { return decode_dist(near_key[0]); }
    __device__ __forceinline__
    float expandedCullDist2 () const
    {
        // retain candidates equal to the current cutoff despite single-precision traversal rounding
        return nextafterf(maxRadius2(), __uint_as_float(0x7f800000U));
    }

    __device__ __forceinline__
    float processCandidate (int idx_candidate, float dist_sq)
    {
        unsigned int idx_old = static_cast<unsigned int>(kdtree_node[idx_candidate].idx_old);
        if (dev_active && dev_active[idx_old] == 0) return expandedCullDist2();

        unsigned int neighbor = static_cast<unsigned int>(_encode_col_neighbor(
            static_cast<int>(idx_old), kdtree_node[idx_candidate].image
        ));
        unsigned long long candidate = encode(dist_sq, neighbor);

        // replace an existing periodic image only when the new image is closer
        int idx_slot = -1;
        if (dedup_needed)
        {
            for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
            {
                unsigned int neighbor_old = decode_neighbor(near_key[idx_neighbor]);
                if (neighbor_old == 0xffffffffU
                    || static_cast<unsigned int>(_get_col_idx_old(static_cast<int>(neighbor_old))) != idx_old)
                    continue;
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

        // restore max-heap order after replacing either the root or a duplicate-image slot
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
