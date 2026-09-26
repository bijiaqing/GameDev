#ifndef GAMEDEV_SWARM_KDTREE_INDEX_HEAP_CUH
#define GAMEDEV_SWARM_KDTREE_INDEX_HEAP_CUH

#include <type_traits> // std::conditional_t

#include <gpu.cuh>

#include <_col_image.cuh>

// retain the exact top-K physical neighbors while filtering inactive particles and periodic duplicate images
template<int K, typename Node
#ifdef GAMEDEV_ROCM
    , bool ExternalStorage = false
#endif // GAMEDEV_ROCM
>
struct idx_old_heap
{
    const Node *kdtree_node;
    const unsigned char *dev_active;
    static_assert(K > 0 && K <= 4096, "KD-tree requires 0 < K <= 4096");
    #ifdef GAMEDEV_ROCM
    // MI300A tuning: private heaps and 64 queries per block are a paired choice
    static constexpr int threads = 64;
    using private_keys = std::conditional_t<ExternalStorage, unsigned long long*, unsigned long long[K]>;
    private_keys near_key;
    __device__ __forceinline__ unsigned long long get_key (int slot) const
    { return near_key[slot]; }
    __device__ __forceinline__ void set_key (int slot, unsigned long long value)
    { near_key[slot] = value; }
    #else  // !GAMEDEV_ROCM
    // keep per-block heap storage at or below 32 KiB as K increases
    static constexpr int threads = K <= 256 ? 16 : K <= 512 ? 8 : K <= 1024 ? 4 : K <= 2048 ? 2 : 1;
    unsigned long long *near_key;
    __device__ __forceinline__ unsigned long long get_key (int slot) const
    { return near_key[threads*slot]; }
    __device__ __forceinline__ void set_key (int slot, unsigned long long value)
    { near_key[threads*slot] = value; }
    #endif // GAMEDEV_ROCM
    bool dedup_needed;

    __device__ explicit idx_old_heap (
        float search_dist, const Node *tree_node, bool dedup_needed = false,
        const unsigned char *dev_active = nullptr
        #ifdef GAMEDEV_ROCM
        , unsigned long long *external_storage = nullptr
        #endif // GAMEDEV_ROCM
        )
        : kdtree_node(tree_node), dev_active(dev_active), dedup_needed(dedup_needed)
    {
        #ifdef GAMEDEV_ROCM
        if constexpr(ExternalStorage) near_key = external_storage;
        #endif // GAMEDEV_ROCM
        #ifndef GAMEDEV_ROCM
        __shared__ unsigned long long shared_key[K*threads];
        near_key = shared_key + threadIdx.x;
        // no barrier is needed because each thread initializes and accesses only its own column
        #endif // !GAMEDEV_ROCM
        // initialize a finite max-heap whose invalid identifiers sort after every physical candidate
        unsigned long long empty = encode(search_dist*search_dist, 0xffffffffU);
        #pragma unroll
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++) set_key(idx_neighbor, empty);
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
        return decode_dist(get_key(idx_neighbor));
    }
    __device__ __forceinline__ int returnIndex (int idx_neighbor) const
    {
        unsigned int neighbor = decode_neighbor(get_key(idx_neighbor));
        return (neighbor == 0xffffffffU) ? -1 : _get_col_idx_old(static_cast<int>(neighbor));
    }
    __device__ __forceinline__ int returnImage (int idx_neighbor) const
    {
        unsigned int neighbor = decode_neighbor(get_key(idx_neighbor));
        return (neighbor == 0xffffffffU) ? 0 : _get_col_image(static_cast<int>(neighbor));
    }
    __device__ __forceinline__ int returnNeighbor (int idx_neighbor) const
    {
        unsigned int neighbor = decode_neighbor(get_key(idx_neighbor));
        return (neighbor == 0xffffffffU) ? -1 : static_cast<int>(neighbor);
    }
    __device__ __forceinline__ float initialCullDist2 () const { return expandedCullDist2(); }
    __device__ __forceinline__ float maxRadius2 () const { return decode_dist(get_key(0)); }
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
                unsigned int neighbor_old = decode_neighbor(get_key(idx_neighbor));
                if (neighbor_old == 0xffffffffU
                    || static_cast<unsigned int>(_get_col_idx_old(static_cast<int>(neighbor_old))) != idx_old)
                    continue;
                idx_slot = idx_neighbor;
                break;
            }
        }

        if (idx_slot >= 0)
        {
            if (candidate >= get_key(idx_slot)) return expandedCullDist2();
        }
        else
        {
            if (candidate >= get_key(0)) return expandedCullDist2();
            idx_slot = 0;
        }

        // retain the selected child value to avoid a dependent scratch reload
        while (true)
        {
            int idx_child = 2*idx_slot + 1;
            if (idx_child >= K)
            {
                set_key(idx_slot, candidate);
                break;
            }
            unsigned long long child = get_key(idx_child);
            if (idx_child + 1 < K)
            {
                unsigned long long right = get_key(idx_child + 1);
                if (right > child)
                {
                    child = right;
                    ++idx_child;
                }
            }
            if (child < candidate)
            {
                set_key(idx_slot, candidate);
                break;
            }
            set_key(idx_slot, child);
            idx_slot = idx_child;
        }
        return expandedCullDist2();
    }
};

#endif // GAMEDEV_SWARM_KDTREE_INDEX_HEAP_CUH
