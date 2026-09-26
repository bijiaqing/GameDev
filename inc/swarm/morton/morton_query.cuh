#ifndef GAMEDEV_MORTON_QUERY_CUH
#define GAMEDEV_MORTON_QUERY_CUH

#include <climits>                         // INT_MAX

#ifdef GAMEDEV_ROCM
#include <gpu.cuh>                  // HIP device qualifiers
#else  // !GAMEDEV_ROCM
#include <gpu.cuh>                  // CUDA device qualifiers
#include <math_constants.h>                // CUDART_INF_F
#endif // GAMEDEV_ROCM

#include <_collision.cuh>
#include <morton/morton_index.cuh>

// sort candidates by original identifier before removing periodic duplicates
template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _morton_id_sort (float *dist_sq, int *idx_old)
{
    for (int width = 2; width <= SORT_SIZE; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int idx_slot = threadIdx.x; idx_slot < SORT_SIZE; idx_slot += BLOCK_SIZE)
            {
                int idx_partner = idx_slot ^ stride;
                if (idx_partner <= idx_slot) continue;
                bool ascending = (idx_slot & width) == 0;
                int idx_physical = (idx_old[idx_slot] == INT_MAX)
                    ? INT_MAX : _get_col_idx_old(idx_old[idx_slot]);
                int idx_partner_physical = (idx_old[idx_partner] == INT_MAX)
                    ? INT_MAX : _get_col_idx_old(idx_old[idx_partner]);
                bool partner_less = idx_partner_physical < idx_physical
                    || (idx_partner_physical == idx_physical
                        && (dist_sq[idx_partner] < dist_sq[idx_slot]
                            || (dist_sq[idx_partner] == dist_sq[idx_slot]
                                && idx_old[idx_partner] < idx_old[idx_slot])));
                bool slot_less = idx_physical < idx_partner_physical
                    || (idx_physical == idx_partner_physical
                        && (dist_sq[idx_slot] < dist_sq[idx_partner]
                            || (dist_sq[idx_slot] == dist_sq[idx_partner]
                                && idx_old[idx_slot] < idx_old[idx_partner])));
                bool swap_pair = ascending ? partner_less : slot_less;
                if (swap_pair)
                {
                    float dist_sq_tmp = dist_sq[idx_slot];
                    dist_sq[idx_slot] = dist_sq[idx_partner];
                    dist_sq[idx_partner] = dist_sq_tmp;
                    int idx_old_tmp = idx_old[idx_slot];
                    idx_old[idx_slot] = idx_old[idx_partner];
                    idx_old[idx_partner] = idx_old_tmp;
                }
            }
            __syncthreads();
        }
    }
}

// query periodic ghost records and collapse multiple images to one physical particle
template<int K, int BLOCK_SIZE, int WORK_SIZE, int STACK_SIZE>
__device__ __forceinline__
void _morton_ghost_topk (
    const morton_view &morton_data, const float3 &query_point, float search_dist,
    bool unique_ids, float *work_dist_sq, int *work_idx_old, int *idx_node_stack,
    int &stack_count, int &idx_node, int &batch_count,
    unsigned int &leaf_visit_count, unsigned int &candidate_count,
    unsigned int &stack_overflow, const unsigned char *dev_active = nullptr)
{
    static_assert(3*K + BLOCK_SIZE <= WORK_SIZE,
        "Morton ghost work array cannot hold the duplicate-safe candidate set");
    constexpr int FAST_SIZE = [] { int n = 1; while (n < 2*K || n < K + BLOCK_SIZE) n *= 2; return n; }();
    static_assert(FAST_SIZE <= WORK_SIZE, "Morton work array cannot hold the padded top-K merge");

    if (unique_ids)
    {
        // skip duplicate filtering when geometry guarantees disjoint physical identifiers
        _morton_topk<K, BLOCK_SIZE, FAST_SIZE, STACK_SIZE>(
            morton_data, query_point, search_dist, work_dist_sq, work_idx_old, idx_node_stack,
            stack_count, idx_node, batch_count,
            leaf_visit_count, candidate_count, stack_overflow, dev_active, COL_IMAGE_COUNT
        );
        return;
    }

    // retain up to three images per physical neighbor before identifier deduplication
    _morton_topk<3*K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
        morton_data, query_point, search_dist, work_dist_sq, work_idx_old, idx_node_stack,
        stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow, dev_active, COL_IMAGE_COUNT
    );

    for (int idx_slot = 3*K + threadIdx.x; idx_slot < WORK_SIZE; idx_slot += BLOCK_SIZE)
    {
        #ifdef GAMEDEV_ROCM
        work_dist_sq[idx_slot] = MORTON_INF_F;
        #else  // !GAMEDEV_ROCM
        work_dist_sq[idx_slot] = CUDART_INF_F;
        #endif // GAMEDEV_ROCM
        work_idx_old[idx_slot] = INT_MAX;
    }
    __syncthreads();

    _morton_id_sort<WORK_SIZE, BLOCK_SIZE>(work_dist_sq, work_idx_old);
    constexpr int slots_per_thread = (WORK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool duplicate[slots_per_thread];
    int idx_local = 0;
    for (int idx_slot = threadIdx.x; idx_slot < WORK_SIZE; idx_slot += BLOCK_SIZE)
    {
        duplicate[idx_local++] = idx_slot > 0 && work_idx_old[idx_slot] != INT_MAX
            && _get_col_idx_old(work_idx_old[idx_slot])
                == _get_col_idx_old(work_idx_old[idx_slot - 1]);
    }
    __syncthreads();

    idx_local = 0;
    for (int idx_slot = threadIdx.x; idx_slot < WORK_SIZE; idx_slot += BLOCK_SIZE)
    {
        if (duplicate[idx_local++])
        {
            #ifdef GAMEDEV_ROCM
            work_dist_sq[idx_slot] = MORTON_INF_F;
            #else  // !GAMEDEV_ROCM
            work_dist_sq[idx_slot] = CUDART_INF_F;
            #endif // GAMEDEV_ROCM
            work_idx_old[idx_slot] = INT_MAX;
        }
    }
    __syncthreads();
    _morton_pair_sort<WORK_SIZE, BLOCK_SIZE>(work_dist_sq, work_idx_old);
}

#endif // GAMEDEV_MORTON_QUERY_CUH
