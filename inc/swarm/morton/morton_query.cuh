#ifndef GAMEDEV_MORTON_QUERY_CUH
#define GAMEDEV_MORTON_QUERY_CUH

#include <climits>                         // INT_MAX

#include <cuda_runtime.h>                  // CUDA device qualifiers
#include <math_constants.h>                // CUDART_INF_F

#include <morton/morton_index.cuh>

template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _morton_id_sort (float *distance, int *index)
{
    for (int width = 2; width <= SORT_SIZE; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int slot = threadIdx.x; slot < SORT_SIZE; slot += BLOCK_SIZE)
            {
                int partner = slot ^ stride;
                if (partner <= slot) continue;
                bool ascending = (slot & width) == 0;
                bool partner_less = index[partner] < index[slot]
                    || (index[partner] == index[slot] && distance[partner] < distance[slot]);
                bool slot_less = index[slot] < index[partner]
                    || (index[slot] == index[partner] && distance[slot] < distance[partner]);
                bool swap_pair = ascending ? partner_less : slot_less;
                if (swap_pair)
                {
                    float dist_tmp = distance[slot];
                    distance[slot] = distance[partner];
                    distance[partner] = dist_tmp;
                    int idx_tmp = index[slot];
                    index[slot] = index[partner];
                    index[partner] = idx_tmp;
                }
            }
            __syncthreads();
        }
    }
}

template<int K, int BLOCK_SIZE, int WORK_SIZE, int STACK_SIZE>
__device__ __forceinline__
void _morton_ghost_topk (const morton_view &view, const float3 &query, float radius,
    bool duplicate_safe, float *work_dist, int *work_idx, int *node_stack,
    int &stack_size, int &idx_node, int &batch_count,
    unsigned int &leaves_visited, unsigned int &candidates_examined,
    unsigned int &stack_overflow)
{
    static_assert(3*K + BLOCK_SIZE <= WORK_SIZE,
        "Morton ghost work array cannot hold the duplicate-safe candidate set");
    static_assert(K + BLOCK_SIZE <= 512,
        "Morton fast work array cannot hold one top-K candidate tile");

    if (duplicate_safe)
    {
        _morton_topk<K, BLOCK_SIZE, 512, STACK_SIZE>(
            view, query, radius, work_dist, work_idx, node_stack,
            stack_size, idx_node, batch_count, leaves_visited, candidates_examined, stack_overflow
        );
        return;
    }

    _morton_topk<3*K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
        view, query, radius, work_dist, work_idx, node_stack,
        stack_size, idx_node, batch_count, leaves_visited, candidates_examined, stack_overflow
    );

    for (int slot = 3*K + threadIdx.x; slot < WORK_SIZE; slot += BLOCK_SIZE)
    {
        work_dist[slot] = CUDART_INF_F;
        work_idx[slot] = INT_MAX;
    }
    __syncthreads();

    _morton_id_sort<WORK_SIZE, BLOCK_SIZE>(work_dist, work_idx);
    constexpr int slots_per_thread = (WORK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool duplicate[slots_per_thread];
    int local_slot = 0;
    for (int slot = threadIdx.x; slot < WORK_SIZE; slot += BLOCK_SIZE)
        duplicate[local_slot++] = slot > 0 && work_idx[slot] == work_idx[slot - 1];
    __syncthreads();

    local_slot = 0;
    for (int slot = threadIdx.x; slot < WORK_SIZE; slot += BLOCK_SIZE)
    {
        if (duplicate[local_slot++])
        {
            work_dist[slot] = CUDART_INF_F;
            work_idx[slot] = INT_MAX;
        }
    }
    __syncthreads();
    _morton_pair_sort<WORK_SIZE, BLOCK_SIZE>(work_dist, work_idx);
}

#endif // GAMEDEV_MORTON_QUERY_CUH
