#pragma once

#include "../../../../../inc/swarm/morton/morton_index.cuh"

// expose one cooperative exact top-K query per CUDA block for tests and standalone consumers
template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void morton_search (int *dev_near_idx_old, float *dev_near_dist_sq, unsigned int *dev_leaf_visit_count,
    unsigned int *dev_candidate_count, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data, float search_dist,
    const unsigned char *dev_active = nullptr)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[SORT_SIZE];
    __shared__ int work_idx_old[SORT_SIZE];
    __shared__ int idx_node_stack[STACK_SIZE];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_count;
    __shared__ unsigned int candidate_total;
    __shared__ unsigned int overflow;

    _morton_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        morton_data, dev_query_point[idx_query], search_dist, work_dist_sq, work_idx_old,
        idx_node_stack, stack_count, idx_node, batch_count,
        leaf_count, candidate_total, overflow, dev_active
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        int idx_out = idx_query*K + idx_neighbor;
        dev_near_idx_old[idx_out] =
            (work_idx_old[idx_neighbor] == INT_MAX) ? -1 : work_idx_old[idx_neighbor];
        dev_near_dist_sq[idx_out] = work_dist_sq[idx_neighbor];
    }
    if (threadIdx.x == 0)
    {
        if (dev_leaf_visit_count) dev_leaf_visit_count[idx_query] = leaf_count;
        if (dev_candidate_count) dev_candidate_count[idx_query] = candidate_total;
        if (dev_stack_overflow) dev_stack_overflow[idx_query] = overflow;
    }
}

// reduce each top-K result to a deterministic checksum for timing without large output transfers
template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void morton_digest (double *dev_checksum, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data, float search_dist)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[SORT_SIZE];
    __shared__ int work_idx_old[SORT_SIZE];
    __shared__ int idx_node_stack[STACK_SIZE];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int overflow;

    _morton_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        morton_data, dev_query_point[idx_query], search_dist, work_dist_sq, work_idx_old,
        idx_node_stack, stack_count, idx_node, batch_count,
        leaf_count, candidate_count, overflow
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            if (work_idx_old[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(work_dist_sq[idx_neighbor])
                + 1.0e-12*static_cast<double>(work_idx_old[idx_neighbor]);
        }
        dev_checksum[idx_query] = value;
        if (dev_stack_overflow) dev_stack_overflow[idx_query] = overflow;
    }
}
