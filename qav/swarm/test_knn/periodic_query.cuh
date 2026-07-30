#ifndef QAV_PERIODIC_QUERY_CUH
#define QAV_PERIODIC_QUERY_CUH

#include <climits>

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F, CUDART_PI_F

#include <morton/morton_index.cuh>

__device__ __forceinline__
float3 _rotate_query_z (const float3 &query, float angle)
{
    float sin_angle;
    float cos_angle;
    sincosf(angle, &sin_angle, &cos_angle);
    return make_float3(
        cos_angle*query.x - sin_angle*query.y,
        sin_angle*query.x + cos_angle*query.y,
        query.z
    );
}

__device__ __forceinline__
float _get_seam_dist (const float3 &query, float angle_offset)
{
    float R = hypotf(query.x, query.y);
    float cos_offset = cosf(angle_offset);
    return (cos_offset >= 0.0f) ? R*fabsf(sinf(angle_offset)) : R;
}

template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _periodic_id_sort (float *distances, int *indices)
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
                bool partner_less = indices[partner] < indices[slot]
                    || (indices[partner] == indices[slot] && distances[partner] < distances[slot]);
                bool slot_less = indices[slot] < indices[partner]
                    || (indices[slot] == indices[partner] && distances[slot] < distances[partner]);
                bool swap_pair = ascending ? partner_less : slot_less;
                if (swap_pair)
                {
                    float dist_tmp = distances[slot];
                    distances[slot] = distances[partner];
                    distances[partner] = dist_tmp;
                    int idx_tmp = indices[slot];
                    indices[slot] = indices[partner];
                    indices[partner] = idx_tmp;
                }
            }
            __syncthreads();
        }
    }
}

template<int K, int BLOCK_SIZE, int WORK_SIZE, int MERGE_SIZE, int STACK_SIZE>
__device__ __forceinline__
int _periodic_topk (const morton_view &view, const float3 &query, float x,
    float radius, float x_min, float x_max,
    float *work_dist, int *work_idx, float *merge_dist, int *merge_idx, int *node_stack,
    int &stack_size, int &idx_node, int &batch_count,
    unsigned int &leaves_visited, unsigned int &candidates_examined,
    unsigned int &stack_overflow, unsigned int &overflow_total)
{
    static_assert(3*K <= MERGE_SIZE, "periodic merge array cannot hold three top-K lists");
    static_assert((MERGE_SIZE & (MERGE_SIZE - 1)) == 0, "periodic merge size must be a power of two");

    if (threadIdx.x == 0) overflow_total = 0;
    __syncthreads();

    float width = x_max - x_min;
    bool periodic = width < 2.0f*CUDART_PI_F - 1.0e-6f;
    bool lower_image = periodic && _get_seam_dist(query, x - x_min) <= radius;
    bool upper_image = periodic && _get_seam_dist(query, x_max - x) <= radius;
    int image_count = 0;

    _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
        view, query, radius, work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );
    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        merge_dist[image_count*K + idx_neighbor] = work_dist[idx_neighbor];
        merge_idx[image_count*K + idx_neighbor] = work_idx[idx_neighbor];
    }
    if (threadIdx.x == 0) overflow_total += stack_overflow;
    image_count++;
    __syncthreads();

    if (lower_image)
    {
        float3 image_query = _rotate_query_z(query, width);
        _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
            view, image_query, radius, work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
            leaves_visited, candidates_examined, stack_overflow
        );
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist[image_count*K + idx_neighbor] = work_dist[idx_neighbor];
            merge_idx[image_count*K + idx_neighbor] = work_idx[idx_neighbor];
        }
        if (threadIdx.x == 0) overflow_total += stack_overflow;
        image_count++;
        __syncthreads();
    }

    if (upper_image)
    {
        float3 image_query = _rotate_query_z(query, -width);
        _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
            view, image_query, radius, work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
            leaves_visited, candidates_examined, stack_overflow
        );
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist[image_count*K + idx_neighbor] = work_dist[idx_neighbor];
            merge_idx[image_count*K + idx_neighbor] = work_idx[idx_neighbor];
        }
        if (threadIdx.x == 0) overflow_total += stack_overflow;
        image_count++;
        __syncthreads();
    }

    // the ordinary one-image result is already sorted and contains no duplicate identifiers
    if (image_count == 1) return image_count;

    // disjoint query balls cannot contain two images of one physical particle, so merge directly
    float3 first_image = _rotate_query_z(query, width);
    float image_dist_sq = _get_morton_dist_sq(query, first_image);
    if (image_count == 2 && image_dist_sq > 4.0f*radius*radius)
    {
        if (threadIdx.x == 0)
        {
            int idx_a = 0;
            int idx_b = K;
            for (int idx_out = 0; idx_out < K; idx_out++)
            {
                bool take_a = idx_a < K && (idx_b >= 2*K
                    || _morton_neighbor_less(merge_dist[idx_a], merge_idx[idx_a],
                        merge_dist[idx_b], merge_idx[idx_b]));
                int idx_in = take_a ? idx_a++ : idx_b++;
                work_dist[idx_out] = merge_dist[idx_in];
                work_idx[idx_out] = merge_idx[idx_in];
            }
        }
        __syncthreads();
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist[idx_neighbor] = work_dist[idx_neighbor];
            merge_idx[idx_neighbor] = work_idx[idx_neighbor];
        }
        __syncthreads();
        return image_count;
    }

    for (int slot = image_count*K + threadIdx.x; slot < MERGE_SIZE; slot += BLOCK_SIZE)
    {
        merge_dist[slot] = CUDART_INF_F;
        merge_idx[slot] = INT_MAX;
    }
    __syncthreads();

    // group equal stable identifiers and retain the closest periodic image of each particle
    _periodic_id_sort<MERGE_SIZE, BLOCK_SIZE>(merge_dist, merge_idx);
    constexpr int slots_per_thread = (MERGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool duplicate[slots_per_thread];
    int local_slot = 0;
    for (int slot = threadIdx.x; slot < MERGE_SIZE; slot += BLOCK_SIZE)
    {
        duplicate[local_slot++] = slot > 0 && merge_idx[slot] == merge_idx[slot - 1];
    }
    __syncthreads();
    local_slot = 0;
    for (int slot = threadIdx.x; slot < MERGE_SIZE; slot += BLOCK_SIZE)
    {
        if (duplicate[local_slot++])
        {
            merge_dist[slot] = CUDART_INF_F;
            merge_idx[slot] = INT_MAX;
        }
    }
    __syncthreads();

    _morton_pair_sort<MERGE_SIZE, BLOCK_SIZE>(merge_dist, merge_idx);
    return image_count;
}

template<int K, int BLOCK_SIZE = 256, int WORK_SIZE = 512, int MERGE_SIZE = 1024, int STACK_SIZE = 256>
__global__
void periodic_morton_query (int *neighbor_idx, float *neighbor_dist, unsigned int *stack_overflows,
    unsigned int *image_counts, const float3 *queries, const float *query_x, int query_count,
    morton_view view, float radius, float x_min, float x_max)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist[WORK_SIZE];
    __shared__ int work_idx[WORK_SIZE];
    __shared__ float merge_dist[MERGE_SIZE];
    __shared__ int merge_idx[MERGE_SIZE];
    __shared__ int node_stack[STACK_SIZE];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;
    __shared__ unsigned int overflow_total;

    int image_count = _periodic_topk<K, BLOCK_SIZE, WORK_SIZE, MERGE_SIZE, STACK_SIZE>(
        view, queries[idx_query], query_x[idx_query], radius, x_min, x_max,
        work_dist, work_idx, merge_dist, merge_idx, node_stack,
        stack_size, idx_node, batch_count, leaves_visited, candidates_examined,
        stack_overflow, overflow_total
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        int idx_out = idx_query*K + idx_neighbor;
        neighbor_idx[idx_out] = (merge_idx[idx_neighbor] == INT_MAX) ? -1 : merge_idx[idx_neighbor];
        neighbor_dist[idx_out] = merge_dist[idx_neighbor];
    }
    if (threadIdx.x == 0)
    {
        if (stack_overflows) stack_overflows[idx_query] = overflow_total;
        if (image_counts) image_counts[idx_query] = image_count;
    }
}

template<int K, int BLOCK_SIZE = 256, int WORK_SIZE = 512, int MERGE_SIZE = 1024, int STACK_SIZE = 256>
__global__
void periodic_morton_checksum (double *checksum, unsigned int *stack_overflows,
    unsigned int *image_counts, const float3 *queries, const float *query_x, int query_count,
    morton_view view, float radius, float x_min, float x_max)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist[WORK_SIZE];
    __shared__ int work_idx[WORK_SIZE];
    __shared__ float merge_dist[MERGE_SIZE];
    __shared__ int merge_idx[MERGE_SIZE];
    __shared__ int node_stack[STACK_SIZE];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;
    __shared__ unsigned int overflow_total;

    int image_count = _periodic_topk<K, BLOCK_SIZE, WORK_SIZE, MERGE_SIZE, STACK_SIZE>(
        view, queries[idx_query], query_x[idx_query], radius, x_min, x_max,
        work_dist, work_idx, merge_dist, merge_idx, node_stack,
        stack_size, idx_node, batch_count, leaves_visited, candidates_examined,
        stack_overflow, overflow_total
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            if (merge_idx[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(merge_dist[idx_neighbor])
                + 1.0e-12*static_cast<double>(merge_idx[idx_neighbor]);
        }
        checksum[idx_query] = value;
        if (stack_overflows) stack_overflows[idx_query] = overflow_total;
        if (image_counts) image_counts[idx_query] = image_count;
    }
}

#endif // QAV_PERIODIC_QUERY_CUH
