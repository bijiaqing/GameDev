#ifndef QAV_PERIODIC_QUERY_CUH
#define QAV_PERIODIC_QUERY_CUH

#include <climits>        // INT_MAX

#include <hip/hip_runtime.h> // float3 and HIP device intrinsics

#include <morton/morton_index.cuh>

// rotate one Cartesian query into an adjacent azimuthal wedge image
__device__ __forceinline__
float3 _rotate_query_z (const float3 &query_point, float angle)
{
    float sin_angle;
    float cos_angle;
    sincosf(angle, &sin_angle, &cos_angle);
    return make_float3(
        cos_angle*query_point.x - sin_angle*query_point.y,
        sin_angle*query_point.x + cos_angle*query_point.y,
        query_point.z
    );
}

// calculate the shortest Cartesian distance to one radial seam plane
__device__ __forceinline__
float _get_seam_dist (const float3 &query_point, float x_offset)
{
    float R = hypotf(query_point.x, query_point.y);
    float cos_offset = cosf(x_offset);
    return (cos_offset >= 0.0f) ? R*fabsf(sinf(x_offset)) : R;
}

// group equal physical identifiers before retaining their nearest periodic image
template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _periodic_id_sort (float *dist_sq, int *idx_old)
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
                bool partner_less = idx_old[idx_partner] < idx_old[idx_slot]
                    || (idx_old[idx_partner] == idx_old[idx_slot] && dist_sq[idx_partner] < dist_sq[idx_slot]);
                bool slot_less = idx_old[idx_slot] < idx_old[idx_partner]
                    || (idx_old[idx_slot] == idx_old[idx_partner] && dist_sq[idx_slot] < dist_sq[idx_partner]);
                bool swap_pair = ascending ? partner_less : slot_less;
                if (swap_pair)
                {
                    float dist_tmp = dist_sq[idx_slot];
                    dist_sq[idx_slot] = dist_sq[idx_partner];
                    dist_sq[idx_partner] = dist_tmp;
                    int idx_tmp = idx_old[idx_slot];
                    idx_old[idx_slot] = idx_old[idx_partner];
                    idx_old[idx_partner] = idx_tmp;
                }
            }
            __syncthreads();
        }
    }
}

// query only geometrically reachable wedge images and merge them into one physical top-K set
template<int K, int BLOCK_SIZE, int WORK_SIZE, int MERGE_SIZE, int STACK_SIZE>
__device__ __forceinline__
int _periodic_topk (const morton_view &morton_data, const float3 &query_point, float x,
    float search_dist, float x_min, float x_max,
    float *work_dist_sq, int *work_idx_old, float *merge_dist_sq, int *merge_idx_old, int *idx_node_stack,
    int &stack_count, int &idx_node, int &batch_count,
    unsigned int &leaf_visit_count, unsigned int &candidate_count,
    unsigned int &stack_overflow, unsigned int &overflow_total)
{
    static_assert(3*K <= MERGE_SIZE, "periodic merge array cannot hold three top-K lists");
    static_assert((MERGE_SIZE & (MERGE_SIZE - 1)) == 0, "periodic merge size must be a power of two");

    if (threadIdx.x == 0) overflow_total = 0;
    __syncthreads();

    float width = x_max - x_min;
    bool periodic = width < 2.0f*MORTON_PI_F - 1.0e-6f;
    bool lower_image = periodic && _get_seam_dist(query_point, x - x_min) <= search_dist;
    bool upper_image = periodic && _get_seam_dist(query_point, x_max - x) <= search_dist;
    int image_count = 0;

    _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
        morton_data, query_point, search_dist, work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow
    );
    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        merge_dist_sq[image_count*K + idx_neighbor] = work_dist_sq[idx_neighbor];
        merge_idx_old[image_count*K + idx_neighbor] = work_idx_old[idx_neighbor];
    }
    if (threadIdx.x == 0) overflow_total += stack_overflow;
    image_count++;
    __syncthreads();

    if (lower_image)
    {
        float3 image_query = _rotate_query_z(query_point, width);
        _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
            morton_data, image_query, search_dist, work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
            leaf_visit_count, candidate_count, stack_overflow
        );
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist_sq[image_count*K + idx_neighbor] = work_dist_sq[idx_neighbor];
            merge_idx_old[image_count*K + idx_neighbor] = work_idx_old[idx_neighbor];
        }
        if (threadIdx.x == 0) overflow_total += stack_overflow;
        image_count++;
        __syncthreads();
    }

    if (upper_image)
    {
        float3 image_query = _rotate_query_z(query_point, -width);
        _morton_topk<K, BLOCK_SIZE, WORK_SIZE, STACK_SIZE>(
            morton_data, image_query, search_dist, work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
            leaf_visit_count, candidate_count, stack_overflow
        );
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist_sq[image_count*K + idx_neighbor] = work_dist_sq[idx_neighbor];
            merge_idx_old[image_count*K + idx_neighbor] = work_idx_old[idx_neighbor];
        }
        if (threadIdx.x == 0) overflow_total += stack_overflow;
        image_count++;
        __syncthreads();
    }

    // the ordinary one-image result contains no duplicate identifiers
    if (image_count == 1) return image_count;

    // disjoint query balls cannot contain two images of one physical particle, so merge directly
    float3 first_image = _rotate_query_z(query_point, width);
    float image_dist_sq = _get_morton_point_dist_sq(query_point, first_image);
    if (image_count == 2 && image_dist_sq > 4.0f*search_dist*search_dist)
    {
        if (threadIdx.x == 0)
        {
            int idx_a = 0;
            int idx_b = K;
            for (int idx_out = 0; idx_out < K; idx_out++)
            {
                bool take_a = idx_a < K && (idx_b >= 2*K
                    || _morton_neighbor_less(merge_dist_sq[idx_a], merge_idx_old[idx_a],
                        merge_dist_sq[idx_b], merge_idx_old[idx_b]));
                int idx_in = take_a ? idx_a++ : idx_b++;
                work_dist_sq[idx_out] = merge_dist_sq[idx_in];
                work_idx_old[idx_out] = merge_idx_old[idx_in];
            }
        }
        __syncthreads();
        for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
        {
            merge_dist_sq[idx_neighbor] = work_dist_sq[idx_neighbor];
            merge_idx_old[idx_neighbor] = work_idx_old[idx_neighbor];
        }
        __syncthreads();
        return image_count;
    }

    for (int idx_slot = image_count*K + threadIdx.x; idx_slot < MERGE_SIZE; idx_slot += BLOCK_SIZE)
    {
        merge_dist_sq[idx_slot] = MORTON_INF_F;
        merge_idx_old[idx_slot] = INT_MAX;
    }
    __syncthreads();

    // group equal original particle idx_old and retain the closest periodic image of each particle
    _periodic_id_sort<MERGE_SIZE, BLOCK_SIZE>(merge_dist_sq, merge_idx_old);
    constexpr int slots_per_thread = (MERGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bool duplicate[slots_per_thread];
    int idx_local = 0;
    for (int idx_slot = threadIdx.x; idx_slot < MERGE_SIZE; idx_slot += BLOCK_SIZE)
    {
        duplicate[idx_local++] = idx_slot > 0 && merge_idx_old[idx_slot] == merge_idx_old[idx_slot - 1];
    }
    __syncthreads();
    idx_local = 0;
    for (int idx_slot = threadIdx.x; idx_slot < MERGE_SIZE; idx_slot += BLOCK_SIZE)
    {
        if (duplicate[idx_local++])
        {
            merge_dist_sq[idx_slot] = MORTON_INF_F;
            merge_idx_old[idx_slot] = INT_MAX;
        }
    }
    __syncthreads();

    _morton_pair_sort<MERGE_SIZE, BLOCK_SIZE>(merge_dist_sq, merge_idx_old);
    return image_count;
}

template<int K, int BLOCK_SIZE = 256, int WORK_SIZE = 512, int MERGE_SIZE = 1024, int STACK_SIZE = 256>
__global__
void periodic_morton_query (int *dev_near_idx_old, float *dev_near_dist_sq,
    unsigned int *dev_stack_overflow_count, unsigned int *dev_image_count,
    const float3 *dev_query_point, const float *dev_query_x, int query_count,
    morton_view morton_data, float search_dist, float x_min, float x_max)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[WORK_SIZE];
    __shared__ int work_idx_old[WORK_SIZE];
    __shared__ float merge_dist_sq[MERGE_SIZE];
    __shared__ int merge_idx_old[MERGE_SIZE];
    __shared__ int idx_node_stack[STACK_SIZE];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;
    __shared__ unsigned int overflow_total;

    int image_count = _periodic_topk<K, BLOCK_SIZE, WORK_SIZE, MERGE_SIZE, STACK_SIZE>(
        morton_data, dev_query_point[idx_query], dev_query_x[idx_query], search_dist, x_min, x_max,
        work_dist_sq, work_idx_old, merge_dist_sq, merge_idx_old, idx_node_stack,
        stack_count, idx_node, batch_count, leaf_visit_count, candidate_count,
        stack_overflow, overflow_total
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        int idx_out = idx_query*K + idx_neighbor;
        dev_near_idx_old[idx_out] = (merge_idx_old[idx_neighbor] == INT_MAX) ? -1 : merge_idx_old[idx_neighbor];
        dev_near_dist_sq[idx_out] = merge_dist_sq[idx_neighbor];
    }
    if (threadIdx.x == 0)
    {
        if (dev_stack_overflow_count) dev_stack_overflow_count[idx_query] = overflow_total;
        if (dev_image_count) dev_image_count[idx_query] = image_count;
    }
}

#endif // QAV_PERIODIC_QUERY_CUH
