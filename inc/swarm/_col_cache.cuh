#ifndef SWARM_COL_CACHE_CUH
#define SWARM_COL_CACHE_CUH

#if defined(COLLISION) && (!defined(BERNOULLI) || defined(KNN_CACHE))

#include <climits>  // INT_MAX
#include <cstddef>  // std::size_t

#include <_collision.cuh>
#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

__host__ __device__ __forceinline__
std::size_t _get_col_offset (int idx_owner, int idx_neighbor)
{
    return static_cast<std::size_t>(idx_owner)*static_cast<std::size_t>(N_K)
        + static_cast<std::size_t>(idx_neighbor);
}

// cache the fixed physical top-K neighborhood for one geometry epoch
#ifdef COLLISION_KDTREE
__global__
void col_cache_get (int *dev_col_neighbor, real *dev_col_measure,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    const unsigned char *dev_col_active, const swarm *dev_particle,
    float image_dist_min)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_kdtree_node[idx_tree].image != 0) return;

    int idx_old_i = dev_kdtree_node[idx_tree].idx_old;
    if (dev_col_active[idx_old_i] == 0)
    {
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
        {
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        }
        dev_col_measure[idx_old_i] = 0.0;
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // deduplicate overlapping wedge images using the same physical-id heap as direct search
    bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*search_dist;
#ifdef GAMEDEV_ROCM
    // Keep mutable private keys in their own allocation so heap control fields can be scalarized.
    unsigned long long private_keys[N_K];
    using cache_heap = idx_old_heap<N_K,kdtree_node,true>;
    cache_heap near_result(search_dist,dev_kdtree_node,!unique_ids,dev_col_active,private_keys);
#else
    using cache_heap = kdtree_heap;
    kdtree_heap near_result(search_dist, dev_kdtree_node, !unique_ids, dev_col_active);
#endif
    kdtree::cct::knn <cache_heap, kdtree_node, kdtree_traits> (
        near_result, dev_kdtree_node[idx_tree].cartesian,
        *dev_kdtree_box, dev_kdtree_node, N_T
    );

    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = near_result.returnIndex(idx_neighbor);
        dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)]
            = near_result.returnNeighbor(idx_neighbor);
        if (idx_old_j >= 0)
            max_dist_sq = fmaxf(max_dist_sq, near_result.returnDist2(idx_neighbor));
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);
    dev_col_measure[idx_old_i] = measure;
}
#else  // COLLISION_MORTON
__global__
void col_cache_get (int *dev_col_neighbor, real *dev_col_measure,
    unsigned int *dev_morton_overflow, const float3 *dev_morton_point,
    const unsigned char *dev_col_active, const swarm *dev_particle,
    morton_view morton_data, bool unique_ids)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    if (dev_col_active[idx_old_i] == 0)
    {
        for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
        {
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        }
        if (threadIdx.x == 0)
        {
            dev_col_measure[idx_old_i] = 0.0;
            dev_morton_overflow[idx_old_i] = 0;
        }
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // No periodic images exist in a full disk. Compile only the required top-K path.
    constexpr int fast_work = [] { int n=1; while(n<2*N_K || n<N_K+MORTON_TPB) n*=2; return n; }();
    constexpr int query_work = X_WEDGE ? MORTON_WORK_SIZE : fast_work;
    __shared__ float work_dist_sq[query_work];
    __shared__ int work_idx_old[query_work];
    __shared__ int idx_node_stack[256];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;

    if constexpr (!X_WEDGE) {
        _morton_topk<N_K, MORTON_TPB, fast_work, 256>(
            morton_data, dev_morton_point[idx_old_i], search_dist,
            work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
            leaf_visit_count, candidate_count, stack_overflow, dev_col_active, COL_IMAGE_COUNT);
    } else
    {
    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        morton_data, dev_morton_point[idx_old_i], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow, dev_col_active
    );
    }

    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        int neighbor = work_idx_old[idx_neighbor];
        dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)]
            = (neighbor == INT_MAX) ? -1 : neighbor;
    }
    if (threadIdx.x == 0)
    {
        float max_dist_sq = 0.0f;
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
        {
            if (work_idx_old[idx_neighbor] != INT_MAX)
                max_dist_sq = fmaxf(max_dist_sq, work_dist_sq[idx_neighbor]);
        }
        real radius = sqrt(static_cast<real>(max_dist_sq));
        real measure = _get_ball_measure(y, z, radius);
        dev_col_measure[idx_old_i] = measure;
        dev_morton_overflow[idx_old_i] = stack_overflow;
    }
}
#endif // COLLISION_KDTREE


#endif // COLLISION && (FROZEN_BATH || KNN_CACHE)

#endif // SWARM_COL_CACHE_CUH
