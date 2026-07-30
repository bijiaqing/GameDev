#ifdef COLLISION

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

// =========================================================================================================================
// kernel: col_rate_calc
// calculate each representative particle's total local collision propensity from its K nearest neighbors
//
// parallelization: one KD-tree thread or one Morton block with cooperative search and pair evaluation
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_col_tree[idx_tree].image != 0) return;

    int idx_old_i = dev_col_tree[idx_tree].index_old;
    dev_col_rate[idx_old_i] = 0.0;
    dev_col_dist[idx_old_i] = 0.0;

    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    if (!_is_particle_active(y, z)) return;

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);
    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);
    bool deduplicate = image_dist_min >= 0.0f && image_dist_min <= 2.0f*max_search_dist;
    candidatelist query_result(max_search_dist, dev_col_tree, deduplicate);
    kdtree::cct::knn <candidatelist, kdtree_node, kdtree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_T
    );

    real col_rate_i = 0.0;
    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = query_result.returnIndex(idx_neighbor);
        if (idx_old_j < 0) continue;
        if (idx_old_j == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        max_dist_sq = fmaxf(max_dist_sq, query_result.returnDist2(idx_neighbor));
        col_rate_i += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, lambda_0
        );
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);
    #ifdef COLLISION_UNIT_VOLUME
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME

    dev_col_dist[idx_old_i] = radius;
    dev_col_rate[idx_old_i] = (measure > 0.0) ? col_rate_i / measure : 0.0;
}
#else  // COLLISION_MORTON
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist, unsigned int *dev_col_overflow,
    const swarm *dev_particle, const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_col_point, morton_view col_morton, bool duplicate_safe,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0
)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    if (threadIdx.x == 0)
    {
        dev_col_rate[idx_old_i] = 0.0;
        dev_col_dist[idx_old_i] = 0.0;
        dev_col_overflow[idx_old_i] = 0;
    }
    __syncthreads();

    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    if (!_is_particle_active(y, z)) return;

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);
    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    __shared__ float work_dist[MORTON_WORK_SIZE];
    __shared__ int work_idx[MORTON_WORK_SIZE];
    __shared__ int node_stack[256];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;
    __shared__ real pair_rate[N_K];
    __shared__ float pair_dist[N_K];

    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        col_morton, dev_col_point[idx_old_i], max_search_dist, duplicate_safe,
        work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    if (threadIdx.x == 0) dev_col_overflow[idx_old_i] = stack_overflow;
    __syncthreads();
    if (stack_overflow != 0) return;

    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        pair_dist[idx_neighbor] = 0.0f;
        int idx_old_j = work_idx[idx_neighbor];
        if (idx_old_j < 0 || idx_old_j == INT_MAX || idx_old_j == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        pair_dist[idx_neighbor] = work_dist[idx_neighbor];
        pair_rate[idx_neighbor] = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, lambda_0
        );
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    real col_rate_i = 0.0;
    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        max_dist_sq = fmaxf(max_dist_sq, pair_dist[idx_neighbor]);
        col_rate_i += pair_rate[idx_neighbor];
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);
    #ifdef COLLISION_UNIT_VOLUME
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME

    dev_col_dist[idx_old_i] = radius;
    dev_col_rate[idx_old_i] = (measure > 0.0) ? col_rate_i / measure : 0.0;
}
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // COLLISION
