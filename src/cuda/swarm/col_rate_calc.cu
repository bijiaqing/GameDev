#ifdef COLLISION

#ifdef KNN_CACHE
#include <cstddef>  // std::size_t
#endif // KNN_CACHE

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

// identify a nonfinite collision propensity or KNN-ball radius before event sampling
__global__
void inf_rate_flag (const real *dev_col_rate, const real *dev_col_dist, int *dev_bad_part)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    if (!isfinite(dev_col_rate[idx]) || !isfinite(dev_col_dist[idx]))
        atomicCAS(dev_bad_part, 0, idx + 1);
}

// =========================================================================================================================
// kernel: col_rate_calc
// calculate each representative particle's total local collision propensity from its K nearest neighbors
//
// parallelization: one KD-tree thread or one Morton block with cooperative search and pair evaluation
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist, const swarm *dev_particle,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_kdtree_node[idx_tree].image != 0) return;

    int idx_old_i = dev_kdtree_node[idx_tree].idx_old;
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

    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // query the local physical top-K set while deduplicating overlapping periodic images when required
    bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*search_dist;
    kdtree_heap near_result(search_dist, dev_kdtree_node, !unique_ids, dev_col_active);
    kdtree::cct::knn <kdtree_heap, kdtree_node, kdtree_traits> (
        near_result, dev_kdtree_node[idx_tree].cartesian,
        *dev_kdtree_box, dev_kdtree_node, N_T
    );

    // sum pair propensities and use the farthest retained neighbor as the KNN-ball radius
    real col_rate_i = 0.0;
    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = near_result.returnIndex(idx_neighbor);
        int image_j = near_result.returnImage(idx_neighbor);
        if (idx_old_j < 0) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        max_dist_sq = fmaxf(max_dist_sq, near_result.returnDist2(idx_neighbor));
        col_rate_i += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0
        );
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);

    dev_col_dist[idx_old_i] = radius;
    dev_col_rate[idx_old_i] = (measure > 0.0) ? col_rate_i / measure : 0.0;
}
#else  // COLLISION_MORTON
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist, unsigned int *dev_morton_overflow,
    const swarm *dev_particle, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_morton_point, morton_view morton_data, bool unique_ids,
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
        dev_morton_overflow[idx_old_i] = 0;
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

    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    __shared__ float work_dist_sq[MORTON_WORK_SIZE];
    __shared__ int work_idx_old[MORTON_WORK_SIZE];
    __shared__ int idx_node_stack[256];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;
    __shared__ real pair_rate[N_K];
    __shared__ float pair_dist_sq[N_K];

    // query and deduplicate the periodic Morton top-K set cooperatively
    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        morton_data, dev_morton_point[idx_old_i], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow, dev_col_active
    );

    if (threadIdx.x == 0) dev_morton_overflow[idx_old_i] = stack_overflow;
    __syncthreads();
    if (stack_overflow != 0) return;

    // evaluate retained pair propensities in parallel before the deterministic reduction
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        pair_dist_sq[idx_neighbor] = 0.0f;
        int neighbor = work_idx_old[idx_neighbor];
        if (neighbor < 0 || neighbor == INT_MAX) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        pair_dist_sq[idx_neighbor] = work_dist_sq[idx_neighbor];
        pair_rate[idx_neighbor] = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0
        );
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    real col_rate_i = 0.0;
    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        max_dist_sq = fmaxf(max_dist_sq, pair_dist_sq[idx_neighbor]);
        col_rate_i += pair_rate[idx_neighbor];
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);

    dev_col_dist[idx_old_i] = radius;
    dev_col_rate[idx_old_i] = (measure > 0.0) ? col_rate_i / measure : 0.0;
}
#endif // COLLISION_KDTREE

#ifdef KNN_CACHE

// =========================================================================================================================
// kernel overload: col_rate_calc
// recalculate frozen Bernoulli rates from the persistent physical-neighbor cache
//
// parallelization: one KD-tree thread or one Morton block per representative particle
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_rate_calc (real *dev_col_rate, const swarm *dev_particle,
    const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0)
{
    int idx_old_i = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_old_i >= N_P) return;

    dev_col_rate[idx_old_i] = 0.0;
    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);
    real measure = dev_col_measure[idx_old_i];
    if (dev_col_active[idx_old_i] == 0 || !_is_in_bounds(loc_x, loc_y, loc_z)
        || !(measure > 0.0)) return;

    real col_rate_i = 0.0;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        std::size_t idx_cache = static_cast<std::size_t>(idx_old_i)*N_K + idx_neighbor;
        int neighbor = dev_col_neighbor[idx_cache];
        if (neighbor < 0) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        col_rate_i += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0
        );
    }
    dev_col_rate[idx_old_i] = col_rate_i / measure;
}
#else  // COLLISION_MORTON
__global__
void col_rate_calc (real *dev_col_rate, const swarm *dev_particle,
    const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    if (threadIdx.x == 0) dev_col_rate[idx_old_i] = 0.0;
    __syncthreads();

    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);
    real measure = dev_col_measure[idx_old_i];
    if (dev_col_active[idx_old_i] == 0 || !_is_in_bounds(loc_x, loc_y, loc_z)
        || !(measure > 0.0)) return;

    __shared__ real pair_rate[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        std::size_t idx_cache = static_cast<std::size_t>(idx_old_i)*N_K + idx_neighbor;
        int neighbor = dev_col_neighbor[idx_cache];
        if (neighbor < 0) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        pair_rate[idx_neighbor] = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0
        );
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    real col_rate_i = 0.0;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        col_rate_i += pair_rate[idx_neighbor];
    }
    dev_col_rate[idx_old_i] = col_rate_i / measure;
}
#endif // COLLISION_KDTREE

#endif // KNN_CACHE

// =========================================================================================================================

#endif // COLLISION
