#ifdef COLLISION

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <periodic_query.cuh>
#endif // COLLISION_MORTON

// =========================================================================================================================
// kernel: col_rate_calc
// calculate each representative particle's total local collision propensity from its K nearest neighbors
//
// parallelization: one KD query thread or one cooperative Morton block per representative
//
// per call:
//   1 query neighbors within the scale-height search radius
//   2 sum pair propensity numerators while excluding the representative itself
//   3 divide by the accessible KNN measure and expose the rate for the host reduction
// =========================================================================================================================

__device__ __forceinline__
unsigned long long _get_neighbor_hash (unsigned int index)
{
    unsigned long long value = static_cast<unsigned long long>(index) + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30))*0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27))*0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

#ifdef COLLISION_KDTREE
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist,
    unsigned long long *dev_col_hash, unsigned int *dev_col_count, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const tree *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, bool make_probe
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_col_tree[idx_tree].image != 0) return;
    
    int idx_old_i = dev_col_tree[idx_tree].index_old;
    
    dev_col_rate[idx_old_i] = 0.0;
    dev_col_dist[idx_old_i] = 0.0;
    if (make_probe)
    {
        dev_col_hash[idx_old_i] = 0;
        dev_col_count[idx_old_i] = 0;
    }

    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    if (!_is_particle_active(y, z)) return;

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    // limit the KNN query to a configured fraction of the local gas scale height
    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);
        
    candidatelist query_result(max_search_dist, dev_col_tree);
    cukd::cct::knn <candidatelist, tree, tree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_T
    );

    real col_rate_i = 0.0; // total collision rate for the representative particle
    float max_dist_sq = 0.0f;
    unsigned long long neighbor_hash = 0;
    unsigned int neighbor_count = 0;

    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        real col_rate_ij = 0.0;
        int idx_query = query_result.returnIndex(idx_neighbor);

        if (idx_query != -1)
        {
            int idx_old_j = dev_col_tree[idx_query].index_old;
            if (idx_old_j == idx_old_i) continue; // skip self-collision
            if (!_is_particle_active(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            )) continue;

            if (make_probe)
            {
                neighbor_hash += _get_neighbor_hash(static_cast<unsigned int>(idx_old_j));
                neighbor_count++;
            }
            float dist_sq = query_result.returnDist2(idx_neighbor);
            max_dist_sq = fmaxf(max_dist_sq, dist_sq);

            col_rate_ij = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
                dev_particle, dev_size_old, dev_numr_old,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                idx_old_i, idx_old_j, lambda_0
            );
        }

        col_rate_i += col_rate_ij;
    }

    // normalize by the accessible measure of the smallest ball containing the returned neighbors
    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);
    #ifdef COLLISION_UNIT_VOLUME
    // bypass KNN geometry only for dimensionless analytic kernel tests
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME
    col_rate_i = (measure > 0.0) ? col_rate_i / measure : 0.0;

    // retain the neighborhood radius and total rate for event execution
    dev_col_dist[idx_old_i] = radius;
    dev_col_rate[idx_old_i] = col_rate_i;
    if (make_probe)
    {
        dev_col_hash[idx_old_i] = neighbor_hash;
        dev_col_count[idx_old_i] = neighbor_count;
    }
}
#else  // COLLISION_MORTON
__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist,
    unsigned long long *dev_col_hash, unsigned int *dev_col_count, unsigned int *dev_col_overflow,
    const swarm *dev_particle, const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_col_point, adaptive_morton_view col_morton,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, bool make_probe
)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    if (threadIdx.x == 0)
    {
        dev_col_rate[idx_old_i] = 0.0;
        dev_col_dist[idx_old_i] = 0.0;
        if (make_probe)
        {
            dev_col_hash[idx_old_i] = 0;
            dev_col_count[idx_old_i] = 0;
        }
        dev_col_overflow[idx_old_i] = 0;
    }

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

    __shared__ float work_dist[512];
    __shared__ int work_idx[512];
    __shared__ float merge_dist[1024];
    __shared__ int merge_idx[1024];
    __shared__ int node_stack[256];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;
    __shared__ unsigned int overflow_total;

    _periodic_topk<N_K, MORTON_TPB, 512, 1024, 256>(
        col_morton, dev_col_point[idx_old_i], static_cast<float>(x),
        max_search_dist, static_cast<float>(X_MIN), static_cast<float>(X_MAX),
        work_dist, work_idx, merge_dist, merge_idx, node_stack,
        stack_size, idx_node, batch_count, leaves_visited, candidates_examined,
        stack_overflow, overflow_total
    );

    if (threadIdx.x != 0) return;

    dev_col_overflow[idx_old_i] = overflow_total;
    if (overflow_total != 0) return;

    real col_rate_i = 0.0;
    float max_dist_sq = 0.0f;
    unsigned long long neighbor_hash = 0;
    unsigned int neighbor_count = 0;

    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = merge_idx[idx_neighbor];
        if (idx_old_j < 0 || idx_old_j == INT_MAX || idx_old_j == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

        if (make_probe)
        {
            neighbor_hash += _get_neighbor_hash(static_cast<unsigned int>(idx_old_j));
            neighbor_count++;
        }
        max_dist_sq = fmaxf(max_dist_sq, merge_dist[idx_neighbor]);
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
    if (make_probe)
    {
        dev_col_hash[idx_old_i] = neighbor_hash;
        dev_col_count[idx_old_i] = neighbor_count;
    }
}
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // COLLISION
