#ifdef COLLISION

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

// =========================================================================================================================
// kernel: col_event_run
// sample and apply at most one frozen-rate Bernoulli collision event per representative particle
//
// parallelization: one KD-tree thread or one Morton block with cooperative search and pair evaluation
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const real *dev_col_dist, const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0,
    real dt_col
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_col_tree[idx_tree].image != 0) return;

    int idx_old_i = dev_col_tree[idx_tree].index_old;
    real col_rate_i = dev_col_rate[idx_old_i];
    if (col_rate_i <= 0.0) return;

    curs rngstate = dev_rngstate[idx_old_i];
    real event_prob = -expm1(-col_rate_i*dt_col);
    if (curand_uniform_double(&rngstate) > event_prob)
    {
        dev_rngstate[idx_old_i] = rngstate;
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    bool deduplicate = image_dist_min >= 0.0f && image_dist_min <= 2.0f*max_search_dist;
    candidatelist query_result(max_search_dist, dev_col_tree, deduplicate);
    kdtree::cct::knn <candidatelist, kdtree_node, kdtree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_T
    );

    real measure = _get_ball_measure(y, z, dev_col_dist[idx_old_i]);
    #ifdef COLLISION_UNIT_VOLUME
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME
    if (measure <= 0.0) return;

    real target = col_rate_i*curand_uniform_double(&rngstate);
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_candidate = query_result.returnIndex(idx_neighbor);
        if (idx_candidate < 0) continue;
        if (idx_candidate == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_candidate].position.y, dev_particle[idx_candidate].position.z
        )) continue;

        idx_old_j = idx_candidate;
        cumulative += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, lambda_0
        ) / measure;
        if (cumulative >= target) break;
    }
    if (idx_old_j < 0) return;

    #ifdef MULTISIZE
    real vrel = 0.0;
    if (COAG_KERNEL == CUSTOM_KERNEL)
    {
        vrel = _get_vrel(dev_particle, dev_size_old, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
    }

    real size_i = dev_size_old[idx_old_i];
    real size_j = dev_size_old[idx_old_j];
    real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);

    if (vrel <= V_FRAG)
    {
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i
            / (size_k*size_k*size_k);
    }
    else
    {
        real frag_sample = curand_uniform_double(&rngstate);
        size_k = fmax(INIT_SMIN, size_k*frag_sample*frag_sample);
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i
            / (size_k*size_k*size_k);
    }
    #endif // MULTISIZE

    dev_rngstate[idx_old_i] = rngstate;
}
#else  // COLLISION_MORTON
__global__
void col_event_run (swarm *dev_particle, curs *dev_rngstate,
    const real *dev_col_rate, const real *dev_col_dist, unsigned int *dev_col_overflow,
    const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_col_point, morton_view col_morton, bool duplicate_safe,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col
)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    __shared__ bool run_event;
    __shared__ curs rngstate;
    if (threadIdx.x == 0)
    {
        dev_col_overflow[idx_old_i] = 0;
        real col_rate_i = dev_col_rate[idx_old_i];
        run_event = col_rate_i > 0.0;
        if (run_event)
        {
            rngstate = dev_rngstate[idx_old_i];
            real event_prob = -expm1(-col_rate_i*dt_col);
            run_event = curand_uniform_double(&rngstate) <= event_prob;
            if (!run_event) dev_rngstate[idx_old_i] = rngstate;
        }
    }
    __syncthreads();
    if (!run_event) return;

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
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

    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        col_morton, dev_col_point[idx_old_i], max_search_dist, duplicate_safe,
        work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    if (threadIdx.x == 0)
    {
        dev_col_overflow[idx_old_i] = stack_overflow;
        if (stack_overflow != 0) dev_rngstate[idx_old_i] = rngstate;
    }
    __syncthreads();
    if (stack_overflow != 0) return;

    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        int idx_candidate = work_idx[idx_neighbor];
        if (idx_candidate < 0 || idx_candidate == INT_MAX || idx_candidate == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_candidate].position.y, dev_particle[idx_candidate].position.z
        )) continue;

        pair_rate[idx_neighbor] = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_candidate, lambda_0
        );
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    real measure = _get_ball_measure(y, z, dev_col_dist[idx_old_i]);
    #ifdef COLLISION_UNIT_VOLUME
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME
    if (measure <= 0.0) return;

    real target = dev_col_rate[idx_old_i]*curand_uniform_double(&rngstate);
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_candidate = work_idx[idx_neighbor];
        if (idx_candidate < 0 || idx_candidate == INT_MAX || idx_candidate == idx_old_i) continue;
        if (!_is_particle_active(
            dev_particle[idx_candidate].position.y, dev_particle[idx_candidate].position.z
        )) continue;

        idx_old_j = idx_candidate;
        cumulative += pair_rate[idx_neighbor] / measure;
        if (cumulative >= target) break;
    }
    if (idx_old_j < 0) return;

    #ifdef MULTISIZE
    real vrel = 0.0;
    if (COAG_KERNEL == CUSTOM_KERNEL)
    {
        vrel = _get_vrel(dev_particle, dev_size_old, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
    }

    real size_i = dev_size_old[idx_old_i];
    real size_j = dev_size_old[idx_old_j];
    real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);
    if (vrel <= V_FRAG)
    {
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i
            / (size_k*size_k*size_k);
    }
    else
    {
        real frag_sample = curand_uniform_double(&rngstate);
        size_k = fmax(INIT_SMIN, size_k*frag_sample*frag_sample);
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i
            / (size_k*size_k*size_k);
    }
    #endif // MULTISIZE

    dev_rngstate[idx_old_i] = rngstate;
}
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // COLLISION
