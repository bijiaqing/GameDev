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

// =========================================================================================================================
// kernel: col_event_run
// sample and apply at most one frozen-rate Bernoulli collision event per representative particle
//
// parallelization: one KD-tree thread or one Morton block with cooperative search and pair evaluation
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const real *dev_col_dist, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    float image_dist_min,
    real lambda_0,
    real dt_col
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_kdtree_node[idx_tree].image != 0) return;

    int idx_old_i = dev_kdtree_node[idx_tree].idx_old;
    real col_rate_i = dev_col_rate[idx_old_i];
    if (col_rate_i <= 0.0) return;

    // sample whether this representative experiences one event during the frozen-rate interval
    curs rngstate = dev_rngstate[idx_old_i];
    real event_prob = -expm1(-col_rate_i*dt_col);
    bool run_event = curand_uniform_double(&rngstate) <= event_prob;
    dev_rngstate[idx_old_i] = rngstate; // commit every consumed draw before any later terminal path
    if (!run_event) return;

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*search_dist;
    kdtree_heap near_result(search_dist, dev_kdtree_node, !unique_ids, dev_col_active);
    kdtree::cct::knn <kdtree_heap, kdtree_node, kdtree_traits> (
        near_result, dev_kdtree_node[idx_tree].cartesian,
        *dev_kdtree_box, dev_kdtree_node, N_T
    );

    real measure = _get_ball_measure(y, z, dev_col_dist[idx_old_i]);
    if (measure <= 0.0) return;

    // select the collision partner from cumulative pair propensity rather than neighbor rank
    real target = col_rate_i*curand_uniform_double(&rngstate);
    dev_rngstate[idx_old_i] = rngstate;
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j_try = near_result.returnIndex(idx_neighbor);
        if (idx_old_j_try < 0) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j_try].position.y, dev_particle[idx_old_j_try].position.z
        )) continue;

        idx_old_j = idx_old_j_try;
        cumulative += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j_try, lambda_0
        ) / measure;
        if (cumulative >= target) break;
    }
    if (idx_old_j < 0) return;

    #ifdef MULTISIZE
    // preserve represented mass while applying coagulation or sampled fragmentation
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
    const real *dev_col_rate, const real *dev_col_dist, unsigned int *dev_morton_overflow,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    const float3 *dev_morton_point, morton_view morton_data, bool unique_ids,
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

    // let one thread advance the particle RNG before launching the cooperative search
    if (threadIdx.x == 0)
    {
        dev_morton_overflow[idx_old_i] = 0;
        real col_rate_i = dev_col_rate[idx_old_i];
        run_event = col_rate_i > 0.0;
        if (run_event)
        {
            rngstate = dev_rngstate[idx_old_i];
            real event_prob = -expm1(-col_rate_i*dt_col);
            run_event = curand_uniform_double(&rngstate) <= event_prob;
            dev_rngstate[idx_old_i] = rngstate; // commit every consumed draw before any later terminal path
        }
    }
    __syncthreads();
    if (!run_event) return;

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
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

    // reconstruct the same frozen neighbor set used by the preceding rate calculation
    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        morton_data, dev_morton_point[idx_old_i], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow, dev_col_active
    );

    if (threadIdx.x == 0)
    {
        dev_morton_overflow[idx_old_i] = stack_overflow;
        if (stack_overflow != 0) dev_rngstate[idx_old_i] = rngstate;
    }
    __syncthreads();
    if (stack_overflow != 0) return;

    // evaluate pair propensities cooperatively before serial inverse-CDF selection
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        int idx_old_j = work_idx_old[idx_neighbor];
        if (idx_old_j < 0 || idx_old_j == INT_MAX) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

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

    real measure = _get_ball_measure(y, z, dev_col_dist[idx_old_i]);
    if (measure <= 0.0) return;

    real target = dev_col_rate[idx_old_i]*curand_uniform_double(&rngstate);
    dev_rngstate[idx_old_i] = rngstate;
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j_try = work_idx_old[idx_neighbor];
        if (idx_old_j_try < 0 || idx_old_j_try == INT_MAX) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j_try].position.y, dev_particle[idx_old_j_try].position.z
        )) continue;

        idx_old_j = idx_old_j_try;
        cumulative += pair_rate[idx_neighbor] / measure;
        if (cumulative >= target) break;
    }
    if (idx_old_j < 0) return;

    #ifdef MULTISIZE
    // preserve represented mass while applying coagulation or sampled fragmentation
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

#ifdef KNN_CACHE

// =========================================================================================================================
// kernel overload: col_event_run
// sample frozen Bernoulli events and select partners from the persistent physical-neighbor cache
//
// parallelization: one KD-tree thread or one Morton block per representative particle
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col)
{
    int idx_old_i = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_old_i >= N_P) return;

    real col_rate_i = dev_col_rate[idx_old_i];
    if (col_rate_i <= 0.0) return;

    curs rngstate = dev_rngstate[idx_old_i];
    real event_prob = -expm1(-col_rate_i*dt_col);
    bool run_event = curand_uniform_double(&rngstate) <= event_prob;
    dev_rngstate[idx_old_i] = rngstate;
    if (!run_event) return;

    real measure = dev_col_measure[idx_old_i];
    if (!(measure > 0.0)) return;

    real target = col_rate_i*curand_uniform_double(&rngstate);
    dev_rngstate[idx_old_i] = rngstate;
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        std::size_t idx_cache = static_cast<std::size_t>(idx_old_i)*N_K + idx_neighbor;
        int idx_old_j_try = dev_col_neighbor[idx_cache];
        if (idx_old_j_try < 0) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j_try].position.y, dev_particle[idx_old_j_try].position.z
        )) continue;

        idx_old_j = idx_old_j_try;
        cumulative += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j_try, lambda_0
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
void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate,
    const int *dev_col_neighbor, const real *dev_col_measure,
    const unsigned char *dev_col_active, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    __shared__ bool run_event;
    __shared__ curs rngstate;
    if (threadIdx.x == 0)
    {
        real col_rate_i = dev_col_rate[idx_old_i];
        run_event = col_rate_i > 0.0;
        if (run_event)
        {
            rngstate = dev_rngstate[idx_old_i];
            real event_prob = -expm1(-col_rate_i*dt_col);
            run_event = curand_uniform_double(&rngstate) <= event_prob;
            dev_rngstate[idx_old_i] = rngstate;
        }
    }
    __syncthreads();
    if (!run_event) return;

    __shared__ real pair_rate[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        pair_rate[idx_neighbor] = 0.0;
        std::size_t idx_cache = static_cast<std::size_t>(idx_old_i)*N_K + idx_neighbor;
        int idx_old_j = dev_col_neighbor[idx_cache];
        if (idx_old_j < 0) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
        )) continue;

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

    real measure = dev_col_measure[idx_old_i];
    if (!(measure > 0.0)) return;

    real target = dev_col_rate[idx_old_i]*curand_uniform_double(&rngstate);
    dev_rngstate[idx_old_i] = rngstate;
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        std::size_t idx_cache = static_cast<std::size_t>(idx_old_i)*N_K + idx_neighbor;
        int idx_old_j_try = dev_col_neighbor[idx_cache];
        if (idx_old_j_try < 0) continue;
        if (!_is_particle_active(
            dev_particle[idx_old_j_try].position.y, dev_particle[idx_old_j_try].position.z
        )) continue;

        idx_old_j = idx_old_j_try;
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

#endif // KNN_CACHE

// =========================================================================================================================

#endif // COLLISION
