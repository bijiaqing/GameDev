#ifdef COLLISION

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: col_event_run
// sample and apply at most one frozen-rate Bernoulli collision event per representative particle
//
// parallelization: one thread per primary KD-tree node with periodic image nodes skipped
//
// per call:
//   1 sample whether the representative collides during dt_col
//   2 rebuild its local KNN list and sample a partner from pair propensities
//   3 update only the live representative using the read-only size and multiplicity snapshot
// =========================================================================================================================

__global__
void col_event_run (swarm *dev_particle, curs *dev_rngstate, const real *dev_col_rate, const real *dev_col_dist,
    const real *dev_size_old, const real *dev_numr_old, const tree *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0,
    real dt_col
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_col_tree[idx_tree].image != 0) return;

    int idx_old_i = dev_col_tree[idx_tree].index_old;
    real col_rate_i = dev_col_rate[idx_old_i];
    if (col_rate_i <= 0.0) return;

    // sample the exact probability of at least one event for the frozen total propensity
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

    // recover the same local neighbor set used to calculate col_rate_i
    candidatelist query_result(max_search_dist);
    cukd::cct::knn <candidatelist, tree, tree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_T
    );

    real measure = _get_ball_measure(y, z, dev_col_dist[idx_old_i]);
    #ifdef COLLISION_UNIT_VOLUME
    // use the analytic unit-volume normalization only in dimensionless kernel tests
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME
    if (measure <= 0.0) return;

    // select one partner by inverse sampling of the pair-propensity sum
    real target = col_rate_i*curand_uniform_double(&rngstate);
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int j = 0; j < N_K; j++)
    {
        int idx_query = query_result.returnIndex(j);
        if (idx_query < 0) continue;

        int idx_candidate = dev_col_tree[idx_query].index_old;
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
    // use the physical relative speed only when deciding a physical-kernel collision outcome
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
        // coagulate both physical grain masses into the updated representative species
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i / (size_k*size_k*size_k);
    }
    else
    {
        // sample the fragment size distribution and conserve the representative swarm mass
        real frag_sample = curand_uniform_double(&rngstate);
        size_k = fmax(INIT_SMIN, size_k*frag_sample*frag_sample);
        dev_particle[idx_old_i].par_size = size_k;
        dev_particle[idx_old_i].par_numr = dev_numr_old[idx_old_i]*size_i*size_i*size_i / (size_k*size_k*size_k);
    }
    #endif // MULTISIZE

    dev_rngstate[idx_old_i] = rngstate;
}

#endif // COLLISION
