#ifdef COLLISION

#include <graffiti_kern.cuh>
#include <helpers_paramphys.cuh>
#include <helpers_collision.cuh>

// =========================================================================================================================
// kernel: col_proc_exec
// sample and apply at most one frozen-rate Bernoulli collision event per representative particle
//
// parallelization: one thread per primary KD-tree node with periodic image nodes skipped
//
// per call:
//   1 sample whether the representative collides during dt_col
//   2 rebuild its local KNN list and sample a partner from pair propensities
//   3 update only the live representative using the read-only particle snapshot
// =========================================================================================================================

__global__
void col_proc_exec (swarm *dev_particle, const swarm *dev_particle_old, curs *dev_rs_swarm, real dt_col,
    const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    int tree_size = (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12) ? 3*N_P : N_P;
    if (idx_tree >= tree_size || dev_col_tree[idx_tree].image != 0) return;

    int idx_old_i = dev_col_tree[idx_tree].index_old;
    real col_rate_i = dev_particle_old[idx_old_i].col_rate;
    if (col_rate_i <= 0.0) return;

    // sample the exact probability of at least one event for the frozen total propensity
    curs rs_swarm = dev_rs_swarm[idx_old_i];
    real event_prob = -expm1(-col_rate_i*dt_col);
    if (curand_uniform_double(&rs_swarm) > event_prob)
    {
        dev_rs_swarm[idx_old_i] = rs_swarm;
        return;
    }

    real x = dev_particle_old[idx_old_i].position.x;
    real y = dev_particle_old[idx_old_i].position.y;
    real z = dev_particle_old[idx_old_i].position.z;
    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // recover the same local neighbor set used to calculate col_rate_i
    candidatelist query_result(max_search_dist);
    cukd::cct::knn <candidatelist, tree, tree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, tree_size);

    real measure = _get_ball_measure(x, y, z, dev_particle_old[idx_old_i].max_dist);
    #ifdef COLLISION_UNIT_VOLUME
    // use the analytic unit-volume normalization only in dimensionless kernel tests
    measure = 1.0;
    #endif
    if (measure <= 0.0) return;

    // select one partner by inverse sampling of the pair-propensity sum
    real target = col_rate_i*curand_uniform_double(&rs_swarm);
    real cumulative = 0.0;
    int idx_old_j = -1;
    for (int j = 0; j < N_K; j++)
    {
        int idx_query = query_result.returnIndex(j);
        if (idx_query < 0) continue;

        int candidate = dev_col_tree[idx_query].index_old;
        if (candidate == idx_old_i) continue;

        idx_old_j = candidate;
        cumulative += _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle_old, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gasdens
            #endif
        ) / measure;
        if (cumulative >= target) break;
    }
    if (idx_old_j < 0) return;

    #ifdef MULTISIZE
    // use the physical relative speed only when deciding a physical-kernel collision outcome
    real v_rel = 0.0;
    if (COAG_KERNEL == CUSTOM_KERNEL)
    {
        v_rel = _get_vrel(dev_particle_old, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gasdens
            #endif
        );
    }

    real s_i = dev_particle_old[idx_old_i].par_size;
    real s_j = dev_particle_old[idx_old_j].par_size;
    real s_k = cbrt(s_i*s_i*s_i + s_j*s_j*s_j);

    if (v_rel <= V_FRAG)
    {
        // coagulate both physical grain masses into the updated representative species
        dev_particle[idx_old_i].par_size  = s_k;
        dev_particle[idx_old_i].par_numr = dev_particle_old[idx_old_i].par_numr*s_i*s_i*s_i/(s_k*s_k*s_k);
    }
    else
    {
        // sample the fragment size distribution and conserve the representative swarm mass
        real sample = curand_uniform_double(&rs_swarm);
        s_k = fmax(INIT_SMIN, s_k*sample*sample);
        dev_particle[idx_old_i].par_size  = s_k;
        dev_particle[idx_old_i].par_numr = dev_particle_old[idx_old_i].par_numr*s_i*s_i*s_i/(s_k*s_k*s_k);
    }
    #endif

    dev_rs_swarm[idx_old_i] = rs_swarm;
}

#endif // COLLISION
