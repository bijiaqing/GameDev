#ifdef COLLISION

#include <_collision.cuh>
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: col_rate_calc
// calculate each representative particle's total local collision propensity from its K nearest neighbors
//
// parallelization: one thread per primary KD-tree node with periodic image nodes skipped
//
// per call:
//   1 query neighbors within the scale-height search radius
//   2 sum pair propensity numerators while excluding the representative itself
//   3 divide by the accessible KNN measure and expose the rate for the host reduction
// =========================================================================================================================

__global__
void col_rate_calc (real *dev_col_rate, real *dev_col_dist, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const tree *dev_col_tree, const bbox *dev_boundbox,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
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

    // limit the KNN query to a configured fraction of the local gas scale height
    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);
        
    candidatelist query_result(max_search_dist);
    cukd::cct::knn <candidatelist, tree, tree_traits> (
        query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_T
    );

    real col_rate_i = 0.0; // total collision rate for the representative particle
    float max_dist_sq = 0.0f;

    for(int j = 0; j < N_K; j++)
    {
        real col_rate_ij = 0.0;
        int idx_query = query_result.returnIndex(j);

        if (idx_query != -1)
        {
            int idx_old_j = dev_col_tree[idx_query].index_old;
            if (idx_old_j == idx_old_i) continue; // skip self-collision
            if (!_is_particle_active(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            )) continue;

            float dist_sq = query_result.returnDist2(j);
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
}

// =========================================================================================================================

#endif // COLLISION
