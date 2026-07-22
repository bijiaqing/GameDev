#ifdef COLLISION

#include <graffiti_kern.cuh>
#include <helpers_paramgrid.cuh>  // for _get_loc_x/y/z, _is_in_bounds, _get_cell_index
#include <helpers_paramphys.cuh>  // for _get_hg
#include <helpers_collision.cuh>  // for candidatelist, KernelType, _get_col_rate_ij

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
void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
)
{
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    int tree_size = (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12) ? 3*N_P : N_P;
    if (idx_tree >= tree_size || dev_col_tree[idx_tree].image != 0) return;
    int idx_old_i = dev_col_tree[idx_tree].index_old;
    dev_col_rate[idx_old_i] = 0.0;

    real x = dev_particle[idx_old_i].position.x;
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    // limit the KNN query to a configured fraction of the local gas scale height
    real R = y*sin(z);
    float max_search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);
        
    candidatelist query_result(max_search_dist);
    cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, tree_size);

    real col_rate_i = 0.0;
    float max_dist2 = 0.0f;

    for(int j = 0; j < N_K; j++)
        {
            real col_rate_ij = 0.0;
            int idx_query = query_result.returnIndex(j);

            if (idx_query != -1)
            {
                int idx_old_j = dev_col_tree[idx_query].index_old;
                if (idx_old_j == idx_old_i) continue;

                float dist2 = query_result.returnDist2(j);
                max_dist2 = fmaxf(max_dist2, dist2);
                col_rate_ij = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (dev_particle, idx_old_i, idx_old_j
                    #ifdef IMPORTGAS
                    , dev_gasdens
                    #endif
                );
            }

            col_rate_i += col_rate_ij;
    }

    // normalize by the accessible measure of the smallest ball containing the returned neighbors
    real radius = sqrtf(static_cast<real>(max_dist2));
    real volume = _get_ball_measure(x, y, z, radius);
    #ifdef COLLISION_UNIT_VOLUME
    // bypass KNN geometry only for dimensionless analytic kernel tests
    volume = 1.0;
    #endif
    col_rate_i = (volume > 0.0) ? col_rate_i/volume : 0.0;

    // retain the neighborhood radius and total rate for event execution
    dev_particle[idx_old_i].max_dist = radius;
    dev_particle[idx_old_i].col_rate = col_rate_i;
    dev_col_rate[idx_old_i] = col_rate_i;
}

// =========================================================================================================================

#endif // COLLISION
