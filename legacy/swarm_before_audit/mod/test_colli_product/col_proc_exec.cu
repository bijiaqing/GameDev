#ifdef COLLISION

#include <graffiti_kern.cuh>
#include <helpers_paramgrid.cuh>  // for _get_loc_x/y/z, _is_in_bounds, _get_cell_index
#include <helpers_paramphys.cuh>  // for _get_hg
#include <helpers_collision.cuh>  // for candidatelist, KernelType, _get_col_rate_ij

// =========================================================================================================================
// Kernel: col_proc_exec
// Purpose: Execute collision process (merge/fragmentation) for selected particles
// Dependencies: helpers_collision.cuh (provides candidatelist, _get_col_rate_ij, _get_vrel via KernelType template)
// =========================================================================================================================

__global__
void col_proc_exec (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, 
    const real *dev_col_rand, const int *dev_col_flag, const tree *dev_col_tree, const bbox *dev_boundbox
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif
)
{
    // calculates the outcome of collision and update the property of the particle
    // if the collision flag of the cell that the particle is in is 1, it examines whether the particle is going to collide
    // if the particle *first* makes real >= rand, it is the particle that is going to collide in the cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_tree < N_P)
    {
        int idx_old_i = dev_col_tree[idx_tree].index_old;

        real x = dev_particle[idx_old_i].position.x;
        real y = dev_particle[idx_old_i].position.y;
        real z = dev_particle[idx_old_i].position.z;

        real loc_x = _get_loc_x(x);
        real loc_y = _get_loc_y(y);
        real loc_z = _get_loc_z(z);

        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return; // particle is out of bounds, do nothing

        int idx_cell = _get_cell_index(loc_x, loc_y, loc_z);

        if (dev_col_flag[idx_cell] == 0)
        {
            return; // no collision in the cell, do nothing
        }
        
        // collision rate of the particle
        real col_rate_i = dev_particle[idx_old_i].col_rate;
        
        // collision parameters of the cell that the particle is in
        real col_rand = dev_col_rand[idx_cell]; // dev_col_rand is now a value between 0 and collision rate of the cell
        real col_expt = atomicAdd(&dev_col_expt[idx_cell], col_rate_i); // atomicAdd returns the value BEFORE the addition

        if (col_expt < col_rand && col_expt + col_rate_i >= col_rand)   // if this is the particle that is going to collide
        {
            curs rs_swarm = dev_rs_swarm[idx_old_i];

            // maximum search distance for KNN neighbor search defined by gas scale height
            real R = y*sin(z);
            float max_search_dist = static_cast<float>(_get_hg(R)*R);
            
            candidatelist query_result(max_search_dist);
            cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_P);

            // col_rand_ij is now a value between 0 and the total collision rate of the particle
            real col_rand_ij = col_rate_i*curand_uniform_double(&rs_swarm);
            real col_expt_ij = 0.0;

            real radius = dev_particle[idx_old_i].max_dist;
            real volume = (4.0/3.0)*M_PI*radius*radius*radius;

            volume = 1.0; // for testing purposes

            int idx_old_j, j = 0;

            // if particle idx_old_j *first* makes expt >= rand, it is the one that idx_old_i is going to collide with
            while (col_expt_ij < col_rand_ij && j < N_K)
            {
                int idx_query = query_result.returnIndex(j);

                if (idx_query != -1)
                {   
                    idx_old_j = dev_col_tree[idx_query].index_old;
                    col_expt_ij += _get_col_rate_ij<static_cast<KernelType>(COAG_KERNEL)>(dev_particle, idx_old_i, idx_old_j
                        #ifdef IMPORTGAS
                        , dev_gasdens
                        #endif
                    ) / volume;
                }

                j++;
            }

            real v_rel = 0.0;
            // real v_rel = _get_vrel(dev_particle, idx_old_i, idx_old_j
            //     #ifdef IMPORTGAS
            //     , dev_gasdens
            //     #endif
            // );

            real s_i = dev_particle[idx_old_i].par_size;
            real s_j = dev_particle[idx_old_j].par_size;
            real s_k = cbrt(s_i*s_i*s_i + s_j*s_j*s_j);

            if (v_rel <= V_FRAG)
            {
                // collide with idx_old_j and MERGE
                dev_particle[idx_old_i].par_size  = s_k;
                dev_particle[idx_old_i].par_numr *= (s_i*s_i*s_i) / (s_k*s_k*s_k);
            }
            else
            {
                // collide with idx_old_j and BREAK-UP
                real rand = curand_uniform_double(&rs_swarm);  // distribution in (0, 1]

                // determine the fragmentation outcome using inverse transform sampling
                // see https://en.wikipedia.org/wiki/Inverse_transform_sampling
                // since n(s) ds = N*s^-3.5 ds is the grain size distribution and 
                // integrate( m(s)*n(s) ds) = mi + mj = si^3 + sj^3, where s is 0 < s < (si^3 + sj^3)^(1/3)
                // we get N = 0.5*(si^3 + sj^3)^(5/6), so that n(s) ds = 0.5*(si^3 + sj^3)^(5/6) s^-3.5 ds
                // since the probability distribution function (PDF) P(s) = m(s)*n(s) / (si^3 + sj^3), 
                // the cumulative distribution function (CDF) becomes C(s) = integrate( P(s) ds) = (si^3 + sj^3)^(-1/6) s^(1/2)
                // which naturally satisfies C(s) in [0,1] when s in [0, (si^3 + sj^3)^(1/3)]
                // Since the definition of inverse function gives C(C^-1(s)) = s,
                // the inverse function of the CDF is C^-1(s) = (si^3 + sj^3)^(1/3) s^2    <--- this is the code below
                // so we draw a random number x from a uniform distribution between 0 and 1,
                // put it in C^-1(x) will give a random s that satisfies the PDF P(s) = m(s)*n(s) / (si^3 + sj^3)

                s_k *= rand*rand;

                dev_particle[idx_old_i].par_size  = s_k;
                dev_particle[idx_old_i].par_numr *= (s_i*s_i*s_i) / (s_k*s_k*s_k);
            }

            dev_rs_swarm[idx_old_i] = rs_swarm;
        }
    }
}

// =================================================================================================================================

#endif // COLLISION
