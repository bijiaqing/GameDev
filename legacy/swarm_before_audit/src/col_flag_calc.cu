#ifdef COLLISION

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: col_flag_calc
// Purpose: Determine which grid cells will have collisions in current timestep
// Dependencies: curand (for random number generation)
// =========================================================================================================================

__global__
void col_flag_calc (int *dev_col_flag, curs *dev_rs_grids, real *dev_col_rand, const real *dev_col_rate, real dt_col)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_G)
    {
        curs rs_grids = dev_rs_grids[idx];
        
        // if expt > rand, there will be a collision in the cell in the current timestep
        real col_rand = curand_uniform_double(&rs_grids); // the range is (0,1]
        real col_expt = dt_col*dev_col_rate[idx]; // total expected number of collisions in the cell within the current timestep

        if (col_expt >= col_rand)
        {
            dev_col_flag[idx] = 1;

            // need to pass on dev_col_rand to determine which particle in the cell collides
            dev_col_rand[idx] = dev_col_rate[idx]*curand_uniform_double(&rs_grids);
        }
        else
        {
            dev_col_flag[idx] = 0;
        }

        dev_rs_grids[idx] = rs_grids;
    }
}

// ========================================================================================================================

#endif // COLLISION
