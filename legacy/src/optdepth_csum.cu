#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: optdepth_csum
// Purpose: Compute cumulative sum of optical depth in the radial direction
// Dependencies: None
// =========================================================================================================================

__global__
void optdepth_csum (real *dev_optdepth) // cumulative sum in the radial direction
{
    int idx_y = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_y < N_X*N_Z)
    {	
        int idx_x = idx_y % N_X;
        int idx_z = idx_y / N_X;

        // cumulative sum in Y direction
        // no race condition since each thread works on a unique Y row
        for (int i = 1; i < N_Y; i++)
        {
            int idx_cell = idx_z*N_X*N_Y + i*N_X + idx_x;
            dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
        }
    }
}

// =========================================================================================================================

#endif // RADIATION
