#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: optdepth_mean
// Purpose: Compute azimuthal average of optical depth (average in X direction)
// Dependencies: None
// =========================================================================================================================

__global__
void optdepth_mean (real *dev_optdepth)
{
    int idx_x = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_x < N_Y*N_Z)
    {	
        int idx_y = idx_x % N_Y;
        int idx_z = idx_x / N_Y;

        real optdepth_sum = 0.0;

        // sum in X direction
        // no race condition since each thread works on a unique X row
        for (int i = 0; i < N_X; i++)
        {
            int idx_cell = idx_z*N_X*N_Y + idx_y*N_X + i;
            optdepth_sum += dev_optdepth[idx_cell];
        }

        real optdepth_avg = optdepth_sum / N_X;

        for (int j = 0; j < N_X; j++)
        {
            int idx_cell = idx_z*N_X*N_Y + idx_y*N_X + j;
            dev_optdepth[idx_cell] = optdepth_avg;
        }
    }
}

// =========================================================================================================================

#endif // RADIATION
