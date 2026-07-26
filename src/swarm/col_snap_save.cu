#ifdef COLLISION

#include <swarm_kern.cuh>

// =========================================================================================================================

__global__
void col_snap_save (real *dev_size_old, real *dev_numr_old, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    dev_size_old[idx] = dev_particle[idx].par_size;
    dev_numr_old[idx] = dev_particle[idx].par_numr;
}

// =========================================================================================================================

#endif // COLLISION
