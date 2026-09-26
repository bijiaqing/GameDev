#include <gpu.cuh>

#ifdef COLLISION
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_bath_init
// freeze the partner reservoir and reset continuation state for one bath
// =====================================================================================================================

__global__
void col_bath_init (const int *owner_ids, int owner_count, real *dev_size_old, real *dev_numr_old, real *dev_col_time,
    int *dev_col_events, unsigned char *dev_col_complete, const swarm *dev_particle)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    dev_size_old[idx] = dev_particle[idx].par_size;
    dev_numr_old[idx] = dev_particle[idx].par_numr;
    dev_col_time[idx] = 0.0;
    dev_col_events[idx] = 0;
    dev_col_complete[idx] = 0;
}

#endif // COLLISION
