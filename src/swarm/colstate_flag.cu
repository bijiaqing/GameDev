#include <gpu.cuh>

#ifdef COLLISION
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: colstate_flag
// retain the collision-entry finite-state guard when a valid geometry package skips site reconstruction
//
// parallelization: one thread per representative particle
// =====================================================================================================================

__global__
void colstate_flag (const swarm *dev_particle, int *dev_bad_part)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    bool finite = isfinite(dev_particle[idx].position.x)
        && isfinite(dev_particle[idx].position.y)
        && isfinite(dev_particle[idx].position.z)
        && isfinite(dev_particle[idx].velocity.x)
        && isfinite(dev_particle[idx].velocity.y)
        && isfinite(dev_particle[idx].velocity.z);
    #ifdef MULTISIZE
    finite = finite && isfinite(dev_particle[idx].par_size) && isfinite(dev_particle[idx].par_numr);
    #endif // MULTISIZE
    if (!finite) atomicCAS(dev_bad_part, 0, idx + 1);
}

#endif // COLLISION
