#include <gpu_compat.cuh>
#if defined(COLLISION) || defined(DIFFUSION)

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: rngstate_init
// initialize one reproducible cuRAND stream for each representative particle
// =========================================================================================================================

__global__
void rngstate_init (curs *dev_rngstate, int seed)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    gpuRandInit(seed, idx, 0, &dev_rngstate[idx]);
}

// =========================================================================================================================

#endif // COLLISION || DIFFUSION
