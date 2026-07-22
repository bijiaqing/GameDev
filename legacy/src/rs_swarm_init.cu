#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: rs_swarm_init
// Purpose: Initialize cuRAND random number generator states for each particle swarm
// Dependencies: curand (curand_init)
// =========================================================================================================================

__global__
void rs_swarm_init (curs *dev_rs_swarm, int seed)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        curand_init(seed, idx, 0, &dev_rs_swarm[idx]);
    }
}

// =========================================================================================================================

#endif // COLLISION or DIFFUSION
