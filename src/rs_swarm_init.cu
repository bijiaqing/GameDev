#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))

#include <graffiti_kern.cuh>

// =========================================================================================================================
// kernel: rs_swarm_init
// initialize one reproducible cuRAND stream for each representative particle
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

#endif // COLLISION || (TRANSPORT && DIFFUSION)
