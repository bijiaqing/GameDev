#include <swarm_kern.cuh>
__global__ void rngstate_init (curs *state, int)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N_P) gpuRandInit(SEED + 1, i, 0, &state[i]);
}
