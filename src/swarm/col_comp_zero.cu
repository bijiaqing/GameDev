#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_comp_zero
// clear the path-integrated compensators of owners entering a new bath
// =====================================================================================================================

__global__
void col_comp_zero (const int *ids, int count, real *hazard,
    real *jump1, real *jump2, real *jumpmax)
{
    int slot = blockIdx.x*blockDim.x + threadIdx.x;
    if (slot >= count) return;
    int i = ids[slot];
    hazard[i] = jump1[i] = jump2[i] = jumpmax[i] = 0;
}

#endif // COLLISION && !BERNOULLI
