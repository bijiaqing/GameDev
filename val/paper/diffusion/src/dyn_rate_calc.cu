#include <swarm_kern.cuh>

// test-only replacement: impose the fixed step DT_MAX instead of the dynamics CFL policy to isolate production diffusion
// the runtime still clips the last step to each output time
__global__ void dyn_rate_calc(real *rate, const swarm *)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= N_P) return;
    rate[i] = 1.0/DT_MAX;
}
