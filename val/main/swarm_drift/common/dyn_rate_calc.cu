#include <swarm_kern.cuh>

// Fixed-step integrator experiment: retain production SSA but bypass its CFL policy.
// The runtime still clips the last step to each output time.
__global__ void dyn_rate_calc(real *rate, const swarm *)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= N_P) return;
    rate[i] = 1.0/DT_MAX;
}
