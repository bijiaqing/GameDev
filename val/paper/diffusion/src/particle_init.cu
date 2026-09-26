#include <swarm_kern.cuh>

// draw the initial log-normal ring from independent streams; evolution uses the production RNG seed 1
__global__ void particle_init(swarm *particle, const real *, const real *, const real *)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= N_P) return;
    curs state;
    gpuRandInit(RING_INIT_SEED, i, 0, &state);
    const real u = RING_LOG_WIDTH*gpuRandNormalDouble(&state);
    particle[i].position = {0.0, R_0*exp(u), 0.5*M_PI};
    particle[i].velocity = {0.0, 0.0, 0.0};
}
