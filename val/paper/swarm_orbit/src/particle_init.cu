#include <swarm_kern.cuh>

// Pericentre of a=1, e=0.5, GM=1. Tangential state stores R*v_phi.
__global__ void particle_init(swarm *particle, const real *, const real *, const real *)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= N_P) return;
    particle[i].position = {0.0, 0.5, 0.5*M_PI};
    particle[i].velocity = {sqrt(0.75), 0.0, 0.0};
}
