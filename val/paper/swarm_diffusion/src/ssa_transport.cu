#include <swarm_kern.cuh>

// Isolate spatial diffusion while retaining the production runtime composition.
__global__ void ssa_transport(swarm *, real) {}
