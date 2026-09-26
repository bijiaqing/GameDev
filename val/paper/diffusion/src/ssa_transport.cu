#include <swarm_kern.cuh>

// test-only no-op transport isolates spatial diffusion while retaining the production runtime composition
__global__ void ssa_transport(swarm *, real) {}
