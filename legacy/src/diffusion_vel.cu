#if defined(TRANSPORT) && defined(DIFFUSION)

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: diffusion_vel
// Purpose: Apply turbulent diffusion effects to particle velocities
// Dependencies: curand (for normal distribution random numbers)
// Status: PLACEHOLDER - To be implemented
// =========================================================================================================================

__global__
void diffusion_vel (swarm *dev_particle, curs *dev_rs_swarm, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        // TODO: Implement velocity diffusion
    }
}

// =========================================================================================================================

#endif // DIFFUSION
