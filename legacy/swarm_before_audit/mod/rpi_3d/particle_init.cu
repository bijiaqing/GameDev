

#include <graffiti_kern.cuh>
#include <helpers_paramphys.cuh>  // for _get_grain_mass

// =========================================================================================================================
// Kernel: particle_init
// Purpose: Initialize particle swarm positions, velocities, sizes, and grain numbers
// Dependencies: helpers_paramphys.cuh (provides _get_grain_mass)
// =========================================================================================================================

__global__
void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        dev_particle[idx].position.x = dev_random_x[idx];
        dev_particle[idx].position.y = dev_random_y[idx];
        dev_particle[idx].position.z = Z_MAX - dev_random_z[idx]; // offset from midplane into upper half
        
        dev_particle[idx].velocity.x = sqrt(G*M_S*dev_random_y[idx]); // specific angular momentum in azimuth
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;
    }
}

// =========================================================================================================================
