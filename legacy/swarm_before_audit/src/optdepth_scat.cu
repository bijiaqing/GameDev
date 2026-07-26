#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>
#include <helpers_scatfield.cuh>

// =========================================================================================================================
// Kernel: optdepth_scat
// Purpose: Scatter particle optical depths to grid using trilinear interpolation
// Dependencies: helpers_scatfield.cuh (provides _particle_to_grid_core template)
// =========================================================================================================================

__global__
void optdepth_scat (real *dev_optdepth, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        _particle_to_grid_core <OPTDEPTH> (dev_optdepth, dev_particle, idx);
    }
}

// =========================================================================================================================

#endif // RADIATION
