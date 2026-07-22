#ifdef SAVE_DENS

#include <graffiti_kern.cuh>
#include <helpers_scatfield.cuh>

// =========================================================================================================================
// kernel: dustdens_scat
// scatter each representative particle's dust mass to the cell-centred grid
// =========================================================================================================================

__global__
void dustdens_scat (real *dev_dustdens, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        _particle_to_grid_core <DUSTDENS> (dev_dustdens, dev_particle, idx);
    }
}

// =========================================================================================================================

#endif // SAVE_DENS
