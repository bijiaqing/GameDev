#ifdef RADIATION

#include <graffiti_kern.cuh>
#include <scatfield.cuh>

// =========================================================================================================================
// kernel: optdepth_scat
// scatter each representative particle's extinction cross section to the cell-centred grid
// =========================================================================================================================

__global__
void optdepth_scat (real *dev_optdepth, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    _particle_to_grid_core <OPTDEPTH> (dev_optdepth, dev_particle, idx);
}

// =========================================================================================================================

#endif // RADIATION
