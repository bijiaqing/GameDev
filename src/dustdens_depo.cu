#ifdef SAVE_DENS

#include <param_phys.cuh>
#include <swarm_grid.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dustdens_depo
// deposit each representative particle's dust mass to the cell-centred grid
// =========================================================================================================================

__global__
void dustdens_depo (real *dev_dustdens, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    real weight = _get_grain_mass(size)*dev_particle[idx].par_numr;
    #else
    real weight = M_D / N_P;
    #endif // MULTISIZE

    _deposit_field(dev_dustdens, dev_particle, idx, weight);
}

// =========================================================================================================================

#endif // SAVE_DENS
