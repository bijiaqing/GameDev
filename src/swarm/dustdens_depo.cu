#ifdef SAVE_DENS

#include <_transport.cuh>
#include <param_phys.cuh>
#include <swarm_grid.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: dustdens_depo
// deposit each representative particle's dust mass to the cell-centered grid
// =====================================================================================================================

__global__
void dustdens_depo (real *dev_dustdens, const swarm *dev_particle, real total_dust_mass)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;
    if (!_is_particle_active(dev_particle[idx].position.y, dev_particle[idx].position.z)) return;

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    real weight = _get_grain_mass(size)*dev_particle[idx].par_numr;
    #else  // MONOSIZE
    real weight = total_dust_mass / N_P;
    #endif // MULTISIZE

    _deposit_field(dev_dustdens, dev_particle, idx, weight);
}

// =====================================================================================================================

#endif // SAVE_DENS
