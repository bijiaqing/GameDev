#ifdef RADIATION

#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_grid.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: optdepth_depo
// deposit each representative particle's extinction cross section to the cell-centered grid
// =====================================================================================================================

__global__
void optdepth_depo (real *dev_optdepth, const swarm *dev_particle, real total_dust_mass)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;
    if (!_is_particle_active(dev_particle[idx].position.y, dev_particle[idx].position.z)) return;

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    real weight = _get_grain_mass(size)*dev_particle[idx].par_numr;
    #else  // MONOSIZE
    real size = S_0;
    real weight = total_dust_mass / N_P;
    #endif // MULTISIZE

    weight *= KAPPA_0 / (size / S_0);

    if (N_Z == 1)
    {
        real R = _get_cyl_R(dev_particle[idx].position.y, dev_particle[idx].position.z);
        real H_g = _get_hg(R)*R;

        // close the vertically integrated disk with a well-mixed gas-scale-height profile
        weight /= sqrt(2.0*M_PI)*H_g;
    }

    _deposit_field(dev_optdepth, dev_particle, idx, weight);
}

// =====================================================================================================================

#endif // RADIATION
