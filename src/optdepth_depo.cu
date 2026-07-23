#ifdef RADIATION

#include <param_phys.cuh>
#include <swarm_grid.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: optdepth_depo
// deposit each representative particle's extinction cross section to the cell-centred grid
// =========================================================================================================================

__global__
void optdepth_depo (real *dev_optdepth, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    real weight = _get_grain_mass(size)*dev_particle[idx].par_numr;
    #else
    real size = S_0;
    real weight = M_D / N_P;
    #endif // MULTISIZE

    weight *= KAPPA_0 / (size / S_0);

    if (N_Z == 1)
    {
        real R = dev_particle[idx].position.y*sin(dev_particle[idx].position.z);
        real h_g = _get_hg(R);
        real H_d = h_g*R;
        #ifdef DIFFUSION
        #ifndef CONST_NU
        real alpha_z = ALPHA / SCHMIDT_Z;
        #else
        real alpha_z = NU/(h_g*h_g*R*R*_get_omegaK(R)*SCHMIDT_Z);
        #endif
        real stokes_mid = ST_0*(size / S_0);
        #ifndef CONST_ST
        stokes_mid /= pow(R / R_0, IDX_P);
        #endif
        H_d *= sqrt(alpha_z/(alpha_z + stokes_mid));
        #endif
        weight /= sqrt(2.0*M_PI)*H_d;
    }

    _deposit_field(dev_optdepth, dev_particle, idx, weight);
}

// =========================================================================================================================

#endif // RADIATION
