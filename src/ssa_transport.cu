#if defined(TRANSPORT) && !defined(RADIATION)

#include <_transport.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: ssa_transport
// complete one gravity-and-drag transport step when radiation does not require a midpoint mesh reconstruction
//
// parallelization: one thread per representative particle
// =========================================================================================================================

__global__
void ssa_transport (swarm *dev_particle,
    #ifdef IMPORTGAS
    const real *dev_gas_dens, const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz,
    #endif
    real dt
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x_i, y_i, z_i;
    real x_1, y_1, z_1;
    real x_j, y_j, z_j;

    real lx_i, vy_i, lz_i;
    real lx_j, vy_j, lz_j;

    // construct the staggered midpoint position from the initial state
    _load_particle(dev_particle, idx, x_i, y_i, z_i, lx_i, vy_i, lz_i);
    _ssa_substep_1(dt, x_i, y_i, z_i, lx_i, vy_i, lz_i, x_1, y_1, z_1);

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    #else
    real size = S_0;
    #endif // MULTISIZE

    real beta = 0.0;

    _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j
        #ifdef IMPORTGAS
        , dev_gas_velx, dev_gas_vely, dev_gas_velz, dev_gas_dens
        #endif
    );

    // enforce boundaries after the completed transport update
    _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
    _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
}

// =========================================================================================================================

#endif // TRANSPORT && !RADIATION
