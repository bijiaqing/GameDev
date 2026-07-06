#if defined(TRANSPORT) && !defined(RADIATION)

#include <graffiti_kern.cuh>
#include <helpers_transport.cuh>

// =========================================================================================================================
// Kernel: ssa_transport
// Purpose: Complete staggered semi-analytic integrator for particle evolution (no radiation pressure)
// Dependencies: helpers_transport.cuh (_load_particle, _save_particle, _if_out_of_box, _ssa_substep_1, _ssa_substep_2)
// =========================================================================================================================

__global__
void ssa_transport (swarm *dev_particle, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx, const real *dev_gasvely, const real *dev_gasvelz
    #endif
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        real x_i, y_i, z_i;
        real x_1, y_1, z_1;
        real x_j, y_j, z_j;
        
        real lx_i, vy_i, lz_i;
        real lx_j, vy_j, lz_j;

        _load_particle(dev_particle, idx, x_i, y_i, z_i, lx_i, vy_i, lz_i);
        _ssa_substep_1(dt, x_i, y_i, z_i, lx_i, vy_i, lz_i, x_1, y_1, z_1);

        #ifdef MULTISIZE
        real size = dev_particle[idx].par_size;
        #else
        real size = S_0;
        #endif // MULTISIZE
        
        real beta = 0.0; // no radiation pressure

        _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j
            #ifdef IMPORTGAS
            , dev_gasvelx, dev_gasvely, dev_gasvelz, dev_gasdens
            #endif
        );

        _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
    }
}

// =========================================================================================================================

#endif // NO RADIATION
