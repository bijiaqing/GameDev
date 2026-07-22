#if defined(TRANSPORT) && defined(RADIATION)

#include <graffiti_kern.cuh>
#include <helpers_transport.cuh>  // for _load_particle, _save_particle, _if_out_of_box, _ssa_substep_2
#include <helpers_paramgrid.cuh>  // for _get_loc_x/y/z
#include <helpers_interpval.cuh>  // for _interp_field

// =========================================================================================================================
// Kernel: ssa_substep_2
// Purpose: Second substep of the staggered semi-analytic integrator for particles
// Dependencies: helpers_transport.cuh (_load_particle, _save_particle, _if_out_of_box, _ssa_substep_2),
//               helpers_paramgrid.cuh (_get_loc_x/y/z),
//               helpers_interpval.cuh (_interp_field)
// =========================================================================================================================

__global__
void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx, const real *dev_gasvely, const real *dev_gasvelz
    #endif
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        real x_1, y_1, z_1;
        real x_j, y_j, z_j;
        
        real lx_i, vy_i, lz_i;
        real lx_j, vy_j, lz_j;
        
        _load_particle(dev_particle, idx, x_1, y_1, z_1, lx_i, vy_i, lz_i);

        // retrieve the optical depth and calculate beta
        real loc_x = _get_loc_x(x_1);
        real loc_y = _get_loc_y(y_1);
        real loc_z = _get_loc_z(z_1);

        real optdepth = _interp_field(dev_optdepth, loc_x, loc_y, loc_z, true);
        
        #ifdef MULTISIZE
        real size = dev_particle[idx].par_size;
        #else
        real size = S_0;
        #endif // MULTISIZE
        real beta = BETA_0*exp(-optdepth) / (size / S_0);

        _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j
            #ifdef IMPORTGAS
            , dev_gasvelx, dev_gasvely, dev_gasvelz, dev_gasdens
            #endif
        );

        _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
    }
}

// ========================================================================================================================

#endif // RADIATION
