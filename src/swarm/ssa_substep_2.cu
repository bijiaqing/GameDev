#ifdef RADIATION

#include <_transport.cuh>
#include <param_grid.cuh>
#include <swarm_grid.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: ssa_substep_2
// apply attenuated radiation, gravity, gas drag, and optional P-R drag at midpoint and complete the transport step
//
// parallelization: one thread per representative particle after midpoint optical-depth reconstruction
// =====================================================================================================================

__global__
void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth,
    #ifdef IMPORTGAS
    const real *dev_gas_dens, const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz,
    #endif // IMPORTGAS
    real beta_taper,
    real dt
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x_1, y_1, z_1;
    real x_j, y_j, z_j;

    real lx_i, vy_i, lz_i;
    real lx_j, vy_j, lz_j;

    _load_particle(dev_particle, idx, x_1, y_1, z_1, lx_i, vy_i, lz_i);
    if (!_is_particle_active(y_1, z_1)) return;

    // interpolate optical depth from radial outer faces at the midpoint particle position
    real loc_x = _get_loc_x(x_1);
    real loc_y = _get_loc_y(y_1);
    real loc_z = _get_loc_z(z_1);

    real optdepth = _interp_field(dev_optdepth, loc_x, loc_y, loc_z, true);

    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    #else  // MONOSIZE
    real size = S_0;
    #endif // MULTISIZE

    // attenuate the size-dependent radiation-to-gravity ratio along the radial ray
    real beta = beta_taper*BETA_0*exp(-optdepth) / (size / S_0);

    _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j
        #ifdef IMPORTGAS
        , dev_gas_velx, dev_gas_vely, dev_gas_velz, dev_gas_dens
        #endif // IMPORTGAS
    );

    // apply boundary rules only after completing the full transport state
    _apply_transport_boundary(x_j, y_j, z_j, lx_j, vy_j, lz_j);
    _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
}

// =====================================================================================================================

#endif // RADIATION
