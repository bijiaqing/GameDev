#ifdef IMPORTGAS

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: gas_lerp_calc
// advance the working gas fields by an incremental linear blend toward the next imported frame
// =========================================================================================================================

__global__
void gas_lerp_calc (real *dev_gas_dens, real *dev_gas_velx, real *dev_gas_vely, real *dev_gas_velz,
    const real *dev_gas_dens_next, const real *dev_gas_velx_next,
    const real *dev_gas_vely_next, const real *dev_gas_velz_next, real gas_blend)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    real gas_keep = 1.0 - gas_blend;
    dev_gas_dens[idx_cell] = gas_keep*dev_gas_dens[idx_cell] + gas_blend*dev_gas_dens_next[idx_cell];
    dev_gas_velx[idx_cell] = gas_keep*dev_gas_velx[idx_cell] + gas_blend*dev_gas_velx_next[idx_cell];
    dev_gas_vely[idx_cell] = gas_keep*dev_gas_vely[idx_cell] + gas_blend*dev_gas_vely_next[idx_cell];
    dev_gas_velz[idx_cell] = gas_keep*dev_gas_velz[idx_cell] + gas_blend*dev_gas_velz_next[idx_cell];
}

#endif // IMPORTGAS
