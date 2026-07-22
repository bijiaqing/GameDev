#ifdef IMPORTGAS

#include <graffiti_kern.cuh>

// =========================================================================================================================
// kernel: gas_lerp_calc
// advance the working gas fields by an incremental linear blend toward the next imported frame
// =========================================================================================================================

__global__
void gas_lerp_calc (real *dev_gas_dens, real *dev_gas_velx, real *dev_gas_vely, real *dev_gas_velz,
    const real *dev_gas_dens_next, const real *dev_gas_velx_next,
    const real *dev_gas_vely_next, const real *dev_gas_velz_next, real blend)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real keep = 1.0 - blend;
    dev_gas_dens[idx] = keep*dev_gas_dens[idx] + blend*dev_gas_dens_next[idx];
    dev_gas_velx[idx] = keep*dev_gas_velx[idx] + blend*dev_gas_velx_next[idx];
    dev_gas_vely[idx] = keep*dev_gas_vely[idx] + blend*dev_gas_vely_next[idx];
    dev_gas_velz[idx] = keep*dev_gas_velz[idx] + blend*dev_gas_velz_next[idx];
}

#endif // IMPORTGAS
