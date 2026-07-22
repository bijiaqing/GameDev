#ifdef IMPORTGAS

#include <graffiti_kern.cuh>

// =========================================================================================================================
// kernel: gas_interp_calc
// advance the working gas fields by an incremental linear blend toward the next imported frame
// =========================================================================================================================

__global__
void gas_interp_calc (real *dev_gasdens, real *dev_gasvelx, real *dev_gasvely, real *dev_gasvelz,
    const real *dev_gasdens_next, const real *dev_gasvelx_next,
    const real *dev_gasvely_next, const real *dev_gasvelz_next, real blend)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    real keep = 1.0 - blend;
    dev_gasdens[idx] = keep*dev_gasdens[idx] + blend*dev_gasdens_next[idx];
    dev_gasvelx[idx] = keep*dev_gasvelx[idx] + blend*dev_gasvelx_next[idx];
    dev_gasvely[idx] = keep*dev_gasvely[idx] + blend*dev_gasvely_next[idx];
    dev_gasvelz[idx] = keep*dev_gasvelz[idx] + blend*dev_gasvelz_next[idx];
}

#endif // IMPORTGAS
