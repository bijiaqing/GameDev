#include <cmath>

#include <fluid_kern.cuh>






__global__
void inf_cell_flag (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif
    int *dev_badstate
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    bool finite = isfinite(dev_dustdens[idx]);
    finite = finite && isfinite(dev_dustmomx[idx]) && isfinite(dev_dustmomy[idx]) && isfinite(dev_dustmomz[idx]);
    finite = finite && isfinite(dev_dustvelx[idx]) && isfinite(dev_dustvely[idx]) && isfinite(dev_dustvelz[idx]);

    #ifdef RADIATION
    finite = finite && isfinite(dev_optdepth[idx]);
    #endif

    if (!finite) atomicCAS(dev_badstate, 0, idx + 1);
}

