#include <fluid_kern.cuh>

__global__
void inf_cell_flag (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif // RADIATION
    int *dev_bad_cell
)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    // test every evolved field in the current cell
    bool finite = isfinite(dev_dustdens[idx_cell]);
    finite = finite && isfinite(dev_dustmomx[idx_cell]) && isfinite(dev_dustmomy[idx_cell])
        && isfinite(dev_dustmomz[idx_cell]);
    finite = finite && isfinite(dev_dustvelx[idx_cell]) && isfinite(dev_dustvely[idx_cell])
        && isfinite(dev_dustvelz[idx_cell]);

    #ifdef RADIATION
    finite = finite && isfinite(dev_optdepth[idx_cell]);
    #endif // RADIATION

    // record the first nonfinite cell with zero reserved for a clean state
    if (!finite) atomicCAS(dev_bad_cell, 0, idx_cell + 1);
}
