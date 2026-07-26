#ifdef RADIATION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: optdepth_calc
// Fluid version: compute optical depth directly from dust volumetric density.
// τ_cell = κ(s) · ρ_d · Δy_cell   where Δy_cell = y0·(dy−1) (radial extent of cell)
// After optdepth_csum this becomes the cumulative optical depth τ(y) from the inner boundary.
// =========================================================================================================================

__global__
void optdepth_calc (real *dev_optdepth, const real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;

    real dy = _get_dy();
    real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
    real dy_len = y0*(dy - 1.0);

    dev_optdepth[idx] = KAPPA_0*dev_dustdens[idx]*dy_len;
}

// =========================================================================================================================

#endif // RADIATION
