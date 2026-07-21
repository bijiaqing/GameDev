#ifndef FLUID_KERN_CUH
#define FLUID_KERN_CUH

#include <const.cuh>

// =========================================================================================================================

__global__ void advect_x_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void advect_y_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_weight_y, real dt
);

__global__ void advect_z_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_weight_z, real dt
);

// =========================================================================================================================

__global__ void cfl_rate_calc (
    real *dev_cfl_rate, const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz
);

// =========================================================================================================================

__global__ void diffus_x_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);

__global__ void diffus_y_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);

__global__ void diffus_z_calc (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);

// =========================================================================================================================

__global__ void inf_cell_flag (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif
    int *dev_badstate
);

// =========================================================================================================================

__global__ void init_rho_calc (
    real *dev_dustdens, const real *dev_initdens
);

__global__ void init_vel_calc (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif
);

// =========================================================================================================================

__global__ void momentum_getv (
    const real *dev_dustdens,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
);

__global__ void momentum_setv (
    const real *dev_dustdens,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz
);

#ifdef RADIATION
__global__ void optdepth_calc (real *dev_optdepth, const real *dev_dustdens);
__global__ void optdepth_csum (real *dev_optdepth);
#endif

// =========================================================================================================================

__global__ void source_update (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz, const real *dev_dustdens,
    #ifdef RADIATION
    const real *dev_optdepth, real beta_taper,
    #endif
    real dt
);

// =========================================================================================================================

#endif
