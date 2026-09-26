#ifndef GAMEDEV_FLUID_KERN_CUH
#define GAMEDEV_FLUID_KERN_CUH

#include <gpu.cuh>

#if defined(VISC_FLOW) && !defined(DIFFUSION)
#error "VISC_FLOW requires DIFFUSION"
#endif // VISC_FLOW && !DIFFUSION

#include <const_defs.cuh>

// compute frozen-coefficient drag relaxation and force weights shared by source_update and analytical tests
__device__ __forceinline__
void _get_drag_weights (real dt, real ts, real &drag_relax, real &drag_decay,
    real &force_weight_old, real &force_weight_new)
{
    real tau = dt / ts;
    drag_relax = -expm1(-tau);
    drag_decay = 1.0 - drag_relax;

    // evaluate drag-weighted force quadrature with a cancellation-safe small-step series
    if (tau < 1.0e-04)
    {
        real tau_sq = tau*tau;
        real tau_cb = tau_sq*tau;

        force_weight_old = dt*(0.5 - tau / 3.0 + tau_sq / 8.0 - tau_cb / 30.0);
        force_weight_new = dt*(0.5 - tau / 6.0 + tau_sq / 24.0 - tau_cb / 120.0);
    }
    else
    {
        force_weight_new = ts*(tau - drag_relax) / tau;
        force_weight_old = ts*drag_relax - force_weight_new;
    }

}

// =====================================================================================================================
// conservative directional transport

#ifdef FLUID_BLOCK_SWEEP

#ifdef GAMEDEV_ROCM
constexpr int TPB_BLOCK = 64;
#else  // !GAMEDEV_ROCM
constexpr int TPB_BLOCK = 32;
#endif // GAMEDEV_ROCM

// index the explicit full-grid planes shared by every block-owned advection sweep
enum block_adv_field
{
    BLOCK_RHOD = 0,
    BLOCK_MX,
    BLOCK_MY,
    BLOCK_MZ,
    BLOCK_LX,
    BLOCK_VY,
    BLOCK_LZ,
    BLOCK_ANTI_RHOD,
    BLOCK_ANTI_MX,
    BLOCK_ANTI_MY,
    BLOCK_ANTI_MZ,
    BLOCK_RHOD_LOW,
    BLOCK_ADV_FIELDS
};

__global__ void advection_xbl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_adv_work, real dt
);

__global__ void advection_ybl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_y, real *dev_adv_work, real dt
);

__global__ void advection_zbl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_z, real *dev_adv_work, real dt
);

#else  // !FLUID_BLOCK_SWEEP

__global__ void advection_xth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void advection_yth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_y, real dt
);

__global__ void advection_zth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_ppm_weight_z, real dt
);

#endif // FLUID_BLOCK_SWEEP

// =====================================================================================================================
// transport timestep rate

__global__ void cfl_rate_calc (
    real *dev_cfl_rate, const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz
);

// =====================================================================================================================
// conservative density diffusion

#ifdef FLUID_BLOCK_SWEEP

__global__ void diffusion_xbl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void diffusion_ybl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void diffusion_zbl (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

#else  // !FLUID_BLOCK_SWEEP

__global__ void diffusion_xth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void diffusion_yth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

__global__ void diffusion_zth (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real dt
);

#endif // FLUID_BLOCK_SWEEP

// =====================================================================================================================
// evolved-state diagnostics

__global__ void inf_cell_flag (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif // RADIATION
    int *dev_bad_cell
);

// =====================================================================================================================
// field initialization

__global__ void init_rho_calc (
    real *dev_dustdens, const real *dev_initdens
);

__global__ void init_vel_calc (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif // DIFFUSION
);

// =====================================================================================================================
// primitive and conserved momentum synchronization

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
// radial optical-depth construction
__global__ void optdepth_calc (real *dev_optdepth, const real *dev_dustdens);
__global__ void optdepth_csum (real *dev_optdepth);
#endif // RADIATION

// =====================================================================================================================
// local drag and external-force update

__global__ void source_update (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    const real *dev_dustdens,
    #ifdef RADIATION
    const real *dev_optdepth, real beta_taper,
    #endif // RADIATION
    real dt
);

// =====================================================================================================================

#endif // GAMEDEV_FLUID_KERN_CUH
