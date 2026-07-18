#ifndef GRAFFITI_KERN_CUH
#define GRAFFITI_KERN_CUH

#include <const.cuh>

// =========================================================================================================================
// Fluid kernel declarations
// =========================================================================================================================

__global__ void f_rho_initial (real *dev_dustdens, const real *dev_initdens);
__global__ void f_vel_initial (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz);

// Exact linear-drag relaxation + second-order exponential endpoint-force update at fixed cell center
__global__ void f_source_term (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz, const real *dev_dustdens,
    #if defined(RADIATION)
    const real *dev_optdepth, real beta_taper,
    #endif
    real dt
);

// FARGO + PPM azimuthal advection — primitives recovered from the current conserved state
__global__ void f_advection_x (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, 
    real dt
);

// Volume-coordinate PPM radial advection — primitives recovered from the current conserved state
// Vacuum cells (rho < RHO_VAC) use lx = sqrt(R) = lx_K to prevent retrograde FARGO residual
__global__ void f_advection_y (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, 
    const real *dev_weight_y, real dt
);

// Volume-coordinate PPM polar advection — primitives recovered from the current conserved state
// Skipped whenever N_Z = 1 because no polar coordinate is active. Same vacuum fallback as Y
__global__ void f_advection_z (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    const real *dev_weight_z, real dt
);

// Positivity-subcycled implicit Crank-Nicolson diffusion of q = rho_d/rho_g.  The reconstructed
// diffusive mass flux carries donor-cell specific momentum in the minimum conservative closure.
__global__ void f_diffusion_x (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);
__global__ void f_diffusion_y (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);
__global__ void f_diffusion_z (
    real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, real dt
);

#ifdef RADIATION
__global__ void optdepth_calc (real *dev_optdepth, const real *dev_dustdens);
__global__ void optdepth_csum (real *dev_optdepth);
#endif // RADIATION

// CFL timestep: one thread per (Y,Z) ring so the X rate uses the FARGO ring-mean residual
__global__ void cfl_rate_calc (
    real *dev_cfl_rate, const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz
);

// Record the first non-finite cell as idx+1 in dev_bad_state (zero means valid).
__global__ void state_finite_check (
    int *dev_bad_state,
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz
    #ifdef RADIATION
    , const real *dev_optdepth, bool check_optdepth
    #endif
);

// Momentum recovery:  v_d = mom/rho  (after advection)
__global__ void f_moment_recv (
    const real *dev_dustdens, 
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz, 
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
);

// Momentum sync: enforce the common vacuum state, then set mom = rho*v_d
__global__ void f_moment_sync (
    const real *dev_dustdens,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz
);

// =========================================================================================================================

#endif // GRAFFITI_KERN_CUH
