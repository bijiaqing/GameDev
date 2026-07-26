#ifndef GRAFFITI_KERN_CUH
#define GRAFFITI_KERN_CUH

// Naming convention: velx/vely/velz and momx/momy/momz are used uniformly.
// Internally, the X and Z pairs are metric-weighted angular quantities:
// velx=R*v_x, momx=dens*velx, velz=y*v_z, and momz=dens*velz.

#include <const.cuh>

// =========================================================================================================================
// Fluid kernel declarations
// =========================================================================================================================

__global__ void f_rho_initial (real *dev_dustdens, const real *dev_initdens);
__global__ void f_vel_initial (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif
);

// Exact linear-drag relaxation + second-order exponential endpoint-force update at fixed cell center
__global__ void f_source_step (
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz, const real *dev_dustdens,
    #ifdef RADIATION
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
// Vacuum cells (dens < RHO_VAC) use velx = sqrt(R) = velx_K to prevent retrograde FARGO residual
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

// Positivity-subcycled implicit Crank-Nicolson diffusion of q = rho_d/rho_g
// The reconstructed diffusive mass flux carries upwind-cell specific momentum in the minimum conservative closure
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

// Record the first non-finite cell as idx+1 in dev_badstate (zero means valid)
// Radiation builds always include optical depth in the verified state
__global__ void finite_verify (
    const real *dev_dustdens,
    const real *dev_dustmomx, const real *dev_dustmomy, const real *dev_dustmomz,
    const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    #ifdef RADIATION
    const real *dev_optdepth,
    #endif
    int *dev_badstate
);

// Primitive recovery from momx, momy, and momz after conservative updates
__global__ void f_moment_getv (
    const real *dev_dustdens,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
);

// Conserved-state rebuild from velx, vely, and velz
__global__ void f_moment_setv (
    const real *dev_dustdens,
    real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    real *dev_dustmomx, real *dev_dustmomy, real *dev_dustmomz
);

// =========================================================================================================================

#endif // GRAFFITI_KERN_CUH
