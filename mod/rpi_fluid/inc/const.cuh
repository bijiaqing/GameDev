#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>    // for M_PI

using real  = double;

// =========================================================================================================================
// code units

const real  G           = 1.0;
const real  M_S         = 1.0;
const real  R_0         = 1.0;

// =========================================================================================================================
// mesh domain size and resolution

constexpr int  N_X      = 1024;             // number of grid cells in X direction (azimuth)
constexpr real X_MIN    = 0.0;              // minimum X boundary (azimuth)
constexpr real X_MAX    = 2.0*M_PI;         // maximum X boundary (azimuth)

constexpr int  N_Y      = 1024;             // number of grid cells in Y direction (radius)
constexpr real Y_MIN    = 0.5;              // minimum Y boundary (radius)
constexpr real Y_MAX    = 2.5;              // maximum Y boundary (radius)

constexpr int  N_Z      = 1;                // number of grid cells in Z direction (colattitude)
constexpr real Z_MIN    = 0.5*M_PI;
constexpr real Z_MAX    = 0.5*M_PI;

static_assert(N_X >= 1 && N_Y >= 1 && N_Z >= 1, "Every grid dimension must contain at least one cell.");
static_assert(N_X == 1 || X_MAX > X_MIN, "An active azimuthal grid requires X_MAX > X_MIN.");
static_assert(Y_MIN > 0.0 && Y_MAX > Y_MIN, "The radial domain must satisfy 0 < Y_MIN < Y_MAX.");
static_assert(Z_MIN >= 0.0 && Z_MAX <= M_PI, "Polar boundaries must lie within [0,pi].");
static_assert(N_Z > 1 || (Z_MIN == 0.5*M_PI && Z_MAX == 0.5*M_PI),
    "N_Z = 1 requires a radial or azimuthal-radial midplane model with Z_MIN = Z_MAX = pi/2.");
static_assert(N_Z == 1 || Z_MAX > Z_MIN, "An active polar grid requires Z_MAX > Z_MIN.");

#ifdef HALFDISK
static_assert(N_Z == 1 || Z_MAX == 0.5*M_PI,
    "HALFDISK requires its reflecting outer polar boundary at Z_MAX = pi/2.");
#else
static_assert(N_Z == 1 || (Z_MIN < 0.5*M_PI && Z_MAX > 0.5*M_PI),
    "Without HALFDISK, an active polar domain must span the midplane pi/2.");
#endif

// A resolved polar direction contains vertical stellar gravity. Without turbulent diffusion,
// the pressureless dust layer has no finite equilibrium thickness and collapses toward the midplane
// Reject that unsupported configuration at compile time
#ifndef DIFFUSION
static_assert(N_Z == 1, "N_Z > 1 requires the DIFFUSION flag for vertical dust support.");
#endif

// =========================================================================================================================
// gas parameters

const real  SIGMA_0     = 1.0e-02;
const real  ASPR_0      = 0.05;
const real  IDX_P       = -1.0;
const real  IDX_Q       = -0.4;

#ifdef DIFFUSION
#ifndef CONST_NU
const real  ALPHA       = 1.0e-03;
#else
const real  NU          = 1.0e-05;
#endif
#endif // DIFFUSION

// =========================================================================================================================
// dust parameters

const real  METAL_Z     = 1.0e-02;     // unconvolved dust-to-gas surface-density ratio
const real  ST_0        = 1.0e-03;     // reference Stokes number at R_0 (can be up to ~1)

#ifdef RADIATION
const real  BETA_0      = 1.0e+01;     // radiation-pressure-to-gravity ratio
const real  KAPPA_0     = 1.0e+05;

// Smoothly turn radiation on over time. This avoids an impulsive source transient from 
// applying the full force to a no-radiation drift-equilibrium initial state.
const real  T_BETA      = 2.0*M_PI;

#endif // RADIATION

#ifdef DIFFUSION
const real  SC_X        = 1.0e+20;      // Schmidt numbers
const real  SC_Y        = 1.0e+20;
const real  SC_Z        = 1.0;

const real  POS_LIMIT   = 0.9;          // maximum explicit-side CN coefficient sum per substep
#endif

// =========================================================================================================================
// time step and output

const int  SAVE_MAX     = 500;

const real DT_OUT       = 2.0*M_PI;     // output interval
const real DT_MAX       = 1.0e-01;      // maximum time step
const real OUTPUT_TIME_TOL = 1.0e-10;   // tolerance for detecting the scheduled output time

constexpr real CFL_NUM  = 0.5;          // CFL safety factor for advection

static_assert(CFL_NUM > 0.0 && CFL_NUM <= 0.5,
    "FARGO nearest-integer shifting requires 0 < CFL_NUM <= 0.5.");

// Cells below this single threshold are treated as numerical vacuum everywhere:
// they use Keplerian azimuthal angular momentum, zero drift, and are excluded from the CFL reduction
// Above it, primitive velocity is recovered exactly as momentum/density without a denominator floor, 
// so mom_d = rho_d*v_d remains an identity
const real RHO_VAC      = 1.0e-15;      // vacuum threshold for density and momentum recovery

// =========================================================================================================================
// CUDA numerical parameters

const int TPB   = 32;
const int N_G   = N_X*N_Y*N_Z;

const int NB_A  = N_G     / TPB + 1;   // per-cell kernels
const int NB_X  = N_Y*N_Z / TPB + 1;   // X-sweep (each thread = one Y-Z column)
const int NB_Y  = N_X*N_Z / TPB + 1;   // Y-sweep (each thread = one X-Z column)
const int NB_Z  = N_X*N_Y / TPB + 1;   // Z-sweep (each thread = one X-Y column)

// =========================================================================================================================

#endif // CONST_CUH
