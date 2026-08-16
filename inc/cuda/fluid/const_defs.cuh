#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath>       // M_PI

using real = double;

// =========================================================================================================================
// code units

const real  G           = 1.0;
const real  M_S         = 1.0;
const real  R_0         = 1.0;

// =========================================================================================================================
// mesh domain size and resolution

constexpr int  N_X      = 1024;
constexpr real X_MIN    = 0.0;
constexpr real X_MAX    = 2.0*M_PI;

constexpr int  N_Y      = 1024;
constexpr real Y_MIN    = 0.5;
constexpr real Y_MAX    = 2.5;

constexpr int  N_Z      = 1;
constexpr real Z_MIN    = 0.5*M_PI;
constexpr real Z_MAX    = 0.5*M_PI;

// =========================================================================================================================
// gas parameters

const real  SIGMA_0     = 1.0e-02;
const real  ASPR_0      = 0.05;
const real  IDX_P       = -1.0;         // radial power-law index of the gas surface density
const real  IDX_Q       = -0.4;         // radial power-law index of the gas temperature

#ifdef DIFFUSION
#ifndef CONST_NU  // CONST_ALPHA
const real  ALPHA       = 1.0e-03;
#else             // CONST_NU
const real  NU          = 1.0e-05;
#endif // CONST_NU
#endif // DIFFUSION

// =========================================================================================================================
// dust parameters

const real  METAL_Z     = 1.0e-02;      // dust-to-gas surface-density ratio for initialization
const real  STOKES_0    = 1.0e-03;

#ifdef RADIATION
const real  BETA_0      = 1.0e+01;      // radiation-pressure-to-gravity ratio
const real  KAPPA_0     = 5.0e+04;      // opacity coefficient
const real  T_BETA      = 2.0*M_PI;     // smoothly turn radiation on over time
#endif

#ifdef DIFFUSION
const real  SCHMIDT_X   = 1.0e+20;      // azimuthal Schmidt number in spherical X
const real  SCHMIDT_Y   = 1.0e+20;      // radial Schmidt number in spherical Y
const real  SCHMIDT_Z   = 1.0;          // polar Schmidt number in spherical Z
const real  POS_LIMIT   = 0.9;          // limit for the dust density positivity limiter
#endif

// =========================================================================================================================
// time step and output

const int  SAVE_MAX     = 100;
const real DT_OUT       = 2.0*M_PI;     // output interval
const real DT_MAX       = 1.0e-01;

constexpr real CFL_DYN  = 0.45;         // CFL number for the explicit advection step (0 < CFL_DYN <= 0.5)
const real RHO_VAC      = 1.0e-30;      // vacuum density for the dust density positivity limiter

// =========================================================================================================================
// CUDA kernel launch parameters

const int TPB   = 64;
const int N_G   = N_X*N_Y*N_Z;

const int NB_G  = N_G     / TPB + 1;
const int NB_X  = N_Y*N_Z / TPB + 1;
const int NB_Y  = N_X*N_Z / TPB + 1;
const int NB_Z  = N_X*N_Y / TPB + 1;

// =========================================================================================================================
// compile-time sanity checks

static_assert(
    N_X > 1,
    "an active azimuthal dimension with N_X > 1 is required"
);
static_assert(
    N_Y > 1,
    "an active radial dimension with N_Y > 1 is required"
);
static_assert(
    N_Z >= 1,
    "the polar dimension must contain at least one cell"
);
static_assert(
    X_MAX > X_MIN,
    "the azimuthal domain must satisfy X_MAX > X_MIN"
);
static_assert(
    Y_MIN > 0.0 && Y_MAX > Y_MIN,
    "the radial domain must satisfy 0 < Y_MIN < Y_MAX"
);
static_assert(
    Z_MIN > 0.0 && Z_MAX < M_PI,
    "the polar domain must lie within (0, pi)"
);
static_assert(
    N_Z > 1 || (Z_MIN == 0.5*M_PI && Z_MAX == 0.5*M_PI),
    "N_Z = 1 requires a model with Z_MIN = Z_MAX = pi/2"
);
static_assert(
    N_Z == 1 || Z_MAX > Z_MIN,
    "an active polar grid requires Z_MAX > Z_MIN"
);

#ifdef HALF_DISK
static_assert(
    N_Z == 1 || Z_MAX == 0.5*M_PI,
    "HALF_DISK requires its reflecting outer polar boundary at Z_MAX = pi/2"
);
#else
static_assert(
    N_Z == 1 || (Z_MIN < 0.5*M_PI && Z_MAX > 0.5*M_PI),
    "an active polar domain without HALF_DISK must span the midplane pi/2"
);
#endif

#ifndef DIFFUSION
static_assert(
    N_Z == 1,
    "N_Z > 1 requires the DIFFUSION flag for vertical dust support"
);
#endif

static_assert(
    CFL_DYN > 0.0 && CFL_DYN <= 0.5,
    "FARGO nearest-integer shifting requires 0 < CFL_DYN <= 0.5"
);

// =========================================================================================================================

#endif
