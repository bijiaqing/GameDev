#ifndef CONST_CUH
#define CONST_CUH

#include <cmath> // M_PI

using real = double;

// The Makefile normally supplies these VERIFY_* macros from run_model.py.  Defaults keep a model directly buildable when a
// particular command-line parameter is omitted.
#ifndef VERIFY_RES
#define VERIFY_RES 64
#endif

#ifndef VERIFY_CFL
#define VERIFY_CFL 0.5
#endif

#ifndef VERIFY_POWER
#define VERIFY_POWER -1.0
#endif

#ifndef VERIFY_SHIFT
#define VERIFY_SHIFT 3.25
#endif

const real G   = 1.0;
const real M_S = 1.0;
const real R_0 = 1.0;

// Refine only the direction under test for isolated kernels and refine both active directions for ring tests.  Four cells in
// an inactive transverse direction are enough to expose indexing mistakes without making every convergence run expensive.
#if defined(VERIFY_X_TRANSPORT) || defined(VERIFY_X_DIFFUSION)
constexpr int N_X = VERIFY_RES;
constexpr int N_Y = 4;
constexpr int N_Z = 1;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_OPTDEPTH)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#elif defined(VERIFY_Y_TRANSPORT_SPH) || defined(VERIFY_Y_DIFFUSION_SPH)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 4;
#elif defined(VERIFY_Z_TRANSPORT) || defined(VERIFY_Z_DIFFUSION)
constexpr int N_X = 4;
constexpr int N_Y = 4;
constexpr int N_Z = VERIFY_RES;
#elif defined(VERIFY_SOURCE_DRAG)
constexpr int N_X = 8;
constexpr int N_Y = 1;
constexpr int N_Z = 1;
#elif defined(VERIFY_RING)
constexpr int N_X = VERIFY_RES;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#else
#error "A VERIFY_* model selector must be defined in flags.mk"
#endif

constexpr real X_MIN = 0.0;
constexpr real X_MAX = 2.0*M_PI;
constexpr real Y_MIN = 0.5;
constexpr real Y_MAX = 2.5;

// Polar diffusion uses a hemisphere with natural zero-flux boundaries.  Other 3D tests avoid the coordinate poles, while a
// 2D radial-azimuthal model is represented by one zero-width cell at the midplane.
#if defined(VERIFY_Z_DIFFUSION)
constexpr real Z_MIN = 0.0;
constexpr real Z_MAX = 0.5*M_PI;
#elif defined(VERIFY_Z_TRANSPORT) || defined(VERIFY_Y_TRANSPORT_SPH) || defined(VERIFY_Y_DIFFUSION_SPH)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = M_PI - 0.35;
#else
constexpr real Z_MIN = 0.5*M_PI;
constexpr real Z_MAX = 0.5*M_PI;
#endif

const real SIGMA_0 = 1.0;
const real ASPR_0  = 0.5;

// Radiation-supported ring equilibria require a gas profile consistent with the chosen beta; the remaining tests use the
// simpler non-radiative exponent.
#ifdef VERIFY_RING_RADIATION
const real IDX_P = 1.2;
#else
const real IDX_P = 2.0;
#endif

const real IDX_Q = -1.0;

#ifdef DIFFUSION
const real NU = 5.0e-2;
#endif

const real METAL_Z = 1.0e-2;
const real STOKES_0 = 1.0e-1;

#ifdef RADIATION
#ifdef VERIFY_RING_RADIATION
// Ring tests isolate a known unattenuated radiation force by setting opacity to zero.  The standalone optical-depth test uses
// unit opacity and beta only to satisfy the shared production parameter interface.
const real BETA_0 = 2.0e-1;
const real KAPPA_0 = 0.0;
#else
const real BETA_0 = 1.0;
const real KAPPA_0 = 1.0;
#endif
const real T_BETA = 1.0;
#endif

#ifdef DIFFUSION
// A Schmidt number of one activates diffusion in the direction being tested.  A numerically enormous value makes diffusion
// negligible in every other direction while preserving the same production kernel interface.
#if defined(VERIFY_X_DIFFUSION) || defined(VERIFY_RING_DIFFUSION)
const real SC_X = 1.0;
#else
const real SC_X = 1.0e300;
#endif

#if defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH)
const real SC_Y = 1.0;
#else
const real SC_Y = 1.0e300;
#endif

#if defined(VERIFY_Z_DIFFUSION)
const real SC_Z = 1.0;
#else
const real SC_Z = 1.0e300;
#endif

const real POS_LIMIT = 0.9;
#endif

const int SAVE_MAX = 1;
const real DT_OUT  = 1.0;
const real DT_MAX  = 1.0;
constexpr real CFL_NUM = VERIFY_CFL;
const real RHO_VAC = 1.0e-15;

// Kernel launch counts correspond to one thread per cell, x ring, y column, or z column.  The extra block is harmless because
// every kernel begins with an out-of-range return.
const int TPB  = 32;
const int N_G  = N_X*N_Y*N_Z;
const int NB_A = N_G     / TPB + 1;
const int NB_X = N_Y*N_Z / TPB + 1;
const int NB_Y = N_X*N_Z / TPB + 1;
const int NB_Z = N_X*N_Y / TPB + 1;

const real VERIFY_Q0  = 1.0;
const real VERIFY_EPS = 0.1;
const int  VERIFY_M   = 2;
const real VERIFY_A   = 0.2;
const real VERIFY_LZ  = 0.15;
const real VERIFY_D   = 5.0e-2;

// Final times are long enough to produce measurable translation or decay but short enough to keep fine-grid suites practical.
#if defined(VERIFY_X_TRANSPORT)
const real VERIFY_TEND = 2.0*M_PI;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH)
const real VERIFY_TEND = 0.25;
#elif defined(VERIFY_Z_TRANSPORT)
const real VERIFY_TEND = 0.30;
#elif defined(VERIFY_SOURCE_DRAG)
const real VERIFY_TEND = 1.0;
#elif defined(VERIFY_RING)
const real VERIFY_TEND = 1.0;
#else
const real VERIFY_TEND = 0.5;
#endif

// Fail during compilation when a test configuration violates assumptions made by the production grid helpers or kernels.
static_assert(N_X > 1, "Verification models require N_X > 1");
static_assert(N_Y >= 1 && N_Z >= 1, "All grid dimensions must be nonempty");
static_assert(X_MAX > X_MIN, "The azimuthal domain must be active");
static_assert(Y_MIN > 0.0 && Y_MAX > Y_MIN, "Invalid radial domain");
static_assert(Z_MIN >= 0.0 && Z_MAX <= M_PI, "Invalid polar domain");
static_assert(N_Z > 1 || (Z_MIN == 0.5*M_PI && Z_MAX == 0.5*M_PI), "N_Z=1 must be the midplane model");
static_assert(N_Z == 1 || Z_MAX > Z_MIN, "An active polar grid needs nonzero extent");

#ifdef HALFDISK
static_assert(N_Z == 1 || Z_MAX == 0.5*M_PI, "HALFDISK must end at the midplane");
#else
static_assert(N_Z == 1 || (Z_MIN < 0.5*M_PI && Z_MAX > 0.5*M_PI), "A full polar model must span the midplane");
#endif

#ifndef DIFFUSION
static_assert(N_Z == 1, "Production guards require DIFFUSION when N_Z>1");
#endif

static_assert(CFL_NUM > 0.0 && CFL_NUM <= 0.5, "Verification CFL must be in (0,0.5]");

#endif
