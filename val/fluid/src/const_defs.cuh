#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath> // M_PI

using real = double;

// test runners may override these nonphysical verification controls while each model keeps its constants in this header
#ifdef TEST_RES
constexpr int VERIFY_RES = TEST_RES;
#else  // !TEST_RES
constexpr int VERIFY_RES = 64;
#endif // TEST_RES

#ifdef TEST_CFL
constexpr real VERIFY_CFL = TEST_CFL;
#else  // !TEST_CFL
constexpr real VERIFY_CFL = 0.5;
#endif // TEST_CFL

#ifdef TEST_POWER
constexpr real VERIFY_POWER = TEST_POWER;
#else  // !TEST_POWER
constexpr real VERIFY_POWER = -1.0;
#endif // TEST_POWER

#ifdef TEST_SHIFT
constexpr real VERIFY_SHIFT = TEST_SHIFT;
#else  // !TEST_SHIFT
constexpr real VERIFY_SHIFT = 3.25;
#endif // TEST_SHIFT

const real G   = 1.0;
const real M_S = 1.0;
const real R_0 = 1.0;

// test constants deliberately replace a production model's physical setup with the smallest grid that isolates one
// claim
// refine only the direction under test for isolated kernels and refine both active directions for ring tests; four
// cells in an inactive transverse direction are enough to expose indexing mistakes without making every convergence
// run expensive
#if defined(VERIFY_STARTUP_3D)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = VERIFY_RES;
#elif defined(VERIFY_DIFFUSION_POSLIMIT) && defined(TEST_DIRECTION_Y)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#elif defined(VERIFY_DIFFUSION_POSLIMIT) && defined(TEST_DIRECTION_Z)
constexpr int N_X = 4;
constexpr int N_Y = 4;
constexpr int N_Z = VERIFY_RES;
#elif defined(VERIFY_DIFFUSION_POSLIMIT)
constexpr int N_X = VERIFY_RES;
constexpr int N_Y = 1;
constexpr int N_Z = 1;
#elif defined(VERIFY_X_TRANSPORT) || defined(VERIFY_X_DIFFUSION) || defined(VERIFY_X_WEDGE_TRANSPORT) \
    || defined(VERIFY_X_WEDGE_DIFFUSION)
constexpr int N_X = VERIFY_RES;
constexpr int N_Y = 4;
constexpr int N_Z = 1;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_OUTFLOW_2D) || defined(VERIFY_Y_DIFFUSION_CYL) \
    || defined(VERIFY_OPTDEPTH) || defined(VERIFY_ATTENUATION_2D)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#elif defined(VERIFY_Y_TRANSPORT_SPH) || defined(VERIFY_Y_DIFFUSION_SPH)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 4;
#elif defined(VERIFY_Z_TRANSPORT) || defined(VERIFY_Z_OUTFLOW) || defined(VERIFY_Z_REFLECT) \
    || defined(VERIFY_Z_DIFFUSION)
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
#else  // other VERIFY_* cases
#error "A VERIFY_* model selector must be defined in flags.mk"
#endif // VERIFY_* case selection

#if defined(VERIFY_X_WEDGE_TRANSPORT) || defined(VERIFY_X_WEDGE_DIFFUSION)
constexpr real X_MIN = -0.4;
constexpr real X_MAX =  0.8;
#else  // !(VERIFY_X_WEDGE_TRANSPORT || VERIFY_X_WEDGE_DIFFUSION)
constexpr real X_MIN = 0.0;
constexpr real X_MAX = 2.0*M_PI;
#endif // VERIFY_X_WEDGE_TRANSPORT || VERIFY_X_WEDGE_DIFFUSION
#if defined(VERIFY_DIFFUSION_POSLIMIT)
constexpr real Y_MIN = 0.9;
constexpr real Y_MAX = 1.1;
#else  // !VERIFY_DIFFUSION_POSLIMIT
#ifdef VERIFY_STARTUP_3D
constexpr real Y_MIN = 0.8;
constexpr real Y_MAX = 1.2;
#else  // !VERIFY_STARTUP_3D
constexpr real Y_MIN = 0.5;
constexpr real Y_MAX = 2.5;
#endif // VERIFY_STARTUP_3D
#endif // VERIFY_DIFFUSION_POSLIMIT

// polar diffusion uses a hemisphere with natural zero-flux boundaries; other 3D tests avoid the coordinate poles, while
// a 2D radial-azimuthal model is represented by one zero-width cell at the midplane
#if defined(VERIFY_Z_DIFFUSION)
constexpr real Z_MIN = 0.0;
constexpr real Z_MAX = 0.5*M_PI;
#elif defined(VERIFY_STARTUP_3D)
constexpr real Z_MIN = 0.5*M_PI - 0.4;
constexpr real Z_MAX = 0.5*M_PI + 0.4;
#elif defined(VERIFY_Z_REFLECT)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = 0.5*M_PI;
#elif defined(VERIFY_Z_TRANSPORT) || defined(VERIFY_Z_OUTFLOW) \
    || (defined(VERIFY_DIFFUSION_POSLIMIT) && defined(TEST_DIRECTION_Z)) || defined(VERIFY_Y_TRANSPORT_SPH) \
    || defined(VERIFY_Y_DIFFUSION_SPH)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = M_PI - 0.35;
#else  // other VERIFY_* cases
constexpr real Z_MIN = 0.5*M_PI;
constexpr real Z_MAX = 0.5*M_PI;
#endif // VERIFY_* case selection

#ifdef VERIFY_STARTUP_3D
const real SIGMA_0 = 1.3;
const real ASPR_0  = 0.12;
#else  // !VERIFY_STARTUP_3D
const real SIGMA_0 = 1.0;
const real ASPR_0  = 0.5;
#endif // VERIFY_STARTUP_3D

// radiation-supported ring equilibria require a gas profile consistent with the chosen beta; the remaining tests use
// the simpler non-radiative exponent
#ifdef VERIFY_STARTUP_3D
// p=3/2 and q=0 remove pressure-supported radial drift, isolating the polar advection-diffusion balance
const real IDX_P = 1.5;
#elif defined(VERIFY_RING_RADIATION)
const real IDX_P = 1.2;
#else  // !(VERIFY_STARTUP_3D || VERIFY_RING_RADIATION)
const real IDX_P = 2.0;
#endif // VERIFY_STARTUP_3D / VERIFY_RING_RADIATION

#ifdef VERIFY_STARTUP_3D
const real IDX_Q = 0.0;
#else  // !VERIFY_STARTUP_3D
const real IDX_Q = -1.0;
#endif // VERIFY_STARTUP_3D

#ifdef DIFFUSION
#ifdef VERIFY_STARTUP_3D
const real ALPHA = 4.0e-3;
#else  // !VERIFY_STARTUP_3D
const real NU = 5.0e-2;
#endif // VERIFY_STARTUP_3D
#endif // DIFFUSION

#ifdef VERIFY_STARTUP_3D
const real METAL_Z = 1.7e-2;
const real STOKES_0 = 3.0e-2;
#else  // !VERIFY_STARTUP_3D
const real METAL_Z = 1.0e-2;
#if defined(VERIFY_X_DIFFUSION) || defined(VERIFY_X_WEDGE_DIFFUSION) || defined(VERIFY_Y_DIFFUSION_CYL) \
    || defined(VERIFY_Y_DIFFUSION_SPH) || defined(VERIFY_Z_DIFFUSION) || defined(VERIFY_DIFFUSION_POSLIMIT)
// the tracer limit keeps the isolated Fourier/Bessel/Legendre references at constant D=nu/Sc
const real STOKES_0 = 0.0;
#else  // other VERIFY_* cases
const real STOKES_0 = 1.0e-1;
#endif // VERIFY_* case selection
#endif // VERIFY_STARTUP_3D

#ifdef RADIATION
#ifdef VERIFY_RING_RADIATION
// ring tests isolate a known unattenuated radiation force by setting opacity to zero; the standalone optical-depth test
// uses unit opacity and beta only to satisfy the shared production parameter interface
const real BETA_0 = 2.0e-1;
const real KAPPA_0 = 0.0;
#else  // !VERIFY_RING_RADIATION
const real BETA_0 = 1.0;
const real KAPPA_0 = 1.0;
#endif // VERIFY_RING_RADIATION
const real T_BETA = 1.0;
#endif // RADIATION

#ifdef DIFFUSION
// a Schmidt number of one activates diffusion in the direction being tested; a numerically enormous value makes
// diffusion negligible in every other direction while preserving the same production kernel interface
#if defined(VERIFY_X_DIFFUSION) || defined(VERIFY_X_WEDGE_DIFFUSION) || defined(VERIFY_RING_DIFFUSION) \
    || (defined(VERIFY_DIFFUSION_POSLIMIT) && !defined(TEST_DIRECTION_Y) && !defined(TEST_DIRECTION_Z))
const real SCHMIDT_X = 1.0;
#else  // other VERIFY_* cases
const real SCHMIDT_X = 1.0e300;
#endif // VERIFY_* case selection

#if defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH) \
    || (defined(VERIFY_DIFFUSION_POSLIMIT) && defined(TEST_DIRECTION_Y))
const real SCHMIDT_Y = 1.0;
#else  // !(VERIFY_Y_DIFFUSION_CYL || VERIFY_Y_DIFFUSION_SPH || (VERIFY_DIFFUSION_POSLIMIT && TEST_DIRECTION_Y))
const real SCHMIDT_Y = 1.0e300;
#endif // VERIFY_Y_DIFFUSION_CYL || VERIFY_Y_DIFFUSION_SPH || (VERIFY_DIFFUSION_POSLIMIT && TEST_DIRECTION_Y)

#ifdef VERIFY_STARTUP_3D
const real SCHMIDT_Z = 2.0;
#elif defined(VERIFY_Z_DIFFUSION)  || (defined(VERIFY_DIFFUSION_POSLIMIT) && defined(TEST_DIRECTION_Z))
const real SCHMIDT_Z = 1.0;
#else  // !(VERIFY_STARTUP_3D || (VERIFY_Z_DIFFUSION || (VERIFY_DIFFUSION_POSLIMIT && TEST_DIRECTION_Z)))
const real SCHMIDT_Z = 1.0e300;
#endif // VERIFY_STARTUP_3D / (VERIFY_Z_DIFFUSION || (VERIFY_DIFFUSION_POSLIMIT && TEST_DIRECTION_Z))

const real POS_LIMIT = 0.9;
#endif // DIFFUSION

const int SAVE_MAX = 1;
const real DT_OUT  = 1.0;
const real DT_MAX  = 1.0;
constexpr real CFL_DYN = VERIFY_CFL;
const real RHO_VAC = 1.0e-15;

// kernel launch counts correspond to one thread per cell, x ring, y column, or z column; the extra block is harmless
// because every kernel begins with an out-of-range return
const int TPB  = 64;
const int N_G  = N_X*N_Y*N_Z;
const int NB_G = N_G     / TPB + 1;
const int NB_X = N_Y*N_Z / TPB + 1;
const int NB_Y = N_X*N_Z / TPB + 1;
const int NB_Z = N_X*N_Y / TPB + 1;

const real VERIFY_Q0  = 1.0;
const real VERIFY_EPS = 0.1;
const int  VERIFY_M   = 2;
const real VERIFY_A   = 0.2;
const real VERIFY_RATE_Z = 0.15;
const real VERIFY_D   = 5.0e-2;

// final times are long enough to produce measurable translation or decay but short enough to keep fine-grid suites
// practical
#if defined(VERIFY_X_TRANSPORT)
const real VERIFY_TEND = 2.0*M_PI;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH)
const real VERIFY_TEND = 0.25;
#elif defined(VERIFY_Y_OUTFLOW_2D)
const real VERIFY_TEND = 1.0;
#elif defined(VERIFY_Z_TRANSPORT)
const real VERIFY_TEND = 0.30;
#elif defined(VERIFY_Z_OUTFLOW) || defined(VERIFY_Z_REFLECT)
const real VERIFY_TEND = 1.0;
#elif defined(VERIFY_SOURCE_DRAG)
const real VERIFY_TEND = 1.0;
#elif defined(VERIFY_RING)
const real VERIFY_TEND = 1.0;
#else  // other VERIFY_* cases
const real VERIFY_TEND = 0.5;
#endif // VERIFY_* case selection

// fail during compilation when a test configuration violates assumptions made by the production grid helpers or kernels
static_assert(N_X > 1, "Verification models require N_X > 1");
static_assert(N_Y >= 1 && N_Z >= 1, "All grid dimensions must be nonempty");
static_assert(X_MAX > X_MIN, "The azimuthal domain must be active");
static_assert(Y_MIN > 0.0 && Y_MAX > Y_MIN, "Invalid radial domain");
static_assert(Z_MIN >= 0.0 && Z_MAX <= M_PI, "Invalid polar domain");
static_assert(N_Z > 1 || (Z_MIN == 0.5*M_PI && Z_MAX == 0.5*M_PI), "N_Z=1 must be the midplane model");
static_assert(N_Z == 1 || Z_MAX > Z_MIN, "An active polar grid needs nonzero extent");

#ifdef HALF_DISK
static_assert(N_Z == 1 || Z_MAX == 0.5*M_PI, "HALF_DISK must end at the midplane");
#else  // !HALF_DISK
static_assert(N_Z == 1 || (Z_MIN < 0.5*M_PI && Z_MAX > 0.5*M_PI), "A full polar model must span the midplane");
#endif // HALF_DISK

#ifndef DIFFUSION
static_assert(N_Z == 1, "Production guards require DIFFUSION when N_Z>1");
#endif // !DIFFUSION

static_assert(CFL_DYN > 0.0 && CFL_DYN <= 0.5, "Verification CFL must be in (0,0.5]");

#endif // !CONST_DEFS_CUH
