#ifndef GAMEDEV_VAL_PAPER_DIFFUSION_CONST_DEFS_CUH
#define GAMEDEV_VAL_PAPER_DIFFUSION_CONST_DEFS_CUH

#include <cmath>  // M_PI
#include <string> // std::string

#include <gpu.cuh>
using curs = gpuRandState;
using real = double;
using real3 = double3;

// equal-mass, single-size radial diffusion test in units with GM = R_0 = Omega_0 = 1
constexpr real G = 1.0, M_S = 1.0, R_0 = 1.0, S_0 = 1.0;
constexpr int N_P = 1048576;
constexpr int N_X = 1, N_Y = 128, N_Z = 1;
constexpr real X_MIN = -M_PI, X_MAX = M_PI;
// exp(-3) and exp(3), as compile-time constants usable by GPU device functions
constexpr real Y_MIN = 0.049787068367863944, Y_MAX = 20.085536923187668;
constexpr real Z_MIN = 0.5*M_PI, Z_MAX = 0.5*M_PI;
constexpr int N_G = N_X*N_Y*N_Z;

// the production closure gives D_R = 1e-4 R^2/(1+St^2); deterministic motion is disabled
constexpr real ASPR_0 = 0.05, IDX_P = -1.0, IDX_Q = 0.5;
constexpr real SIGMA_0 = 1.0, METAL_Z = 0.01, RHO_0 = 1.0;
constexpr real STOKES_0 = DIFFUSION_STOKES;
constexpr real ALPHA = 0.04, SCHMIDT_X = 1.0, SCHMIDT_R = 1.0, SCHMIDT_Z = 1.0;
constexpr real RING_LOG_WIDTH = 0.05;
constexpr int RING_INIT_SEED = 17;
constexpr real DT_MAX = 1.0 + STOKES_0*STOKES_0;
constexpr real CFL_DYN = 0.45; // recorded for completeness; model dyn_rate_calc fixes dt
constexpr int SAVE_MAX = 10;
constexpr real DT_OUT = 100.0*(1.0 + STOKES_0*STOKES_0);
constexpr int LIN_BASE = 1;

struct swarm { real3 position; real3 velocity; };
static_assert(sizeof(swarm) == 6*sizeof(real), "analysis expects six packed doubles per particle");
constexpr int TPB = 64;
constexpr int NB_P = N_P / TPB + 1, NB_G = N_G / TPB + 1;
constexpr int NB_X = N_Y*N_Z / TPB + 1, NB_Y = N_X*N_Z / TPB + 1;
#endif // GAMEDEV_VAL_PAPER_DIFFUSION_CONST_DEFS_CUH
