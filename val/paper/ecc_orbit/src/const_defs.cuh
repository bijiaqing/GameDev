#ifndef GAMEDEV_VAL_PAPER_ECC_ORBIT_CONST_DEFS_CUH
#define GAMEDEV_VAL_PAPER_ECC_ORBIT_CONST_DEFS_CUH

#include <cmath>  // M_PI
#include <string> // std::string
using real = double;
using real3 = double3;

// GM = semimajor axis = 1; exact zero-drag eccentric orbit
constexpr real G = 1.0, M_S = 1.0, R_0 = 1.0, S_0 = 1.0;
constexpr int N_P = 1;
constexpr int N_X = 8, N_Y = 16, N_Z = 1;
constexpr real X_MIN = -M_PI, X_MAX = M_PI;
constexpr real Y_MIN = 0.3, Y_MAX = 2.0;
constexpr real Z_MIN = 0.5*M_PI, Z_MAX = 0.5*M_PI;
constexpr int N_G = N_X*N_Y*N_Z;

// gas parameters are required by the production host interface; the orbit update ignores drag
constexpr real ASPR_0 = 0.05, IDX_P = 1.0, IDX_Q = -1.0;
constexpr real SIGMA_0 = 1.0, METAL_Z = 0.01, RHO_0 = 1.0;
constexpr real STOKES_0 = 1.0; // unused by the zero-drag specialization
constexpr real ORBIT_PERIOD = 2.0*M_PI;
constexpr real DT_MAX = ORBIT_PERIOD / ORBIT_STEPS_PER_PERIOD;
constexpr real CFL_DYN = 0.45; // recorded for completeness; model dyn_rate_calc fixes dt
constexpr int SAVE_MAX = 2000;
constexpr real DT_OUT = ORBIT_PERIOD / 20.0;
constexpr int LIN_BASE = 1;

struct swarm { real3 position; real3 velocity; };
constexpr int TPB = 64;
constexpr int NB_P = N_P / TPB + 1, NB_G = N_G / TPB + 1;
constexpr int NB_X = N_Y*N_Z / TPB + 1, NB_Y = N_X*N_Z / TPB + 1;
#endif // GAMEDEV_VAL_PAPER_ECC_ORBIT_CONST_DEFS_CUH
