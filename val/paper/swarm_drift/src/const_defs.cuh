#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH
#include <cmath>
#include <string>
using real = double;
using real3 = double3;

// GM = R_0 = Omega_0 = 1. Single-size, deterministic particle transport only.
const real G = 1.0, M_S = 1.0, R_0 = 1.0, S_0 = 1.0;
const int N_P = 1;
const int N_X = 8, N_Y = 16, N_Z = 1;
const real X_MIN = -M_PI, X_MAX = M_PI;
const real Y_MIN = 0.5, Y_MAX = 1.5;
const real Z_MIN = 0.5*M_PI, Z_MAX = 0.5*M_PI;
const int N_G = N_X*N_Y*N_Z;

// Match F&M Eq. 39's gas speed, despite the different pressure-support closure.
const real ASPR_0 = 0.05, IDX_P = 1.0, IDX_Q = -1.0;
const real SIGMA_0 = 1.0, METAL_Z = 0.01, RHO_0 = 1.0;
const real STOKES_0 = DRIFT_STOKES;
const real DT_MAX = DRIFT_DT;
const real CFL_DYN = 0.45; // recorded for completeness; model dyn_rate_calc fixes dt
const int SAVE_MAX = 10;
const real DT_OUT = (STOKES_0 > 1.0 ? STOKES_0 : 1.0);
const int LIN_BASE = 1;

struct swarm { real3 position; real3 velocity; };
const int TPB = 64;
const int NB_P = N_P/TPB + 1, NB_G = N_G/TPB + 1;
const int NB_X = N_Y*N_Z/TPB + 1, NB_Y = N_X*N_Z/TPB + 1;
#endif
