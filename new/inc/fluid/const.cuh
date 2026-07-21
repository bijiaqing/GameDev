#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>

using real  = double;




const real  G           = 1.0;
const real  M_S         = 1.0;
const real  R_0         = 1.0;




constexpr int  N_X      = 1024;
constexpr real X_MIN    = 0.0;
constexpr real X_MAX    = 2.0*M_PI;

constexpr int  N_Y      = 1024;
constexpr real Y_MIN    = 0.5;
constexpr real Y_MAX    = 2.5;

constexpr int  N_Z      = 1;
constexpr real Z_MIN    = 0.5*M_PI;
constexpr real Z_MAX    = 0.5*M_PI;




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
#endif




const real  METAL_Z     = 1.0e-02;
const real  STOKES_0    = 1.0e-03;

#ifdef RADIATION
const real  BETA_0      = 1.0e+01;
const real  KAPPA_0     = 5.0e+04;

const real  T_BETA      = 2.0*M_PI;

#endif

#ifdef DIFFUSION
const real  SC_X        = 1.0e+20;
const real  SC_Y        = 1.0e+20;
const real  SC_Z        = 1.0;

const real  POS_LIMIT   = 0.9;
#endif




const int  SAVE_MAX     = 100;

const real DT_OUT       = 2.0*M_PI;
const real DT_MAX       = 1.0e-01;

constexpr real CFL_NUM  = 0.5;

const real RHO_VAC      = 1.0e-30;




const int TPB   = 32;
const int N_G   = N_X*N_Y*N_Z;

const int NB_A  = N_G     / TPB + 1;
const int NB_X  = N_Y*N_Z / TPB + 1;
const int NB_Y  = N_X*N_Z / TPB + 1;
const int NB_Z  = N_X*N_Y / TPB + 1;




static_assert(
    N_X > 1,
    "rpi_fluid requires an active azimuthal dimension with N_X > 1"
);
static_assert(
    N_Y >= 1 && N_Z >= 1,
    "The radial and polar grid dimensions must contain at least one cell"
);
static_assert(
    X_MAX > X_MIN,
    "The azimuthal domain must satisfy X_MAX > X_MIN"
);
static_assert(
    Y_MIN > 0.0 && Y_MAX > Y_MIN,
    "The radial domain must satisfy 0 < Y_MIN < Y_MAX"
);
static_assert(
    Z_MIN >= 0.0 && Z_MAX <= M_PI,
    "Polar boundaries must lie within [0,pi]"
);
static_assert(
    N_Z > 1 || (Z_MIN == 0.5*M_PI && Z_MAX == 0.5*M_PI),
    "N_Z = 1 requires a 2D azimuthal-radial midplane model with Z_MIN = Z_MAX = pi/2"
);
static_assert(
    N_Z == 1 || Z_MAX > Z_MIN,
    "An active polar grid requires Z_MAX > Z_MIN"
);

#ifdef HALFDISK
static_assert(
    N_Z == 1 || Z_MAX == 0.5*M_PI,
    "HALFDISK requires its reflecting outer polar boundary at Z_MAX = pi/2"
);
#else
static_assert(
    N_Z == 1 || (Z_MIN < 0.5*M_PI && Z_MAX > 0.5*M_PI),
    "Without HALFDISK, an active polar domain must span the midplane pi/2"
);
#endif

#ifndef DIFFUSION
static_assert(
    N_Z == 1,
    "N_Z > 1 requires the DIFFUSION flag for vertical dust support"
);
#endif

static_assert(
    CFL_NUM > 0.0 && CFL_NUM <= 0.5,
    "FARGO nearest-integer shifting requires 0 < CFL_NUM <= 0.5"
);



#endif
