// Model constants; all collision evolution uses the current production implementation.
#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath>
#include <string>

#if defined(COLLISION) || defined(DIFFUSION)
#include <gpu.cuh>
#endif

#ifdef COLLISION_KDTREE
#include <kdtree/builder.h>
#endif

#if defined(COLLISION) || defined(DIFFUSION)
using curs = gpuRandState;
#endif

#ifdef COLLISION_KDTREE
using kdtree_boxf = kdtree::box_t<float3>;
#endif

using real  = double;
using real3 = double3;

const real  G           = 1.0;
const real  M_S         = 1.0;
const real  R_0         = 1.0;
const real  S_0         = 1.0;

#ifndef SWEEP_N_P
#define SWEEP_N_P 1000000
#endif
const int   N_P         = SWEEP_N_P;

const int   N_X         = 2;
const real  X_MIN       = -M_PI;
const real  X_MAX       = +M_PI;

const int   N_Y         = 100;
const real  Y_MIN       = 0.5;
const real  Y_MAX       = 1.5;

const int   N_Z         = 1;
const real  Z_MIN       = 0.5*M_PI;
const real  Z_MAX       = 0.5*M_PI;

const int   N_G         = N_X*N_Y*N_Z;

#ifdef COLLISION
const bool  X_WEDGE     = N_X > 1
    && static_cast<float>(X_MAX) - static_cast<float>(X_MIN) < 6.28318530717958647692f - 1.0e-6f;
#endif

#ifndef DIFFUSION
static_assert(N_Z == 1, "N_Z > 1 requires DIFFUSION");
#endif

const real  SIGMA_0     = 1.0e-02;
const real  METAL_Z     = 1.0e-02;
const real  ASPR_0      = 0.05;
const real  IDX_P       = -1.0;
const real  IDX_Q       = -0.4;

#if defined(COLLISION) || defined(DIFFUSION)
#ifdef CONST_NU
const real  NU          = 1.0e-05;
#else
const real  ALPHA       = 1.0e-04;
#endif
#endif

#ifdef COLLISION
#ifdef CODE_UNIT
const real  REYNOLDS_0  = 1.0e+08;
#else
const real  M_MOL       = 2.3*1.66054e-24;
const real  X_SEC       = 2.0e-15;
#endif
#endif

const real  STOKES_0    = 1.0e-03;

const real  RHO_0       = 6.0/M_PI;

#ifdef RADIATION
const real  BETA_0      = 1.0e+01;
const real  KAPPA_0     = 1.0;
const real  T_BETA      = 2.0*M_PI;

#ifdef PR_EFFECT
const real  C_LIGHT     = 1.0e+04;
#endif
#endif

#ifdef DIFFUSION
const real  SCHMIDT_X   = 1.0;
const real  SCHMIDT_R   = 1.0;
#endif

#if defined(DIFFUSION) || defined(COLLISION)
const real  SCHMIDT_Z   = 1.0;
#endif

#ifdef COLLISION
const int   COAG_KERNEL = 0;
#ifndef SWEEP_N_K
#define SWEEP_N_K 16
#endif
const int   N_K         = SWEEP_N_K;

const real  H_SEARCH    = 128.0;
const real  V_FRAG      = 1.0;
#ifdef BERNOULLI
const real  CFL_COL     = 0.01;
#endif

#ifndef BERNOULLI
const int   COL_BATH_TPB  = 64;
const int   COL_EVENT_CAP = 32;
const int   COL_BIN_X     = 8;
const int   COL_BIN_Y     = 4;
const int   COL_BIN_Z     = 2;
const int   COL_BIN_S     = 64;
const int   COL_BIN_MIN   = 64;
#ifndef SWEEP_COL_BATH_EPS
#define SWEEP_COL_BATH_EPS 1.0e-2
#endif
constexpr real COL_BATH_EPS   = SWEEP_COL_BATH_EPS;
constexpr real COL_BATH_ALPHA = 1.0e-3;
#endif
#endif

#ifdef COLLISION_MORTON
const int   MORTON_TPB         = 256;
const int   MORTON_LEAF_TARGET = 128;
const int   MORTON_MAX_LEVEL   = 20;
const int   MORTON_WORK_SIZE   = 1024;

static_assert(3*N_K + MORTON_TPB <= MORTON_WORK_SIZE,
    "Morton work storage must hold three periodic images of every KNN slot");
#endif

#ifdef MULTISIZE
const real INIT_SMIN    = 1.0e+00;
const real INIT_SMAX    = 1.0e+00;
#endif

#if defined(COLLISION) && !defined(BERNOULLI)

#endif

const int  SAVE_MAX     = 9;

const real DT_OUT       = 0.1;

#ifdef TRANSPORT
const real DT_MAX       = 0.1;
const real CFL_DYN      = 0.45;
#endif

#if defined(LOGTIMING) || defined(LOGOUTPUT)
const int  LOG_BASE     = 10;
#else
const int  LIN_BASE     = 1;
#endif

struct swarm
{
    real3   position;
    real3   velocity;

    #ifdef MULTISIZE
    real    par_size;
    real    par_numr;
    #endif
};

#ifdef COLLISION_KDTREE
struct kdtree_node
{
    float3  cartesian;
    int     idx_old;
    int     split_dim;
    int     image;
};

struct kdtree_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };

    static inline __host__ __device__ const point_t &get_point (const kdtree_node &node) { return node.cartesian; }
    static inline __host__ __device__ float get_coord (const kdtree_node &node, int dim) { return kdtree::get_coord(node.cartesian, dim); }
    static inline __host__ __device__ int get_dim (const kdtree_node &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (kdtree_node &node, int dim) { node.split_dim = dim; }
};
#endif

const int TPB = 64;

const int NB_P = N_P     / TPB + 1;
const int NB_G = N_G     / TPB + 1;
const int NB_X = N_Y*N_Z / TPB + 1;
const int NB_Y = N_X*N_Z / TPB + 1;

#ifdef COLLISION_KDTREE
const int N_T  = X_WEDGE ? 3*N_P : N_P;
const int NB_T = N_T     / TPB + 1;
#endif

constexpr real COL_SIZE_MIN = 0.5, COL_SIZE_MAX = 8.0;
constexpr real COL_BATH_MAX = 1.0e100;
constexpr real BENCHMARK_MASS = 1.0e30;
static_assert(SEED >= 0 && SEED < 10, "Campaign seeds are 0 through 9");
#endif
