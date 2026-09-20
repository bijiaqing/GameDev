#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath>  // M_PI
#include <string> // std::string

#if defined(COLLISION) || defined(DIFFUSION)
#include <gpu.cuh>  // gpuRandState
#endif

#ifdef COLLISION_KDTREE
#include <kdtree/builder.h>
#endif

using real = double;
using real3 = double3;

#if defined(COLLISION) || defined(DIFFUSION)
using curs = gpuRandState;
#endif

#ifdef COLLISION_KDTREE
using kdtree_boxf = kdtree::box_t<float3>;
#endif

#ifdef TEST_RES
constexpr int VERIFY_RES = TEST_RES;
#else
constexpr int VERIFY_RES = 64;
#endif

constexpr real G = 1.0;
constexpr real M_S = 1.0;
constexpr real R_0 = 1.0;
constexpr real S_0 = 1.0;

// keep publication samples large only where stochastic statistics require them
#if defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D)  || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_3D)
constexpr int N_P = 2048;
#elif defined(TEST_ORBIT_ECC_2D) || defined(TEST_ORBIT_BETA_2D) || defined(TEST_ORBIT_INC_3D)
constexpr int N_P = 4;
#elif defined(TEST_DRAG_PATH_1D)
constexpr int N_P = 3;
#elif defined(TEST_DIFFUSION_1D) || defined(TEST_DIFFUSION_2D) || defined(TEST_DIFFUSION_3D)  || defined(TEST_DIFFUSION_WEDGE_2D) || defined(TEST_DIFFUSION_WEDGE_3D)
constexpr int N_P = 16*VERIFY_RES*VERIFY_RES;
#elif defined(TEST_INITIAL_3D)
constexpr int N_P = 65536;
#elif defined(TEST_PRDRAG_2D)
constexpr int N_P = 8;
#elif defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS) || defined(TEST_COLPHYS_3D)
constexpr int N_P = 2;
#else
constexpr int N_P = 64;
#endif

// activate only coordinates exercised by the retained publication claim
#if defined(TEST_COLCHAIN_3D)
constexpr int N_X = 8;
constexpr int N_Y = 16;
constexpr int N_Z = 8;
#elif defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D)  || defined(TEST_COLCHAIN_WEDGE_2D)
constexpr int N_X = 32;
constexpr int N_Y = 32;
constexpr int N_Z = 1;
#elif defined(TEST_INITIAL_3D)
constexpr int N_X = 1;
constexpr int N_Y = 96;
constexpr int N_Z = VERIFY_RES;
#elif defined(TEST_ORBIT_INC_3D)
constexpr int N_X = 32;
constexpr int N_Y = 16;
constexpr int N_Z = 16;
#elif defined(TEST_DIFFUSION_3D) || defined(TEST_DIFFUSION_WEDGE_3D)  || defined(TEST_COLPHYS_3D)
constexpr int N_X = 8;
constexpr int N_Y = 16;
constexpr int N_Z = 16;
#elif defined(TEST_DIFFUSION_1D) || defined(TEST_DRAG_PATH_1D)
constexpr int N_X = 1;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#elif defined(TEST_DIFFUSION_2D) || defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS)
constexpr int N_X = 16;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#else
constexpr int N_X = 32;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#endif

#if defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_DIFFUSION_WEDGE_2D)  || defined(TEST_DIFFUSION_WEDGE_3D)
constexpr real X_MIN = -0.1;
constexpr real X_MAX = 0.1;
#elif defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS) || defined(TEST_COLPHYS_3D)
constexpr real X_MIN = -0.5;
constexpr real X_MAX = 0.5;
#else
constexpr real X_MIN = -M_PI;
constexpr real X_MAX = M_PI;
#endif
constexpr real Y_MIN = 0.5;
constexpr real Y_MAX = 1.5;

#ifdef TEST_INITIAL_3D
constexpr real Z_MIN = 0.5*M_PI - 0.01;
constexpr real Z_MAX = 0.5*M_PI + 0.01;
#elif defined(TEST_ORBIT_INC_3D) || defined(TEST_DIFFUSION_3D)  || defined(TEST_DIFFUSION_WEDGE_3D) || defined(TEST_COLCHAIN_3D)  || defined(TEST_COLPHYS_3D)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = M_PI - 0.35;
#else
constexpr real Z_MIN = 0.5*M_PI;
constexpr real Z_MAX = 0.5*M_PI;
#endif

constexpr int N_G = N_X*N_Y*N_Z;

#ifdef COLLISION
constexpr bool X_WEDGE = N_X > 1
    && static_cast<float>(X_MAX) - static_cast<float>(X_MIN) < 6.28318530717958647692f - 1.0e-6f;
#endif

constexpr real SIGMA_0 = 1.0;
constexpr real METAL_Z = 1.0e-2;
constexpr real ASPR_0 = 0.05;
constexpr real IDX_P = 2.0;
constexpr real IDX_Q = -1.0;
constexpr real STOKES_0 = 0.2;
constexpr real RHO_0 = 1.0;

#if defined(DIFFUSION) || defined(COLLISION)
#ifdef CONST_NU
#ifdef TEST_ORBIT_INC_3D
constexpr real NU = 0.0;
#else
constexpr real NU = 2.0e-2;
#endif
#else
constexpr real ALPHA = 1.0e-4;
#endif
#endif

#ifdef DIFFUSION
#ifdef TEST_DIFFUSION_1D
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0;
#elif defined(TEST_DIFFUSION_2D) || defined(TEST_DIFFUSION_WEDGE_2D)  || defined(TEST_DIFFUSION_WEDGE_3D)
constexpr real SCHMIDT_X = 1.0;
constexpr real SCHMIDT_R = 1.0e300;
#elif defined(TEST_DIFFUSION_3D)
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0;
#else
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0e300;
#endif
#if defined(TEST_DIFFUSION_3D) || defined(TEST_INITIAL_3D)
constexpr real SCHMIDT_Z = 1.0;
#else
constexpr real SCHMIDT_Z = 1.0e300;
#endif
#endif

#if defined(COLLISION) && !defined(DIFFUSION)
constexpr real SCHMIDT_Z = 1.0;
#endif

#ifdef RADIATION
constexpr real BETA_0 = 0.2;
constexpr real KAPPA_0 = 0.7;
constexpr real T_BETA = 1.0;
#ifdef PR_EFFECT
constexpr real C_LIGHT = 25.0;
#endif
#endif

#ifdef MULTISIZE
constexpr real INIT_SMIN = 0.05;
constexpr real INIT_SMAX = 6.4;
#endif

#ifdef COLLISION
#ifdef CODE_UNIT
constexpr real REYNOLDS_0 = 1.0e8;
#else
constexpr real M_MOL = 2.3*1.66054e-24;
constexpr real X_SEC = 2.0e-15;
#endif
#if defined(TEST_COLCHAIN_FRAG_2D) || defined(TEST_COLCHAIN_WEDGE_2D)  || defined(TEST_COLPHYS_CODE)  || defined(TEST_COLPHYS_CGS) || defined(TEST_COLPHYS_3D)
constexpr int COAG_KERNEL = 3;
#else
constexpr int COAG_KERNEL = 0;
#endif
#if defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D)  || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_3D)
constexpr int N_K = 200;
#else
constexpr int N_K = 2;
#endif
constexpr real H_SEARCH = 1.0;
#ifdef TEST_COLCHAIN_FRAG_2D
constexpr real V_FRAG = 0.0;
#else
constexpr real V_FRAG = 1.0;
#endif
#ifdef BERNOULLI
#ifdef TEST_CFL_COL
constexpr real CFL_COL = TEST_CFL_COL;
#else
constexpr real CFL_COL = 0.01;
#endif
#endif // BERNOULLI

#ifndef BERNOULLI
constexpr int COL_BATH_TPB = 256;
#ifdef TEST_CHAIN_CAP
constexpr int COL_EVENT_CAP = TEST_CHAIN_CAP;
#else
constexpr int COL_EVENT_CAP = 32;
#endif
constexpr int COL_BIN_X = 8;
constexpr int COL_BIN_Y = 4;
constexpr int COL_BIN_Z = 2;
constexpr int COL_BIN_S = 8;
constexpr int COL_BIN_MIN = 64;
constexpr real COL_BATH_MAX = 0.05;
constexpr real COL_BATH_EPS = 0.06;
constexpr real COL_BATH_ALPHA = 1.0e-3;
#endif
#endif

#ifdef COLLISION_MORTON
constexpr int MORTON_TPB = 256;
constexpr int MORTON_LEAF_TARGET = 128;
constexpr int MORTON_MAX_LEVEL = 20;
constexpr int MORTON_WORK_SIZE = 1024;
#endif

constexpr int SAVE_MAX = 1;
#ifdef TEST_DT_OUT
constexpr real DT_OUT = TEST_DT_OUT;
#elif defined(TEST_COLCHAIN_3D)
constexpr real DT_OUT = 1.0e-2;
#elif defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_FRAG_2D)
// The query-local physical kernel needs more time than the synthetic-kernel chain test to produce events.
constexpr real DT_OUT = 1.0;
#elif defined(TEST_COLCHAIN_2D)
constexpr real DT_OUT = 1.0e-3;
#else
constexpr real DT_OUT = 1.0;
#endif
constexpr real DT_MAX = 1.0;
constexpr real CFL_DYN = 0.5;
constexpr int LIN_BASE = 1;

struct swarm
{
    real3 position;
    real3 velocity;
#ifdef MULTISIZE
    real par_size;
    real par_numr;
#endif
};

#ifdef COLLISION_KDTREE
struct kdtree_node
{
    float3 cartesian;
    int idx_old;
    int split_dim;
    int image;
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

constexpr int TPB = 64;
constexpr int NB_P = N_P / TPB + 1;
constexpr int NB_G = N_G / TPB + 1;
constexpr int NB_X = N_Y*N_Z / TPB + 1;
constexpr int NB_Y = N_X*N_Z / TPB + 1;

#ifdef COLLISION_KDTREE
constexpr int N_T = X_WEDGE ? 3*N_P : N_P;
constexpr int NB_T = N_T / TPB + 1;
#endif

static_assert(N_X > 0 && N_Y > 1, "swarm verification requires a radial grid and at least one x cell");
static_assert(N_Z == 1 || Z_MAX > Z_MIN, "active z grids require nonzero extent");

#endif
