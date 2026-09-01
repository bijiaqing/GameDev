#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath>  // M_PI
#include <string> // std::string

#if defined(COLLISION) || defined(DIFFUSION)
#include <hiprand/hiprand_kernel.h>
#endif

#ifdef COLLISION_KDTREE
#include <kdtree/builder.h>
#endif

using real = double;
using real3 = double3;

#if defined(COLLISION) || defined(DIFFUSION)
using curs = hiprandState;
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

// particle counts are test samples rather than production population choices: use only enough representatives to cover the
// analytical cases, except for stochastic and performance tests that need a statistically meaningful ensemble
#ifdef PERF_PARTICLES
constexpr int N_P = PERF_PARTICLES;
#elif defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D) \
    || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_3D) \
    || defined(TEST_COLCHAIN_RESTART_2D) || defined(TEST_COLREUSE_2D)
constexpr int N_P = 2048;
#elif defined(TEST_IMPORT_3D) || defined(TEST_PARINIT_3D)
constexpr int N_P = 64;
#elif defined(TEST_PERF_COLLISION_2D)
#ifndef PERF_PARTICLES
#define PERF_PARTICLES 100000
#endif
constexpr int N_P = PERF_PARTICLES;
#elif defined(TEST_SETTLE_DIFFUSE_3D)
constexpr int N_P = 65536;
#elif defined(TEST_DYNRATE_3D)
constexpr int N_P = 64;
#elif defined(TEST_ORBIT_ECC_2D) || defined(TEST_ORBIT_BETA_2D) || defined(TEST_ORBIT_INC_3D)
constexpr int N_P = 4;
#elif defined(TEST_DRAG_PATH_1D)
constexpr int N_P = 3;
#elif defined(TEST_ABSORB_PATH_1D)
constexpr int N_P = 4;
#elif defined(TEST_DIFFUSION_1D) || defined(TEST_DIFFUSION_2D) || defined(TEST_DIFFUSION_3D)
constexpr int N_P = 16*VERIFY_RES*VERIFY_RES;
#elif defined(TEST_INITIAL_3D)
constexpr int N_P = 65536;
#elif defined(TEST_RESTART_2D)
constexpr int N_P = 256;
#elif defined(TEST_GRID_1D)
constexpr int N_P = VERIFY_RES;
#elif defined(TEST_GRID_2D)
constexpr int N_P = VERIFY_RES*VERIFY_RES;
#elif defined(TEST_GRID_3D)
constexpr int N_P = 4*VERIFY_RES*VERIFY_RES;
#elif defined(TEST_DRAG_1D) || defined(TEST_VISCFLOW_1D) || defined(TEST_RADIATION_1D) \
    || defined(TEST_PRDRAG_1D) || defined(TEST_IMPORT_1D) \
    || defined(TEST_DRAG_2D) || defined(TEST_RADIATION_2D) || defined(TEST_PRDRAG_2D)
constexpr int N_P = 8;
#elif defined(TEST_COLLISION_1D) || defined(TEST_COLLISION_2D) || defined(TEST_COLLISION_3D) \
    || defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS)
constexpr int N_P = 2;
#elif defined(TEST_BOUNDARY_1D) || defined(TEST_BOUNDARY_2D) \
    || defined(TEST_BOUNDARY_3D) || defined(TEST_BOUNDARY_HALF)
constexpr int N_P = 1;
#else
constexpr int N_P = 64;
#endif

// activate only the coordinates needed by each claim and keep inactive dimensions explicit to expose indexing mistakes
#if defined(TEST_IMPORT_3D) || defined(TEST_PARINIT_3D)
constexpr int N_X = 4;
constexpr int N_Y = 8;
constexpr int N_Z = 4;
#elif defined(TEST_COLCHAIN_3D)
constexpr int N_X = 8;
constexpr int N_Y = 16;
constexpr int N_Z = 8;
#elif defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D) \
    || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_RESTART_2D) \
    || defined(TEST_COLREUSE_2D)
constexpr int N_X = 32;
constexpr int N_Y = 32;
constexpr int N_Z = 1;
#elif defined(TEST_PERF_COLLISION_2D)
constexpr int N_X = 100;
constexpr int N_Y = 100;
constexpr int N_Z = 1;
#elif defined(TEST_GRID_1D)
constexpr int N_X = 1;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#elif defined(TEST_GRID_2D)
constexpr int N_X = VERIFY_RES;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = 1;
#elif defined(TEST_GRID_3D)
constexpr int N_X = 4;
constexpr int N_Y = VERIFY_RES;
constexpr int N_Z = VERIFY_RES;
#elif defined(TEST_INITIAL_3D)
constexpr int N_X = 1;
constexpr int N_Y = 96;
constexpr int N_Z = VERIFY_RES;
#elif defined(TEST_ORBIT_INC_3D)
constexpr int N_X = 32;
constexpr int N_Y = 16;
constexpr int N_Z = 16;
#elif defined(TEST_DYNRATE_3D) || defined(TEST_DIFFUSION_3D) \
    || defined(TEST_SETTLE_DIFFUSE_3D) || defined(TEST_COLLISION_3D)
constexpr int N_X = 8;
constexpr int N_Y = 16;
constexpr int N_Z = 16;
#elif defined(TEST_BOUNDARY_3D) || defined(TEST_BOUNDARY_HALF)
constexpr int N_X = 8;
constexpr int N_Y = 16;
constexpr int N_Z = 8;
#elif defined(TEST_DIFFUSION_1D) || defined(TEST_COLLISION_1D) || defined(TEST_BOUNDARY_1D)
constexpr int N_X = 1;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#elif defined(TEST_DIFFUSION_2D) || defined(TEST_RESTART_2D) \
    || defined(TEST_COLLISION_2D) || defined(TEST_BOUNDARY_2D) \
    || defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS)
constexpr int N_X = 16;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#elif defined(TEST_ORBIT_1D) || defined(TEST_DRAG_1D) || defined(TEST_DRAG_PATH_1D) \
    || defined(TEST_ABSORB_PATH_1D) \
    || defined(TEST_VISCFLOW_1D) || defined(TEST_RADIATION_1D) \
    || defined(TEST_PRDRAG_1D) || defined(TEST_IMPORT_1D)
constexpr int N_X = 1;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#else
constexpr int N_X = 32;
constexpr int N_Y = 16;
constexpr int N_Z = 1;
#endif

#if defined(TEST_BOUNDARY_2D) || defined(TEST_COLCHAIN_WEDGE_2D)
constexpr real X_MIN = -0.1;
constexpr real X_MAX = 0.1;
#else
constexpr real X_MIN = -M_PI;
constexpr real X_MAX = M_PI;
#endif
constexpr real Y_MIN = 0.5;
constexpr real Y_MAX = 1.5;

#if defined(TEST_INITIAL_3D)
constexpr real Z_MIN = 0.5*M_PI - 0.01;
constexpr real Z_MAX = 0.5*M_PI + 0.01;
#elif defined(TEST_GRID_3D) || defined(TEST_IMPORT_3D) || defined(TEST_PARINIT_3D) || defined(TEST_ORBIT_INC_3D) \
    || defined(TEST_DYNRATE_3D) \
    || defined(TEST_DIFFUSION_3D) || defined(TEST_SETTLE_DIFFUSE_3D) \
    || defined(TEST_COLLISION_3D) || defined(TEST_COLCHAIN_3D) \
    || defined(TEST_BOUNDARY_3D)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = M_PI - 0.35;
#elif defined(TEST_BOUNDARY_HALF)
constexpr real Z_MIN = 0.35;
constexpr real Z_MAX = 0.5*M_PI;
#else
constexpr real Z_MIN = 0.5*M_PI;
constexpr real Z_MAX = 0.5*M_PI;
#endif

constexpr int N_G = N_X*N_Y*N_Z;

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
#if defined(TEST_DYNRATE_3D)
constexpr real SCHMIDT_X = 1.0;
constexpr real SCHMIDT_R = 1.0;
#elif defined(TEST_DIFFUSION_1D)
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0;
#elif defined(TEST_DIFFUSION_2D)
constexpr real SCHMIDT_X = 1.0;
constexpr real SCHMIDT_R = 1.0e300;
#elif defined(TEST_RESTART_2D)
constexpr real SCHMIDT_X = 1.0;
constexpr real SCHMIDT_R = 1.0;
#elif defined(TEST_DIFFUSION_3D)
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0;
#elif defined(TEST_SETTLE_DIFFUSE_3D)
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0e300;
#else
constexpr real SCHMIDT_X = 1.0e300;
constexpr real SCHMIDT_R = 1.0e300;
#endif
constexpr real SCHMIDT_Z =
#if defined(TEST_DYNRATE_3D) || defined(TEST_DIFFUSION_3D) \
    || defined(TEST_SETTLE_DIFFUSE_3D) || defined(TEST_INITIAL_3D)
    1.0;
#else
    1.0e300;
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
#else  // PHYSICAL_UNIT
constexpr real M_MOL = 2.3*1.66054e-24;
constexpr real X_SEC = 2.0e-15;
#endif // CODE_UNIT
constexpr int COAG_KERNEL =
#if defined(TEST_COLCHAIN_FRAG_2D) || defined(TEST_COLPHYS_CODE) || defined(TEST_COLPHYS_CGS)
    3;
#else
    0;
#endif // TEST_COLCHAIN_FRAG_2D || TEST_COLPHYS_CODE || TEST_COLPHYS_CGS
#ifdef TEST_N_K
constexpr int N_K = TEST_N_K;
#elif defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D) \
    || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_3D) \
    || defined(TEST_COLCHAIN_RESTART_2D) || defined(TEST_COLREUSE_2D) \
    || defined(TEST_PERF_COLLISION_2D)
constexpr int N_K = 200;
#else
constexpr int N_K = 2;
#endif
#ifdef TEST_H_SEARCH
constexpr real H_SEARCH = TEST_H_SEARCH;
#elif defined(TEST_ABSORB_PATH_1D)
constexpr real H_SEARCH = 10.0;
#else  // OTHER TESTS
constexpr real H_SEARCH = 1.0;
#endif // TEST_ABSORB_PATH_1D
constexpr real V_FRAG =
#ifdef TEST_COLCHAIN_FRAG_2D
    0.0;
#else
    1.0;
#endif // TEST_COLCHAIN_FRAG_2D
constexpr real CFL_COL =
#ifdef TEST_CFL_COL
    TEST_CFL_COL;
#else
    0.01;
#endif // TEST_CFL_COL

#if defined(COLLISION) && !defined(BERNOULLI)
constexpr int COL_BATH_TPB = 256;
constexpr int COL_EVENT_CAP =
#ifdef TEST_CHAIN_CAP
    TEST_CHAIN_CAP;
#else
    32;
#endif // TEST_CHAIN_CAP
constexpr int COL_BIN_X = 8;
constexpr int COL_BIN_Y = 4;
constexpr int COL_BIN_Z = 2;
constexpr int COL_BIN_S = 8;
constexpr int COL_BIN_MIN = 64;
constexpr real COL_BATH_MAX =
#ifdef TEST_BATH_MAX
    TEST_BATH_MAX;
#else
    0.05;
#endif // TEST_BATH_MAX
constexpr real COL_BATH_EPS =
#ifdef TEST_BATH_EPS
    TEST_BATH_EPS;
#else
    0.06;
#endif // TEST_BATH_EPS
constexpr real COL_BATH_ALPHA = 1.0e-3;
constexpr real COL_SIZE_MIN = 0.5*INIT_SMIN;
constexpr real COL_SIZE_MAX = 8.0*INIT_SMAX;
#endif // COLLISION && !BERNOULLI
#endif

#ifdef COLLISION_MORTON
constexpr int MORTON_TPB = 256;
constexpr int MORTON_LEAF_TARGET = 128;
constexpr int MORTON_MAX_LEVEL = 20;
constexpr int MORTON_WORK_SIZE = 1024;
#endif

#ifdef TEST_COLCHAIN_RESTART_2D
constexpr int SAVE_MAX = 2;
#elif defined(TEST_PERF_COLLISION_2D) && defined(TEST_SAVE_MAX)
constexpr int SAVE_MAX = TEST_SAVE_MAX;
#else
constexpr int SAVE_MAX = 1;
#endif // TEST_COLCHAIN_RESTART_2D
#if defined(TEST_DT_OUT)
constexpr real DT_OUT = TEST_DT_OUT;
#elif defined(TEST_COLCHAIN_3D)
constexpr real DT_OUT = 1.0e-2;
#elif defined(TEST_COLCHAIN_2D) || defined(TEST_COLCHAIN_FRAG_2D) \
    || defined(TEST_COLCHAIN_WEDGE_2D) || defined(TEST_COLCHAIN_RESTART_2D) \
    || defined(TEST_COLREUSE_2D)
constexpr real DT_OUT = 1.0e-3;
#else
constexpr real DT_OUT = 1.0;
#endif // TEST_DT_OUT
constexpr real DT_MAX =
#ifdef TEST_COLREUSE_2D
    2.5e-4;
#else
    1.0;
#endif // TEST_COLREUSE_2D
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
constexpr int N_T = (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-6) ? 3*N_P : N_P;
constexpr int NB_T = N_T / TPB + 1;
#endif

static_assert(N_X > 0 && N_Y > 1, "swarm verification requires a radial grid and at least one x cell");
static_assert(N_Z == 1 || Z_MAX > Z_MIN, "active z grids require nonzero extent");

#endif
