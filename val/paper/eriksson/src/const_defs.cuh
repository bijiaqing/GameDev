#ifndef GAMEDEV_VAL_PAPER_ERIKSSON_CONST_DEFS_CUH
#define GAMEDEV_VAL_PAPER_ERIKSSON_CONST_DEFS_CUH

#include <cmath>  // M_PI
#include <string> // std::string

#include <gpu.cuh>
using curs = gpuRandState;
#ifdef COLLISION_KDTREE
#include <kdtree/builder.h> // kdtree::box_t, kdtree::builder, kdtree::get_coord
using kdtree_boxf = kdtree::box_t<float3>;
#endif // COLLISION_KDTREE
using real = double;
using real3 = double3;

// all dimensional values are CGS; CODE_UNIT must remain disabled for Brownian motion
constexpr real AU = 1.495978707e13, YEAR = 31557600.0;
constexpr real G = 6.67430e-8, M_S = 1.98847e33, R_0 = AU;
constexpr real S_0 = 1.0e-4; // diameter, not the radius used by Eriksson et al.
constexpr int N_P = 1048576;
constexpr int N_X = 1, N_Y = 128, N_Z = 64;
constexpr real X_MIN = -M_PI, X_MAX = M_PI;
constexpr real Y_MIN = 5.0*AU, Y_MAX = 50.0*AU;
constexpr real Z_MIN = 0.5*M_PI - 0.2, Z_MAX = 0.5*M_PI + 0.2;
constexpr int N_G = N_X*N_Y*N_Z;
constexpr bool X_WEDGE = false;

constexpr real SIGMA_0 = 1410.4014065096128, METAL_Z = 0.01;
// h(1 au) from T=209.7926358245702 K, mu=2.34 proton masses
constexpr real ASPR_0 = 0.0288821994330985, IDX_P = -1.0, IDX_Q = -0.5;
constexpr real ALPHA = COAG_ALPHA;
constexpr real M_MOL = 2.34*1.67262192369e-24, X_SEC = 2.0e-15;
constexpr real RHO_0 = 1.0;
constexpr real STOKES_0 = M_PI*RHO_0*S_0 / (4.0*SIGMA_0);
constexpr real SCHMIDT_X = 1.0, SCHMIDT_R = 1.0, SCHMIDT_Z = 1.0;
constexpr real INIT_SMIN = 1.0e-4, INIT_SMAX = INIT_SMIN; // fixed 0.5-micron radius monomers

constexpr int COAG_KERNEL = 3, N_K = 256;
constexpr real H_SEARCH = 1.0, V_FRAG = 100.0;
// use the production backend default chain widths
#ifdef GAMEDEV_ROCM
constexpr int COL_BATH_TPB = 128;
#else  // !GAMEDEV_ROCM
constexpr int COL_BATH_TPB = 64;
#endif // GAMEDEV_ROCM
constexpr int COL_EVENT_CAP = 32;
constexpr int COL_BIN_X = 1, COL_BIN_Y = 16, COL_BIN_Z = 8, COL_BIN_S = 32;
constexpr int COL_BIN_MIN = 64;
constexpr real COL_BATH_MAX = YEAR;
constexpr real COL_BATH_EPS = 0.08, COL_BATH_ALPHA = 1.0e-3;
static_assert(N_K == 256 && COL_BATH_EPS == 0.08, "Benchmark settings must match across variants");
constexpr int MORTON_TPB = 64, MORTON_LEAF_TARGET = 128;
constexpr int neighbor_pow2 (int n)
{
    int p = 1;
    while (p < n) p *= 2;
    return p;
}
constexpr int MORTON_MAX_LEVEL = 20, MORTON_WORK_SIZE = neighbor_pow2(3*N_K + MORTON_TPB);
static_assert(3*N_K + MORTON_TPB <= MORTON_WORK_SIZE);

constexpr int SAVE_MAX = COAG_OUTPUTS, LIN_BASE = 1;
constexpr real DT_OUT = 100.0*YEAR, DT_MAX = YEAR, CFL_DYN = 0.45;
struct swarm { real3 position; real3 velocity; real par_size; real par_numr; };
static_assert(sizeof(swarm) == 8*sizeof(real));
#ifdef COLLISION_KDTREE
struct kdtree_node                          // KD-tree node consumed by kdtree::builder
{
    float3  cartesian;                      // Cartesian position of the physical particle or periodic image
    int     idx_old;                        // stable particle-array index before KD-tree reordering
    int     split_dim;                      // splitting dimension of the tree node
    int     image;                          // zero for a physical node and nonzero for a periodic image
};

struct kdtree_traits                        // traits for kdtree::builder
{
    using point_t = float3;
    enum { has_explicit_dim = true };

    // expose point coordinates and split dimensions through the KD-tree traits interface
    static inline __host__ __device__ const point_t &get_point (const kdtree_node &node) { return node.cartesian; }
    static inline __host__ __device__ float get_coord (const kdtree_node &node,
        int dim) { return kdtree::get_coord(node.cartesian, dim); }
    static inline __host__ __device__ int get_dim (const kdtree_node &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (kdtree_node &node, int dim) { node.split_dim = dim; }
};
#endif // COLLISION_KDTREE
#ifdef COLLISION_KDTREE
constexpr int N_T = X_WEDGE ? 3*N_P : N_P;
constexpr int NB_T = N_T / 64 + 1;
#endif // COLLISION_KDTREE
constexpr int TPB = 64;
constexpr int NB_P = N_P / TPB + 1, NB_G = N_G / TPB + 1;
constexpr int NB_X = N_Y*N_Z / TPB + 1, NB_Y = N_X*N_Z / TPB + 1;
#endif // GAMEDEV_VAL_PAPER_ERIKSSON_CONST_DEFS_CUH
