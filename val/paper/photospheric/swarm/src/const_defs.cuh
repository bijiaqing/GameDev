#ifndef GAMEDEV_VAL_PAPER_PHOTOSPHERIC_SWARM_CONST_DEFS_CUH
#define GAMEDEV_VAL_PAPER_PHOTOSPHERIC_SWARM_CONST_DEFS_CUH

#include <cmath>                            // M_PI
#include <string>                           // std::string

#include <gpu.cuh>

#ifdef COLLISION_KDTREE
#include <kdtree/builder.h>                 // kdtree::get_coord, kdtree::box_t
#endif // COLLISION_KDTREE

#if defined(COLLISION) || defined(DIFFUSION)
using curs = gpuRandState;
#endif // COLLISION || DIFFUSION

#ifdef COLLISION_KDTREE
using kdtree_boxf = kdtree::box_t<float3>;  // axis-aligned float bounding box type for KD-tree
#endif // COLLISION_KDTREE

using real  = double;                       // code real type
using real3 = double3;                      // double3 is a built-in CUDA type

// =====================================================================================================================
// code units
// =====================================================================================================================

constexpr real  G           = 1.0;              // gravitational constant
constexpr real  M_S         = 1.0;              // mass of the central star
constexpr real  R_0         = 1.0;              // reference radius of the disk
constexpr real  S_0         = 1.0;              // reference grain diameter, independent of the disk reference radius

// =====================================================================================================================
// mesh domain size and resolution
// =====================================================================================================================

constexpr int   N_P         = 1'000'000'000;          // total number of representative particles

constexpr int  N_X      = 1536;
constexpr real X_MIN    = 0.0;
constexpr real X_MAX    = 0.5*M_PI;

constexpr int  N_Y      = 1024;
constexpr real Y_MIN    = 0.5;
constexpr real Y_MAX    = 1.5;

constexpr int   N_Z         = 1;                // number of grid cells in Z direction (colattitude)
constexpr real  Z_MIN       = 0.5*M_PI;         // minimum Z boundary (colattitude)
constexpr real  Z_MAX       = 0.5*M_PI;         // maximum Z boundary (colattitude)

constexpr int   N_G         = N_X*N_Y*N_Z;      // total number of grid cells

#ifdef COLLISION
constexpr bool  X_WEDGE     = N_X > 1
    && static_cast<float>(X_MAX) - static_cast<float>(X_MIN) < 6.28318530717958647692f - 1.0e-06f;
#endif // COLLISION

#ifndef DIFFUSION
static_assert(N_Z == 1, "N_Z > 1 requires DIFFUSION");
#endif // NO DIFFUSION

// =====================================================================================================================
// gas parameters
// =====================================================================================================================

constexpr real  SIGMA_0     = 1.0e-02;          // reference gas surface density at R_0
constexpr real  METAL_Z     = 1.0e-02;          // total dust-to-gas surface-density ratio for initialization
constexpr real  ASPR_0      = 0.05;             // the reference aspect ratio of the gas disk
constexpr real  IDX_P       = -0.5;             // the radial power-law index of the gas surface density profile
constexpr real  IDX_Q       =  0.0;             // radial power-law index of the gas temperature (vertically isothermal)

#if defined(COLLISION) || defined(DIFFUSION)
#ifdef CONST_NU
constexpr real  NU          = 1.0e-05;          // the kinematic viscosity parameter of the gas
#else  // CONST_ALPHA
constexpr real  ALPHA       = 1.0e-04;          // the Shakura-Sunayev viscosity parameter of the gas
#endif // CONST_NU
#endif // COLLISION || DIFFUSION

#ifdef COLLISION
#ifdef CODE_UNIT
constexpr real  REYNOLDS_0  = 1.0e+08;          // reference Reynolds number at R_0
#else  // PHYSICAL_UNIT
constexpr real  M_MOL       = 2.3*1.66054e-24;  // mean molecular weight of the gas in grams
constexpr real  X_SEC       = 2.0e-15;          // the cross section of H2 gas in cm^2
#endif // CODE_UNIT
#endif // COLLISION

// =====================================================================================================================
// dust parameters for dynamics
// =====================================================================================================================

constexpr real  STOKES_0    = 1.0e-03;          // midplane Stokes number at R_0 for dust with the reference size

constexpr real  RHO_0       = 1.0;              // compact-grain internal density

#ifdef RADIATION
constexpr real  BETA_0      = 1.0e+01;          // the reference ratio between the radiation pressure and the gravity
constexpr real  KAPPA_0     = 5.0e+04;          // the reference gray opacity of the dust
constexpr real  T_BETA      = 2.0*M_PI;         // duration of the smooth radiation startup

#ifdef PR_EFFECT
constexpr real  C_LIGHT     = 1.0e+04;          // speed of light in orbital code velocity units
#endif // PR_EFFECT
#endif // RADIATION

#ifdef DIFFUSION
constexpr real  SCHMIDT_X   = 1.0;              // the Schmidt number for cylindrical azimuthal diffusion
constexpr real  SCHMIDT_R   = 1.0;              // the Schmidt number for cylindrical radial diffusion
#endif // DIFFUSION

#if defined(DIFFUSION) || defined(COLLISION)
constexpr real  SCHMIDT_Z   = 1.0;              // the Schmidt number for cylindrical vertical diffusion
#endif // DIFFUSION || COLLISION

#ifdef COLLISION
constexpr int   COAG_KERNEL = 0;                // 0-2 = normalized synthetic kernels; 3 = physical kernel
constexpr int   N_K         = 200;              // number of candidate slots returned by each KNN query

constexpr real  H_SEARCH    = 1.0;              // KNN search radius in units of the local gas scale height
constexpr real  V_FRAG      = 1.0;              // the fragmentation velocity for dust collision
#ifdef BERNOULLI
constexpr real  CFL_COL     = 0.01;             // maximum collision propensity per representative and batch
#endif // BERNOULLI

#ifndef BERNOULLI
// backend default threads per owner chain; a model const_defs.cuh may choose another width
#ifdef GAMEDEV_ROCM
constexpr int   COL_BATH_TPB  = 128;
#else  // !GAMEDEV_ROCM
constexpr int   COL_BATH_TPB  = 64;
#endif // GAMEDEV_ROCM
constexpr int   COL_EVENT_CAP = 32;              // accepted events permitted per representative and continuation launch
constexpr int   COL_BIN_X     = 8;               // azimuthal controller bins before reduced-dimension collapse
constexpr int   COL_BIN_Y     = 4;               // radial controller bins
constexpr int   COL_BIN_Z     = 2;               // polar controller bins before reduced-dimension collapse
constexpr int   COL_BIN_S     = 8;               // logarithmic grain-size controller bins
constexpr int   COL_BIN_MIN   = 64;              // target minimum representatives after adjacent size-bin merging
constexpr real  COL_BATH_MAX  = 0.05;            // maximum frozen-reservoir bath duration
constexpr real  COL_BATH_EPS  = 0.02;            // log-size refresh and distribution-audit tolerance
constexpr real  COL_BATH_ALPHA = 1.0e-03;         // family-wise confidence-tail probability for realized audits
#endif // FROZEN_BATH
#endif // COLLISION

#ifdef COLLISION_MORTON
constexpr int   MORTON_TPB         = 64;       // cooperative threads assigned to one Morton query
constexpr int   MORTON_LEAF_TARGET = 128;       // target records per adaptive leaf
constexpr int   MORTON_MAX_LEVEL   = 20;        // maximum adaptive subdivision depth
// shared slots for duplicate-safe top-K selection
constexpr int   MORTON_WORK_SIZE = []()
{
    int n = 1;
    while (n < 3*N_K + MORTON_TPB)
    {
        n *= 2;
    }
    return n;
}();

static_assert(3*N_K + MORTON_TPB <= MORTON_WORK_SIZE,
    "Morton work storage must hold three periodic images of every KNN slot");
#endif // COLLISION_MORTON

// =====================================================================================================================
// dust initialization parameters
// =====================================================================================================================

#ifdef MULTISIZE
constexpr real INIT_SMIN    = 1.0e+00;          // minimum grain size for particle initialization
constexpr real INIT_SMAX    = 1.0e+00;          // maximum grain size for particle initialization
#endif // MULTISIZE

#if defined(COLLISION) && !defined(BERNOULLI)
#endif // COLLISION && !BERNOULLI

// =====================================================================================================================
// time step and output parameters
// =====================================================================================================================

constexpr int  SAVE_MAX     = 20;               // total number of outputs for mesh fields

constexpr real DT_OUT       = 2.0*M_PI;

#ifdef TRANSPORT
constexpr real DT_MAX       = 1.0e-01;
constexpr real CFL_DYN      = 0.45;             // maximum fraction of a local mesh scale crossed in one dynamics step
#endif // TRANSPORT

#if defined(LOGTIMING) || defined(LOGOUTPUT)
constexpr int  LOG_BASE     = 10;               // logarithmic base for LOGTIMING steps or LOGOUTPUT particle output
#else  // LINEAR
constexpr int  LIN_BASE     = 10;               // save particle data every LIN_BASE density outputs
#endif // LOGTIMING || LOGOUTPUT

// =====================================================================================================================
// structures
// =====================================================================================================================

struct swarm                                // representative-particle state
{
    real3   position;                       // x = azimuth, y = spherical radius, z = polar angle
    real3   velocity;                       // x = l_phi, y = v_r, z = l_theta

    #ifdef MULTISIZE
    real    par_size;                       // diameter of one physical grain in the represented species
    real    par_numr;                       // number of physical grains represented by this particle
    #endif // MULTISIZE
};

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

// =====================================================================================================================
// cuda numerical parameters
// =====================================================================================================================

constexpr int TPB = 64; // number of threads per block

constexpr int NB_P = N_P     / TPB + 1;         // number of blocks for swarm-level parallelization
constexpr int NB_G = N_G     / TPB + 1;         // number of blocks for grid-level  parallelization
constexpr int NB_X = N_Y*N_Z / TPB + 1;         // number of blocks for X-direction parallelization
constexpr int NB_Y = N_X*N_Z / TPB + 1;         // number of blocks for Y-direction parallelization

#ifdef COLLISION_KDTREE
constexpr int N_T  = X_WEDGE ? 3*N_P : N_P;    // physical and periodic-image tree nodes
constexpr int NB_T = N_T     / TPB + 1;         // number of blocks for tree-level parallelization
#endif // COLLISION_KDTREE

// =====================================================================================================================

#endif // GAMEDEV_VAL_PAPER_PHOTOSPHERIC_SWARM_CONST_DEFS_CUH
