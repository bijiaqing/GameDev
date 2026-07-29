#ifndef CONST_DEFS_CUH
#define CONST_DEFS_CUH

#include <cmath>                            // M_PI
#include <string>                           // std::string

#if defined(COLLISION) || defined(DIFFUSION)
#include <curand_kernel.h>                  // curandState
#endif // COLLISION || DIFFUSION

#ifdef COLLISION_KDTREE
#include "cukd/builder.h"                   // cukd::get_coord, cukd::box_t
#endif // COLLISION_KDTREE

#if defined(COLLISION) || defined(DIFFUSION)
using curs = curandState;
#endif // COLLISION || DIFFUSION

#ifdef COLLISION_KDTREE
using bbox = cukd::box_t<float3>;           // axis-aligned bounding box type for KD-tree
#endif // COLLISION_KDTREE

using real  = double;                       // code real type
using real3 = double3;                      // double3 is a built-in CUDA type

// =========================================================================================================================
// code units
// =========================================================================================================================

const real  G           = 1.0;              // gravitational constant
const real  M_S         = 1.0;              // mass of the central star
const real  R_0         = 1.0;              // reference radius of the disk
const real  S_0         = 1.0;              // reference grain diameter, independent of the disk reference radius

// =========================================================================================================================
// mesh domain size and resolution
// =========================================================================================================================

const int   N_P         = 100000;            // total number of representative particles

const int   N_X         = 32;              // number of grid cells in X direction (azimuth)
const real  X_MIN       = -M_PI;            // minimum X boundary (azimuth)
const real  X_MAX       = +M_PI;            // maximum X boundary (azimuth)

const int   N_Y         = 64;              // number of grid cells in Y direction (radius)
const real  Y_MIN       = 0.5;              // minimum Y boundary (radius)
const real  Y_MAX       = 1.5;              // maximum Y boundary (radius)

const int   N_Z         = 32;                // number of grid cells in Z direction (colattitude)
const real  Z_MIN       = 0.45*M_PI;         // minimum Z boundary (colattitude)
const real  Z_MAX       = 0.55*M_PI;         // maximum Z boundary (colattitude)

const int   N_G         = N_X*N_Y*N_Z;      // total number of grid cells

#ifndef DIFFUSION
static_assert(N_Z == 1, "N_Z > 1 requires DIFFUSION");
#endif // NO DIFFUSION

// =========================================================================================================================
// gas parameters
// =========================================================================================================================

const real  SIGMA_0     = 1.0e-02;          // reference gas surface density at R_0
const real  METAL_Z     = 1.0e-02;          // total dust-to-gas surface-density ratio for initialization
const real  ASPR_0      = 0.05;             // the reference aspect ratio of the gas disk
const real  IDX_P       = -1.0;             // the radial power-law index of the gas surface density profile
const real  IDX_Q       = -0.4;             // the radial power-law index of the gas temperature profile (vertically isothermal)

#if defined(COLLISION) || defined(DIFFUSION)
#ifdef CONST_NU
const real  NU          = 1.0e-05;          // the kinematic viscosity parameter of the gas
#else  // CONST_ALPHA
const real  ALPHA       = 1.0e-04;          // the Shakura-Sunayev viscosity parameter of the gas
#endif // CONST_NU
#endif // COLLISION || DIFFUSION

#ifdef COLLISION
#ifdef CODE_UNIT
const real  REYNOLDS_0        = 1.0e+08;          // reference Reynolds number at R_0
#else  // PHYSICAL_UNIT
const real  M_MOL       = 2.3*1.66054e-24;  // mean molecular weight of the gas in grams
const real  X_SEC       = 2.0e-15;          // the cross section of H2 gas in cm^2
#endif // CODE_UNIT
#endif // COLLISION

// =========================================================================================================================
// dust parameters for dynamics
// =========================================================================================================================

const real  STOKES_0    = 1.0e-03;          // midplane Stokes number at R_0 for dust with the reference size

const real  RHO_0       = 1.0;              // compact-grain internal density

#ifdef RADIATION
const real  BETA_0      = 1.0e+01;          // the reference ratio between the radiation pressure and the gravity
const real  KAPPA_0     = 1.0;              // the reference gray opacity of the dust
const real  T_BETA      = 2.0*M_PI;         // duration of the smooth radiation startup

#ifdef PR_EFFECT
const real  C_LIGHT     = 1.0e+04;          // speed of light in orbital code velocity units
#endif // PR_EFFECT
#endif // RADIATION

#ifdef DIFFUSION
const real  SCHMIDT_X   = 1.0;              // the Schmidt number for cylindrical azimuthal diffusion
const real  SCHMIDT_R   = 1.0;              // the Schmidt number for cylindrical radial diffusion
#endif // DIFFUSION

#if defined(DIFFUSION) || defined(COLLISION)
const real  SCHMIDT_Z   = 1.0;              // the Schmidt number for cylindrical vertical diffusion
#endif // DIFFUSION || COLLISION

#ifdef COLLISION
const int   COAG_KERNEL = 0;                // coagulation kernels: 0 = constant, 1 = linear, 2 = product, 3 = custom
const int   N_K         = 200;              // number of candidate slots returned by each KNN query

const real  H_SEARCH    = 1.0;              // KNN search radius in units of the local gas scale height
const real  V_FRAG      = 1.0;              // the fragmentation velocity for dust collision
const real  CFL_COL     = 0.02;             // maximum collision propensity per representative and batch

#ifdef COLLISION_MORTON
const int   MORTON_TPB   = 256;             // cooperative threads assigned to one Morton query
const int   MORTON_LEAF  = 128;             // target records per adaptive leaf
const int   MORTON_LEVEL = 20;              // maximum adaptive subdivision depth
static_assert(N_K <= 256, "The current cooperative Morton workspace supports N_K <= 256");
#endif // COLLISION_MORTON
#endif // COLLISION

// =========================================================================================================================
// dust initialization parameters
// =========================================================================================================================

#ifdef MULTISIZE
const real INIT_SMIN    = 5.0e-01;          // minimum grain size for particle initialization
const real INIT_SMAX    = 2.0e+00;          // maximum grain size for particle initialization
#endif // MULTISIZE

// =========================================================================================================================
// time step and output parameters
// =========================================================================================================================

const int  SAVE_MAX     = 1;                // total number of outputs for mesh fields

const real DT_OUT       = 1.0e-04;

#ifdef TRANSPORT
const real DT_MAX       = 0.1;
const real CFL_DYN      = 0.5;              // maximum fraction of a local mesh scale crossed in one dynamics step
#endif // TRANSPORT

#if defined(LOGTIMING) || defined(LOGOUTPUT)
const int  LOG_BASE     = 10;               // logarithmic base for time stepping (LOGTIMING) or particle output (LOGOUTPUT)
#else  // LINEAR
const int  LIN_BASE     = 1;                // save particle data every LIN_BASE iterations
#endif // LOGTIMING || LOGOUTPUT

// =========================================================================================================================
// structures
// =========================================================================================================================

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
struct tree                                 // KD-tree node consumed by cukd::builder
{
    float3  cartesian;                      // Cartesian position of the physical particle or periodic image
    int     index_old;                      // stable particle-array index before KD-tree reordering
    int     split_dim;                      // splitting dimension of the tree node
    int     image;                          // zero for a physical node and nonzero for a periodic image
};

struct tree_traits                          // traits for cukd::builder
{
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    // expose point coordinates and split dimensions through the cuKD traits interface
    static inline __host__ __device__ const point_t &get_point (const tree &node) { return node.cartesian; }
    static inline __host__ __device__ float get_coord (const tree &node, int dim) { return cukd::get_coord(node.cartesian, dim); }
    static inline __host__ __device__ int get_dim (const tree &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (tree &node, int dim) { node.split_dim = dim; }
};
#endif // COLLISION_KDTREE

// =========================================================================================================================
// cuda numerical parameters
// =========================================================================================================================

const int TPB = 32; // number of threads per block

const int NB_P = N_P     / TPB + 1;         // number of blocks for swarm-level parallelization
const int NB_G = N_G     / TPB + 1;         // number of blocks for grid-level  parallelization
const int NB_X = N_Y*N_Z / TPB + 1;         // number of blocks for X-direction parallelization
const int NB_Y = N_X*N_Z / TPB + 1;         // number of blocks for Y-direction parallelization

#ifdef COLLISION_KDTREE
const int N_T  = (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12) ? 3*N_P : N_P; // physical and periodic-image tree nodes
const int NB_T = N_T     / TPB + 1;         // number of blocks for tree-level parallelization
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // CONST_DEFS_CUH
