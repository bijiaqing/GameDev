#ifndef PARAM_GRID_CUH
#define PARAM_GRID_CUH

#include <cmath>  // cos, fmin, log, pow, sin

#include <const_defs.cuh>

// =========================================================================================================================
// swarm mesh primitives
// =========================================================================================================================

__host__ __device__ __forceinline__
real _get_dx() { return (X_MAX - X_MIN) / static_cast<real>(N_X); }

__host__ __device__ __forceinline__
real _get_dy() { return pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); }

__host__ __device__ __forceinline__
real _get_dz() { return (Z_MAX - Z_MIN) / static_cast<real>(N_Z); }

// return the cylindrical radius with an exact vertically integrated branch
__host__ __device__ __forceinline__
real _get_cyl_R (real y, real z) { return (N_Z > 1) ? y*sin(z) : y; }

// return the cylindrical height with an exact vertically integrated branch
__host__ __device__ __forceinline__
real _get_cyl_Z (real y, real z) { return (N_Z > 1) ? y*cos(z) : 0.0; }

// smallest cylindrical radius covered by the spherical domain
__host__ __device__ __forceinline__
real _get_init_Rmin()
{
    if (N_Z == 1) return Y_MIN;

    return Y_MIN*fmin(sin(Z_MIN), sin(Z_MAX));
}

// =========================================================================================================================
// cell faces
// =========================================================================================================================

__host__ __device__ __forceinline__
real _get_yface (int iy) { return Y_MIN*pow(_get_dy(), static_cast<real>(iy)); }

__host__ __device__ __forceinline__
real _get_zface (int iz) { return Z_MIN + _get_dz()*static_cast<real>(iz); }

// =========================================================================================================================
// active geometry and cell measures
// =========================================================================================================================

// radial measure power: 2 for a vertically integrated disk and 3 when the polar dimension is active
__host__ __device__ __forceinline__
real _get_mesh_dim() { return 2.0 + static_cast<real>(N_Z > 1); }

// azimuthal cell measure or the complete 2pi extent of an axisymmetric cell
__host__ __device__ __forceinline__
real _get_vol_x ()
{
    real vol_x = _get_dx();

    return (N_X > 1) ? vol_x : 2.0*M_PI;
}

// radial cell measure y0^d*(dy^d - 1)/d
__host__ __device__ __forceinline__
real _get_vol_y (int iy)
{
    real mesh_dim = _get_mesh_dim();
    real vol_y = pow(_get_yface(iy), mesh_dim)*(pow(_get_dy(), mesh_dim) - 1.0) / mesh_dim;

    return vol_y;
}

// sin-weighted polar cell measure with a neutral value for an inactive polar dimension
__host__ __device__ __forceinline__
real _get_vol_z (int iz)
{
    real vol_z = cos(_get_zface(iz)) - cos(_get_zface(iz + 1));

    return (N_Z > 1) ? vol_z : 1.0;
}

// =========================================================================================================================
// continuous grid coordinates
// =========================================================================================================================

// convert azimuth to its continuous uniform-grid coordinate
__device__ __forceinline__
real _get_loc_x (real x)
{ return (N_X > 1) ? (x - X_MIN) / _get_dx() : 0.0; }

// convert spherical radius to its continuous logarithmic-grid coordinate
__device__ __forceinline__
real _get_loc_y (real y)
{ return (N_Y > 1) ? log(y / Y_MIN) / log(_get_dy()) : 0.0; }

// convert polar angle to its continuous uniform-grid coordinate
__device__ __forceinline__
real _get_loc_z (real z)
{ return (N_Z > 1) ? (z - Z_MIN) / _get_dz() : 0.0; }

// =========================================================================================================================
// grid bounds and indexing
// =========================================================================================================================

// test whether continuous grid coordinates lie inside every active dimension
__device__ __forceinline__
bool _is_in_bounds (real loc_x, real loc_y, real loc_z)
{
    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);
    
    return in_x && in_y && in_z;
}

// =========================================================================================================================

#endif // PARAM_GRID_CUH
