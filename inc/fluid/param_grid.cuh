#ifndef PARAM_GRID_CUH
#define PARAM_GRID_CUH

#include <cmath> // cos, fmin, pow, sin

#include <const_defs.cuh>

// =========================================================================================================================
// fluid mesh primitives
// =========================================================================================================================

// grid spacing
// -------------------------------------------------------------------------------------------------------------------------

__host__ __device__ __forceinline__
real _get_dx() { return (X_MAX - X_MIN) / static_cast<real>(N_X); }

__host__ __device__ __forceinline__
real _get_dy() { return pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); }

__host__ __device__ __forceinline__
real _get_dz() { return (Z_MAX - Z_MIN) / static_cast<real>(N_Z); }

// smallest cylindrical radius covered by the spherical domain
__host__ __device__ __forceinline__
real _get_init_Rmin()
{
    if (N_Z == 1) return Y_MIN;

    return Y_MIN*fmin(sin(Z_MIN), sin(Z_MAX));
}

// active geometry
// -------------------------------------------------------------------------------------------------------------------------

// radial measure power: 2 for a vertically integrated disk and 3 when the polar dimension is active
__host__ __device__ __forceinline__
real _get_mesh_dim() { return 2.0 + static_cast<real>(N_Z > 1); }

// cell faces and centers
// -------------------------------------------------------------------------------------------------------------------------

// radial faces and logarithmic center
__host__ __device__ __forceinline__
real _get_yface (int iy) { return Y_MIN*pow(_get_dy(), static_cast<real>(iy)); }

__host__ __device__ __forceinline__
real _get_ycent (int iy) { return Y_MIN*pow(_get_dy(), static_cast<real>(iy) + 0.5); }

// polar faces and coordinate center
__host__ __device__ __forceinline__
real _get_zface (int iz) { return Z_MIN + static_cast<real>(iz)*_get_dz(); }

__host__ __device__ __forceinline__
real _get_zcent (int iz) { return Z_MIN + (static_cast<real>(iz) + 0.5)*_get_dz(); }

// cell measures
// -------------------------------------------------------------------------------------------------------------------------

// radial face-area factor y^(d-1)
__host__ __device__ __forceinline__
real _get_area_y (int iy) { return pow(_get_yface(iy), _get_mesh_dim() - 1.0); }

// radial cell measure y0^d*(dy^d - 1)/d
__host__ __device__ __forceinline__
real _get_vol_y (int iy)
{
    real dy = _get_dy();
    real mesh_dim = _get_mesh_dim();

    return pow(_get_yface(iy), mesh_dim)*(pow(dy, mesh_dim) - 1.0) / mesh_dim;
}

// sin-weighted polar cell measure with a neutral value for an inactive polar dimension
__host__ __device__ __forceinline__
real _get_vol_z (int iz)
{ return (N_Z > 1) ? cos(_get_zface(iz)) - cos(_get_zface(iz + 1)) : 1.0; }

// =========================================================================================================================

#endif
