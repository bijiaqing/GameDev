#ifndef PARAMGRID_CUH
#define PARAMGRID_CUH

#include <const.cuh>

// =========================================================================================================================
// continuous grid coordinates
// =========================================================================================================================

// convert azimuth to its continuous uniform-grid coordinate
__device__ __forceinline__
real _get_loc_x (real x)
{
    return (N_X > 1) ? (static_cast<real>(N_X)*   (x - X_MIN) /    (X_MAX - X_MIN)) : 0.0;
}

// convert spherical radius to its continuous logarithmic-grid coordinate
__device__ __forceinline__
real _get_loc_y (real y)
{
    return (N_Y > 1) ? (static_cast<real>(N_Y)*log(y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0;
}

// convert polar angle to its continuous uniform-grid coordinate
__device__ __forceinline__
real _get_loc_z (real z)
{
    return (N_Z > 1) ? (static_cast<real>(N_Z)*   (z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0;
}

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

// flatten the cell containing a valid continuous grid position
__device__ __forceinline__
int _get_cell_index (real loc_x, real loc_y, real loc_z)
{
    return static_cast<int>(loc_z)*N_X*N_Y + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
}

// =========================================================================================================================
// grid measure
// =========================================================================================================================

// calculate the exact spherical cell measure for the active one-, two-, or three-dimensional geometry
__device__ __forceinline__
real _get_grid_volume (int idx_cell, real *y0_ptr = nullptr, real *dy_ptr = nullptr)
{
    // recover the radial and polar indices from the flattened cell index
    int idx_x = idx_cell % N_X;
    int idx_y = (idx_cell / N_X) % N_Y;
    int idx_z = idx_cell / (N_X*N_Y);
    
    real idx_dim = static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1) + 1.0;

    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real vol_x = (N_X > 1) ? dx : 1.0;
    
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real y0 = Y_MIN*pow(dy, static_cast<real>(idx_y));
    real vol_y = pow(y0, idx_dim)*(pow(dy, idx_dim) - 1.0) / idx_dim;
    
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
    real z0 = Z_MIN + dz*static_cast<real>(idx_z);
    real vol_z = (N_Z > 1) ? (cos(z0) - cos(z0 + dz)) : 1.0;
    
    if (y0_ptr) *y0_ptr = y0;
    if (dy_ptr) *dy_ptr = dy;
    
    return vol_x*vol_y*vol_z;
}

// =========================================================================================================================

#endif // NOT PARAMGRID_CUH
