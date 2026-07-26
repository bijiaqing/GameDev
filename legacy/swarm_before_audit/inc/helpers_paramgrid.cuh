#ifndef HELPERS_PARAMGRID_CUH
#define HELPERS_PARAMGRID_CUH

#include <const.cuh>

// =========================================================================================================================
// Grid Coordinate Conversion Functions
// Purpose: Convert physical coordinates to grid cell coordinates
// =========================================================================================================================

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
real _get_loc_x (real x)
{
    return (N_X > 1) ? (static_cast<real>(N_X)*   (x - X_MIN) /    (X_MAX - X_MIN)) : 0.0;
}

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
real _get_loc_y (real y)
{
    return (N_Y > 1) ? (static_cast<real>(N_Y)*log(y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0;
}

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
real _get_loc_z (real z)
{
    return (N_Z > 1) ? (static_cast<real>(N_Z)*   (z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0;
}

// =========================================================================================================================
// Grid Boundary and Indexing Functions
// Purpose: Check bounds and calculate grid cell indices
// =========================================================================================================================

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
bool _is_in_bounds (real loc_x, real loc_y, real loc_z)
{
    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);
    
    return in_x && in_y && in_z;
}

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
int _get_cell_index (real loc_x, real loc_y, real loc_z)
{
    return static_cast<int>(loc_z)*N_X*N_Y + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
}

// =========================================================================================================================
// Grid Cell Volume Calculation
// Purpose: Calculate spherical grid cell volume in cylindrical coordinates
// Used by: optdepth_calc, dustdens_calc, col_proc_exec, col_rate_calc
// =========================================================================================================================

// Order: O0 | Dependencies: None (only constants)
__device__ __forceinline__
real _get_grid_volume (int idx_cell, real *y0_ptr = nullptr, real *dy_ptr = nullptr)
{
    // Convert 1D cell index to 3D grid indices
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

#endif // NOT HELPERS_PARAMGRID_CUH
