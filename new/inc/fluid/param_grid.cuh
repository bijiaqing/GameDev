#ifndef PARAM_GRID_CUH
#define PARAM_GRID_CUH

#include <const.cuh>

// =========================================================================================================================

__host__ __device__ __forceinline__
real _get_dx() { return (X_MAX - X_MIN) / static_cast<real>(N_X); }

__host__ __device__ __forceinline__
real _get_dy() { return pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); }

__host__ __device__ __forceinline__
real _get_dz() { return (Z_MAX - Z_MIN) / static_cast<real>(N_Z); }

// the power of y used in the volume calculation, which is 3 for 3D and 2 for 2D
__host__ __device__ __forceinline__
real _get_powy() { return (N_Z > 1) ? 3.0 : 2.0; }

// =========================================================================================================================

#endif
