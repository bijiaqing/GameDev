#ifndef SWARM_DIFFUSION_CUH
#define SWARM_DIFFUSION_CUH

#ifdef DIFFUSION

#include <const_defs.cuh>

// =========================================================================================================================
// density-diffusion drift
// =========================================================================================================================

// calculate the cylindrical Ito and variable-diffusivity drift for diffusion of dust density
__device__ __forceinline__
real _get_diff_drift_R (real R, real coeff_R)
{
    #ifdef CONST_NU
    real idx_diff = 0.0;
    #else  // CONST_ALPHA
    real idx_diff = IDX_Q + 1.5;
    #endif // CONST_NU

    return coeff_R*(idx_diff + 1.0) / R;
}

#endif // DIFFUSION

// =========================================================================================================================

#endif // SWARM_DIFFUSION_CUH
