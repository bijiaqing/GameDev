#ifndef VERIFY_HELPERS_CUH
#define VERIFY_HELPERS_CUH

// Reuse every production helper except the gas density and diffusivity hooks
// needed by the constant-coefficient operator benchmarks. Renaming while the
// production header is included avoids copying the PPM/HLL implementation.
#define _get_rhog  _verify_prod_rhog
#ifdef DIFFUSION
#define _get_nu    _verify_prod_nu
#define _get_alpha _verify_prod_alpha
#endif

#include "../../inc/helpers.cuh"

#undef _get_rhog
#ifdef DIFFUSION
#undef _get_nu
#undef _get_alpha
#endif

__device__ __forceinline__
real _get_rhog (real R, real Z, real h_g)
{
#ifdef VERIFY_UNIFORM_GAS
    (void)R;
    (void)Z;
    (void)h_g;
    return 1.0;
#else
    return _verify_prod_rhog(R, Z, h_g);
#endif
}

#ifdef DIFFUSION
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
#ifdef VERIFY_CONSTANT_DIFFUSIVITY
    (void)R;
    (void)h_g;
    return VERIFY_D;
#else
    return _verify_prod_nu(R, h_g);
#endif
}

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
#ifdef VERIFY_CONSTANT_DIFFUSIVITY
    return VERIFY_D / (h_g*h_g*R*R*_get_omegaK(R));
#else
    return _verify_prod_alpha(R, h_g);
#endif
}
#endif

#endif // VERIFY_HELPERS_CUH
