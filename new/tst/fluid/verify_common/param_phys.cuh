#ifndef VERIFY_PARAM_PHYS_CUH
#define VERIFY_PARAM_PHYS_CUH

// Import the production physical prescriptions under private verification names.  The temporary macro substitutions rename
// the function definitions while the production header is parsed; after undefining the macros, this file can provide wrappers
// with the original names and selectively replace only the physics required by an analytical test.
#define _get_rhog  _verify_prod_rhog
#ifdef DIFFUSION
#define _get_nu    _verify_prod_nu
#define _get_alpha _verify_prod_alpha
#endif

#include "../../../inc/fluid/param_phys.cuh"

#undef _get_rhog
#ifdef DIFFUSION
#undef _get_nu
#undef _get_alpha
#endif

__device__ __forceinline__
real _get_rhog (real R, real Z, real h_g)
{
    // Isolated diffusion eigenmodes assume uniform gas so density and dust-to-gas ratio obey the same equation.  All other
    // tests call the unchanged production gas-density prescription through its private alias.
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
    // Constant diffusivity gives the Fourier, radial-Bessel, and Legendre modes simple exponential analytical decay rates.
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
    // Convert the requested constant physical diffusivity back into the alpha value expected by production initialization
    // helpers, maintaining nu = alpha*h_g^2*R^2*Omega_K.
#ifdef VERIFY_CONSTANT_DIFFUSIVITY
    return VERIFY_D / (h_g*h_g*R*R*_get_omegaK(R));
#else
    return _verify_prod_alpha(R, h_g);
#endif
}
#endif

#endif
