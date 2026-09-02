#ifndef VERIFY_PARAM_PHYS_CUH
#define VERIFY_PARAM_PHYS_CUH

// import the production physical prescriptions under private verification names; the temporary macro substitutions rename
// the function definitions while the production header is parsed; after undefining the macros, this file can provide wrappers
// with the original names and selectively replace only the physics required by an analytical test
#ifdef DIFFUSION
#define _get_nu    _test_prod_nu
#define _get_alpha _test_prod_alpha
#endif // DIFFUSION

#include "../../../../inc/comm/fluid/param_phys.cuh"

#ifdef DIFFUSION
#undef _get_nu
#undef _get_alpha
#endif // DIFFUSION

#ifdef DIFFUSION
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    // constant diffusivity gives the Fourier, radial-Bessel, and Legendre modes simple exponential analytical decay rates
#ifdef VERIFY_CONSTANT_DIFFUSIVITY
    (void)R;
    (void)h_g;
    return VERIFY_D;
#else
    return _test_prod_nu(R, h_g);
#endif // VERIFY_CONSTANT_DIFFUSIVITY
}

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    // convert the requested constant physical diffusivity back into the alpha value expected by production initialization
    // helpers, maintaining nu = alpha*h_g^2*R^2*Omega_K
#ifdef VERIFY_CONSTANT_DIFFUSIVITY
    return VERIFY_D / (h_g*h_g*R*R*_get_omegaK(R));
#else
    return _test_prod_alpha(R, h_g);
#endif // VERIFY_CONSTANT_DIFFUSIVITY
}
#endif // DIFFUSION

#endif // VERIFY_PARAM_PHYS_CUH
