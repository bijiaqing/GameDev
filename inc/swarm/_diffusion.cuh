#ifndef SWARM_DIFFUSION_CUH
#define SWARM_DIFFUSION_CUH
#ifdef DIFFUSION
#include <param_phys.cuh>

// return the cylindrical volume-measure drift D/R plus the radial viscosity gradient dD/dR at fixed Stokes number
// constant alpha gives nu proportional to R^(q+3/2), hence dD/dR = D*(q+3/2)/R
__device__ __forceinline__
real _get_diff_drift_R (real R, real D)
{
    #ifdef CONST_NU
    return D / R;
    #else  // CONST_ALPHA
    return D*(IDX_Q + 2.5) / R;
    #endif // CONST_NU
}

// return analytic d ln(rho_g)/dR at fixed Z, or d ln(Sigma_g)/dR in the vertically integrated model
__device__ __forceinline__
real _get_diff_gradlnrho_R (real R, real Z, real h)
{
    if constexpr (N_Z == 1) return IDX_P / R;
    real r = sqrt(R*R + Z*Z);
    return (IDX_P - 0.5*IDX_Q - 1.5) / R
        +(Z*Z / (r*r*r) - (IDX_Q + 1.0)*(R / r - 1.0) / R) / (h*h);
}
// return analytic d ln(rho_g)/dZ at fixed R from the exact spherical stratification
__device__ __forceinline__
real _get_diff_gradlnrho_Z (real R, real Z, real h)
{
    return -R*Z / (h*h*pow(R*R + Z*Z, 1.5));
}

#ifdef IMPORTGAS
// differentiate the same clamped, periodic trilinear interpolant used by drag
// return (d_phi ln rho, d_R ln rho, d_Z ln rho), with phi measured in radians
__device__ __forceinline__
real3 _get_diff_import_gradient (real x, real y, real z, const real *gas)
{
    real lx = _get_loc_x(x), ly = _get_loc_y(y), lz = _get_loc_z(z);
    #ifdef HALF_DISK
    lz = fmin(lz, static_cast<real>(N_Z) - 1.e-6);
    #endif // HALF_DISK
    int cell = static_cast<int>(lx) + N_X*static_cast<int>(ly) + N_X*N_Y*static_cast<int>(lz);
    auto t = _3d_interp(lx, ly, lz);
    // derivatives of the separable weights with respect to phi, spherical y, and polar z
    real dx = t.next_x ? ((lx - floor(lx) >= 0.5) ? 1.0 : -1.0) / _get_dx() : 0.0;
    real step = t.next_y > 0 ? _get_dy() - 1.0 : 1.0 / _get_dy() - 1.0;
    real dy = t.next_y ? (1.0 + t.frac_y*step) / (y*step) : 0.0;
    real dz = t.next_z ? (t.next_z > 0 ? 1.0 : -1.0) / _get_dz() : 0.0;
    real rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    for (int k = 0; k < 2; ++k)for (int j = 0; j < 2; ++j)for (int i = 0; i < 2; ++i)
    {
        real v = gas[cell + i*t.next_x + j*t.next_y + k*t.next_z];
        real wx = i ? t.frac_x : 1.0 - t.frac_x, wy = j ? t.frac_y : 1.0 - t.frac_y, wz = k ? t.frac_z : 1.0 - t.frac_z;
        rho += v*wx*wy*wz;
        gx += v*(i ? dx : -dx)*wy*wz;
        gy += v*wx*(j ? dy : -dy)*wz;
        gz += v*wx*wy*(k ? dz : -dz);
    }
    assert(isfinite(rho) && rho > 0.0);
    // rotate the spherical (y, z) gradient into cylindrical (R, Z) components
    return {gx / rho, (sin(z)*gy + cos(z)*gz / y) / rho, (cos(z)*gy - sin(z)*gz / y) / rho};
}
#endif // IMPORTGAS

// Stokes-suppressed viscosity nu/(1+St^2) and deterministic Ito drifts divided by each directional D
struct dust_diffusion { real nu, drift_R_per_D, drift_Z_per_D, drift_phi_per_D; };
// spatial derivatives hold grain size fixed; collisions update size between operators
__device__ __forceinline__
dust_diffusion _get_dust_diffusion (real x, real y, real z, real size
#ifdef IMPORTGAS
    , const real *gas
#endif // IMPORTGAS
)
{
    real R = _get_cyl_R(y, z), Z = _get_cyl_Z(y, z), h = _get_hg(R);
    real st = _get_stokes(R, Z, h, size
        #ifdef IMPORTGAS
        , x, y, z, gas
        #endif // IMPORTGAS
    );
    // f is the finite-Stokes suppression of the gas diffusivity
    real f = 1.0 / (1.0 + st*st);
    #ifdef IMPORTGAS
    real3 g = _get_diff_import_gradient(x, y, z, gas);
    #else  // ANALYTIC_GAS
    real3 g = {0.0, _get_diff_gradlnrho_R(R, Z, h), (N_Z > 1) ? _get_diff_gradlnrho_Z(R, Z, h) : 0.0};
    #endif // IMPORTGAS
    // Epstein St is proportional to 1/(rho_g H_g), or 1/Sigma_g in 2D; the 3D radial term adds the H_g gradient
    real st_R = (N_Z > 1) ? -g.y - (IDX_Q + 3.0) / (2.0*R) : -g.y;
    real st_Z = -g.z, st_phi = -g.x;
    #if defined(CONST_ST) && !defined(IMPORTGAS)
    st_R = st_Z = st_phi = 0.0;
    #endif // CONST_ST && !IMPORTGAS
    #ifdef DIFFUSE_CONCENTRATION
    constexpr real concentration = 1.0;
    #else  // DENSITY_DIFFUSION
    constexpr real concentration = 0.0;
    #endif // DIFFUSE_CONCENTRATION
    // d ln f = -2*(1-f) d ln St supplies the Stokes part of the diffusivity gradient
    // concentration diffusion adds D*grad(ln rho_g) to each drift
    real factor_grad = 2.0*(1.0 - f);
    return {_get_nu(R, h)*f,
        _get_diff_drift_R(R, 1.0) - factor_grad*st_R + concentration*g.y,
        -factor_grad*st_Z + concentration*g.z,
        -factor_grad*st_phi + concentration*g.x};
}
#endif // DIFFUSION
#endif // SWARM_DIFFUSION_CUH
