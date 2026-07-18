#include <curand_kernel.h>
#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernels: f_rho_initial, f_vel_initial
// Purpose: Set the initial dust density, followed by a no-radiation drag-pressure drift state.
//
// Density: ρ_d(R,Z) = Σ_d(R) / (√(2π)·H_d) · exp(−Z²/(2·H_d²))
//          where Σ_d(R) is interpolated from the uniformly sampled cylindrical-radius
//          convolved profile dev_initdens,
//          and H_d = H_g·√(δ_z/(δ_z+St_mid)), δ_z=α/Sc_z
//          (Gaussian turbulent settling-diffusion equilibrium).
//          For N_X>1, a 10% Gaussian perturbation depending only on azimuth is added to break
//          azimuthal symmetry without injecting independent radial or vertical structure.
//
// Drift velocity (before radiation is ramped on):
//   v_R   = 2 St (u_phi-v_K)/(1+St²),
//   v_phi = u_phi - 0.5 St v_R,  u_phi=v_K√(1−2η).
// For N_Z>1, v_Z is chosen from the continuum vertical diffusion balance of the initialized
// Gaussian column, then (v_R,v_Z) is transformed to the stored spherical (v_r,l_theta) form.
// =========================================================================================================================

__global__
void f_rho_initial (real *dev_dustdens, const real *dev_initdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    // ---- Decompose flat index into 3-D cell indices ----
    int ix = idx % N_X;
    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);   

    // ---- Cell centre coordinates ----

    real yc = Y_MIN*pow(_get_dy(), iy + 0.5); // geometric centre (volume centroid)
    real zc = (N_Z > 1) ? (Z_MIN + (iz + 0.5)*_get_dz()) : 0.5*(Z_MIN + Z_MAX);

    real R = yc*sin(zc);
    real Z = yc*cos(zc);

    // ---- Gas parameters at cell centre ----
    real h_g = _get_hg(R);
    // ---- Dust scale height (turbulent diffusion equilibrium) ----
    #ifdef DIFFUSION
    real delta_z = _get_alpha(R, h_g) / SC_Z;
    // One Gaussian column requires one scale height at each cylindrical radius, so use the
    // midplane Stokes number rather than the cell-local value at Z.  For constant grain size
    // in Epstein drag, _get_St gives St_mid = ST_0*(R/R_0)^(-IDX_P).
    real St_mid = _get_St(R, 0.0, h_g);
    real H_d = h_g*R*sqrt(delta_z / (delta_z + St_mid));
    #else
    real H_d = h_g*R;
    #endif

    // Interpolate Sigma_d at cylindrical R, not spherical grid radius yc.  The convolved
    // profile is tabulated at N_Y+1 uniform nodes over [Y_MIN,Y_MAX]; cells outside that
    // cylindrical support start with no dust.  This is the only surface-to-volume conversion.
    real sigma_d = 0.0;
    if (R >= Y_MIN && R <= Y_MAX)
    {
        real du = (Y_MAX - Y_MIN) / static_cast<real>(N_Y);
        int i_u = static_cast<int>((R - Y_MIN) / du);
        if (i_u >= N_Y) i_u = N_Y - 1;
        real frac_u = (R - (Y_MIN + i_u*du)) / du;
        sigma_d = (1.0 - frac_u)*dev_initdens[i_u] + frac_u*dev_initdens[i_u + 1];
    }

    real rho_d = sigma_d / (sqrt(2.0*M_PI)*H_d)*exp(-Z*Z / (2.0*H_d*H_d));

    if (N_X > 1)
    {
        // Seed only by ix: every (R,Z) cell at a given azimuth receives the same multiplier.
        // No post-perturbation mass renormalization is applied.
        curandState rng;
        curand_init(static_cast<unsigned long long>(ix), 0ULL, 0ULL, &rng);
        real xi = curand_normal_double(&rng);   // ξ(ix) ~ N(0,1)
        rho_d = fmax(rho_d*(1.0 + 0.1*xi), 0.0);
    }

    dev_dustdens[idx] = rho_d;
}

__global__
void f_vel_initial (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real yc = Y_MIN*pow(_get_dy(), iy + 0.5);
    real zc = (N_Z > 1) ? (Z_MIN + (iz + 0.5)*_get_dz()) : 0.5*(Z_MIN + Z_MAX);

    real sin_z = sin(zc);
    real cos_z = cos(zc);
    real R = yc*sin_z;
    real Z = yc*cos_z;

    real h_g = _get_hg(R);
    real omega = _get_omegaK(R);
    real v_K = R*omega;
    real eta = _get_eta(R, Z, h_g);
    real u_phi = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real St = _get_St(R, Z, h_g);

    // Standard test-particle drift equilibrium for gas with zero radial velocity.  Radiation
    // is deliberately absent here and is turned on smoothly by f_source_term.
    real v_R = 2.0*St*(u_phi - v_K) / (1.0 + St*St);
    real v_phi = u_phi - 0.5*St*v_R;

    real v_Z = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        real H_g = h_g*R;
        real delta_z = _get_alpha(R, h_g) / SC_Z;
        real St_mid = _get_St(R, 0.0, h_g);
        real H_d = H_g*sqrt(delta_z / (delta_z + St_mid));
        real D_z = _get_nu(R, h_g) / SC_Z;

        // Zero continuum vertical dust flux for the initialized Gaussian profiles:
        // rho_d*v_Z - D_z*rho_g*d(rho_d/rho_g)/dZ = 0.
        v_Z = -D_z*Z*(1.0/(H_d*H_d) - 1.0/(H_g*H_g));
    }
    #endif // DIFFUSION

    // Cylindrical-to-spherical meridional transformation:
    // e_R = sin(theta)e_r + cos(theta)e_theta,
    // e_Z = cos(theta)e_r - sin(theta)e_theta.
    real v_r = v_R*sin_z + v_Z*cos_z;
    real v_theta = v_R*cos_z - v_Z*sin_z;

    dev_dustvelx[idx] = R*v_phi;
    dev_dustvely[idx] = v_r;
    dev_dustvelz[idx] = yc*v_theta;
}

// =========================================================================================================================
