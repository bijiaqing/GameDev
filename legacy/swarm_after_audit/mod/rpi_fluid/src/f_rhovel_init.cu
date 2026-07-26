#include <curand_kernel.h>
#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernels: f_rho_initial, f_vel_initial
// Purpose: Set the initial dust density, followed by a no-radiation drag-pressure drift state.
//
// Density: rho_d = rho_g*q_d, where rho_g uses the exact hydrostatic stratification and
//          q_d adds the settled dust concentration relative to the gas.  This reduces to the
//          familiar Gaussian dust layer in the thin-disk limit.
//          Sigma_d(R) is interpolated from the uniformly sampled cylindrical-radius dev_initdens,
//          and H_d = H_g·√(δ_z/(δ_z+St_mid)), δ_z=α/Sc_z
//          sets the intended near-midplane settling-diffusion scale height.
//          A 10% Gaussian perturbation depending only on azimuth is added to break azimuthal
//          symmetry without injecting independent radial or vertical structure.
//
// Drift velocity (before radiation is ramped on):
//   v_R = 2 St (vg_x-v_K)/(1+St²),
//   v_x = vg_x - 0.5 St v_R,  vg_x=v_K√(1−2η).
// For N_Z>1, the spherical-polar velocity is initialized directly from
//          v_z = (D_z/y)*d ln(rho_d/rho_g)/dz,
// so its advective mass flux balances the same coordinate-aligned polar diffusion operator used later.
// =========================================================================================================================

#ifdef DIFFUSION
static __device__ __forceinline__
real _get_init_ratio (const real *dev_dustdens, int idx, real yc, real zc)
{
    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);
    real h_g = _get_hg(Rc);
    real rhog = _get_rhog(Rc, Zc, h_g);

    return dev_dustdens[idx] / rhog;
}
#endif // DIFFUSION

__global__
void f_rho_initial (real *dev_dustdens, const real *dev_initdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int ix = idx % N_X;
    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X * N_Y);

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real h_g = _get_hg(Rc);

    real H_g = h_g*Rc;

    #ifdef DIFFUSION
    real alpha_z = _get_alpha(Rc, h_g) / SC_Z;
    real St_mid  = _get_St(Rc, 0.0, h_g);
    real H_d     = H_g*sqrt(alpha_z / (alpha_z + St_mid));
    #else
    real H_d     = H_g;
    #endif

    real sigma_d = 0.0;
    if (Rc >= Y_MIN && Rc <= Y_MAX)
    {
        real du = (Y_MAX - Y_MIN) / static_cast<real>(N_Y);
        int iu = static_cast<int>((Rc - Y_MIN) / du);
        if (iu >= N_Y) iu = N_Y - 1;

        real frac_u = (Rc - (Y_MIN + iu*du)) / du;
        sigma_d = (1.0 - frac_u)*dev_initdens[iu] + frac_u*dev_initdens[iu + 1];
    }

    real sigma_g = SIGMA_0*pow(Rc / R_0, IDX_P);
    real rhog = _get_rhog(Rc, Zc, h_g);
    real ratio_mid = sigma_d*H_g / (sigma_g*H_d);
    real settle_exp = exp(-0.5*Zc*Zc*(1.0/(H_d*H_d) - 1.0/(H_g*H_g)));
    real dens = rhog*ratio_mid*settle_exp;

    // seed only by ix: every (R,Z) cell at a given azimuth receives the same multiplier
    curandState rng;
    curand_init(static_cast<unsigned long long>(ix), 0ULL, 0ULL, &rng);
    real xi = curand_normal_double(&rng);   // ξ(ix) ~ N(0,1)
    dens = fmax(dens*(1.0 + 0.1*xi), 0.0);

    dev_dustdens[idx] = dens;
}

__global__
void f_vel_initial (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz
    #ifdef DIFFUSION
    , const real *dev_dustdens
    #endif
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real h_g = _get_hg(Rc);
    real omega = _get_omegaK(Rc);
    real v_K = Rc*omega;
    real eta = _get_eta(Rc, Zc, h_g);
    real vg_x = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real St = _get_St(Rc, Zc, h_g);

    real v_R = 2.0*St*(vg_x - v_K) / (1.0 + St*St);
    real v_x = vg_x - 0.5*St*v_R;

    real speed_z_diff = 0.0;
    #ifdef DIFFUSION
    if (N_Z > 1)
    {
        real Dz = _get_nu(Rc, h_g) / SC_Z;
        real ratio = _get_init_ratio(dev_dustdens, idx, yc, zc);
        real grad_ratio;

        if (iz == 0)
        {
            int idx_next = idx + N_X*N_Y;
            real ratio_next = _get_init_ratio(dev_dustdens, idx_next, yc, zc + dz);
            grad_ratio = (ratio_next - ratio) / dz;
        }
        else if (iz == N_Z - 1)
        {
            int idx_prev = idx - N_X*N_Y;
            real ratio_prev = _get_init_ratio(dev_dustdens, idx_prev, yc, zc - dz);
            grad_ratio = (ratio - ratio_prev) / dz;
        }
        else
        {
            int idx_prev = idx - N_X*N_Y;
            int idx_next = idx + N_X*N_Y;
            real ratio_prev = _get_init_ratio(dev_dustdens, idx_prev, yc, zc - dz);
            real ratio_next = _get_init_ratio(dev_dustdens, idx_next, yc, zc + dz);
            grad_ratio = (ratio_next - ratio_prev) / (2.0*dz);
        }

        if (ratio > 0.0) speed_z_diff = Dz*grad_ratio / (yc*ratio);
    }
    #endif // DIFFUSION

    real v_y = v_R*sin(zc);
    real v_z = v_R*cos(zc) + speed_z_diff;

    dev_dustvelx[idx] = Rc*v_x;
    dev_dustvely[idx] = v_y;
    dev_dustvelz[idx] = yc*v_z;
}

// =========================================================================================================================
