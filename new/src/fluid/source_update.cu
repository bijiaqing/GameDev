#include <fluid_kern.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// kernel: source_update
// purpose: update dust primitives under gas drag, gravity, radiation pressure, and spherical geometric forces
//
// parallelization: one thread per grid cell
//
// per call:
//   1 near-vacuum state recovery
//   2 exact exponential drag relaxation
//   3 drag-weighted old-to-new force quadrature
//   4 sequential azimuthal, polar, and radial primitive updates
// =========================================================================================================================

#ifdef RADIATION
// interpolate cumulative outer-face optical depth to the logarithmic cell center
static __device__ __forceinline__
real _interp_optdepth (real tau_i, real tau_o)
{
    real frac_c = 1.0 / (sqrt(_get_dy()) + 1.0);

    return tau_i + frac_c*(tau_o - tau_i);
}
#endif // RADIATION

// evaluate spherical radial force, radial centrifugal acceleration, and polar geometric torque
static __device__ __forceinline__
void _get_force_term (real yc, real zc, real Rc, real lx, real lz, real beta, real &Fy, real &Fcy, real &Tcz)
{
    Fy  = -(1.0 - beta)*_get_omegaK(yc)*_get_omegaK(yc)*yc;
    Fcy = lx*lx / Rc / Rc / yc + lz*lz / yc / yc / yc;
    Tcz = (N_Z > 1) ? lx*lx / Rc / Rc / sin(zc)*cos(zc) : 0.0;
}

__global__
void source_update (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    const real *dev_dustdens,
    #ifdef RADIATION
    const real *dev_optdepth,
    real beta_taper,
    #endif
    real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X*N_Y);

    real yc = _get_ycent(iy);
    real zc = _get_zcent(iz);

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    // reset near-vacuum cells to the fallback primitive state
    if (dev_dustdens[idx] < RHO_VAC)
    {
        dev_dustvelx[idx] = sqrt(G*M_S*fmax(Rc, 0.0));
        dev_dustvely[idx] = 0.0;
        dev_dustvelz[idx] = 0.0;

        return;
    }

    real omega = _get_omegaK(Rc);
    real h_g = _get_hg(Rc);

    // construct exact exponential drag relaxation coefficients
    real stokes = _get_stokes(Rc, Zc, h_g);
    real ts = stokes / omega;
    real drag_h = dt / ts;

    real drag_relax = -expm1(-drag_h);
    real drag_decay = 1.0 - drag_relax;

    // evaluate drag-weighted force quadrature with a cancellation-safe small-step series
    real force_weight_n, force_weight_new;
    if (drag_h < 1.0e-4)
    {
        real drag_h2 = drag_h*drag_h;
        real drag_h3 = drag_h2*drag_h;

        force_weight_n   = dt*(0.5 - drag_h/3.0 + drag_h2/8.0  - drag_h3/30.0);
        force_weight_new = dt*(0.5 - drag_h/6.0 + drag_h2/24.0 - drag_h3/120.0);
    }
    else
    {
        force_weight_new = ts*(drag_h - drag_relax) / drag_h;
        force_weight_n   = ts*drag_relax - force_weight_new;
    }

    // attenuate radiation pressure by the optical depth interpolated to the logarithmic cell center
    #ifdef RADIATION
    real tau_i = (iy > 0) ? dev_optdepth[idx - N_X] : 0.0;
    real tau_o = dev_optdepth[idx];
    real beta  = beta_taper*BETA_0*exp(-_interp_optdepth(tau_i, tau_o));
    #else
    real beta  = 0.0;
    #endif

    // load dust primitives and construct the local gas equilibrium state
    real lx = dev_dustvelx[idx];
    real vy = dev_dustvely[idx];
    real lz = (N_Z > 1) ? dev_dustvelz[idx] : 0.0;

    real eta = _get_eta(Rc, Zc, h_g);
    real lx_g = Rc*Rc*omega*sqrt(fmax(1.0 - 2.0*eta, 0.0));

    #ifdef VISC_ACCRETION
    real vgas_R = _get_visc_vel(Rc, Zc, h_g);
    real vy_g = vgas_R*sin(zc);
    real lz_g = (N_Z > 1) ? yc*vgas_R*cos(zc) : 0.0;
    #else  // PURE_ROTATION
    real vy_g = 0.0;
    real lz_g = 0.0;
    #endif // VISC_ACCRETION

    // evaluate forces at the old state and relax azimuthal specific angular momentum
    real Fy_n, Fcy_n, Tcz_n, lx_new;
    _get_force_term(yc, zc, Rc, lx, lz, beta, Fy_n, Fcy_n, Tcz_n);

    lx_new  = drag_decay*lx;
    lx_new += drag_relax*lx_g;

    // re-evaluate polar torque and update polar specific angular momentum
    real Fy_tmp, Fcy_tmp, Tcz_new, lz_new;
    _get_force_term(yc, zc, Rc, lx_new, lz, beta, Fy_tmp, Fcy_tmp, Tcz_new);

    lz_new  = drag_decay*lz;
    lz_new += drag_relax*lz_g;
    lz_new += force_weight_n*Tcz_n;
    lz_new += force_weight_new*Tcz_new;
    if (N_Z == 1) lz_new = 0.0;

    // re-evaluate radial forces and update radial velocity
    real Fy_new, Fcy_new, vy_new;
    _get_force_term(yc, zc, Rc, lx_new, lz_new, beta, Fy_new, Fcy_new, Tcz_new);

    vy_new  = drag_decay*vy;
    vy_new += drag_relax*vy_g;
    vy_new += force_weight_n*(Fy_n + Fcy_n);
    vy_new += force_weight_new*(Fy_new + Fcy_new);

    // write the updated primitive state to global memory
    dev_dustvelx[idx] = lx_new;
    dev_dustvely[idx] = vy_new;
    dev_dustvelz[idx] = lz_new;
}
