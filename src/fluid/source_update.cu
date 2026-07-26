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
real _interp_optdepth (real optdepth_i, real optdepth_o)
{
    real frac_c = 1.0 / (sqrt(_get_dy()) + 1.0);

    return optdepth_i + frac_c*(optdepth_o - optdepth_i);
}
#endif // RADIATION

// evaluate spherical radial force, radial centrifugal acceleration, and polar geometric torque
static __device__ __forceinline__
void _get_force_term (real y, real z, real R, real lx, real lz, real beta,
    real &grav_y, real &cent_y, real &torq_z)
{
    grav_y = -(1.0 - beta)*_get_omegaK(y)*_get_omegaK(y)*y;
    cent_y = lx*lx / R / R / y + lz*lz / y / y / y;
    torq_z = (N_Z > 1) ? lx*lx / R / R / sin(z)*cos(z) : 0.0;
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
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);

    real R = y*sin(z);
    real Z = y*cos(z);

    // reset near-vacuum cells to the fallback primitive state
    if (dev_dustdens[idx_cell] < RHO_VAC)
    {
        dev_dustvelx[idx_cell] = sqrt(G*M_S*fmax(R, 0.0));
        dev_dustvely[idx_cell] = 0.0;
        dev_dustvelz[idx_cell] = 0.0;

        return;
    }

    real omega = _get_omegaK(R);
    real h_g = _get_hg(R);

    // construct exact exponential drag relaxation coefficients
    real stokes = _get_stokes(R, Z, h_g);
    real ts = stokes / omega;
    real tau = dt / ts;

    real drag_relax = -expm1(-tau);
    real drag_decay = 1.0 - drag_relax;

    // evaluate drag-weighted force quadrature with a cancellation-safe small-step series
    real force_weight_old, force_weight_new;
    if (tau < 1.0e-4)
    {
        real tau_sq = tau*tau;
        real tau_cb = tau_sq*tau;

        force_weight_old = dt*(0.5 - tau/3.0 + tau_sq/8.0  - tau_cb/30.0);
        force_weight_new = dt*(0.5 - tau/6.0 + tau_sq/24.0 - tau_cb/120.0);
    }
    else
    {
        force_weight_new = ts*(tau - drag_relax) / tau;
        force_weight_old = ts*drag_relax - force_weight_new;
    }

    // attenuate radiation pressure by the optical depth interpolated to the logarithmic cell center
    #ifdef RADIATION
    real optdepth_i = (iy > 0) ? dev_optdepth[idx_cell - N_X] : 0.0;
    real optdepth_o = dev_optdepth[idx_cell];
    real beta = beta_taper*BETA_0*exp(-_interp_optdepth(optdepth_i, optdepth_o));
    #else
    real beta  = 0.0;
    #endif

    // load dust primitives and construct the local gas equilibrium state
    real lx = dev_dustvelx[idx_cell];
    real vy = dev_dustvely[idx_cell];
    real lz = (N_Z > 1) ? dev_dustvelz[idx_cell] : 0.0;

    real eta = _get_eta(R, Z, h_g);
    real lx_g = R*R*omega*sqrt(fmax(1.0 - 2.0*eta, 0.0));

    #ifdef VISC_ACCRETION
    real vR_g = _get_visc_vel(R, Z, h_g);
    real vy_g = vR_g*sin(z);
    real lz_g = (N_Z > 1) ? y*vR_g*cos(z) : 0.0;
    #else  // PURE_ROTATION
    real vy_g = 0.0;
    real lz_g = 0.0;
    #endif // VISC_ACCRETION

    // evaluate forces at the old state and relax azimuthal specific angular momentum
    real grav_yold, cent_yold, torq_zold, lx_new;
    _get_force_term(y, z, R, lx, lz, beta, grav_yold, cent_yold, torq_zold);

    lx_new  = drag_decay*lx;
    lx_new += drag_relax*lx_g;

    // re-evaluate polar torque and update polar specific angular momentum
    real grav_ytmp, cent_ytmp, torq_znew, lz_new;
    _get_force_term(y, z, R, lx_new, lz, beta, grav_ytmp, cent_ytmp, torq_znew);

    lz_new  = drag_decay*lz;
    lz_new += drag_relax*lz_g;
    lz_new += force_weight_old*torq_zold;
    lz_new += force_weight_new*torq_znew;
    if (N_Z == 1) lz_new = 0.0;

    // re-evaluate radial forces and update radial velocity
    real grav_ynew, cent_ynew, vy_new;
    _get_force_term(y, z, R, lx_new, lz_new, beta, grav_ynew, cent_ynew, torq_znew);

    vy_new  = drag_decay*vy;
    vy_new += drag_relax*vy_g;
    vy_new += force_weight_old*(grav_yold + cent_yold);
    vy_new += force_weight_new*(grav_ynew + cent_ynew);

    // write the updated primitive state to global memory
    dev_dustvelx[idx_cell] = lx_new;
    dev_dustvely[idx_cell] = vy_new;
    dev_dustvelz[idx_cell] = lz_new;
}
