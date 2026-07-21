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

// evaluate spherical radial force, radial centrifugal acceleration, and polar geometric torque
static __device__ __forceinline__
void _get_force_term (real yc, real zc, real Rc, real velx, real velz, real beta, real &Fy, real &Fcy, real &Tcz)
{
    Fy  = -(1.0 - beta)*_get_omegaK(yc)*_get_omegaK(yc)*yc;
    Fcy = velx*velx / Rc / Rc / yc + velz*velz / yc / yc / yc;
    Tcz = velx*velx / Rc / Rc / sin(zc)*cos(zc);
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

    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

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

    // attenuate radiation pressure by the cell-centred optical depth
    #ifdef RADIATION
    real tau_i = (iy > 0) ? dev_optdepth[idx - N_X] : 0.0;
    real tau_o = dev_optdepth[idx];
    real beta  = beta_taper*BETA_0*exp(-0.5*(tau_i + tau_o));
    #else
    real beta  = 0.0;
    #endif

    // load dust primitives and construct the local gas equilibrium state
    real velx = dev_dustvelx[idx];
    real vely = dev_dustvely[idx];
    real velz = dev_dustvelz[idx];

    real eta = _get_eta(Rc, Zc, h_g);
    real velx_g = Rc*Rc*omega*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real vely_g = 0.0;
    real velz_g = 0.0;

    // evaluate forces at the old state and relax azimuthal specific angular momentum
    real Fy_n, Fcy_n, Tcz_n, velx_new;
    _get_force_term(yc, zc, Rc, velx, velz, beta, Fy_n, Fcy_n, Tcz_n);

    velx_new  = drag_decay*velx;
    velx_new += drag_relax*velx_g;

    // re-evaluate polar torque and update polar specific angular momentum
    real Fy_tmp, Fcy_tmp, Tcz_new, velz_new;
    _get_force_term(yc, zc, Rc, velx_new, velz, beta, Fy_tmp, Fcy_tmp, Tcz_new);

    velz_new  = drag_decay*velz;
    velz_new += drag_relax*velz_g;
    velz_new += force_weight_n*Tcz_n;
    velz_new += force_weight_new*Tcz_new;

    // re-evaluate radial forces and update radial velocity
    real Fy_new, Fcy_new, vely_new;
    _get_force_term(yc, zc, Rc, velx_new, velz_new, beta, Fy_new, Fcy_new, Tcz_new);

    vely_new  = drag_decay*vely;
    vely_new += drag_relax*vely_g;
    vely_new += force_weight_n*(Fy_n + Fcy_n);
    vely_new += force_weight_new*(Fy_new + Fcy_new);

    // write the updated primitive state to global memory
    dev_dustvelx[idx] = velx_new;
    dev_dustvely[idx] = vely_new;
    dev_dustvelz[idx] = velz_new;
}
