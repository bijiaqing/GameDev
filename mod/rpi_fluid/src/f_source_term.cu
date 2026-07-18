#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_source_term
// Purpose: second-order exponential drag/force update at each cell centre.
// Linear drag is integrated exactly.  The remaining force is represented linearly between
// its old and new endpoint values, using exponential quadrature weights.  The angular-momentum
// form is triangular: lx is solved first, then lz, then vy, so the new endpoint forces are
// available without iteration.  Unlike Crank-Nicolson, the drag transient decays to zero when
// dt/ts -> infinity instead of alternating with amplification approaching -1.
//
// Velocity variables (spherical angular-momentum formulation):
//   dev_dustvelx  =  ℓ_φ = v_φ · r·sinθ   (specific azimuthal angular momentum)
//   dev_dustvely  =  v_r  (radial velocity)
//   dev_dustvelz  =  ℓ_θ = v_θ · r         (specific polar angular momentum)
// =========================================================================================================================

__global__
void f_source_term (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz, 
    const real *dev_dustdens,
    #ifdef RADIATION
    const real *dev_optdepth,
    real beta_taper,
    #endif
    real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;

    // ---- Cell centre coordinates ----
    int iy = (idx / N_X) % N_Y;
    int iz = idx / (N_X * N_Y);

    real yc =              Y_MIN*pow(_get_dy(), iy + 0.5)  ;
    real zc = (N_Z > 1) ? (Z_MIN +   _get_dz()*(iz + 0.5)) : 0.5*(Z_MIN + Z_MAX);

    real R = yc*sin(zc);
    real Z = yc*cos(zc);

    // Numerical vacuum carries no dynamically meaningful momentum.  Keep its transport state
    // at the same fallback used by primitive recovery and exclude it from the source solve.
    if (dev_dustdens[idx] < RHO_VAC)
    {
        dev_dustvelx[idx] = sqrt(G*M_S*fmax(R, 0.0));
        dev_dustvely[idx] = 0.0;
        dev_dustvelz[idx] = 0.0;
        return;
    }

    // ---- Local gas parameters ----
    real omega = _get_omegaK(R);
    real h_g = _get_hg(R);

    real St = _get_St(R, Z, h_g);
    real ts = St / omega;
    real drag_h = dt / ts;

    // Exact homogeneous drag factors.  expm1 preserves 1-exp(-h) when h is small and safely
    // tends to one when h is large.
    real drag_relax = -expm1(-drag_h);
    real drag_decay = 1.0 - drag_relax;

    // Exponential endpoint-force quadrature:
    //   u_new = E*u + (1-E)*u_g + force_weight_n*F_n + force_weight_new*F_new.
    // Direct evaluation of the weights loses digits through cancellation for h << 1, where
    // both weights approach dt/2, so use their Taylor series in that regime.
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

    #if defined(RADIATION)
    real tau_i = (iy > 0) ? dev_optdepth[idx - N_X] : 0.0;
    real tau_o = dev_optdepth[idx];
    real beta  = beta_taper*BETA_0*exp(-0.5*(tau_i + tau_o));
    #else
    real beta  = 0.0;
    #endif

    // ---- Current dust velocity ----
    real lx = dev_dustvelx[idx];
    real vy = dev_dustvely[idx];
    real lz = dev_dustvelz[idx];

    // ---- Gas velocity at cell centre ----
    real lxg, vyg, lzg;
    real eta = _get_eta(R, Z, h_g);
    lxg = R*R*omega*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    vyg = 0.0;
    lzg = 0.0;

    // Old endpoint forces
    real Fy_n, Fcy_n, Tcz_n;
    _get_force_term(yc, zc, R, lx, lz, beta, Fy_n, Fcy_n, Tcz_n);

    // lx is pure linear drag and is therefore exact for every dt/ts.
    real lx_new = drag_decay*lx + drag_relax*lxg;

    // Tcz depends on lx but not on lz; evaluate the new endpoint before solving lz.
    real Fy_tmp, Fcy_tmp, Tcz_new;
    _get_force_term(yc, zc, R, lx_new, lz, beta, Fy_tmp, Fcy_tmp, Tcz_new);
    real lz_new = drag_decay*lz + drag_relax*lzg
                + force_weight_n*Tcz_n + force_weight_new*Tcz_new;

    // With lx_new and lz_new known, the radial centrifugal endpoint is explicit.
    real Fy_new, Fcy_new;
    _get_force_term(yc, zc, R, lx_new, lz_new, beta, Fy_new, Fcy_new, Tcz_new);
    real vy_new = drag_decay*vy + drag_relax*vyg
                + force_weight_n*(Fy_n + Fcy_n) + force_weight_new*(Fy_new + Fcy_new);

    dev_dustvelx[idx] = lx_new;
    dev_dustvely[idx] = vy_new;
    dev_dustvelz[idx] = lz_new;
}

// =========================================================================================================================
