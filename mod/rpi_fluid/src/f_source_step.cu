#include <graffiti_kern.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// Kernel: f_source_step
// Purpose: second-order exponential drag/force update at each cell centre.
// Linear drag is integrated exactly.  The remaining force is represented linearly between
// its old and new endpoint values, using exponential quadrature weights.  The angular-momentum
// form is triangular: velx is solved first, then velz, then vely, so the new endpoint forces are
// available without iteration.  Unlike Crank-Nicolson, the drag transient decays to zero when
// dt/ts -> infinity instead of alternating with amplification approaching -1.
//
// Velocity variables (spherical angular-momentum formulation):
//   dev_dustvelx  =  ℓ_x = v_x · R   (specific azimuthal angular momentum)
//   dev_dustvely  =  v_y             (spherical-radial velocity)
//   dev_dustvelz  =  ℓ_z = v_z · y   (specific polar angular momentum)
// =========================================================================================================================

static __device__ __forceinline__
void _get_force_term (real yc, real zc, real Rc, real velx, real velz, real beta, real &Fy, real &Fcy, real &Tcz)
{
    Fy  = -(1.0 - beta)*_get_omegaK(yc)*_get_omegaK(yc)*yc;
    Fcy = velx*velx / Rc / Rc / yc + velz*velz / yc / yc / yc;
    Tcz = velx*velx / Rc / Rc / sin(zc)*cos(zc);
}

__global__
void f_source_step (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
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

    // numerical vacuum carries no dynamically meaningful momentum
    // keep its transport state at the same fallback used by primitive recovery and exclude it from the source solve
    if (dev_dustdens[idx] < RHO_VAC)
    {
        dev_dustvelx[idx] = sqrt(G*M_S*fmax(Rc, 0.0));
        dev_dustvely[idx] = 0.0;
        dev_dustvelz[idx] = 0.0;
        
        return;
    }

    real omega = _get_omegaK(Rc);
    real h_g = _get_hg(Rc);

    real St = _get_St(Rc, Zc, h_g);
    real ts = St / omega;
    real drag_h = dt / ts;

    // expm1 preserves 1-exp(-h) when h is small and safely tends to one when h is large
    real drag_relax = -expm1(-drag_h);
    real drag_decay = 1.0 - drag_relax;

    // exponential endpoint-force quadrature:
    // u_new = E*u + (1-E)*u_g + force_weight_n*F_n + force_weight_new*F_new
    // direct evaluation of the weights loses digits through cancellation for h << 1, 
    // where both weights approach dt/2, so use their Taylor series in that regime
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

    #ifdef RADIATION
    real tau_i = (iy > 0) ? dev_optdepth[idx - N_X] : 0.0;
    real tau_o = dev_optdepth[idx];
    real beta  = beta_taper*BETA_0*exp(-0.5*(tau_i + tau_o));
    #else
    real beta  = 0.0;
    #endif

    real velx = dev_dustvelx[idx];
    real vely = dev_dustvely[idx];
    real velz = dev_dustvelz[idx];

    real eta = _get_eta(Rc, Zc, h_g);
    real velx_g = Rc*Rc*omega*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real vely_g = 0.0;
    real velz_g = 0.0;

    // velx is pure linear drag and is therefore exact for every dt/ts.
    real Fy_n, Fcy_n, Tcz_n, velx_new;
    _get_force_term(yc, zc, Rc, velx, velz, beta, Fy_n, Fcy_n, Tcz_n);

    velx_new  = drag_decay*velx;
    velx_new += drag_relax*velx_g;

    // Tcz depends on velx but not on velz; evaluate the new endpoint before solving velz
    real Fy_tmp, Fcy_tmp, Tcz_new, velz_new;
    _get_force_term(yc, zc, Rc, velx_new, velz, beta, Fy_tmp, Fcy_tmp, Tcz_new);
    
    velz_new  = drag_decay*velz;
    velz_new += drag_relax*velz_g;
    velz_new += force_weight_n*Tcz_n;
    velz_new += force_weight_new*Tcz_new;

    // with velx_new and velz_new known, the radial centrifugal endpoint is explicit
    real Fy_new, Fcy_new, vely_new;
    _get_force_term(yc, zc, Rc, velx_new, velz_new, beta, Fy_new, Fcy_new, Tcz_new);

    vely_new  = drag_decay*vely;
    vely_new += drag_relax*vely_g;
    vely_new += force_weight_n*(Fy_n + Fcy_n);
    vely_new += force_weight_new*(Fy_new + Fcy_new);

    dev_dustvelx[idx] = velx_new;
    dev_dustvely[idx] = vely_new;
    dev_dustvelz[idx] = velz_new;
}

// =========================================================================================================================
