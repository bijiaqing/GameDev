#ifdef TRANSPORT

#include <graffiti_kern.cuh>
#include <helpers_paramphys.cuh>
#ifdef DIFFUSION
#include <helpers_diffusion.cuh>
#endif // TRANSPORT

// =========================================================================================================================
// kernel: dt_rate_calc
// calculate one conservative inverse dynamics timestep from all active local motion and diffusion scales
//
// parallelization: one thread per representative particle followed by a host-side maximum reduction
//
// constraints:
//   1 orbital and particle mesh-crossing rates
//   2 imported-gas mesh-crossing rates at both temporal endpoints
//   3 acceleration displacement from gravity, radiation, and centrifugal forces
//   4 stochastic and deterministic diffusion displacement in every active direction
// =========================================================================================================================

__global__
void dt_rate_calc (real *dev_dt_rate, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gasdens, const real *dev_gasvelx,
      const real *dev_gasvely, const real *dev_gasvelz,
      const real *dev_gasdens_next, const real *dev_gasvelx_next,
      const real *dev_gasvely_next, const real *dev_gasvelz_next
    #endif
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x  = dev_particle[idx].position.x;
    real y  = dev_particle[idx].position.y;
    real z  = dev_particle[idx].position.z;
    real lx = dev_particle[idx].velocity.x;
    real vy = dev_particle[idx].velocity.y;
    real lz = dev_particle[idx].velocity.z;

    // construct the local physical cell scales from the spherical mesh
    real R = y*sin(z);
    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dl = log(Y_MAX / Y_MIN) / static_cast<real>(N_Y);
    real dz = (N_Z > 1) ? (Z_MAX - Z_MIN) / static_cast<real>(N_Z) : 1.0;

    // limit orbital phase evolution and particle crossing of every active mesh direction
    real omega = _get_omegaK(R);
    real rate = omega / CFL_DYN;
    if (N_X > 1) rate = fmax(rate, omega/(dx*CFL_DYN));
    if (N_X > 1) rate = fmax(rate, abs(lx)/(R*R*dx*CFL_DYN));
    rate = fmax(rate, abs(vy)/(y*dl*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(lz)/(y*y*dz*CFL_DYN));

    #ifdef IMPORTGAS
    // use the faster gas motion from either bracketing snapshot
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);
    real gas_x = _interp_field(dev_gasvelx, loc_x, loc_y, loc_z);
    real gas_y = _interp_field(dev_gasvely, loc_x, loc_y, loc_z);
    real gas_z = _interp_field(dev_gasvelz, loc_x, loc_y, loc_z);
    gas_x = fmax(abs(gas_x), abs(_interp_field(dev_gasvelx_next, loc_x, loc_y, loc_z)));
    gas_y = fmax(abs(gas_y), abs(_interp_field(dev_gasvely_next, loc_x, loc_y, loc_z)));
    gas_z = fmax(abs(gas_z), abs(_interp_field(dev_gasvelz_next, loc_x, loc_y, loc_z)));
    if (N_X > 1) rate = fmax(rate, abs(gas_x)/(R*dx*CFL_DYN));
    rate = fmax(rate, abs(gas_y)/(y*dl*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(gas_z)/(y*dz*CFL_DYN));
    #endif

    real beta = 0.0;
    #ifdef RADIATION
    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    #else
    real size = S_0;
    #endif
    #if defined(COLLISION) && defined(MULTISIZE)
    // include the strongest permitted radiation force after fragmentation
    size = fmin(size, INIT_SMIN);
    #endif
    beta = BETA_0/(size / S_0);
    #endif

    // require constant-acceleration displacement to remain below a local mesh fraction
    real force_y = -(1.0 - beta)*G*M_S/(y*y);
    real cent_y = lx*lx/(R*R*y) + lz*lz/(y*y*y);
    real accel_y = abs(force_y + cent_y);
    rate = fmax(rate, sqrt(accel_y/(2.0*CFL_DYN*y*dl)));
    if (N_Z > 1)
    {
        real accel_z = abs(lx*lx/(R*R)*cos(z)/sin(z))/y;
        rate = fmax(rate, sqrt(accel_z/(2.0*CFL_DYN*y*dz)));
    }

    #ifdef DIFFUSION
    // evaluate the logarithmic gas-density gradients entering diffusion drift
    real h_g = _get_hg(R);
    real nu = _get_nu(R, h_g);
    real term_x, term_R, term_Z;
    _get_term_grad_cyl(x, y, z, term_x, term_R, term_Z
        #ifdef IMPORTGAS
        , dev_gasdens
        #endif
    );
    #ifdef IMPORTGAS
    // retain the larger drift magnitude from the two imported temporal endpoints
    real next_x, next_R, next_Z;
    _get_term_grad_cyl(x, y, z, next_x, next_R, next_Z, dev_gasdens_next);
    if (abs(next_x) > abs(term_x)) term_x = next_x;
    if (abs(next_R) > abs(term_R)) term_R = next_R;
    if (abs(next_Z) > abs(term_Z)) term_Z = next_Z;
    #endif
    // assemble the cylindrical diffusion coefficients and deterministic drifts
    real coeff_R = nu / SC_R;
    real coeff_Z = (N_Z > 1) ? nu / SC_Z : 0.0;
    real drift_R = coeff_R*(term_R + 1.0/R);
    #ifndef CONST_NU
    drift_R += coeff_R*(IDX_Q + 1.5)/R;
    #endif
    real drift_Z = coeff_Z*term_Z;

    // project cylindrical diffusion variance and drift onto spherical radial and polar directions
    real sin_z = sin(z);
    real cos_z = cos(z);
    real cell_y = y*dl;
    real coeff_y = coeff_R*sin_z*sin_z + coeff_Z*cos_z*cos_z;
    real drift_y = drift_R*sin_z + drift_Z*cos_z;
    rate = fmax(rate, 2.0*coeff_y/(CFL_DYN*CFL_DYN*cell_y*cell_y));
    rate = fmax(rate, abs(drift_y)/(CFL_DYN*cell_y));
    if (N_X > 1)
    {
        real cell_x = R*dx;
        rate = fmax(rate, 2.0*(nu / SC_X)/(CFL_DYN*CFL_DYN*cell_x*cell_x));
        real drift_x = (nu / SC_X)*term_x/R;
        rate = fmax(rate, abs(drift_x)/(CFL_DYN*cell_x));
    }
    if (N_Z > 1)
    {
        real cell_z = y*dz;
        real coeff_z = coeff_R*cos_z*cos_z + coeff_Z*sin_z*sin_z;
        real drift_z = drift_R*cos_z - drift_Z*sin_z;
        rate = fmax(rate, 2.0*coeff_z/(CFL_DYN*CFL_DYN*cell_z*cell_z));
        rate = fmax(rate, abs(drift_z)/(CFL_DYN*cell_z));
    }
    #endif

    dev_dt_rate[idx] = rate;
}

#endif
