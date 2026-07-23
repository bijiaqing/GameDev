#ifdef TRANSPORT

#ifdef DIFFUSION
#include <_diffusion.cuh>
#endif // DIFFUSION
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dt_rates_calc
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
void dt_rates_calc (real *dev_dt_rates, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gas_dens, const real *dev_gas_velx
    , const real *dev_gas_vely, const real *dev_gas_velz
    , const real *dev_gas_dens_next, const real *dev_gas_velx_next
    , const real *dev_gas_vely_next, const real *dev_gas_velz_next
    #endif // IMPORTGAS
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
    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();

    // limit orbital phase evolution and particle crossing of every active mesh direction
    real omega = _get_omegaK(R);
    real rate = omega / CFL_DYN;
    if (N_X > 1) rate = fmax(rate, omega / (dx*CFL_DYN));
    if (N_X > 1) rate = fmax(rate, abs(lx) / (R*R*dx*CFL_DYN));
    if (N_Y > 1) rate = fmax(rate, abs(vy) / (y*log(dy)*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(lz) / (y*y*dz*CFL_DYN));

    #ifdef IMPORTGAS
    // use the faster gas motion from either bracketing snapshot
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    real gas_x = _interp_field(dev_gas_velx, loc_x, loc_y, loc_z);
    real gas_y = _interp_field(dev_gas_vely, loc_x, loc_y, loc_z);
    real gas_z = _interp_field(dev_gas_velz, loc_x, loc_y, loc_z);

    gas_x = fmax(abs(gas_x), abs(_interp_field(dev_gas_velx_next, loc_x, loc_y, loc_z)));
    gas_y = fmax(abs(gas_y), abs(_interp_field(dev_gas_vely_next, loc_x, loc_y, loc_z)));
    gas_z = fmax(abs(gas_z), abs(_interp_field(dev_gas_velz_next, loc_x, loc_y, loc_z)));
    
    if (N_X > 1) rate = fmax(rate, abs(gas_x) / (R*dx*CFL_DYN));
    if (N_Y > 1) rate = fmax(rate, abs(gas_y) / (y*log(dy)*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(gas_z) / (y*dz*CFL_DYN));
    #endif // IMPORTGAS

    real beta = 0.0;
    #ifdef RADIATION
    #ifdef MULTISIZE
    real size = dev_particle[idx].par_size;
    #else  // MONOSIZE
    real size = S_0;
    #endif // MULTISIZE
    #if defined(COLLISION) && defined(MULTISIZE)
    // include the strongest permitted radiation force after fragmentation
    size = fmin(size, INIT_SMIN);
    #endif // COLLISION && MULTISIZE
    beta = BETA_0 / (size / S_0);
    #endif // RADIATION

    // require constant-acceleration displacement to remain below a local mesh fraction
    real force_y = -(1.0 - beta)*G*M_S / (y*y);
    real cent_y = lx*lx / (R*R*y) + lz*lz / (y*y*y);
    real accel_y = abs(force_y + cent_y);
    
    rate = fmax(rate, sqrt(accel_y / (2.0*CFL_DYN*y*log(dy))));
    if (N_Z > 1)
    {
        real accel_z = abs(lx*lx / (R*R)*cos(z) / sin(z)) / y;
        rate = fmax(rate, sqrt(accel_z / (2.0*CFL_DYN*y*dz)));
    }

    #ifdef DIFFUSION
    // evaluate the logarithmic gas-density gradients entering diffusion drift
    real h_g = _get_hg(R);
    real nu = _get_nu(R, h_g);
    real term_x, term_R, term_Z;
    _get_term_grad_cyl(x, y, z, term_x, term_R, term_Z
        #ifdef IMPORTGAS
        , dev_gas_dens
        #endif // IMPORTGAS
    );

    real next_x = term_x;
    real next_R = term_R;
    real next_Z = term_Z;
    #ifdef IMPORTGAS
    // evaluate both imported endpoints before bounding each resulting drift magnitude
    _get_term_grad_cyl(x, y, z, next_x, next_R, next_Z, dev_gas_dens_next);
    #endif // IMPORTGAS

    #ifdef CONST_NU
    real idx_nu = 0.0;
    #else  // CONST_ALPHA
    real idx_nu = IDX_Q + 1.5;
    #endif // CONST_NU

    // assemble cylindrical diffusion coefficients and deterministic physical drift speeds
    real coeff_x = nu / SCHMIDT_X;
    real coeff_R = nu / SCHMIDT_R;
    real coeff_Z = (N_Z > 1) ? nu / SCHMIDT_Z : 0.0;

    real drift_R = coeff_R*(term_R + (idx_nu + 1.0) / R);
    real drift_Z = coeff_Z*term_Z;
    real next_drift_R = coeff_R*(next_R + (idx_nu + 1.0) / R);
    real next_drift_Z = coeff_Z*next_Z;

    // project cylindrical diffusion variance and drift onto spherical radial and polar directions
    real sin_z = sin(z);
    real cos_z = cos(z);
    real cell_y = y*log(dy);
    real coeff_y = coeff_R*sin_z*sin_z + coeff_Z*cos_z*cos_z;
    real drift_y = drift_R*sin_z + drift_Z*cos_z;
    real next_drift_y = next_drift_R*sin_z + next_drift_Z*cos_z;
    
    rate = fmax(rate, 2.0*coeff_y / (CFL_DYN*CFL_DYN*cell_y*cell_y));
    rate = fmax(rate, abs(drift_y) / (CFL_DYN*cell_y));
    rate = fmax(rate, abs(next_drift_y) / (CFL_DYN*cell_y));

    if (N_X > 1)
    {
        real cell_x = R*dx;
        real drift_x = coeff_x*term_x/R;
        real next_drift_x = coeff_x*next_x/R;
        
        rate = fmax(rate, 2.0*coeff_x / (CFL_DYN*CFL_DYN*cell_x*cell_x));
        rate = fmax(rate, abs(drift_x) / (CFL_DYN*cell_x));
        rate = fmax(rate, abs(next_drift_x) / (CFL_DYN*cell_x));
    }

    if (N_Z > 1)
    {
        real cell_z = y*dz;
        real coeff_z = coeff_R*cos_z*cos_z + coeff_Z*sin_z*sin_z;
        real drift_z = drift_R*cos_z - drift_Z*sin_z;
        real next_drift_z = next_drift_R*cos_z - next_drift_Z*sin_z;
        
        rate = fmax(rate, 2.0*coeff_z / (CFL_DYN*CFL_DYN*cell_z*cell_z));
        rate = fmax(rate, abs(drift_z) / (CFL_DYN*cell_z));
        rate = fmax(rate, abs(next_drift_z) / (CFL_DYN*cell_z));
    }
    #endif // DIFFUSION

    dev_dt_rates[idx] = rate;
}

#endif // TRANSPORT
