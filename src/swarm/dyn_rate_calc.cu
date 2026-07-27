#ifdef TRANSPORT

#ifdef DIFFUSION
#include <_diffusion.cuh>
#endif // DIFFUSION
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: dyn_rate_calc
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
void dyn_rate_calc (real *dev_dt_rate, const swarm *dev_particle
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz
    , const real *dev_gas_velx_next, const real *dev_gas_vely_next, const real *dev_gas_velz_next
    #endif // IMPORTGAS
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x  = dev_particle[idx].position.x;
    real y  = dev_particle[idx].position.y;
    real z  = dev_particle[idx].position.z;

    if (!_is_particle_active(y, z))
    {
        dev_dt_rate[idx] = 0.0;
        return;
    }

    real lx = dev_particle[idx].velocity.x;
    real vy = dev_particle[idx].velocity.y;
    real lz = dev_particle[idx].velocity.z;

    // construct the local physical cell scales from the spherical mesh
    real R = y*sin(z);
    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();
    int iy = static_cast<int>(log(y / Y_MIN) / log(dy));
    iy = (iy >= N_Y) ? N_Y - 1 : iy;
    real dr = _get_yface(iy)*(dy - 1.0);

    // limit orbital phase evolution and particle crossing of every active mesh direction
    // the radial scale is the exact width of the logarithmic cell containing the particle
    real omega = _get_omegaK(R);
    real rate = omega / CFL_DYN;
    if (N_X > 1) rate = fmax(rate, omega / (dx*CFL_DYN));
    if (N_X > 1) rate = fmax(rate, abs(lx) / (R*R*dx*CFL_DYN));
    if (N_Y > 1) rate = fmax(rate, abs(vy) / (dr*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(lz) / (y*y*dz*CFL_DYN));

    #ifdef IMPORTGAS
    // use the faster gas motion from either bracketing snapshot
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    real vx_g = _interp_field(dev_gas_velx, loc_x, loc_y, loc_z);
    real vy_g = _interp_field(dev_gas_vely, loc_x, loc_y, loc_z);
    real vz_g = _interp_field(dev_gas_velz, loc_x, loc_y, loc_z);

    vx_g = fmax(abs(vx_g), abs(_interp_field(dev_gas_velx_next, loc_x, loc_y, loc_z)));
    vy_g = fmax(abs(vy_g), abs(_interp_field(dev_gas_vely_next, loc_x, loc_y, loc_z)));
    vz_g = fmax(abs(vz_g), abs(_interp_field(dev_gas_velz_next, loc_x, loc_y, loc_z)));
    
    if (N_X > 1) rate = fmax(rate, abs(vx_g) / (R*dx*CFL_DYN));
    if (N_Y > 1) rate = fmax(rate, abs(vy_g) / (dr*CFL_DYN));
    if (N_Z > 1) rate = fmax(rate, abs(vz_g) / (y*dz*CFL_DYN));
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
    real grav_y = -(1.0 - beta)*G*M_S / (y*y);
    real cent_y = lx*lx / (R*R*y) + lz*lz / (y*y*y);
    real accel_y = abs(grav_y + cent_y);
    
    rate = fmax(rate, sqrt(accel_y / (2.0*CFL_DYN*dr)));
    if (N_Z > 1)
    {
        real torq_z = lx*lx / (R*R)*cos(z) / sin(z);
        real accel_z = abs(torq_z) / y;
        rate = fmax(rate, sqrt(accel_z / (2.0*CFL_DYN*y*dz)));
    }

    #ifdef VISC_ACCRETION
    {
        // include the analytic gas target velocity before drag can transfer it to the dust
        real h_g = _get_hg(R);
        real vR_g = _get_visc_vel(R, y*cos(z), h_g);

        if (N_Y > 1) rate = fmax(rate, abs(vR_g*sin(z)) / (dr*CFL_DYN));
        if (N_Z > 1) rate = fmax(rate, abs(vR_g*cos(z)) / (y*dz*CFL_DYN));
    }
    #endif // VISC_ACCRETION

    #ifdef DIFFUSION
    real h_g = _get_hg(R);
    real nu = _get_nu(R, h_g);

    // assemble cylindrical diffusion coefficients and deterministic physical drift speeds
    real diff_x = nu / SCHMIDT_X;
    real diff_R = nu / SCHMIDT_R;
    real diff_Z = (N_Z > 1) ? nu / SCHMIDT_Z : 0.0;

    // retain zero placeholders for future azimuthal and vertical diffusivity gradients
    real grad_x = 0.0; // partial derivative of diff_x with respect to x
    real grad_Z = 0.0; // partial derivative of diff_Z with respect to Z
    real drift_x = grad_x / (R*R);
    real drift_R = _get_diff_drift_R(R, diff_R);
    real drift_Z = grad_Z;

    // project cylindrical diffusion variance and drift onto spherical radial and polar directions
    real sin_z = sin(z);
    real cos_z = cos(z);
    real diff_y = diff_R*sin_z*sin_z + diff_Z*cos_z*cos_z;
    real drift_y = drift_R*sin_z + drift_Z*cos_z;
    
    rate = fmax(rate, 2.0*diff_y / (CFL_DYN*CFL_DYN*dr*dr));
    rate = fmax(rate, abs(drift_y) / (CFL_DYN*dr));

    if (N_X > 1)
    {
        real cell_x = R*dx;
        
        rate = fmax(rate, 2.0*diff_x / (CFL_DYN*CFL_DYN*cell_x*cell_x));
        rate = fmax(rate, abs(drift_x) / (CFL_DYN*dx));
    }

    if (N_Z > 1)
    {
        real cell_z = y*dz;
        real diff_z = diff_R*cos_z*cos_z + diff_Z*sin_z*sin_z;
        real drift_z = drift_R*cos_z - drift_Z*sin_z;
        
        rate = fmax(rate, 2.0*diff_z / (CFL_DYN*CFL_DYN*cell_z*cell_z));
        rate = fmax(rate, abs(drift_z) / (CFL_DYN*cell_z));
    }
    #endif // DIFFUSION

    dev_dt_rate[idx] = rate;
}

#endif // TRANSPORT
