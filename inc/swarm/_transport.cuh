#ifndef SWARM_TRANSPORT_CUH
#define SWARM_TRANSPORT_CUH

#include <const_defs.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_grid.cuh>

// =========================================================================================================================
// particle state access
// =========================================================================================================================

// load one particle's spherical position and stored momentum-like velocity variables
__device__ __forceinline__
void _load_particle (const swarm *dev_particle, int idx, real &x, real &y, real &z, real &lx, real &vy, real &lz)
{
    x  = dev_particle[idx].position.x;
    y  = dev_particle[idx].position.y;
    z  = dev_particle[idx].position.z;
    lx = dev_particle[idx].velocity.x;
    vy = dev_particle[idx].velocity.y;
    lz = dev_particle[idx].velocity.z;
}

// store one particle's spherical position and momentum-like velocity variables
__device__ __forceinline__
void _save_particle (swarm *dev_particle, int idx, real x, real y, real z, real lx, real vy, real lz)
{
    dev_particle[idx].position.x =  x;
    dev_particle[idx].position.y =  y;
    dev_particle[idx].position.z =  z;
    dev_particle[idx].velocity.x = lx;
    dev_particle[idx].velocity.y = vy;
    dev_particle[idx].velocity.z = lz;
}

// test whether a particle remains inside the active radial and polar transport domain
__device__ __forceinline__
bool _is_particle_active (real y, real z)
{
    if (y < Y_MIN || y >= Y_MAX) return false;
    #ifdef HALFDISK
    if (N_Z > 1 && (z < Z_MIN || z > Z_MAX)) return false;
    #else
    if (N_Z > 1 && (z < Z_MIN || z >= Z_MAX)) return false;
    #endif // HALFDISK

    return true;
}

// wrap the active azimuthal coordinate and lock an inactive azimuthal dimension
__device__ __forceinline__
void _apply_periodic_x (real &x)
{
    if (N_X == 1)
    {
        x = 0.5*(X_MIN + X_MAX);
    }
    else
    {
        while (x >= X_MAX)
        {
            x -= X_MAX - X_MIN;
        }
        while (x < X_MIN)
        {
            x += X_MAX - X_MIN;
        }
    }
}

// park one absorbed representative outside the active radial domain
__device__ __forceinline__
void _absorb_particle (real &y, real &z, real &lx, real &vy, real &lz)
{
    y = 0.0;
    z = 0.5*M_PI;
    lx = 0.0;
    vy = 0.0;
    lz = 0.0;
}

// apply periodic azimuth, transport outflow, and the optional reflecting midplane
__device__ __forceinline__
void _apply_transport_boundary (real &x, real &y, real &z, real &lx, real &vy, real &lz)
{
    _apply_periodic_x(x);

    if (N_Z == 1)
    {
        z = 0.5*M_PI;
        lz = 0.0;
    }

    if (y < Y_MIN || y >= Y_MAX)
    {
        _absorb_particle(y, z, lx, vy, lz);
        return;
    }

    #ifdef HALFDISK
    if (N_Z > 1 && z > Z_MAX)
    {
        z = M_PI - z;
        lz = -lz;
    }
    #endif // HALFDISK

    #ifdef HALFDISK
    bool polar_exit = N_Z > 1 && (z < Z_MIN || z > Z_MAX);
    #else
    bool polar_exit = N_Z > 1 && (z < Z_MIN || z >= Z_MAX);
    #endif // HALFDISK

    if (polar_exit) _absorb_particle(y, z, lx, vy, lz);
}

// reflect stochastic boundary crossings to impose zero radial and polar diffusive flux
__device__ __forceinline__
void _apply_diffusion_boundary (real &x, real &y, real &z)
{
    _apply_periodic_x(x);

    while (y < Y_MIN || y > Y_MAX)
    {
        if (y < Y_MIN) y = 2.0*Y_MIN - y;
        if (y > Y_MAX) y = 2.0*Y_MAX - y;
    }
    if (y >= Y_MAX) y = Y_MAX - 1.0e-12*(Y_MAX - Y_MIN);

    if constexpr (N_Z == 1)
    {
        z = 0.5*M_PI;
    }
    else
    {
        while (z < Z_MIN || z > Z_MAX)
        {
            if (z < Z_MIN) z = 2.0*Z_MIN - z;
            if (z > Z_MAX) z = 2.0*Z_MAX - z;
        }
        if (z >= Z_MAX) z = Z_MAX - 1.0e-12*(Z_MAX - Z_MIN);
    }
}

// =========================================================================================================================
// staggered semi-analytic transport
// =========================================================================================================================

// drift the spherical position through the first half-step with the initial velocity
__device__ __forceinline__
void _ssa_substep_1 (real dt, real x_i, real y_i, real z_i, real lx_i, real vy_i, real lz_i, 
    real &x_1, real &y_1, real &z_1)
{
    // advance from the initial state i to the staggered midpoint position 1
    y_1 = y_i + 0.5*vy_i*dt;
    if constexpr (N_Z == 1)
    {
        z_1 = 0.5*M_PI;
        x_1 = (N_X > 1) ? x_i + 0.5*lx_i*dt / y_i / y_1 : 0.5*(X_MIN + X_MAX);
        return;
    }

    z_1 = z_i + 0.5*lz_i*dt / y_i / y_1;
    x_1 = x_i + 0.5*lx_i*dt / y_i / y_1 / sin(z_i) / sin(z_1);
}

// calculate radial gravity, radial centrifugal acceleration, and polar centrifugal torque
__device__ __forceinline__
void _get_force_term (real y, real z, real R, real lx, real lz, real beta, real &grav_y, real &cent_y, real &torq_z)
{
    grav_y = -(1.0 - beta)*_get_omegaK(y)*_get_omegaK(y)*y;
    cent_y = lx*lx / R / R / y + lz*lz / y / y / y;
    torq_z = (N_Z > 1) ? lx*lx / R / R / sin(z)*cos(z) : 0.0;
}

// integrate gas and optional P-R drag analytically at the midpoint and complete the velocity and position update
__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i, real x_1, real y_1, real z_1, 
    real &x_j, real &y_j, real &z_j, real &lx_j, real &vy_j, real &lz_j
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz, const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    real R_1 = _get_cyl_R(y_1, z_1);
    real Z_1 = _get_cyl_Z(y_1, z_1);
    
    real h_g = _get_hg(R_1);
    real omega = _get_omegaK(R_1);
    
    // evaluate the gas velocity used by the midpoint drag solve
    real lx_g1, vy_g1, lz_g1;
    
    #ifdef IMPORTGAS
    if ((dev_gas_velx != nullptr) && (dev_gas_vely != nullptr) && (dev_gas_velz != nullptr))
    {
        // convert imported linear gas velocities to the stored angular-momentum convention
        real loc_x = _get_loc_x(x_1);
        real loc_y = _get_loc_y(y_1);
        real loc_z = _get_loc_z(z_1);
        
        lx_g1 = _interp_field(dev_gas_velx, loc_x, loc_y, loc_z)*R_1;
        vy_g1 = _interp_field(dev_gas_vely, loc_x, loc_y, loc_z);
        lz_g1 = (N_Z > 1) ? _interp_field(dev_gas_velz, loc_x, loc_y, loc_z)*y_1 : 0.0;
    }
    else
    #endif // IMPORTGAS
    {
        real eta = _get_eta(R_1, Z_1, h_g);
        lx_g1 = sqrt(fmax(1.0 - 2.0*eta, 0.0))*omega*R_1*R_1;

        #ifdef VISC_FLOW
        real vR_g = _get_visc_vel(R_1, Z_1, h_g);
        vy_g1 = (N_Z > 1) ? vR_g*sin(z_1) : vR_g;
        lz_g1 = (N_Z > 1) ? y_1*vR_g*cos(z_1) : 0.0;
        #else  // PURE_ROTATION
        vy_g1 = 0.0;
        lz_g1 = 0.0;
        #endif // VISC_FLOW
    }

    // convert the local Stokes number to stopping time
    real ts_1 = _get_stokes(R_1, Z_1, h_g, size
        #ifdef IMPORTGAS
        , x_1, y_1, z_1, dev_gas_dens
        #endif // IMPORTGAS
    ) / omega;
    #ifndef PR_EFFECT
    real tau_1 = dt / ts_1;
    #endif // NO PR_EFFECT

    // evaluate midpoint forces with the initial angular momenta
    real grav_y1, cent_y1, torq_z1;
    _get_force_term(y_1, z_1, R_1, lx_i, lz_i, beta, grav_y1, cent_y1, torq_z1);

    // obtain a midpoint velocity with the exact frozen-coefficient drag response
    #ifdef PR_EFFECT
    // combine gas drag with gamma x/z and 2*gamma y P-R damping
    real inv_ts1 = 1.0 / ts_1;
    real pr_rate1 = beta*G*M_S / (C_LIGHT*y_1*y_1);

    real rate_x1 = inv_ts1 + pr_rate1;
    real rate_y1 = inv_ts1 + 2.0*pr_rate1;
    real rate_z1 = inv_ts1 + pr_rate1;

    real relax_x1 = -expm1(-0.5*rate_x1*dt);
    real relax_y1 = -expm1(-0.5*rate_y1*dt);
    real relax_z1 = -expm1(-0.5*rate_z1*dt);

    real resp_x1 = relax_x1 / rate_x1;
    real resp_y1 = relax_y1 / rate_y1;
    real resp_z1 = relax_z1 / rate_z1;

    real lx_1 = (1.0 - relax_x1)*lx_i + resp_x1*inv_ts1*lx_g1;
    real vy_1 = (1.0 - relax_y1)*vy_i + resp_y1*(inv_ts1*vy_g1 + grav_y1 + cent_y1);
    real lz_1 = (1.0 - relax_z1)*lz_i + resp_z1*(inv_ts1*lz_g1 + torq_z1);
    #else  // NO PR_EFFECT
    real lx_1 = lx_i + (lx_g1 - lx_i)*(1.0 - exp(-0.5*tau_1));
    real vy_1 = vy_i + ((grav_y1 + cent_y1)*ts_1 + vy_g1 - vy_i)*(1.0 - exp(-0.5*tau_1));
    real lz_1 = lz_i + (torq_z1*ts_1 + lz_g1 - lz_i)*(1.0 - exp(-0.5*tau_1));
    #endif // PR_EFFECT

    // reevaluate centrifugal terms with the midpoint angular momenta while reusing position-dependent beta
    real grav_y2, cent_y2, torq_z2;
    _get_force_term(y_1, z_1, R_1, lx_1, lz_1, beta, grav_y2, cent_y2, torq_z2);

    // complete the full-step frozen-coefficient drag response with midpoint forces
    #ifdef PR_EFFECT
    real relax_xj = -expm1(-rate_x1*dt);
    real relax_yj = -expm1(-rate_y1*dt);
    real relax_zj = -expm1(-rate_z1*dt);

    real resp_xj = relax_xj / rate_x1;
    real resp_yj = relax_yj / rate_y1;
    real resp_zj = relax_zj / rate_z1;

    lx_j = (1.0 - relax_xj)*lx_i + resp_xj*inv_ts1*lx_g1;
    vy_j = (1.0 - relax_yj)*vy_i + resp_yj*(inv_ts1*vy_g1 + grav_y2 + cent_y2);
    lz_j = (1.0 - relax_zj)*lz_i + resp_zj*(inv_ts1*lz_g1 + torq_z2);
    #else  // NO PR_EFFECT
    lx_j = lx_i + (lx_g1 - lx_i)*(1.0 - exp(-tau_1));
    vy_j = vy_i + ((grav_y2 + cent_y2)*ts_1 + vy_g1 - vy_i)*(1.0 - exp(-tau_1));
    lz_j = lz_i + (torq_z2*ts_1 + lz_g1 - lz_i)*(1.0 - exp(-tau_1));
    #endif // PR_EFFECT

    // drift from the midpoint position to the final state j
    y_j = y_1 + 0.5*vy_j*dt;
    if constexpr (N_Z == 1)
    {
        z_j = 0.5*M_PI;
        lz_j = 0.0;
        x_j = (N_X > 1) ? x_1 + 0.5*lx_j*dt / y_1 / y_j : 0.5*(X_MIN + X_MAX);
        return;
    }

    z_j = z_1 + 0.5*lz_j*dt / y_1 / y_j;
    x_j = x_1 + 0.5*lx_j*dt / y_1 / y_j / sin(z_1) / sin(z_j);
}

// =========================================================================================================================

#endif // SWARM_TRANSPORT_CUH
