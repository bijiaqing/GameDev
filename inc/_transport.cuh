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

// recycle one imported-gas particle to the outer midplane boundary with Keplerian rotation
__device__ __forceinline__
void _eject_to_outer (real &y, real &z, real &lx, real &vy, real &lz)
{
    y  = Y_MAX - 1e-6*Y_MAX;
    z  = 0.5*M_PI;
    lx = _get_omegaK(y)*y*y;
    vy = 0.0;
    lz = 0.0;
}

// apply periodic azimuth, imported-gas recycling, and optional midplane reflection
__device__ __forceinline__
void _if_out_of_box (real &x, real &y, real &z, real &lx, real &vy, real &lz)
{
    if (N_X == 1) 
    {
        x = 0.5*(X_MIN + X_MAX);
    }
    else // wrap azimuth into [X_MIN, X_MAX)
    {
        while (x >= X_MAX) x -= X_MAX - X_MIN;
        while (x <  X_MIN) x += X_MAX - X_MIN;
    }

    if (N_Z == 1)
    {
        z = 0.5*M_PI;
        lz = 0.0;
    }

    #ifdef IMPORTGAS
    if (y < Y_MIN || y >= Y_MAX) // recycle radial exits to the outer injection boundary
    {
        _eject_to_outer(y, z, lx, vy, lz);
    }
    if (N_Z > 1 && z < Z_MIN) // recycle polar exits when the imported mesh covers the full disk
    {
        _eject_to_outer(y, z, lx, vy, lz);
    }
    #ifndef HALFDISK
    else if (N_Z > 1 && z >= Z_MAX) // recycle polar exits when the imported mesh covers the full disk
    {
        _eject_to_outer(y, z, lx, vy, lz);
    }
    #endif // NOT HALFDISK
    #endif // IMPORTGAS

    #ifdef HALFDISK
    if (N_Z > 1 && z > Z_MAX) // mirror across midplane (theta = pi/2)
    {
        z  = M_PI - z;
        lz = -lz;      // reverse polar angular momentum
    }
    #endif // HALFDISK
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
    z_1 = z_i + 0.5*lz_i*dt / y_i / y_1;
    x_1 = x_i + 0.5*lx_i*dt / y_i / y_1 / sin(z_i) / sin(z_1);
}

// calculate radial gravity, radial centrifugal acceleration, and polar centrifugal torque
__device__ __forceinline__
void _get_force_term (real y, real z, real R, real l_x, real l_z, real beta, real &F_y, real &Fc_y, real &Tc_z)
{
    F_y  = -(1.0 - beta)*_get_omegaK(y)*_get_omegaK(y)*y;
    Fc_y = l_x*l_x / R / R / y + l_z*l_z / y / y / y;
    Tc_z = (N_Z > 1) ? l_x*l_x / R / R / sin(z) * cos(z) : 0.0;
}

// integrate drag analytically at the midpoint and complete the velocity and position update
__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i, real x_1, real y_1, real z_1, 
    real &x_j, real &y_j, real &z_j, real &lx_j, real &vy_j, real &lz_j
    #ifdef IMPORTGAS
    , const real *dev_gas_velx, const real *dev_gas_vely, const real *dev_gas_velz, const real *dev_gas_dens
    #endif
)
{
    real R_1 = y_1*sin(z_1);
    real Z_1 = y_1*cos(z_1);
    
    real h_g = _get_hg(R_1);
    real omega = _get_omegaK(R_1);
    
    // evaluate the gas velocity used by the midpoint drag solve
    real lxg_1, vyg_1, lzg_1;
    
    #ifdef IMPORTGAS
    if ((dev_gas_velx != nullptr) && (dev_gas_vely != nullptr) && (dev_gas_velz != nullptr))
    {
        // convert imported linear gas velocities to the stored angular-momentum convention
        real loc_x = _get_loc_x(x_1);
        real loc_y = _get_loc_y(y_1);
        real loc_z = _get_loc_z(z_1);
        
        lxg_1 = _interp_field(dev_gas_velx, loc_x, loc_y, loc_z)*y_1*sin(z_1);
        vyg_1 = _interp_field(dev_gas_vely, loc_x, loc_y, loc_z);
        lzg_1 = (N_Z > 1) ? _interp_field(dev_gas_velz, loc_x, loc_y, loc_z)*y_1 : 0.0;
    }
    else
    #endif
    {
        real eta = _get_eta(R_1, Z_1, h_g);
        
        lxg_1 = sqrt(1.0 - 2.0*eta)*omega*R_1*R_1;
        vyg_1 = 0.0;
        lzg_1 = 0.0;
    }

    // convert the local Stokes number to stopping time
    real ts_1 = _get_St(R_1, Z_1, size, h_g
        #ifdef IMPORTGAS
        , x_1, y_1, z_1, dev_gas_dens
        #endif
    ) / omega;
    real tau_1 = dt / ts_1;

    // evaluate midpoint forces with the initial angular momenta
    real Fy_1, Fcy_1, Tcz_1;
    _get_force_term(y_1, z_1, R_1, lx_i, lz_i, beta, Fy_1, Fcy_1, Tcz_1);

    // obtain a midpoint velocity with the exact frozen-coefficient drag response
    real lx_1 = lx_i + (lxg_1 - lx_i)*(1.0 - exp(-0.5*tau_1));
    real vy_1 = vy_i + ((Fy_1 + Fcy_1)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-0.5*tau_1));
    real lz_1 = lz_i + (Tcz_1*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-0.5*tau_1));

    // reevaluate centrifugal terms with the midpoint angular momenta while reusing position-dependent beta
    real Fy_2, Fcy_2, Tcz_2;
    _get_force_term(y_1, z_1, R_1, lx_1, lz_1, beta, Fy_2, Fcy_2, Tcz_2);

    // complete the full-step frozen-coefficient drag response with midpoint forces
    lx_j = lx_i + (lxg_1 - lx_i)*(1.0 - exp(-tau_1));
    vy_j = vy_i + ((Fy_2 + Fcy_2)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-tau_1));
    lz_j = lz_i + (Tcz_2*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-tau_1));

    // drift from the midpoint position to the final state j
    y_j = y_1 + 0.5*vy_j*dt;
    z_j = z_1 + 0.5*lz_j*dt / y_1 / y_j;
    x_j = x_1 + 0.5*lx_j*dt / y_1 / y_j / sin(z_1) / sin(z_j);
}

// =========================================================================================================================

#endif // SWARM_TRANSPORT_CUH
