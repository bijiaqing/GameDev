#ifndef DIFFUSION_CUH
#define DIFFUSION_CUH

#ifdef DIFFUSION

#ifdef IMPORTGAS
#include <cassert>
#endif // IMPORTGAS

#include <const.cuh>
#include <paramgrid.cuh>
#include <interpval.cuh>
#include <paramphys.cuh>

// =========================================================================================================================
// cylindrical gas-density gradients
// =========================================================================================================================

// calculate the logarithmic gas-density derivatives needed by the cylindrical diffusion SDE
__device__ __forceinline__
void _get_term_grad_cyl (real x, real y, real z, real &term_x, real &term_R, real &term_Z
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
)
{   
    #ifdef IMPORTGAS

    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    real rho = _interp_field(dev_gas_dens, loc_x, loc_y, loc_z);

    if (rho <= 0.0)
    {
        printf("ERROR: Invalid gas density rhog = %e at (x,y,z) = (%e,%e,%e)\n", rho, x, y, z);
        assert(false);
    }

    real drho_dx, drho_dy, drho_dz;

    if (N_X > 1 && rho > 0.0)
    {
        real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
        
        real x_p = x + dx;
        real x_m = x - dx;

        if (x_p >= X_MAX) x_p -= (X_MAX - X_MIN);
        if (x_m <  X_MIN) x_m += (X_MAX - X_MIN);
        
        real rho_xp = _interp_field(dev_gas_dens, _get_loc_x(x_p), loc_y, loc_z);
        real rho_xm = _interp_field(dev_gas_dens, _get_loc_x(x_m), loc_y, loc_z);
        
        drho_dx = (rho_xp - rho_xm) / (2.0*dx);
    }
    else // if azimuthal direction disabled
    {
        drho_dx = 0.0;
    }
    
    if (N_Y > 1)
    {
        real dy = y*(log(Y_MAX / Y_MIN) / static_cast<real>(N_Y));
        
        if (loc_y < 0.5) // use a one-sided difference next to the inner radial boundary
        {
            real rho_yp = _interp_field(dev_gas_dens, loc_x, _get_loc_y(y + dy), loc_z);
            
            drho_dy = (rho_yp - rho) / dy;
        }
        else if (loc_y > static_cast<real>(N_Y) - 0.5) // use a one-sided difference next to the outer radial boundary
        {
            real rho_ym = _interp_field(dev_gas_dens, loc_x, _get_loc_y(y - dy), loc_z);
            
            drho_dy = (rho - rho_ym) / dy;
        }
        else // use a centred difference in the radial interior
        {
            real rho_yp = _interp_field(dev_gas_dens, loc_x, _get_loc_y(y + dy), loc_z);
            real rho_ym = _interp_field(dev_gas_dens, loc_x, _get_loc_y(y - dy), loc_z);
            
            drho_dy = (rho_yp - rho_ym) / (2.0*dy);
        }
    }
    else
    {
        drho_dy = 0.0;
    }

    if (N_Z > 1)
    {
        real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
        
        if (loc_z < 0.5) // use a one-sided difference next to the lower polar boundary
        {
            real rho_zp = _interp_field(dev_gas_dens, loc_x, loc_y, _get_loc_z(z + dz));
            
            drho_dz = (rho_zp - rho) / dz;
        }
        else if (loc_z > static_cast<real>(N_Z) - 0.5) // use a one-sided difference next to the upper polar boundary
        {
            real rho_zm = _interp_field(dev_gas_dens, loc_x, loc_y, _get_loc_z(z - dz));
            
            drho_dz = (rho - rho_zm) / dz;
        }
        else // use a centred difference in the polar interior
        {
            real rho_zp = _interp_field(dev_gas_dens, loc_x, loc_y, _get_loc_z(z + dz));
            real rho_zm = _interp_field(dev_gas_dens, loc_x, loc_y, _get_loc_z(z - dz));
            
            drho_dz = (rho_zp - rho_zm) / (2.0*dz);
        }
    }
    else
    {
        drho_dz = 0.0;
    }
    
    real sin_z = sin(z);
    real cos_z = cos(z);

    // rotate the spherical radial and polar derivatives into cylindrical R and Z derivatives
    real drho_dR = drho_dy*sin_z + drho_dz*cos_z / y;
    real drho_dZ = drho_dy*cos_z - drho_dz*sin_z / y;

    term_x = drho_dx / rho;
    term_R = drho_dR / rho;
    term_Z = drho_dZ / rho;

    #else  // analytic gas
    
    // the analytic disk is axisymmetric
    term_x = 0.0;

    real R = y*sin(z);
    real Z = y*cos(z);
    real h_g = _get_hg(R);
    real idx_rhog = IDX_P - 0.5*IDX_Q - 1.5;
    real strat_pot = R / y - 1.0;
    real inv_hg2 = 1.0 / (h_g*h_g);
    real inv_y3 = 1.0 / (y*y*y);

    term_R  = idx_rhog / R;
    term_R += inv_hg2*(Z*Z*inv_y3 - (IDX_Q + 1.0)*strat_pot / R);
    term_Z  = -inv_hg2*R*Z*inv_y3;
    
    #endif // IMPORTGAS
}

#endif // DIFFUSION

#endif // DIFFUSION_CUH
