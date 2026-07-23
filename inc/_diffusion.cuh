#ifndef SWARM_DIFFUSION_CUH
#define SWARM_DIFFUSION_CUH

#ifdef DIFFUSION

#ifdef IMPORTGAS
#include <cassert>      // assert
#endif // IMPORTGAS

#include <const_defs.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_grid.cuh>

// =========================================================================================================================
// cylindrical gas-density gradients
// =========================================================================================================================

#ifdef IMPORTGAS
// interpolate the gas measure matching volume density in 3D and surface density in 2D
__device__ __forceinline__
real _get_diff_gasdens (real x, real y, real z, const real *dev_gas_dens)
{
    real gasdens = _interp_field(dev_gas_dens, _get_loc_x(x), _get_loc_y(y), _get_loc_z(z));

    if (N_Z == 1)
    {
        real R = y*sin(z);
        gasdens *= sqrt(2.0*M_PI)*_get_hg(R)*R;
    }

    return gasdens;
}
#endif // IMPORTGAS

// calculate the logarithmic gas-density derivatives needed by the cylindrical diffusion SDE
__device__ __forceinline__
void _get_term_grad_cyl (real x, real y, real z, real &term_x, real &term_R, real &term_Z
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
)
{   
    #ifdef IMPORTGAS

    real loc_y = _get_loc_y(y);

    real gasdens = _get_diff_gasdens(x, y, z, dev_gas_dens);

    if (gasdens <= 0.0)
    {
        printf("ERROR: Invalid gas density = %e at (x,y,z) = (%e,%e,%e)\n", gasdens, x, y, z);
        assert(false);
    }

    real dgas_dx, dgas_dy, dgas_dz;

    if (N_X > 1)
    {
        real dx = _get_dx();
        
        real x_p = x + dx;
        real x_m = x - dx;

        if (x_p >= X_MAX) x_p -= (X_MAX - X_MIN);
        if (x_m <  X_MIN) x_m += (X_MAX - X_MIN);
        
        real gas_xp = _get_diff_gasdens(x_p, y, z, dev_gas_dens);
        real gas_xm = _get_diff_gasdens(x_m, y, z, dev_gas_dens);
        
        dgas_dx = (gas_xp - gas_xm) / (2.0*dx);
    }
    else // if azimuthal direction disabled
    {
        dgas_dx = 0.0;
    }
    
    if (N_Y > 1)
    {
        real dy = y*log(_get_dy());
        
        if (loc_y < 0.5) // use a one-sided difference next to the inner radial boundary
        {
            real gas_yp = _get_diff_gasdens(x, y + dy, z, dev_gas_dens);
            
            dgas_dy = (gas_yp - gasdens) / dy;
        }
        else if (loc_y > static_cast<real>(N_Y) - 0.5) // use a one-sided difference next to the outer radial boundary
        {
            real gas_ym = _get_diff_gasdens(x, y - dy, z, dev_gas_dens);
            
            dgas_dy = (gasdens - gas_ym) / dy;
        }
        else // use a centred difference in the radial interior
        {
            real gas_yp = _get_diff_gasdens(x, y + dy, z, dev_gas_dens);
            real gas_ym = _get_diff_gasdens(x, y - dy, z, dev_gas_dens);
            
            dgas_dy = (gas_yp - gas_ym) / (2.0*dy);
        }
    }
    else
    {
        dgas_dy = 0.0;
    }

    if (N_Z > 1)
    {
        real loc_z = _get_loc_z(z);
        real dz = _get_dz();
        
        if (loc_z < 0.5) // use a one-sided difference next to the lower polar boundary
        {
            real gas_zp = _get_diff_gasdens(x, y, z + dz, dev_gas_dens);
            
            dgas_dz = (gas_zp - gasdens) / dz;
        }
        else if (loc_z > static_cast<real>(N_Z) - 0.5) // use a one-sided difference next to the upper polar boundary
        {
            real gas_zm = _get_diff_gasdens(x, y, z - dz, dev_gas_dens);
            
            dgas_dz = (gasdens - gas_zm) / dz;
        }
        else // use a centred difference in the polar interior
        {
            real gas_zp = _get_diff_gasdens(x, y, z + dz, dev_gas_dens);
            real gas_zm = _get_diff_gasdens(x, y, z - dz, dev_gas_dens);
            
            dgas_dz = (gas_zp - gas_zm) / (2.0*dz);
        }
    }
    else
    {
        dgas_dz = 0.0;
    }
    
    real sin_z = sin(z);
    real cos_z = cos(z);

    // rotate the spherical radial and polar derivatives into cylindrical R and Z derivatives
    real dgas_dR = dgas_dy*sin_z + dgas_dz*cos_z / y;
    real dgas_dZ = dgas_dy*cos_z - dgas_dz*sin_z / y;

    term_x = dgas_dx / gasdens;
    term_R = dgas_dR / gasdens;
    term_Z = dgas_dZ / gasdens;

    #else  // ANALYTIC_GAS
    
    // the analytic disk is axisymmetric
    term_x = 0.0;

    real R = y*sin(z);
    real Z = y*cos(z);

    if (N_Z == 1)
    {
        term_R = IDX_P / R;
        term_Z = 0.0;
        return;
    }

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

// =========================================================================================================================

#endif // SWARM_DIFFUSION_CUH
