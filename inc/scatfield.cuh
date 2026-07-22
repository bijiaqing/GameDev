#ifndef SCATFIELD_CUH
#define SCATFIELD_CUH

#include <const.cuh>
#include <paramgrid.cuh>
#include <interpval.cuh>
#include <paramphys.cuh>

// =========================================================================================================================
// particle-to-grid scattering
// =========================================================================================================================

enum FieldType
{
    #ifdef SAVE_DENS
    DUSTDENS,
    #endif // SAVE_DENS
    #ifdef RADIATION
    OPTDEPTH,
    #endif // RADIATION
    
    FIELDTYPE_NONE                          // keep the enum valid when no scattered field is enabled
};

// scatter one particle's mass or opacity-weighted mass to its trilinear grid stencil
template <FieldType field_type> __device__ __forceinline__
void _particle_to_grid_core (real *dev_grid_out, const swarm *dev_particle, int idx)
{
    real loc_x = _get_loc_x(dev_particle[idx].position.x);
    real loc_y = _get_loc_y(dev_particle[idx].position.y);
    real loc_z = _get_loc_z(dev_particle[idx].position.z);

    #ifdef HALFDISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1e-6); // clamp midplane particles into last Z cell
    #endif // HALFDISK

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    int idx_cell = _get_cell_index(loc_x, loc_y, loc_z);
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp(loc_x, loc_y, loc_z);

    #ifdef MULTISIZE
    real s = dev_particle[idx].par_size;
    #else
    real s = S_0;
    #endif // MULTISIZE
    real weight = 0.0;

    #ifdef RADIATION
    if (field_type == OPTDEPTH)
    {
        weight  = _get_grain_mass(s);
        #ifdef MULTISIZE
        weight *= dev_particle[idx].par_numr;
        #else
        weight *= M_D / N_P / _get_grain_mass(S_0);
        #endif // MULTISIZE
        weight *= KAPPA_0 / (s / S_0); // convert represented mass to extinction cross section

        if (N_Z == 1)
        {
            // close the vertically integrated model with the local Gaussian midplane density
            real R = dev_particle[idx].position.y*sin(dev_particle[idx].position.z);
            real h_g = _get_hg(R);
            real H_d = h_g*R;
            #ifdef DIFFUSION
            #ifndef CONST_NU
            real alpha_z = ALPHA / SCHMIDT_Z;
            #else
            real alpha_z = NU/(h_g*h_g*R*R*_get_omegaK(R)*SCHMIDT_Z);
            #endif
            real stokes_mid = ST_0*(s / S_0);
            #ifndef CONST_ST
            stokes_mid /= pow(R / R_0, IDX_P);
            #endif
            H_d *= sqrt(alpha_z/(alpha_z + stokes_mid));
            #endif
            weight /= sqrt(2.0*M_PI)*H_d;
        }
    }
    else
    #endif // RADIATION
    #ifdef SAVE_DENS
    if (field_type == DUSTDENS)
    {
        weight  = _get_grain_mass(s);
        #ifdef MULTISIZE
        weight *= dev_particle[idx].par_numr;
        #else
        weight *= M_D / N_P / _get_grain_mass(S_0);
        #endif // MULTISIZE
    }
    else
    #endif // SAVE_DENS
    {
        return;
    }

    // deposit the conserved particle weight to all corners of the interpolation stencil
    atomicAdd(&dev_grid_out[idx_cell                           ], (1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid_out[idx_cell + next_x                  ],        frac_x *(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid_out[idx_cell          + next_y         ], (1.0 - frac_x)*       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid_out[idx_cell + next_x + next_y         ],        frac_x *       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid_out[idx_cell                   + next_z], (1.0 - frac_x)*(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid_out[idx_cell + next_x          + next_z],        frac_x *(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid_out[idx_cell          + next_y + next_z], (1.0 - frac_x)*       frac_y *       frac_z *weight);
    atomicAdd(&dev_grid_out[idx_cell + next_x + next_y + next_z],        frac_x *       frac_y *       frac_z *weight);
}

// =========================================================================================================================

#endif // SCATFIELD_CUH
