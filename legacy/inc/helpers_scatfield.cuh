#ifndef HELPERS_SCATFIELD_CUH
#define HELPERS_SCATFIELD_CUH

#include <const.cuh>
#include <helpers_paramgrid.cuh>
#include <helpers_interpval.cuh>
#include <helpers_paramphys.cuh>

// =========================================================================================================================
// Grid Field Scattering (Order 2)
// Purpose: Scatter particle properties onto grid using trilinear interpolation
// =========================================================================================================================

enum FieldType                              // field type for particle-to-grid scattering
{
    #ifdef SAVE_DENS
    DUSTDENS,                               // dust density field
    #endif // SAVE_DENS
    #if defined(TRANSPORT) && defined(RADIATION)
    OPTDEPTH,                               // optical depth field
    #endif // RADIATION
    
    FIELDTYPE_NONE                          // to prevent empty enum
};

// Order: O2 | Dependencies: _get_loc_x/y/z [O0], _is_in_bounds [O0], _get_cell_index [O0], _3d_interp [O1], _get_grain_mass [O0]
// Flags: TRANSPORT+RADIATION for OPTDEPTH; SAVE_DENS for DUSTDENS
// Purpose: Scatter particle mass/opacity to 8 neighboring grid cells using trilinear weights
template <FieldType field_type> __device__ __forceinline__
void _particle_to_grid_core (real *dev_grid, const swarm *dev_particle, int idx)
{
    real loc_x = _get_loc_x(dev_particle[idx].position.x);
    real loc_y = _get_loc_y(dev_particle[idx].position.y);
    real loc_z = _get_loc_z(dev_particle[idx].position.z);

    #ifdef HALFDISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1e-6); // clamp midplane particles into last Z cell
    #endif // HALFDISK

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return; // particle is out of bounds, do nothing

    int idx_cell = _get_cell_index(loc_x, loc_y, loc_z);
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp(loc_x, loc_y, loc_z);

    #ifdef MULTISIZE
    real s = dev_particle[idx].par_size;
    #else
    real s = S_0;
    #endif // MULTISIZE
    real weight = 0.0;

    #if defined(TRANSPORT) && defined(RADIATION)
    if (field_type == OPTDEPTH)
    {
        weight  = _get_grain_mass(s);
        #ifdef MULTISIZE
        weight *= dev_particle[idx].par_numr;
        #else
        weight *= M_D / N_P / _get_grain_mass(S_0);
        #endif // MULTISIZE
        weight *= KAPPA_0 / (s / S_0); // cross section per unit mass
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
        return; // unknown field type, do nothing
    }

    atomicAdd(&dev_grid[idx_cell                           ], (1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x                  ],        frac_x *(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell          + next_y         ], (1.0 - frac_x)*       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y         ],        frac_x *       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell                   + next_z], (1.0 - frac_x)*(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x          + next_z],        frac_x *(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell          + next_y + next_z], (1.0 - frac_x)*       frac_y *       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y + next_z],        frac_x *       frac_y *       frac_z *weight);
}

// =========================================================================================================================

#endif // HELPERS_SCATFIELD_CUH
