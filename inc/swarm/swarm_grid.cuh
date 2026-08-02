#ifndef SWARM_GRID_CUH
#define SWARM_GRID_CUH

#include <cfloat>                   // DBL_MAX

#include <const_defs.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// interpolation stencil
// =========================================================================================================================

struct interp
{
    int  next_x, next_y, next_z;    // flattened offsets to neighboring cells
    real frac_x, frac_y, frac_z;    // weights assigned to neighboring cells
};

// =========================================================================================================================
// one-dimensional interpolation weights
// =========================================================================================================================

// construct periodic azimuthal cell-centred interpolation weights
__device__ __forceinline__
void _1d_interp_x (real loc_x, real deci_x, real &frac_x, int &next_x)
{
    if (N_X == 1)
    {
        frac_x = 0.0;
        next_x = 0;
    }
    else
    {
        real ref_x = 0.5;
        bool edge_x = loc_x < ref_x || loc_x > static_cast<real>(N_X) + ref_x - 1.0;

        if (!edge_x)
        {
            if (deci_x >= ref_x)
            {
                frac_x = deci_x - ref_x;
                next_x = 1;
            }
            else
            {
                frac_x = ref_x - deci_x;
                next_x = -1;
            }
        }
        else
        {
            // wrap the neighboring stencil cell across the periodic seam
            if (deci_x >= ref_x)
            {
                frac_x = deci_x - ref_x;
                next_x = 1 - N_X;
            }
            else
            {
                frac_x = ref_x - deci_x;
                next_x = N_X - 1;
            }
        }
    }
}

// construct logarithmic-radial interpolation weights for cell centres or outer faces
__device__ __forceinline__
void _1d_interp_y (real loc_y, real deci_y, real &frac_y, int &next_y, bool outer_edge = false)
{
    if (N_Y == 1)
    {
        frac_y = 0.0;
        next_y = 0;
    }
    else
    {
        real dy = _get_dy();
        real ref_y;
        
        if (outer_edge)
        {
            ref_y = 1.0;
        }
        else
        {
            real mesh_dim = _get_mesh_dim();
            
            // place cell-centred values at the exact centroid of the represented disk measure
            ref_y = log((mesh_dim / (mesh_dim + 1.0))*(pow(dy, mesh_dim + 1.0) - 1.0) / (pow(dy, mesh_dim) - 1.0)) / log(dy);
        }
        
        bool edge_y = loc_y < ref_y || loc_y > static_cast<real>(N_Y) + ref_y - 1.0;

        if (outer_edge)
        {
            if (!edge_y)
            {
                frac_y = (dy - pow(dy, deci_y)) / (dy - 1.0);
                next_y = -N_X;
            }
            else
            {
                frac_y = (dy - pow(dy, deci_y)) / (dy - 1.0);
                next_y = 0;         // apply the inner-face optical-depth zero after interpolation
            }
        }
        else
        {
            if (!edge_y)
            {
                if (deci_y >= ref_y)
                {
                    frac_y = (pow(dy, deci_y - ref_y) - 1.0) / (dy - 1.0);
                    next_y = N_X;
                }
                else
                {
                    frac_y = (pow(dy, deci_y - ref_y) - 1.0) / (1.0 / dy - 1.0);
                    next_y = -N_X;
                }
            }
            else
            {
                // clamp a cell-centred stencil rather than reaching beyond the radial domain
                frac_y = 0.0;
                next_y = 0;
            }
        }
    }
}

// construct nonperiodic polar cell-centred interpolation weights
__device__ __forceinline__
void _1d_interp_z (real loc_z, real deci_z, real &frac_z, int &next_z)
{
    if (N_Z == 1)
    {
        frac_z = 0.0;
        next_z = 0;
    }
    else
    {
        real ref_z = 0.5;
        bool edge_z = loc_z < ref_z || loc_z > static_cast<real>(N_Z) + ref_z - 1.0;
        
        if (!edge_z)
        {
            if (deci_z >= ref_z)
            {
                frac_z = deci_z - ref_z;
                next_z = N_X*N_Y;
            }
            else
            {
                frac_z = ref_z - deci_z;
                next_z = -N_X*N_Y;
            }
        }
        else
        {
            // clamp the stencil rather than reaching beyond the polar domain
            frac_z = 0.0;
            next_z = 0;
        }
    }
}

// =========================================================================================================================
// multidimensional interpolation stencil
// =========================================================================================================================

// combine the directional weights and flattened neighbor offsets into one trilinear stencil
__device__ __forceinline__
interp _3d_interp (real loc_x, real loc_y, real loc_z, bool outer_edge = false)
{
    // optical depth uses radial outer-face locations while ordinary fields use cell centroids

    real frac_x, frac_y, frac_z;
    int  next_x, next_y, next_z;

    real deci_x = loc_x - floor(loc_x);
    real deci_y = loc_y - floor(loc_y);
    real deci_z = loc_z - floor(loc_z);

    _1d_interp_x(loc_x, deci_x, frac_x, next_x);
    _1d_interp_y(loc_y, deci_y, frac_y, next_y, outer_edge);
    _1d_interp_z(loc_z, deci_z, frac_z, next_z);

    return {next_x, next_y, next_z, frac_x, frac_y, frac_z};
}

// =========================================================================================================================
// grid-to-particle interpolation
// =========================================================================================================================

// interpolate a scalar grid field at continuous index coordinates with optional radial outer-face centring
__device__ __forceinline__
real _interp_field (const real *dev_field_in, real loc_x, real loc_y, real loc_z, bool outer_edge = false)
{
    #ifdef HALF_DISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1.0e-6);
    #endif // HALF_DISK

    if (outer_edge) // optical depth is defined on radial outer faces
    {
        if (loc_y < 0) return 0.0;
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return DBL_MAX; // suppress radiation outside the optical-depth mesh
    }
    else
    {
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return 0.0;
    }

    int ix = static_cast<int>(loc_x);
    int iy = static_cast<int>(loc_y);
    int iz = static_cast<int>(loc_z);
    int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp(loc_x, loc_y, loc_z, outer_edge);

    real value = 0.0;

    value += dev_field_in[idx_cell                           ]*(1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z);
    value += dev_field_in[idx_cell + next_x                  ]*       frac_x *(1.0 - frac_y)*(1.0 - frac_z);
    value += dev_field_in[idx_cell          + next_y         ]*(1.0 - frac_x)*       frac_y *(1.0 - frac_z);
    value += dev_field_in[idx_cell + next_x + next_y         ]*       frac_x *       frac_y *(1.0 - frac_z);
    value += dev_field_in[idx_cell                   + next_z]*(1.0 - frac_x)*(1.0 - frac_y)*       frac_z ;
    value += dev_field_in[idx_cell + next_x          + next_z]*       frac_x *(1.0 - frac_y)*       frac_z ;
    value += dev_field_in[idx_cell +          next_y + next_z]*(1.0 - frac_x)*       frac_y *       frac_z ;
    value += dev_field_in[idx_cell + next_x + next_y + next_z]*       frac_x *       frac_y *       frac_z ;

    if (outer_edge && loc_y < 1.0) value *= 1.0 - frac_y; // interpolate from zero optical depth at the inner face

    return value;
}

// =========================================================================================================================
// particle-to-grid deposition
// =========================================================================================================================

// deposit one particle weight to its cell-centred trilinear grid stencil
__device__ __forceinline__
void _deposit_field (real *dev_grid_out, const swarm *dev_particle, int idx, real weight)
{
    real loc_x = _get_loc_x(dev_particle[idx].position.x);
    real loc_y = _get_loc_y(dev_particle[idx].position.y);
    real loc_z = _get_loc_z(dev_particle[idx].position.z);

    #ifdef HALF_DISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1e-6);
    #endif // HALF_DISK

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    int ix = static_cast<int>(loc_x);
    int iy = static_cast<int>(loc_y);
    int iz = static_cast<int>(loc_z);
    int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp(loc_x, loc_y, loc_z);

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

#endif // SWARM_GRID_CUH
