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
    int  next_x, next_y, next_z;    // flattened offsets to the neighboring cells
    real frac_x, frac_y, frac_z;    // weights assigned to the neighboring cells
};

// =========================================================================================================================
// one-dimensional interpolation weights
// =========================================================================================================================

// construct periodic azimuthal cell-centred interpolation weights
__device__ __forceinline__
void _1d_interp_x (real loc_x, real deci_x, real &frac_x, int &next_x)
{
    if (N_X == 1)                   // if there is only one cell in X
    {
        frac_x = 0.0;               // the share for the current cell is '1.0 - frac_x'
        next_x = 0;                 // no other cells to share the particle
    }
    else
    {
        real ref_x = 0.5;
        bool edge_x = loc_x < ref_x || loc_x > static_cast<real>(N_X) + ref_x - 1.0;

        if (not edge_x)             // still in the interior of the X domain
        {
            if (deci_x >= ref_x)    // share with the cell on the right
            {
                frac_x = deci_x - ref_x;
                next_x = 1;
            }
            else                    // share with the cell on the left
            {
                frac_x = ref_x - deci_x;
                next_x = -1;
            }
        }
        else                        // too close to the inner or the outer X boundary 
        {
            if (deci_x >= ref_x)    // too close to the outer X boundary
            {
                frac_x = deci_x - ref_x;
                next_x = 1 - N_X;   // share with the first cell of its row
            }
            else                    // too close to the inner X boundary
            {
                frac_x = ref_x - deci_x;
                next_x = N_X - 1;   // share with the last  cell of its row
            }
        }
    }
}

// construct logarithmic-radial interpolation weights for cell centres or outer faces
__device__ __forceinline__
void _1d_interp_y (real loc_y, real deci_y, real &frac_y, int &next_y, bool outer_edge = false)
{
    if (N_Y == 1)                   // if there is only one cell in Y
    {
        frac_y = 0.0;               // the share for the current cell is '1.0 - frac_y'
        next_y = 0;                 // no other cells to share the particle
    }
    else
    {
        real d_y = _get_dy();
        real ref_y;
        
        if (outer_edge)
        {
            ref_y = 1.0;            // outer edge of Y cell
        }
        else
        {
            real mesh_dim = _get_mesh_dim();
            
            // place cell-centred values at the exact centroid of the represented disk measure
            ref_y = log((mesh_dim / (mesh_dim + 1.0))*(pow(d_y, mesh_dim + 1.0) - 1.0) / (pow(d_y, mesh_dim) - 1.0)) / log(d_y);
        }
        
        bool edge_y = loc_y < ref_y || loc_y > static_cast<real>(N_Y) + ref_y - 1.0;

        if (outer_edge)
        {
            if (not edge_y)         // still in the interior of the Y domain
            {
                frac_y = (d_y - pow(d_y, deci_y)) / (d_y - 1.0);
                next_y = -N_X;      // share with the cell on its left
            }
            else                    // at the Y domain boundaries
            {
                frac_y = (d_y - pow(d_y, deci_y)) / (d_y - 1.0);
                next_y = 0;         // the inner-face zero is applied after interpolation
            }
        }
        else
        {
            if (not edge_y)         // still in the interior of the Y domain
            {
                if (deci_y >= ref_y) // share with the cell on the right
                {
                    frac_y = (pow(d_y, deci_y - ref_y) - 1.0) / (d_y - 1.0);
                    next_y = N_X;   // the index distance to the next Y cell on the right is N_X
                }
                else                // share with the cell on the left
                {
                    frac_y = (pow(d_y, deci_y - ref_y) - 1.0) / (1.0 / d_y - 1.0);
                    next_y = -N_X;
                }
            }
            else                    // at the Y domain boundaries
            {
                frac_y = 0.0;       // the current cell take it all like N_Y = 1
                next_y = 0;
            }
        }
    }
}

// construct nonperiodic polar cell-centred interpolation weights
__device__ __forceinline__
void _1d_interp_z (real loc_z, real deci_z, real &frac_z, int &next_z)
{
    if (N_Z == 1)                   // if there is only one cell in Z
    {
        frac_z = 0.0;               // the share for the current cell is '1.0 - frac_z'
        next_z = 0;                 // no other cells to share the particle
    }
    else
    {
        real ref_z = 0.5;
        bool edge_z = loc_z < ref_z || loc_z > static_cast<real>(N_Z) + ref_z - 1.0;
        
        if (not edge_z)             // still in the interior of the Z domain
        {
            if (deci_z >= ref_z)
            {
                frac_z = deci_z - ref_z;
                next_z = N_X*N_Y;   // the index distance to the next Z cell on the right is N_X*N_Y
            }
            else
            {
                frac_z = ref_z - deci_z;
                next_z = -N_X*N_Y;
            }
        }
        else                        // at the Z domain boundaries, the current cell take it all like N_Z = 1
        {
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
    #ifdef HALFDISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1.0e-6);
    #endif // HALFDISK

    if (outer_edge) // optical depth is defined on radial outer faces
    {
        if (loc_y < 0) return 0.0;
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return DBL_MAX; // suppress radiation outside the optical-depth mesh
    }
    else
    {
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return 0.0;
    }

    int idx_cell = static_cast<int>(loc_z)*N_X*N_Y + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
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

    #ifdef HALFDISK
    loc_z = fmin(loc_z, static_cast<real>(N_Z) - 1e-6);
    #endif // HALFDISK

    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return;

    int idx_cell = static_cast<int>(loc_z)*N_X*N_Y + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
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
