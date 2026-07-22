#ifndef HELPERS_INTERPVAL_CUH
#define HELPERS_INTERPVAL_CUH

#include <cfloat>                   // for DBL_MAX

#include <const.cuh>
#include <helpers_paramgrid.cuh>

// =========================================================================================================================
// Interpolation Data Structure
// =========================================================================================================================

struct interp
{
    int  next_x, next_y, next_z;    // indices of the next grid cell in each direction
    real frac_x, frac_y, frac_z;    // fractional distance to the next grid cell in each direction
};

// =========================================================================================================================
// 1D Interpolation Helpers
// Purpose: Calculate interpolation weights for each dimension (periodic X, logarithmic Y, linear Z)
// =========================================================================================================================

// Order: O0 | Dependencies: None (only constants)
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
        real m_x = 0.5;
        bool edge_x = loc_x < m_x || loc_x > static_cast<real>(N_X) + m_x - 1.0;

        if (not edge_x)             // still in the interior of the X domain
        {
            if (deci_x >= m_x)      // share with the cell on the right
            {
                frac_x = deci_x - m_x;
                next_x = 1;
            }
            else                    // share with the cell on the left
            {
                frac_x = m_x - deci_x;
                next_x = -1;
            }
        }
        else                        // too close to the inner or the outer X boundary 
        {
            if (deci_x >= m_x)      // too close to the outer X boundary
            {
                frac_x = deci_x - m_x;
                next_x = 1 - N_X;   // share with the first cell of its row
            }
            else                    // too close to the inner X boundary
            {
                frac_x = m_x - deci_x;
                next_x = N_X - 1;   // share with the last  cell of its row
            }
        }
    }
}

// Order: O0 | Dependencies: None (only constants)
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
        real d_y = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
        real m_y;
        
        if (outer_edge)
        {
            m_y = 1.0;              // outer edge of Y cell
        }
        else
        {
            real idx_dim = static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1) + 1.0;
            
            // volume centroid: r_c = (D/(D+1))*(r_out^(D+1) - r_in^(D+1)) / (r_out^D - r_in^D)
            m_y = log((idx_dim / (idx_dim + 1.0))*(pow(d_y, idx_dim + 1.0) - 1.0) / (pow(d_y, idx_dim) - 1.0)) / log(d_y);
        }
        
        bool edge_y = loc_y < m_y || loc_y > static_cast<real>(N_Y) + m_y - 1.0;

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
                next_y = 0;         // the particles are unfortunately 100% self-shadowed
            }
        }
        else
        {
            if (not edge_y)         // still in the interior of the Y domain
            {
                if (deci_y >= m_y)  // share with the cell on the right
                {
                    frac_y = (pow(d_y, deci_y - m_y) - 1.0) / (d_y - 1.0);
                    next_y = N_X;   // the index distance to the next Y cell on the right is N_X
                }
                else                // share with the cell on the left
                {
                    frac_y = (pow(d_y, deci_y - m_y) - 1.0) / (1.0 / d_y - 1.0);
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

// Order: O0 | Dependencies: None (only constants)
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
        real m_z = 0.5;
        bool edge_z = loc_z < m_z || loc_z > static_cast<real>(N_Z) + m_z - 1.0;
        
        if (not edge_z)             // still in the interior of the Z domain
        {
            if (deci_z >= m_z)
            {
                frac_z = deci_z - m_z;
                next_z = N_X*N_Y;   // the index distance to the next Z cell on the right is N_X*N_Y
            }
            else
            {
                frac_z = m_z - deci_z;
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
// 3D Interpolation Helpers
// Purpose: Combine 1D interpolations into 3D trilinear interpolation
// =========================================================================================================================

// Order: O1 | Dependencies: _1d_interp_x/y/z [O0]
__device__ __forceinline__
interp _3d_interp (real loc_x, real loc_y, real loc_z, bool outer_edge = false)
{
    // this function exists because the optical depth field is defined at the outer radial boundary of each cell
    // the particle needs to be interpolated based on the location of the radial cell edges to get the shadow

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
// Field Interpolation Function
// Purpose: Trilinear interpolation of scalar field values from grid to particle positions
// =========================================================================================================================

// Order: O2 | Dependencies: _is_in_bounds [O0], _get_cell_index [O0], _3d_interp [O1]
__device__ __forceinline__
real _interp_field (const real *dev_field, real loc_x, real loc_y, real loc_z, bool outer_edge = false)
{
    if (outer_edge) // meaning we are interpolating optical depth
    {
        // Optical depth interpolation (outer_edge = true)
        if (loc_y < 0) return 0.0;
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return DBL_MAX; // out of bounds, return a large value
    }
    else
    {
        // Standard field interpolation (outer_edge = false)
        if (!_is_in_bounds(loc_x, loc_y, loc_z)) return 0.0; // out of bounds, returns 0.0
    }

    int idx_cell = _get_cell_index(loc_x, loc_y, loc_z);
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp(loc_x, loc_y, loc_z, outer_edge);

    real value = 0.0;

    value += dev_field[idx_cell                           ]*(1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z);
    value += dev_field[idx_cell + next_x                  ]*       frac_x *(1.0 - frac_y)*(1.0 - frac_z);
    value += dev_field[idx_cell          + next_y         ]*(1.0 - frac_x)*       frac_y *(1.0 - frac_z);
    value += dev_field[idx_cell + next_x + next_y         ]*       frac_x *       frac_y *(1.0 - frac_z);
    value += dev_field[idx_cell                   + next_z]*(1.0 - frac_x)*(1.0 - frac_y)*       frac_z ;
    value += dev_field[idx_cell + next_x          + next_z]*       frac_x *(1.0 - frac_y)*       frac_z ;
    value += dev_field[idx_cell          + next_y + next_z]*(1.0 - frac_x)*       frac_y *       frac_z ;
    value += dev_field[idx_cell + next_x + next_y + next_z]*       frac_x *       frac_y *       frac_z ;

    return value;
}

// =========================================================================================================================

#endif // HELPERS_INTERPVAL_CUH
