#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <const.cuh>

// =========================================================================================================================
// Grid Spacing Helpers
// Purpose: Centralise the three cell-spacing formulas so every kernel uses the same expression.
//
//   dx = uniform azimuthal cell width    (radians)
//   dy = logarithmic radial cell ratio   q = (Y_MAX/Y_MIN)^{1/N_Y}; inner face of cell iy = Y_MIN*q^iy
//   dz = uniform polar cell width        (radians)
// =========================================================================================================================

__host__ __device__ __forceinline__
real _get_dx() { return    (X_MAX - X_MIN)     / static_cast<real>(N_X); }

__host__ __device__ __forceinline__
real _get_dy() { return pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); }

__host__ __device__ __forceinline__
real _get_dz() { return    (Z_MAX - Z_MIN)     / static_cast<real>(N_Z); }

// =========================================================================================================================
// Radial Jacobian dimension:
//   N_Z == 1: 2D azimuthal-radial disk, dV_y = y dy   (dimension 2)
//   N_Z >  1: full 3D spherical grid,    dV_y = y^2 dy (dimension 3)
// The presence of the polar coordinate determines whether the radial geometry is cylindrical or spherical

__host__ __device__ __forceinline__
real _get_powy() { return (N_Z > 1) ? 3.0 : 2.0; }

// =========================================================================================================================

__device__ __forceinline__
real _get_omegaK (real R)
{
    return sqrt(G*M_S / R / R / R);
}

__device__ __forceinline__
real _get_hg (real R)
{
    return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0));
}

__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{
    return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z)));
}

// Exact vertical hydrostatic stratification for a radially locally-isothermal disk:
// rho_g(R,Z)/rho_g(R,0) = exp[(R/sqrt(R^2+Z^2)-1)/h_g^2]

__device__ __forceinline__
real _get_gas_strat (real R, real Z, real h_g)
{
    real y = sqrt(R*R + Z*Z);
    return exp((R / y - 1.0) / (h_g*h_g));
}

__device__ __forceinline__
real _get_rhog (real R, real Z, real h_g)
{
    real H_g = h_g*R;
    real sigma_g = SIGMA_0*pow(R / R_0, IDX_P);
    real rho_mid = sigma_g / (sqrt(2.0*M_PI)*H_g);

    return rho_mid*_get_gas_strat(R, Z, h_g);
}

__device__ __forceinline__
real _get_St (real R, real Z, real h_g)
{
    real St = ST_0;
    St /= pow(R / R_0, IDX_P);
    St /= _get_gas_strat(R, Z, h_g);

    return St;
}

#ifdef DIFFUSION
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifndef CONST_NU
    real nu = ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    return nu;
    #else // CONST_NU
    return NU;
    #endif // NOT CONST_NU
}

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU
    return ALPHA;
    #else  // CONST_NU
    real alpha = NU / (h_g*h_g*R*R*_get_omegaK(R));
    return alpha;
    #endif // NOT CONST_NU
}
#endif // DIFFUSION

// =========================================================================================================================
// Recover primitive dust variables from the conserved state using one consistent vacuum convention
// Vacuum conserved fields are reset as well, preserving momx=dens*velx, momy=dens*vely, and momz=dens*velz

__device__ __forceinline__
void _recover_dust_state (real dens, real R, real &momx, real &momy, real &momz, real &velx, real &vely, real &velz)
{
    if (dens < RHO_VAC)
    {
        // velx_K = R^2*Omega_K = sqrt(G*M_S*R)
        // This algebraically equivalent form remains finite at R = 0, unlike R^2*sqrt(G*M_S/R^3)
        velx = sqrt(G*M_S*fmax(R, 0.0));
        vely = 0.0;
        velz = 0.0;

        momx = dens*velx;
        momy = 0.0;
        momz = 0.0;
    }
    else
    {
        velx = momx / dens;
        vely = momy / dens;
        velz = momz / dens;
    }
}

// =========================================================================================================================
// PPM (Piecewise Parabolic Method) device helpers
// Reference: Colella & Woodward (1984), J. Comput. Phys. 54, 174
//
// Overview of the helper functions:
//   _ppm_edge              -- uniform-grid 4th-order face value used by the periodic X sweep
//   _ppm_edges_nonuniform  -- geometry-aware face values from precomputed FV weights (Y/Z)
//   _ppm_limit             -- Colella-Woodward monotonicity limiter for one cell
//   _ppm_state_R           -- time-averaged face state for v > 0 (sweeps right portion of upwind cell)
//   _ppm_state_L           -- time-averaged face state for v < 0 (sweeps left  portion of upwind cell)
//   _ppm_face_value        -- combines limit + state selection: the common per-field flux core
//
// The uniform formula for _ppm_edge is exact for polynomials up to cubic when q values are FV averages:
//   q_{i+1/2} = (7*(q_i + q_{i+1}) - (q_{i-1} + q_{i+2})) / 12

// =========================================================================================================================
// 4th-order face value between FV cell averages q0 and qp1 using neighbours qm1 and qp2
// Clamped to local monotone range [min(q0,qp1), max(q0,qp1)] to avoid oscillations

__device__ __forceinline__
static real _ppm_edge (real qm1, real q0, real qp1, real qp2)
{
    real qe = (7.0*(q0 + qp1) - (qm1 + qp2)) / 12.0;
    real lo = fmin(q0, qp1);
    real hi = fmax(q0, qp1);
    return fmax(lo, fmin(hi, qe));
}

// =========================================================================================================================
// Geometry-aware edge reconstruction on a nonuniform finite-volume mesh
// Four precomputed coefficients are stored per face
// Interior faces multiply cells iface-2 ... iface+1
// the first and last interior faces use the first two entries for their adjacent cells
// Boundary faces retain the nearest cell value for the existing outflow boundary treatment

__device__ __forceinline__
static void _ppm_edges_nonuniform (const real *cellval, const real *face_weight, real *edge, int ncells)
{
    edge[0] = cellval[0];

    for (int iface = 1; iface < ncells; iface++)
    {
        const real *weight = face_weight + 4*iface;
        if (iface >= 2 && iface <= ncells - 2)
        {
            edge[iface] = weight[0]*cellval[iface - 2] + weight[1]*cellval[iface - 1]
                        + weight[2]*cellval[iface]     + weight[3]*cellval[iface + 1];
        }
        else
        {
            edge[iface] = weight[0]*cellval[iface - 1] + weight[1]*cellval[iface];
        }

        real edge_min = fmin(cellval[iface - 1], cellval[iface]);
        real edge_max = fmax(cellval[iface - 1], cellval[iface]);

        edge[iface] = fmax(edge_min, fmin(edge_max, edge[iface]));
    }

    edge[ncells] = cellval[ncells - 1];
}

// =========================================================================================================================
// Colella-Woodward monotonicity limiter: adjust ql (left edge) and qr (right edge)
// so the parabola in the cell is monotone and bounded
// Computes
//   dq = qr - ql          (amplitude)
//   q6 = 6*q0 - 3*(ql+qr) (curvature parameter in the PPM parabola)
// after the adjustment

__device__ __forceinline__
static void _ppm_limit (real q0, real &ql, real &qr, real &dq, real &q6)
{
    dq = qr - ql;
    q6 = 6.0*q0 - 3.0*(ql + qr);

    if ((qr - q0)*(q0 - ql) <= 0.0)  // extremum inside cell: degenerate to constant
    {
        ql = qr = q0;
        dq = 0.0;
        q6 = 0.0;
        return;
    }

    if ( dq*q6 > dq*dq)  //  left edge overshoot
    {
        ql = 3.0*q0 - 2.0*qr;
        dq = qr - ql;
        q6 = 6.0*q0 - 3.0*(ql + qr);
    }

    if (-dq*q6 > dq*dq)  // right edge overshoot
    {
        qr = 3.0*q0 - 2.0*ql;
        dq = qr - ql;
        q6 = 6.0*q0 - 3.0*(ql + qr);
    }
}

// =========================================================================================================================
// Time-averaged PPM face state for v > 0: integrate over [1-cfl, 1] of the upwind cell.
//   cfl = v * dt / dx_cell   (local CFL in [0,1])
//   qb  = right edge of upwind cell

__device__ __forceinline__
static real _ppm_state_R (real qb, real dq, real q6, real cfl)
{
    return qb - 0.5*cfl*(dq - (1.0 - 2.0*cfl/3.0)*q6);
}

// =========================================================================================================================
// Time-averaged PPM face state for v < 0: integrate over [0, cfl] of the upwind cell.
//   cfl = |v| * dt / dx_cell   (local CFL in [0,1])
//   qa  = left edge of upwind cell

__device__ __forceinline__
static real _ppm_state_L (real qa, real dq, real q6, real cfl)
{
    return qa + 0.5*cfl*(dq + (1.0 - 2.0*cfl/3.0)*q6);
}

// =========================================================================================================================
// Combines edge lookup + monotonicity limiting + upwind time-averaging into a single call
// This is the common core repeated for every advected field (density, velx, vely, velz) in f_advection_x/y/z.cu:
// given the precomputed face-reconstruction array `edge` and the raw cell-average array `cellval`,
// look up the two edges bounding the upwind cell (indices iupL, iupL+1 == iupR), apply the Colella-Woodward limiter,
// then return the time-averaged face value swept across the face over the fraction `cfl` of the upwind cell
// No density-floor clamp is applied here — callers that need dens >= 0 (i.e. the density flux itself) apply
// fmax(..., 0.0) to the returned value
// momentum-component callers use it as-is since velocities/angular momenta may legitimately be negative

__device__ __forceinline__
static real _ppm_face_value (const real *edge, const real *cellval, int iupL, int iupR, bool upwind_on_left, real cfl)
{
    real val_L = edge[iupL];
    real val_R = edge[iupR];
    real dval, coeff_curv;

    _ppm_limit(cellval[iupL], val_L, val_R, dval, coeff_curv);

    return upwind_on_left ?
        _ppm_state_R(val_R, dval, coeff_curv, cfl) :
        _ppm_state_L(val_L, dval, coeff_curv, cfl) ;
}

// =========================================================================================================================
// Local invariant-domain bounds for a primitive field. The one-cell stencil is the domain of
// dependence of the first-order HLL update used as the safe state in the antidiffusive limiter.

__device__ __forceinline__
static void _local_bounds (const real *value, int idx, int count, real &value_min, real &value_max)
{
    int idx_min = (idx > 0) ? idx - 1 : idx;
    int idx_max = (idx < count - 1) ? idx + 1 : idx;

    value_min = value[idx_min];
    value_max = value[idx_min];
    for (int ii = idx_min + 1; ii <= idx_max; ii++)
    {
        value_min = fmin(value_min, value[ii]);
        value_max = fmax(value_max, value[ii]);
    }

    real bound_pad = 1.0e-12*fmax(1.0, fmax(fabs(value_min), fabs(value_max)));
    value_min -= bound_pad;
    value_max += bound_pad;
}

__device__ __forceinline__
static void _local_bounds_periodic (const real *value, int idx, int count, real &value_min, real &value_max)
{
    int idx_prev = (idx - 1 + count) % count;
    int idx_next = (idx + 1) % count;

    value_min = fmin(value[idx_prev], fmin(value[idx], value[idx_next]));
    value_max = fmax(value[idx_prev], fmax(value[idx], value[idx_next]));

    real bound_pad = 1.0e-12*fmax(1.0, fmax(fabs(value_min), fabs(value_max)));
    value_min -= bound_pad;
    value_max += bound_pad;
}

// Maximum fraction of one conservative face correction that keeps a cell in the convex set
//   dens >= 0 and value_min*dens <= momentum <= value_max*dens
// for all three stored specific momenta. The small reserve prevents roundoff from landing exactly
// outside a constraint. Applying the minimum fraction from both face neighbours retains one common
// equal-and-opposite conservative flux.

__device__ __forceinline__
static void _restrict_scale (real available, real change, real &scale)
{
    if (change >= 0.0) return;
    scale = (available > 0.0) ? fmin(scale, (1.0 - 1.0e-12)*available / (-change)) : 0.0;
}

__device__ __forceinline__
static real _invariant_scale (
    real dens, real momx, real momy, real momz,
    real corr_dens, real corr_momx, real corr_momy, real corr_momz,
    real velx_min, real velx_max,
    real vely_min, real vely_max,
    real velz_min, real velz_max)
{
    real scale = 1.0;

    _restrict_scale(dens, corr_dens, scale);
    _restrict_scale(momx - velx_min*dens, corr_momx - velx_min*corr_dens, scale);
    _restrict_scale(velx_max*dens - momx, velx_max*corr_dens - corr_momx, scale);
    _restrict_scale(momy - vely_min*dens, corr_momy - vely_min*corr_dens, scale);
    _restrict_scale(vely_max*dens - momy, vely_max*corr_dens - corr_momy, scale);
    _restrict_scale(momz - velz_min*dens, corr_momz - velz_min*corr_dens, scale);
    _restrict_scale(velz_max*dens - momz, velz_max*corr_dens - corr_momz, scale);

    return fmax(0.0, fmin(1.0, scale));
}

// =========================================================================================================================
// TODO(pressureless-flux): For quantitative dust-clumping studies, compare this HLL flux with the pressureless
// dust Riemann flux of Huang & Bai (2022, ApJS 262, 11; doi:10.3847/1538-4365/ac76cb)
// Their normal-velocity branches are:
//   vL > 0, vR > 0 : use F_L
//   vL < 0, vR < 0 : use F_R
//   vL < 0, vR > 0 : use zero flux (diverging streams / interface vacuum)
//   vL > 0, vR < 0 : use F_L + F_R (converging, interpenetrating dust streams)
// If added, apply exactly the same branch to density and all three momentum components, retain
// the invariant-domain face limiting in the geometric sweeps, and expose HLL versus Huang-Bai as
// a compile-time choice for clump-growth and resolution-convergence comparisons

// =========================================================================================================================
// HLL flux for the pressureless advective system.  Primitive left/right interface states are
// converted to conserved states here, ensuring that mass and all conserved components use one
// common Riemann flux.  `speed` is the physical transport speed in the swept coordinate.

__device__ __forceinline__
static void _pressureless_hll_flux (
    real speed_L, real speed_R,
    real dens_L, real velx_L, real vely_L, real velz_L,
    real dens_R, real velx_R, real vely_R, real velz_R,
    real &flux_dens, real &flux_momx, real &flux_momy, real &flux_momz)
{
    real momx_L = dens_L*velx_L;
    real momy_L = dens_L*vely_L;
    real momz_L = dens_L*velz_L;
    real momx_R = dens_R*velx_R;
    real momy_R = dens_R*vely_R;
    real momz_R = dens_R*velz_R;

    if (speed_L >= 0.0 && speed_R >= 0.0)
    {
        flux_dens = speed_L*dens_L;
        flux_momx = speed_L*momx_L;
        flux_momy = speed_L*momy_L;
        flux_momz = speed_L*momz_L;
        return;
    }

    if (speed_L <= 0.0 && speed_R <= 0.0)
    {
        flux_dens = speed_R*dens_R;
        flux_momx = speed_R*momx_R;
        flux_momy = speed_R*momy_R;
        flux_momz = speed_R*momz_R;
        return;
    }

    real wave_L   = fmin(speed_L, speed_R);
    real wave_R   = fmax(speed_L, speed_R);
    real inv_span = 1.0 / (wave_R - wave_L);

    flux_dens = (wave_R*speed_L*dens_L - wave_L*speed_R*dens_R + wave_L*wave_R*(dens_R - dens_L))*inv_span;
    flux_momx = (wave_R*speed_L*momx_L - wave_L*speed_R*momx_R + wave_L*wave_R*(momx_R - momx_L))*inv_span;
    flux_momy = (wave_R*speed_L*momy_L - wave_L*speed_R*momy_R + wave_L*wave_R*(momy_R - momy_L))*inv_span;
    flux_momz = (wave_R*speed_L*momz_L - wave_L*speed_R*momz_R + wave_L*wave_R*(momz_R - momz_L))*inv_span;
}

// =========================================================================================================================

#endif // HELPERS_CUH
