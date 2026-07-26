#ifndef FLUID_TRANSPORT_CUH
#define FLUID_TRANSPORT_CUH

#include <const_defs.cuh>

// =========================================================================================================================
// dust state recovery

// recover primitive velocities from conserved momenta, with a Keplerian fallback in near-vacuum cells
__device__ __forceinline__
void _recover_dust_state (real rhod, real R, real &mx, real &my, real &mz, real &lx, real &vy, real &lz)
{
    if (rhod < RHO_VAC)
    {
        lx = sqrt(G*M_S*fmax(R, 0.0));
        vy = 0.0;
        lz = 0.0;

        mx = rhod*lx;
        my = 0.0;
        mz = 0.0;
    }
    else
    {
        lx = mx / rhod;
        vy = my / rhod;
        lz = mz / rhod;
    }
}

// =========================================================================================================================
// PPM face reconstruction

// reconstruct a bounded PPM face value on a uniform grid from four neighboring cells (for x direction)
__device__ __forceinline__
static real _ppm_face_uniform (real q_m1, real q_0, real q_p1, real q_p2)
{
    real qe = (7.0*(q_0 + q_p1) - (q_m1 + q_p2)) / 12.0;
    real lo = fmin(q_0, q_p1);
    real hi = fmax(q_0, q_p1);
    return fmax(lo, fmin(hi, qe));
}

// reconstruct bounded face values on a nonuniform grid using precomputed PPM weights (for y and z directions)
__device__ __forceinline__
static void _ppm_faces_nonuniform (const real *cell_val, const real *face_weight, real *face, int cell_count)
{
    face[0] = cell_val[0];

    for (int idx_face = 1; idx_face < cell_count; idx_face++)
    {
        const real *weight = face_weight + 4*idx_face;
        if (idx_face >= 2 && idx_face <= cell_count - 2)
        {
            face[idx_face]  = weight[0]*cell_val[idx_face - 2] + weight[1]*cell_val[idx_face - 1];
            face[idx_face] += weight[2]*cell_val[idx_face]     + weight[3]*cell_val[idx_face + 1];
        }
        else
        {
            face[idx_face] = weight[0]*cell_val[idx_face - 1] + weight[1]*cell_val[idx_face];
        }

        real face_min = fmin(cell_val[idx_face - 1], cell_val[idx_face]);
        real face_max = fmax(cell_val[idx_face - 1], cell_val[idx_face]);

        face[idx_face] = fmax(face_min, fmin(face_max, face[idx_face]));
    }

    face[cell_count] = cell_val[cell_count - 1];
}

// =========================================================================================================================
// PPM profile limiting and upwind tracing

// limit a cell parabolic profile to prevent new extrema and oscillations
__device__ __forceinline__
static void _ppm_limit (real q_0, real &q_L, real &q_R, real &dq, real &q6)
{
    dq = q_R - q_L;
    q6 = 6.0*q_0 - 3.0*(q_L + q_R);

    if ((q_R - q_0)*(q_0 - q_L) <= 0.0)
    {
        q_L = q_R = q_0;
        dq = 0.0;
        q6 = 0.0;
        return;
    }

    if ( dq*q6 > dq*dq)
    {
        q_L = 3.0*q_0 - 2.0*q_R;
        dq = q_R - q_L;
        q6 = 6.0*q_0 - 3.0*(q_L + q_R);
    }

    if (-dq*q6 > dq*dq)
    {
        q_R = 3.0*q_0 - 2.0*q_L;
        dq = q_R - q_L;
        q6 = 6.0*q_0 - 3.0*(q_L + q_R);
    }
}

// integrate the PPM profile over the right-side upwind domain of dependence
__device__ __forceinline__
static real _ppm_state_R (real q_R, real dq, real q6, real cfl)
{ return q_R - 0.5*cfl*(dq - (1.0 - 2.0*cfl/3.0)*q6); }

// integrate the PPM profile over the left-side upwind domain of dependence
__device__ __forceinline__
static real _ppm_state_L (real q_L, real dq, real q6, real cfl)
{ return q_L + 0.5*cfl*(dq + (1.0 - 2.0*cfl/3.0)*q6); }

// produce the time-averaged upwind face state for the Riemann solver
__device__ __forceinline__
static real _ppm_face_value (const real *face, const real *cell_val,
    int idx_up_L, int idx_up_R, bool upwind_on_left, real cfl)
{
    real val_L = face[idx_up_L];
    real val_R = face[idx_up_R];
    real dval, coeff_curv;

    _ppm_limit(cell_val[idx_up_L], val_L, val_R, dval, coeff_curv);

    return upwind_on_left ?
        _ppm_state_R(val_R, dval, coeff_curv, cfl) :
        _ppm_state_L(val_L, dval, coeff_curv, cfl) ;
}

// =========================================================================================================================
// invariant-domain correction limiting

// find local extrema from a cell and its periodic neighbors (for x direction)
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

// find local extrema from a cell and its nonperiodic neighbors (for y and z directions)
__device__ __forceinline__
static void _local_bounds (const real *value, int idx, int count, real &value_min, real &value_max)
{
    int idx_min = (idx > 0) ? idx - 1 : idx;
    int idx_max = (idx < count - 1) ? idx + 1 : idx;

    value_min = value[idx_min];
    value_max = value[idx_min];
    for (int idx_bound = idx_min + 1; idx_bound <= idx_max; idx_bound++)
    {
        value_min = fmin(value_min, value[idx_bound]);
        value_max = fmax(value_max, value[idx_bound]);
    }

    real bound_pad = 1.0e-12*fmax(1.0, fmax(fabs(value_min), fabs(value_max)));
    value_min -= bound_pad;
    value_max += bound_pad;
}

// reduce a correction scale to preserve one positivity or boundedness constraint
__device__ __forceinline__
static void _restrict_scale (real available, real change, real &scale)
{
    if (change >= 0.0) return;
    scale = (available > 0.0) ? fmin(scale, (1.0 - 1.0e-12)*available / (-change)) : 0.0;
}

// find one correction scale that preserves positive density and locally bounded velocities
__device__ __forceinline__
static real _invariant_scale (
    real rhod, real mx, real my, real mz,
    real corr_rhod, real corr_mx, real corr_my, real corr_mz,
    real lx_min, real lx_max,
    real vy_min, real vy_max,
    real lz_min, real lz_max)
{
    real scale = 1.0;

    _restrict_scale(rhod,                 corr_rhod,                      scale);
    _restrict_scale(mx - lx_min*rhod, corr_mx - lx_min*corr_rhod, scale);
    _restrict_scale(lx_max*rhod - mx, lx_max*corr_rhod - corr_mx, scale);
    _restrict_scale(my - vy_min*rhod, corr_my - vy_min*corr_rhod, scale);
    _restrict_scale(vy_max*rhod - my, vy_max*corr_rhod - corr_my, scale);
    _restrict_scale(mz - lz_min*rhod, corr_mz - lz_min*corr_rhod, scale);
    _restrict_scale(lz_max*rhod - mz, lz_max*corr_rhod - corr_mz, scale);

    return fmax(0.0, fmin(1.0, scale));
}

// =========================================================================================================================
// pressureless Riemann flux

// compute conservative dust density and momentum fluxes with the pressureless HLL solver
__device__ __forceinline__
static void _pressureless_hll_flux (
    real speed_L, real speed_R,
    real rhod_L, real lx_L, real vy_L, real lz_L,
    real rhod_R, real lx_R, real vy_R, real lz_R,
    real &flux_rhod, real &flux_mx, real &flux_my, real &flux_mz)
{
    real mx_L = rhod_L*lx_L;
    real my_L = rhod_L*vy_L;
    real mz_L = rhod_L*lz_L;
    real mx_R = rhod_R*lx_R;
    real my_R = rhod_R*vy_R;
    real mz_R = rhod_R*lz_R;

    if (speed_L >= 0.0 && speed_R >= 0.0)
    {
        flux_rhod = speed_L*rhod_L;
        flux_mx = speed_L*mx_L;
        flux_my = speed_L*my_L;
        flux_mz = speed_L*mz_L;
        return;
    }

    if (speed_L <= 0.0 && speed_R <= 0.0)
    {
        flux_rhod = speed_R*rhod_R;
        flux_mx = speed_R*mx_R;
        flux_my = speed_R*my_R;
        flux_mz = speed_R*mz_R;
        return;
    }

    real wave_L   = fmin(speed_L, speed_R);
    real wave_R   = fmax(speed_L, speed_R);
    real inv_span = 1.0 / (wave_R - wave_L);

    flux_rhod = (wave_R*speed_L*rhod_L - wave_L*speed_R*rhod_R + wave_L*wave_R*(rhod_R - rhod_L))*inv_span;
    flux_mx = (wave_R*speed_L*mx_L - wave_L*speed_R*mx_R + wave_L*wave_R*(mx_R - mx_L))*inv_span;
    flux_my = (wave_R*speed_L*my_L - wave_L*speed_R*my_R + wave_L*wave_R*(my_R - my_L))*inv_span;
    flux_mz = (wave_R*speed_L*mz_L - wave_L*speed_R*mz_R + wave_L*wave_R*(mz_R - mz_L))*inv_span;
}

// =========================================================================================================================

#endif
