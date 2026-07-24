#ifndef ADVECTION_CUH
#define ADVECTION_CUH

#include <const.cuh>

// =========================================================================================================================
// dust state recovery

// recover primitive velocities from conserved momenta, with a Keplerian fallback in near-vacuum cells
__device__ __forceinline__
void _recover_dust_state (real dens, real R, real &mx, real &my, real &mz, real &lx, real &vy, real &lz)
{
    if (dens < RHO_VAC)
    {
        lx = sqrt(G*M_S*fmax(R, 0.0));
        vy = 0.0;
        lz = 0.0;

        mx = dens*lx;
        my = 0.0;
        mz = 0.0;
    }
    else
    {
        lx = mx / dens;
        vy = my / dens;
        lz = mz / dens;
    }
}

// =========================================================================================================================
// PPM face reconstruction

// reconstruct a bounded PPM face value on a uniform grid from four neighboring cells (for x direction)
__device__ __forceinline__
static real _ppm_face_uniform (real qm1, real q0, real qp1, real qp2)
{
    real qe = (7.0*(q0 + qp1) - (qm1 + qp2)) / 12.0;
    real lo = fmin(q0, qp1);
    real hi = fmax(q0, qp1);
    return fmax(lo, fmin(hi, qe));
}

// reconstruct bounded face values on a nonuniform grid using precomputed PPM weights (for y and z directions)
__device__ __forceinline__
static void _ppm_faces_nonuniform (const real *cell_val, const real *face_weight, real *face, int cell_count)
{
    face[0] = cell_val[0];

    for (int iface = 1; iface < cell_count; iface++)
    {
        const real *weight = face_weight + 4*iface;
        if (iface >= 2 && iface <= cell_count - 2)
        {
            face[iface]  = weight[0]*cell_val[iface - 2] + weight[1]*cell_val[iface - 1];
            face[iface] += weight[2]*cell_val[iface]     + weight[3]*cell_val[iface + 1];
        }
        else
        {
            face[iface]  = weight[0]*cell_val[iface - 1] + weight[1]*cell_val[iface];
        }

        real face_min = fmin(cell_val[iface - 1], cell_val[iface]);
        real face_max = fmax(cell_val[iface - 1], cell_val[iface]);

        face[iface] = fmax(face_min, fmin(face_max, face[iface]));
    }

    face[cell_count] = cell_val[cell_count - 1];
}

// =========================================================================================================================
// PPM profile limiting and upwind tracing

// limit a cell parabolic profile to prevent new extrema and oscillations
__device__ __forceinline__
static void _ppm_limit (real q0, real &ql, real &qr, real &dq, real &q6)
{
    dq = qr - ql;
    q6 = 6.0*q0 - 3.0*(ql + qr);

    if ((qr - q0)*(q0 - ql) <= 0.0)
    {
        ql = qr = q0;
        dq = 0.0;
        q6 = 0.0;
        return;
    }

    if ( dq*q6 > dq*dq)
    {
        ql = 3.0*q0 - 2.0*qr;
        dq = qr - ql;
        q6 = 6.0*q0 - 3.0*(ql + qr);
    }

    if (-dq*q6 > dq*dq)
    {
        qr = 3.0*q0 - 2.0*ql;
        dq = qr - ql;
        q6 = 6.0*q0 - 3.0*(ql + qr);
    }
}

// integrate the PPM profile over the right-side upwind domain of dependence
__device__ __forceinline__
static real _ppm_state_R (real qb, real dq, real q6, real cfl)
{ return qb - 0.5*cfl*(dq - (1.0 - 2.0*cfl/3.0)*q6); }

// integrate the PPM profile over the left-side upwind domain of dependence
__device__ __forceinline__
static real _ppm_state_L (real qa, real dq, real q6, real cfl)
{ return qa + 0.5*cfl*(dq + (1.0 - 2.0*cfl/3.0)*q6); }

// produce the time-averaged upwind face state for the Riemann solver
__device__ __forceinline__
static real _ppm_face_value (const real *face, const real *cell_val, int iupL, int iupR, bool upwind_on_left, real cfl)
{
    real val_L = face[iupL];
    real val_R = face[iupR];
    real dval, coeff_curv;

    _ppm_limit(cell_val[iupL], val_L, val_R, dval, coeff_curv);

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
    for (int ii = idx_min + 1; ii <= idx_max; ii++)
    {
        value_min = fmin(value_min, value[ii]);
        value_max = fmax(value_max, value[ii]);
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
    real dens, real mx, real my, real mz,
    real corr_dens, real corr_mx, real corr_my, real corr_mz,
    real lx_min, real lx_max,
    real vy_min, real vy_max,
    real lz_min, real lz_max)
{
    real scale = 1.0;

    _restrict_scale(dens,                 corr_dens,                      scale);
    _restrict_scale(mx - lx_min*dens, corr_mx - lx_min*corr_dens, scale);
    _restrict_scale(lx_max*dens - mx, lx_max*corr_dens - corr_mx, scale);
    _restrict_scale(my - vy_min*dens, corr_my - vy_min*corr_dens, scale);
    _restrict_scale(vy_max*dens - my, vy_max*corr_dens - corr_my, scale);
    _restrict_scale(mz - lz_min*dens, corr_mz - lz_min*corr_dens, scale);
    _restrict_scale(lz_max*dens - mz, lz_max*corr_dens - corr_mz, scale);

    return fmax(0.0, fmin(1.0, scale));
}

// =========================================================================================================================
// pressureless Riemann flux

// compute conservative dust density and momentum fluxes with the pressureless HLL solver
__device__ __forceinline__
static void _pressureless_hll_flux (
    real speed_L, real speed_R,
    real dens_L, real lx_L, real vy_L, real lz_L,
    real dens_R, real lx_R, real vy_R, real lz_R,
    real &flux_dens, real &flux_mx, real &flux_my, real &flux_mz)
{
    real mx_L = dens_L*lx_L;
    real my_L = dens_L*vy_L;
    real mz_L = dens_L*lz_L;
    real mx_R = dens_R*lx_R;
    real my_R = dens_R*vy_R;
    real mz_R = dens_R*lz_R;

    if (speed_L >= 0.0 && speed_R >= 0.0)
    {
        flux_dens = speed_L*dens_L;
        flux_mx = speed_L*mx_L;
        flux_my = speed_L*my_L;
        flux_mz = speed_L*mz_L;
        return;
    }

    if (speed_L <= 0.0 && speed_R <= 0.0)
    {
        flux_dens = speed_R*dens_R;
        flux_mx = speed_R*mx_R;
        flux_my = speed_R*my_R;
        flux_mz = speed_R*mz_R;
        return;
    }

    real wave_L   = fmin(speed_L, speed_R);
    real wave_R   = fmax(speed_L, speed_R);
    real inv_span = 1.0 / (wave_R - wave_L);

    flux_dens = (wave_R*speed_L*dens_L - wave_L*speed_R*dens_R + wave_L*wave_R*(dens_R - dens_L))*inv_span;
    flux_mx = (wave_R*speed_L*mx_L - wave_L*speed_R*mx_R + wave_L*wave_R*(mx_R - mx_L))*inv_span;
    flux_my = (wave_R*speed_L*my_L - wave_L*speed_R*my_R + wave_L*wave_R*(my_R - my_L))*inv_span;
    flux_mz = (wave_R*speed_L*mz_L - wave_L*speed_R*mz_R + wave_L*wave_R*(mz_R - mz_L))*inv_span;
}

// =========================================================================================================================

#endif
