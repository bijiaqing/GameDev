#ifndef ADVECTION_CUH
#define ADVECTION_CUH

#include <const.cuh>

__device__ __forceinline__
void _recover_dust_state (real dens, real R, real &momx, real &momy, real &momz, real &velx, real &vely, real &velz)
{
    if (dens < RHO_VAC)
    {
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

__device__ __forceinline__
static real _ppm_edge (real qm1, real q0, real qp1, real qp2)
{
    real qe = (7.0*(q0 + qp1) - (qm1 + qp2)) / 12.0;
    real lo = fmin(q0, qp1);
    real hi = fmax(q0, qp1);
    return fmax(lo, fmin(hi, qe));
}








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






__device__ __forceinline__
static real _ppm_state_R (real qb, real dq, real q6, real cfl)
{
    return qb - 0.5*cfl*(dq - (1.0 - 2.0*cfl/3.0)*q6);
}






__device__ __forceinline__
static real _ppm_state_L (real qa, real dq, real q6, real cfl)
{
    return qa + 0.5*cfl*(dq + (1.0 - 2.0*cfl/3.0)*q6);
}











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



#endif
