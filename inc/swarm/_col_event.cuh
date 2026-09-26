#ifndef GAMEDEV_SWARM_COL_EVENT_CUH
#define GAMEDEV_SWARM_COL_EVENT_CUH

// collision outcome sampling and per-category event bookkeeping for the collision chain

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_types.cuh>

__host__ __device__ inline void _record_event_work (event_work &work, int category, real log_mass)
{
    ++work.count[category];
    work.log_mass[category] += log_mass;
}
// group G identical tiny sticking projectiles into one event so each packet adds at most 0.01% target mass
__host__ __device__ inline real _sticking_packet (real q, bool fragmentation)
{
    return !fragmentation && q > 0.0 && q <= 1.0e-06
        ? fmax(1.0, floor(1.0e-04 / q)) : 1.0;
}
// projectile-to-target grain mass ratio q for compact grains of equal material density
__host__ __device__ inline real _sticking_mass_ratio (real size_i, real size_j)
{
    real ratio = size_j / size_i;
    return ratio*ratio*ratio;
}
// packets preserve the frozen-state mean mass growth but inflate its variance
// lower the 1e-4 packet bound if distribution comparisons show a bias

// return the sampled-to-physical rate factor and conditional absolute log-diameter jump moments
// erosion superposes grouped remnant transitions and ungrouped debris transitions
__host__ __device__ inline real _erosion_outcome_moments (real si, real sj,
    bool high_speed, real &mean, real &second, real &maximum)
{
    real q = _sticking_mass_ratio(si, sj);
    real G = _sticking_packet(q, false);
    // low-speed sticking: one packet of G projectiles at rate lambda/G
    if (!high_speed)
    {
        mean = log1p(G*q) / 3.0;
        second = mean*mean;
        maximum = mean;
        return 1.0 / G;
    }
    // high-speed erosion of a much larger target: remnant loses G*q of its mass or the owner becomes debris
    if (q <= 0.1)
    {
        real remnant = (1.0 - q) / G;
        real debris = q;
        real factor = remnant + debris;
        real jr = -log1p(-G*q) / 3.0;
        real jd = -log(q) / 3.0;
        mean = (remnant*jr + debris*jd) / factor;
        second = (remnant*jr*jr + debris*jd*jd) / factor;
        maximum = fmax(jr, jd);
        return factor;
    }
    // catastrophic fragmentation draws diameter [sqrt(s_min)+U*(sqrt(si)-sqrt(s_min))]^2
    // with L=log(si/s_min)/2, the moments integrate -2*log(y) over y in [exp(-L),1]
    real L = 0.5*log(si / INIT_SMIN);
    if (L < 1.0e-03)
    {
        // use series to avoid cancellation when the target is near the monomer floor
        mean = L - L*L / 6.0 + L*L*L*L / 360.0;
        second = L*L*(4.0/3.0 - L / 3.0 + L*L / 90.0 + L*L*L / 180.0);
    }
    else
    {
        real tail = exp(-L) / (-expm1(-L));
        mean = 2.0*(1.0 - L*tail);
        second = 8.0 - (4.0*L*L + 8.0*L)*tail;
    }
    maximum = 2.0*L;
    return 1.0;
}

// sample one outcome and return the new owner diameter; categories are four sticking q bins,
// fragmentation, remnant erosion, and debris erosion
// u is used only for high-speed events, and the caller supplies one independent draw
__host__ __device__ inline real _sample_erosion_outcome (real si, real sj,
    bool high_speed, real u, int &category, real &log_mass)
{
    real q = _sticking_mass_ratio(si, sj);
    real G = _sticking_packet(q, false);
    if (!high_speed)
    {
        category = q <= 1.0e-06 ? 0 : q <= 1.0e-04 ? 1 : q <= 1.0e-02 ? 2 : 3;
        log_mass = log1p(G*q);
        return cbrt(si*si*si + G*sj*sj*sj);
    }
    if (q <= 0.1)
    {
        real factor = (1.0 - q) / G + q;
        if (u < q / factor)
        {
            category = 6;
            log_mass = log(q);
            return sj;
        }
        category = 5;
        log_mass = log1p(-G*q);
        return si*cbrt(1.0 - G*q);
    }
    category = 4;
    real lower = sqrt(INIT_SMIN);
    real root = lower + u*(sqrt(si) - lower);
    real size = fmin(si, fmax(INIT_SMIN, root*root));
    log_mass = 3.0*log(size / si);
    return size;
}

#endif // COLLISION && !BERNOULLI

#endif // GAMEDEV_SWARM_COL_EVENT_CUH
