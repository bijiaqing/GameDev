// frozen-bath collision chain: active-owner dispatch, local refresh durations, compact continuations, and cached rates
#ifndef GAMEDEV_SWARM_COL_CHAIN_CUH
#define GAMEDEV_SWARM_COL_CHAIN_CUH

#if defined(COLLISION) && !defined(BERNOULLI)

#include <algorithm>  // std::max, std::min
#include <climits>    // INT_MAX
#include <cmath>      // std::abs, std::isfinite, std::log, std::sqrt
#include <cstddef>    // std::size_t
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <vector>     // std::vector

#include <_col_cache.cuh>
#include <_collision.cuh>

// owner-position gas coefficients reused by every pair evaluated for that owner
// Z height, omega Keplerian frequency, vn pressure-drift speed, radial and strat Stokes scalings,
// cs sound speed, re_inv_sqrt inverse square-root Reynolds number, vg_sq turbulent velocity scale squared
struct query_environment
{
    real Z, omega, vn, radial, strat, cs, re_inv_sqrt, vg_sq;
};
#if !defined(IMPORTGAS) && !defined(CODE_UNIT) && !defined(CONST_ST)
// analytic physical-unit gas allows the environment to be cached once per collision half-operator
#define COL_QUERY_ENV_CACHE
// positions and gas are fixed throughout one collision half-operator
__device__ __forceinline__ query_environment _cache_query_environment (const swarm &p)
{
    real R = _get_cyl_R(p.position.y, p.position.z);
    real Z = _get_cyl_Z(p.position.y, p.position.z), h = _get_hg(R);
    real omega = _get_omegaK(R), cs = _get_cs(R, h), alpha = _get_alpha(R, h);
    real strat = _get_gas_strat(R, Z, h);
    return {Z, omega, -_get_eta(R, Z, h)*R*omega, pow(R / R_0, IDX_P), strat, cs,
        _get_re_inv_sqrt(R, alpha, _get_sigma_g(R)*strat), 1.5*alpha*cs*cs};
}
// reproduce _get_stokes for analytic gas from the cached radial and vertical scalings
__device__ __forceinline__ real _cached_stokes (const query_environment &e, real size)
{
    real st = STOKES_0*(size / S_0);
    st /= e.radial;
    st /= e.strat;
    return st;
}
// reproduce the _get_vrel_t regime algebra with precomputed gas coefficients
__device__ __forceinline__ real _cached_turbulence (const query_environment &e, real stokes_i, real stokes_j)
{
    real re_inv_sqrt = e.re_inv_sqrt, vg_sq = e.vg_sq;
    real stokes_large, stokes_small, eps;

    if (stokes_i >= stokes_j)
    {
        stokes_large = stokes_i;
        stokes_small = stokes_j;
    }
    else
    {
        stokes_large = stokes_j;
        stokes_small = stokes_i;
    }

    eps = stokes_small / stokes_large;

    // y_a = t_star / t_stop = 1.6 is the solution to y_star when St << 1
    // y_s is an empirical polynomial fit to the exact solution of y_star (eq. 21d)
    real y_a = 1.6;
    real y_s = 1.6015125;

    // taken from DustPy
    y_s += -0.63119577*stokes_large;
    y_s +=  0.32938936*stokes_large*stokes_large;
    y_s += -0.29847604*stokes_large*stokes_large*stokes_large;

    real vrel_sq = 0.0;

    if (stokes_large < 0.2*re_inv_sqrt)
    {
        // regime 1: very small particles (t_stop_large << t_small) following eq. 27

        vrel_sq = vg_sq*(stokes_large - stokes_small)*(stokes_large - stokes_small) / re_inv_sqrt;
    }
    else if (stokes_large < re_inv_sqrt / y_a)
    {
        // regime 2: transition near t_small boundary (t_stop_large ~ t_small) following eq. 26

        vrel_sq = vg_sq*(stokes_large - stokes_small) / (stokes_large + stokes_small);
        vrel_sq *= (stokes_large / (1.0 + re_inv_sqrt / stokes_large) - stokes_small / (1.0 + re_inv_sqrt
            / stokes_small));
    }
    else if (stokes_large < 5.0*re_inv_sqrt)
    {
        // regime 3: intermediate coupling (t_small < t_stop_large < 5*t_small)

        real coeff = 0.0;
        // coefficient of delta_VI^2  following eq. 17
        coeff  = (stokes_large - stokes_small) / (stokes_large + stokes_small);
        coeff *= (stokes_large / (1.0 + y_a) - stokes_small*stokes_small / (stokes_small + y_a*stokes_large));
        // coefficient of delta_VII^2 following eq. 18
        coeff += 2.0*(y_a*stokes_large - re_inv_sqrt) + stokes_large / (1.0 + y_a);
        coeff -= stokes_large*stokes_large / (stokes_large + re_inv_sqrt);
        coeff += stokes_small*stokes_small / (y_a*stokes_large + stokes_small);
        coeff -= stokes_small*stokes_small / (stokes_small + re_inv_sqrt);

        vrel_sq = vg_sq*coeff;
    }
    else if (stokes_large < 0.2)
    {
        // regime 4: fully intermediate regime (5t_small < t_stop_large < 0.2t_large) following eq. 28

        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_a - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_a) + eps*eps*eps / (y_a + eps)));
    }
    else if (stokes_large < 1.0)
    {
        // regime 5: transition near t_large boundary (0.2t_large < t_stop_large < t_large)
        // following eq. 28, but uses the empirical y_s fit instead of the fixed y_a = 1.6

        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_s - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_s) + eps*eps*eps / (y_s + eps)));
    }
    else
    {
        // regime 6: heavy particles (t_stop_large >= t_large) following eq. 29

        vrel_sq = vg_sq*(1.0 / (1.0 + stokes_large) + 1.0 / (1.0 + stokes_small));
    }

    if (vrel_sq < 0.0)
    {
        printf("ERROR: negative vrel_sq in _get_vrel_t\n");
        assert(false);
    }

    return sqrt(vrel_sq);
}


// reproduce _get_vrel_pair with drift, settling, turbulence, and Brownian terms from the cached environment
__device__ __forceinline__ real _cached_pair_velocity (const query_environment &e, real size_i, real size_j)
{
    real si = _cached_stokes(e, size_i), sj = _cached_stokes(e, size_j);
    real fi = 1.0 / (1.0 + si*si), fj = 1.0 / (1.0 + sj*sj);
    real dvr = 2.0*e.vn*(si*fi - sj*fj), dvphi = e.vn*(fi - fj);
    real dvz = e.Z*e.omega*(fmin(si, 0.5) - fmin(sj, 0.5));
    real vt = _cached_turbulence(e, si, sj);
    real mi = _get_grain_mass(size_i), mj = _get_grain_mass(size_j);
    real vb = fmin(sqrt(8.0*e.cs*e.cs*M_MOL*(mi + mj) / (M_PI*mi*mj)), e.cs);
    return sqrt(dvr*dvr + dvphi*dvphi + dvz*dvz + vt*vt + vb*vb);
}
#endif // COL_QUERY_ENV_CACHE
// bath-start total rate, first and second log-size jump-rate moments, and largest single jump for one owner
// a negative rate marks an invalid start state that must take the full chain path
struct cached_rate_moments { real rate, first, second, maximum; };
#include <algorithm> // std::max, std::min, std::min_element
#include <cmath>     // std::abs, std::isfinite, std::log, std::sqrt, M_PI
#include <stdexcept> // std::runtime_error

// requested refresh duration and its binding constraint: 0 horizon, 1 mean change, 2 fluctuation
struct change_bound { double duration; int reason; };

// A is the mass-weighted mean absolute log-size jump rate
// B is the mass-weighted second log-size jump moment rate, not the variance of the mean
// frozen rates give a mean accumulated absolute change h*A and a compound-Poisson fluctuation scale sqrt(h*B);
// bound each by epsilon
inline change_bound change_limit (double A, double B, double epsilon, double horizon)
{
    if (!std::isfinite(A) || A < 0 || !std::isfinite(B) || B < 0
        || !std::isfinite(epsilon) || !(epsilon > 0)
        || !std::isfinite(horizon) || !(horizon > 0))
        throw std::runtime_error("invalid change-based refresh inputs");
    change_bound result{horizon, 0};
    if (A > 0 && epsilon / A < result.duration) result = {epsilon / A, 1};
    if (B > 0 && epsilon*epsilon / B < result.duration) result = {epsilon*epsilon / B, 2};
    if (!(result.duration > 0)) throw std::runtime_error("change-based timestep underflow");
    return result;
}
// per-owner event counts and log-mass changes by outcome category, recorded only with COL_DIAGNOSTICS
constexpr int EVENT_CATEGORIES = 7;
struct event_work { unsigned long long count[EVENT_CATEGORIES]; real log_mass[EVENT_CATEGORIES]; };
__host__ __device__ inline void _record_event_work (event_work &work, int category, real log_mass)
{
    ++work.count[category];
    work.log_mass[category] += log_mass;
}
// group G identical tiny sticking projectiles into one event so each packet adds at most 0.01% target mass
__host__ __device__ inline real _sticking_packet (real q, bool fragmentation)
{
    return !fragmentation && q > 0.0 && q <= 1.e-6
        ? fmax(1.0, floor(1.e-4 / q)) : 1.0;
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
        real remnant = (1.0 - q) / G, debris = q, factor = remnant + debris;
        real jr = -log1p(-G*q) / 3.0, jd = -log(q) / 3.0;
        mean = (remnant*jr + debris*jd) / factor;
        second = (remnant*jr*jr + debris*jd*jd) / factor;
        maximum = fmax(jr, jd);
        return factor;
    }
    // catastrophic fragmentation draws diameter [sqrt(s_min)+U*(sqrt(si)-sqrt(s_min))]^2
    // with L=log(si/s_min)/2, the moments integrate -2*log(y) over y in [exp(-L),1]
    real L = 0.5*log(si / INIT_SMIN);
    if (L < 1.e-3)
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
        category = q <= 1.e-6 ? 0 : q <= 1.e-4 ? 1 : q <= 1.e-2 ? 2 : 3;
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
#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

// retain mass-weighted bath-start rate moments for one merged controller bin
struct col_rate_bin
{
    real mass;
    real weighted_rate;
    real weighted_change;
    real weighted_second;
    int owner_count;
    int invalid_count;
};

struct col_audit_accum
{
    real mass;
    real predicted_f;
    real predicted_e;
    real predicted_var_f;
    real predicted_var_e;
    real predicted_g;
    real predicted_var_g;
    real maximum_weight;
    real maximum_g_jump;
    real touched;
    real event_weight;
    real growth;
    real end_mass;
    int owner_count;
    int invalid_count;
};

struct col_bath_state
{
    real limit_scale = 1.0;
    int activity_streak = 0;
    int quiet_streak = 0;
    int floor_streak = 0;
};

struct col_bath_result
{
    real max_f = 0.0;
    real max_e = 0.0;
    real max_touched = 0.0;
    real max_events = 0.0;
    real max_g = 0.0;
    real max_g_upper = 0.0;
    real d_bath = 0.0;
    bool activity_overshoot = false;
    bool distribution_overshoot = false;
    bool persistent_overshoot = false;
};

struct col_bath_record
{
    int group_index = -1;
    int operator_index = 0;
    int bath_index = 0;
    int merged_bins = 0;
    int continuation_launches = 0;
    real duration = 0.0;
    real limit_before = 1.0;
    real limit_after = 1.0;
    col_bath_result result;
};

struct col_controller_summary
{
    int operator_count = 0;
    int bath_count = 0;
    int wave_count = 0;
    int continuation_launches = 0;
    int activity_overshoots = 0;
    int distribution_overshoots = 0;
    int persistent_overshoots = 0;
    real minimum_duration = std::numeric_limits<real>::infinity();
    real maximum_duration = 0.0;
    real minimum_limit_scale = 1.0;
    real maximum_f = 0.0;
    real maximum_e = 0.0;
    real maximum_touched = 0.0;
    real maximum_events = 0.0;
    real maximum_g = 0.0;
    real maximum_g_upper = 0.0;
    real maximum_d_bath = 0.0;
    std::vector<col_bath_record> baths;
};

// moving log-size bin bounds per spatial group
// scratch extrema are reset globally; retained bounds change only for refreshed groups
constexpr int moving_groups = ((N_X > 1) ? COL_BIN_X : 1)*COL_BIN_Y*((N_Z > 1) ? COL_BIN_Z : 1);
static __device__ unsigned long long moving_min[moving_groups], moving_max[moving_groups];
static __device__ real moving_lower[moving_groups], moving_upper[moving_groups];

static __global__ void col_size_zero ()
{
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g < moving_groups)
    {
        moving_min[g] = __double_as_longlong(INFINITY);
        moving_max[g] = 0;
    }
}
static __global__ void col_size_scan (const int *ids, int count, const swarm *particles,
    const int *spatial, const unsigned char *active)
{
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if (slot >= count) return;
    int i = ids[slot];
    real size = particles[i].par_size;
    if (!active[i] || !(size > 0) || !isfinite(size)) return;
    // positive doubles have the same ordering as their unsigned bit patterns
    auto bits = static_cast<unsigned long long>(__double_as_longlong(size));
    atomicMin(&moving_min[spatial[i]], bits);
    atomicMax(&moving_max[spatial[i]], bits);
}
// widen the observed size range to [0.5*s_min, 8*s_max] so the next interval's growth stays inside the bins
static __global__ void col_size_bnds ()
{
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g < moving_groups && moving_max[g] != 0)
    {
        moving_lower[g] = 0.5*__longlong_as_double(moving_min[g]);
        moving_upper[g] = 8.0*__longlong_as_double(moving_max[g]);
    }
}
// map a grain diameter to its logarithmic controller bin inside one spatial group, clamping outliers
__device__ __forceinline__ int _get_col_sizebin (real size, int group)
{
    if (!isfinite(size) || !(size > 0)) return 0;
    real fraction = log(size / moving_lower[group]) / log(moving_upper[group] / moving_lower[group]);
    return max(0, min(COL_BIN_S - 1, static_cast<int>(floor(fraction*COL_BIN_S))));
}

// draw a uniform deviate strictly below one so -log(U) and inverse-CDF selection stay finite
__device__ __forceinline__
real _get_col_uniform (curs *rngstate)
{
    #ifdef GAMEDEV_CUDA
    return fmin(curand_uniform_double(rngstate), nextafter(1.0, 0.0));
    #else  // GAMEDEV_ROCM
    return fmin(hiprand_uniform_double(rngstate), nextafter(1.0, 0.0));
    #endif // GAMEDEV_CUDA
}

// atomically raise a double to value with a compare-and-swap loop
__device__ __forceinline__
void _col_atomic_max (real *address, real value)
{
    auto integer = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *integer;
    while (value > __longlong_as_double(static_cast<long long>(old)))
    {
        unsigned long long assumed = old;
        old = atomicCAS(integer, assumed,
            static_cast<unsigned long long>(__double_as_longlong(value)));
        if (old == assumed) break;
    }
}

// evaluate one pair with the same synthetic or physical normalization as _get_col_rate_ij
template <kernel_type kernel> __device__ __forceinline__
real _get_col_chain_rate (const swarm *dev_particle, real size_i,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    int idx_old_i, int idx_old_j, int image_j, real lambda_0, real &vrel,
    const query_environment *environment)
{
    vrel = 0.0;
    real size_j = dev_size_old[idx_old_j];
    // include the owner's own swarm when i == j, using the large-number approximation N_i - 1 ~= N_i
    real numr_j = dev_numr_old[idx_old_j];
    if constexpr (kernel == CONSTANT_KERNEL)
    {
        return lambda_0*numr_j;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        return lambda_0*numr_j*(_get_grain_mass(size_i) + _get_grain_mass(size_j));
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        return lambda_0*numr_j*_get_grain_mass(size_i)*_get_grain_mass(size_j);
    }
    else if constexpr (kernel == CUSTOM_KERNEL)
    {
        #ifdef COL_QUERY_ENV_CACHE
        vrel = _cached_pair_velocity(environment[idx_old_i], size_i, size_j);
        #else  // !COL_QUERY_ENV_CACHE
        vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j, image_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        #endif // COL_QUERY_ENV_CACHE
        real rate = numr_j*vrel*M_PI*(size_i + size_j)*(size_i + size_j) / 4.0;
        if constexpr (N_Z == 1)
        {
            real R_i = _get_cyl_R(
                dev_particle[idx_old_i].position.y, dev_particle[idx_old_i].position.z
            );
            real R_j = _get_cyl_R(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            );
            real H_gi = R_i*_get_hg(R_i);
            real H_gj = R_j*_get_hg(R_j);
            rate /= sqrt(2.0*M_PI*(H_gi*H_gi + H_gj*H_gj));
        }
        return rate;
    }
    else
    {
        assert(false);
        return 0.0;
    }
}

// freeze the partner reservoir and reset continuation state for one bath
__global__
void col_bath_init (const int *owner_ids, int owner_count, real *dev_size_old, real *dev_numr_old, real *dev_col_time,
    int *dev_col_events, unsigned char *dev_col_complete, const swarm *dev_particle)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    dev_size_old[idx] = dev_particle[idx].par_size;
    dev_numr_old[idx] = dev_particle[idx].par_numr;
    dev_col_time[idx] = 0.0;
    dev_col_events[idx] = 0;
    dev_col_complete[idx] = 0;
}

// assign fixed geometry bins used by every bath controller audit
__global__
void col_space_bin (int *dev_col_spatial, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;
    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    int bin_x = 0;
    int bin_y = static_cast<int>(floor((y - Y_MIN) / (Y_MAX - Y_MIN)*COL_BIN_Y));
    int bin_z = 0;
    if constexpr (N_X > 1)
        bin_x = static_cast<int>(floor((x - X_MIN) / (X_MAX - X_MIN)*COL_BIN_X));
    if constexpr (N_Z > 1)
    {
        real extent_z = Z_MAX - Z_MIN;
        bin_z = static_cast<int>(floor((z - Z_MIN) / extent_z*COL_BIN_Z));
    }
    bin_x = (bin_x < 0) ? 0 : ((bin_x >= COL_BIN_X) ? COL_BIN_X - 1 : bin_x);
    bin_y = (bin_y < 0) ? 0 : ((bin_y >= COL_BIN_Y) ? COL_BIN_Y - 1 : bin_y);
    bin_z = (bin_z < 0) ? 0 : ((bin_z >= COL_BIN_Z) ? COL_BIN_Z - 1 : bin_z);
    int count_x = (N_X > 1) ? COL_BIN_X : 1;
    int count_y = COL_BIN_Y;
    dev_col_spatial[idx] = bin_x + count_x*(bin_y + count_y*bin_z);
}

// calculate bath-start rates used by the pre-bath duration controller
__global__
void col_bath_rate (const int *owner_ids, int owner_count, real *dev_col_rate, real *change_rate, real *second_rate,
    const swarm *dev_particle, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const query_environment *environment, cached_rate_moments *cached)
{
    int slot = blockIdx.x;
    if (slot >= owner_count) return;
    int idx_old_i = owner_ids[slot];

    // evaluate every cached pair against the frozen reservoir, one neighbor slot per thread
    __shared__ real rate_work[N_K];
    __shared__ real change_work[N_K], second_work[N_K], maximum_work[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        rate_work[idx_neighbor] = 0.0;
        change_work[idx_neighbor] = second_work[idx_neighbor] = maximum_work[idx_neighbor] = 0.0;
        int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
        if (dev_col_active[idx_old_i] == 0 || neighbor < 0
            || !(dev_col_measure[idx_old_i] > 0.0)) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);

        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];
        real vrel = 0.0;
        real pair_rate = _get_col_chain_rate <static_cast<kernel_type>(COAG_KERNEL)> (
            dev_particle, size_i, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment
        ) / dev_col_measure[idx_old_i];
        real mean = 0, second = 0, maximum = 0;
        pair_rate *= _erosion_outcome_moments(size_i, size_j, vrel >= V_FRAG, mean, second, maximum);
        maximum_work[idx_neighbor] = pair_rate > 0.0 ? maximum : 0.0;
        change_work[idx_neighbor] = pair_rate*mean;
        second_work[idx_neighbor] = pair_rate*second;
        rate_work[idx_neighbor] = pair_rate;
    }
    #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
    // MI300A fast path: fold 256 slots to 64 in shared memory, then reduce one 64-lane wavefront by shuffles
    if constexpr(N_K == 256 && COL_BATH_TPB == 128)
    {
        const int t = threadIdx.x;
        __shared__ unsigned char validity[128];
        __syncthreads();
        validity[t] = isfinite(rate_work[t]) && rate_work[t] >= 0.0 && isfinite(rate_work[t + 128]) && rate_work[t
            + 128] >= 0.0;
        rate_work[t] += rate_work[t + 128];
        change_work[t] += change_work[t + 128];
        second_work[t] += second_work[t + 128];
        maximum_work[t] = fmax(maximum_work[t], maximum_work[t + 128]);
        __syncthreads();
        if (t < 64)
        {
          real r = rate_work[t] + rate_work[t + 64], a = change_work[t] + change_work[t + 64];
          real b = second_work[t] + second_work[t + 64], m = fmax(maximum_work[t], maximum_work[t + 64]);
          int valid = validity[t] && validity[t + 64];
          for (int delta = 32; delta; delta /= 2)
          {
            r += __shfl_down(r, delta, 64);
            a += __shfl_down(a, delta, 64);
            b += __shfl_down(b, delta, 64);
            m = fmax(m, __shfl_down(m, delta, 64));
            int other = __shfl_down(valid, delta, 64);
            valid = valid && other;
          }
          if (t == 0){cached[idx_old_i] = {valid ? r : -1.0, a, b,
              m};dev_col_rate[idx_old_i] = r;change_rate[idx_old_i] = a;second_rate[idx_old_i] = b;}
        }
        return;
    }
    else
    #endif // GAMEDEV_ROCM && __gfx942__
    {
    // reduce the rate moments cooperatively for general N_K and block widths
    __shared__ unsigned char valid_work[N_K];
    for (int j = threadIdx.x; j < N_K; j += blockDim.x)
    {
        valid_work[j] = isfinite(rate_work[j]) && rate_work[j] >= 0.0;
    }
    __syncthreads();

    // fold the upper half into the lower half, including odd active counts
    for (int count = N_K; count > 1; count = (count + 1) / 2)
    {
        int half = (count + 1) / 2;
        for (int j = threadIdx.x; j < count / 2; j += blockDim.x)
        {
            rate_work[j] += rate_work[j + half];
            change_work[j] += change_work[j + half];
            second_work[j] += second_work[j + half];
            maximum_work[j] = fmax(maximum_work[j], maximum_work[j + half]);
            valid_work[j] = valid_work[j] && valid_work[j + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        real rate = rate_work[0], change = change_work[0], second = second_work[0];
        cached[idx_old_i] = {valid_work[0] ? rate : -1.0, change, second, maximum_work[0]};
        dev_col_rate[idx_old_i] = rate;
        change_rate[idx_old_i] = change;
        second_rate[idx_old_i] = second;
    }
    }
}

// count occupied size bins before merging statistically undersampled tails
__global__
void col_count_bin (const int *owner_ids, int owner_count, int *dev_col_count, const swarm *dev_particle,
    const int *dev_col_spatial, const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    atomicAdd(dev_col_count + idx_raw, 1);
}

// accumulate mass-weighted rates for the pre-bath duration bound
__global__
void col_rate_bins (const int *owner_ids, int owner_count, col_rate_bin *dev_col_bin, const swarm *dev_particle,
    const real *dev_col_rate, const real *change_rate, const real *second_rate, const int *dev_col_spatial,
    const int *dev_col_binmap,
    const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    int idx_bin = dev_col_binmap[idx_raw];
    real weight = dev_particle[idx].par_numr*_get_grain_mass(dev_particle[idx].par_size);
    atomicAdd(&dev_col_bin[idx_bin].owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0)
        || !isfinite(dev_col_rate[idx]) || dev_col_rate[idx] < 0.0
        || !isfinite(change_rate[idx]) || change_rate[idx] < 0
        || !isfinite(second_rate[idx]) || second_rate[idx] < 0)
    {
        atomicAdd(&dev_col_bin[idx_bin].invalid_count, 1);
        return;
    }
    atomicAdd(&dev_col_bin[idx_bin].mass, weight);
    atomicAdd(&dev_col_bin[idx_bin].weighted_rate, weight*dev_col_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_change, weight*change_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_second, weight*second_rate[idx]);
}

// evaluate the cached no-event path with one thread per owner
// complete owners whose first waiting time spans the interval and queue only the rest for col_chain_run
__global__ void col_skip_scan (const int *ids, int count, curs *rng,
    const unsigned char *active, const real *measure, const int *spatial, const real *steps,
    const cached_rate_moments *cached, real *time, int *events, unsigned char *complete,
    real *hazard, real *jump1, real *jump2, real *jumpmax, int *queue, int *queued)
{
    int slot = blockIdx.x*blockDim.x + threadIdx.x;
    if (slot >= count) return;
    int i = ids[slot];
    real end = steps[spatial[i]];
    if (!active[i] || !(measure[i] > 0.0))
    {
        time[i] = end;
        complete[i] = 1;
        return;
    }
    if (complete[i]) return;
    if (time[i] == 0.0 && events[i] == 0)
    {
        const auto c = cached[i];
        if (isfinite(c.rate) && c.rate >= 0.0)
        {
            if (c.rate == 0.0)
            {
                time[i] = end;
                complete[i] = 1;
                return;
            }
            curs state = rng[i];
            real wait = -log(_get_col_uniform(&state)) / c.rate;
            real remaining = end - time[i];
            if (isfinite(wait) && wait > 0.0 && wait >= remaining)
            {
                hazard[i] += c.rate*remaining;
                jump1[i] += c.first*remaining;
                jump2[i] += c.second*remaining;
                jumpmax[i] = fmax(jumpmax[i], c.maximum);
                time[i] = end;
                complete[i] = 1;
                rng[i] = state;
                return;
            }
            // leave the trial RNG draw uncommitted because the full chain repeats the same draw
        }
    }
    queue[atomicAdd(queued, 1)] = i;
}


// evolve every owner against one frozen reservoir with bounded continuation
//
// parallelization: one block per queued owner; threads evaluate neighbor pair rates and thread 0 advances the chain
//
// per loop iteration:
//   1 recompute all owner-dependent pair rates and jump moments from the current owner size
//   2 reduce the total rate, jump moments, and last positive slot
//   3 draw one waiting time; stop at the bath end or apply one sampled event and update owner size and number
// the loop stops after COL_EVENT_CAP accepted events; unfinished owners are appended to the next continuation queue
__global__
void col_chain_run (const int *owner_ids, int owner_count, swarm *dev_particle, curs *dev_rngstate, int *dev_col_error,
    int *dev_col_unfinished, real *dev_col_time, int *dev_col_events,
    unsigned char *dev_col_complete, real *dev_col_hazard,
    real *dev_col_jump1_int, real *dev_col_jump2_int, real *dev_col_jumpmax_int,
    const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const real *group_step, const int *spatial,
    int *unfinished_ids, int *error_flag, event_work *work,
    const query_environment *environment, const cached_rate_moments *cached)
{
    int slot = blockIdx.x;
    if (slot >= owner_count) return;
    int idx_old_i = owner_ids[slot];
    real bath_end = group_step[spatial[idx_old_i]];

    __shared__ real pair_rate[N_K];
    __shared__ real pair_jump1[N_K];
    __shared__ real pair_jump2[N_K];
    __shared__ real pair_jumpmax[N_K];
    __shared__ real size_i;
    __shared__ real numr_i;
    __shared__ real time_i;
    __shared__ real hazard_i;
    __shared__ real jump1_i;
    __shared__ real jump2_i;
    __shared__ real jumpmax_i;
    #ifdef GAMEDEV_CUDA
    __shared__ curs rngstate;
    #else  // GAMEDEV_ROCM
    curs rngstate; // keep the HIP RNG object local because shared objects cannot be initialized
    #endif // GAMEDEV_CUDA
    __shared__ int event_count;
    __shared__ int accepted;
    #ifdef COL_DIAGNOSTICS
    __shared__ event_work event_stats;
    #endif // COL_DIAGNOSTICS
    __shared__ bool keep_running;

    if (threadIdx.x == 0)
    {
        size_i = dev_particle[idx_old_i].par_size;
        numr_i = dev_particle[idx_old_i].par_numr;
        time_i = dev_col_time[idx_old_i];
        hazard_i = dev_col_hazard[idx_old_i];
        jump1_i = dev_col_jump1_int[idx_old_i];
        jump2_i = dev_col_jump2_int[idx_old_i];
        jumpmax_i = dev_col_jumpmax_int[idx_old_i];
        rngstate = dev_rngstate[idx_old_i];
        event_count = dev_col_events[idx_old_i];
        accepted = 0;
        #ifdef COL_DIAGNOSTICS
        event_stats = work[idx_old_i];
        #endif // COL_DIAGNOSTICS
        keep_running = dev_col_complete[idx_old_i] == 0;
        if (dev_col_active[idx_old_i] == 0 || !(dev_col_measure[idx_old_i] > 0.0))
        {
            time_i = bath_end;
            dev_col_complete[idx_old_i] = 1;
            keep_running = false;
        }
    }
    __syncthreads();

    // use the bath-start cache only before the first event of this bath, because it matches the published reservoir
    // after an event or in a continuation the owner has changed and every rate must be recomputed
    if (threadIdx.x == 0 && keep_running && time_i == 0.0 && event_count == 0)
    {
        const auto c = cached[idx_old_i];
        if (isfinite(c.rate) && c.rate >= 0.0)
        {
            if (c.rate == 0.0)
            {
                time_i = bath_end;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            else
            {
                curs before = rngstate;
                real wait = -log(_get_col_uniform(&rngstate)) / c.rate;
                real remaining = bath_end - time_i;
                if (isfinite(wait) && wait > 0.0 && wait >= remaining)
                {
                    hazard_i += c.rate*remaining;
                    jump1_i += c.first*remaining;
                    jump2_i += c.second*remaining;
                    jumpmax_i = fmax(jumpmax_i, c.maximum);
                    time_i = bath_end;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else
                {
                    // restore the draw so the full path handles events and invalid waits with the same deviate
                    rngstate = before;
                }
            }
        }
    }
    __syncthreads();

    while (true)
    {
        if (threadIdx.x == 0 && keep_running && accepted >= COL_EVENT_CAP)
        {
            keep_running = false;
        }
        __syncthreads();
        if (!keep_running) break;

        for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
        {
            int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
            pair_rate[idx_neighbor] = 0.0;
            pair_jump1[idx_neighbor] = 0.0;
            pair_jump2[idx_neighbor] = 0.0;
            pair_jumpmax[idx_neighbor] = 0.0;
            if (neighbor >= 0)
            {
                int idx_old_j = _get_col_idx_old(neighbor);
                int image_j = _get_col_image(neighbor);
                real size_j = dev_size_old[idx_old_j];
                real vrel = 0.0;
                real pair_value = _get_col_chain_rate <static_cast<kernel_type>(COAG_KERNEL)> (
                    dev_particle, size_i, dev_size_old, dev_numr_old,
                    #ifdef IMPORTGAS
                    dev_gas_dens,
                    #endif // IMPORTGAS
                    idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment
                ) / dev_col_measure[idx_old_i];
                real mean = 0.0, second = 0.0, maximum = 0.0;
                pair_value *= _erosion_outcome_moments(size_i, size_j, vrel >= V_FRAG, mean, second, maximum);
                pair_rate[idx_neighbor] = pair_value;
                pair_jump1[idx_neighbor] = pair_value*mean;
                pair_jump2[idx_neighbor] = pair_value*second;
                pair_jumpmax[idx_neighbor] = (pair_value > 0.0) ? maximum : 0.0;
            }
        }
        constexpr bool wave_reduction_shape = N_K == 256 && COL_BATH_TPB == 128;
        // keep the generic allocation for other shapes; the shape choice is compile-time
        #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
        constexpr int total_slots = wave_reduction_shape ? 128 : N_K;
        #else  // !(GAMEDEV_ROCM && __gfx942__)
        constexpr int total_slots = N_K;
        #endif // GAMEDEV_ROCM && __gfx942__
        __shared__ real total_work[total_slots];
        __shared__ int last_work[total_slots];
        #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
        // MI300A fast path for N_K=256 and 128 threads, as in col_bath_rate
        if constexpr(wave_reduction_shape)
        {
          const int t = threadIdx.x;
          __syncthreads();
          real left = pair_rate[t], right = pair_rate[t + 128];
          total_work[t] = left + right;
          last_work[t] = (!isfinite(left) || left < 0.0 || !isfinite(right) || right < 0.0)
              ? -2 : (right > 0.0 ? t + 128 : (left > 0.0 ? t : -1));
          pair_jump1[t] += pair_jump1[t + 128];
          pair_jump2[t] += pair_jump2[t + 128];
          pair_jumpmax[t] = fmax(pair_jumpmax[t], pair_jumpmax[t + 128]);
          __syncthreads();
          if (t < 64)
          {
            real r = total_work[t] + total_work[t + 64], a = pair_jump1[t] + pair_jump1[t + 64];
            real b = pair_jump2[t] + pair_jump2[t + 64], m = fmax(pair_jumpmax[t], pair_jumpmax[t + 64]);
            int l = last_work[t], other = last_work[t + 64];
            l = (l == -2 || other == -2) ? -2 : max(l, other);
            for (int delta = 32; delta; delta /= 2)
            {
              r += __shfl_down(r, delta, 64);
              a += __shfl_down(a, delta, 64);
              b += __shfl_down(b, delta, 64);
              m = fmax(m, __shfl_down(m, delta, 64));
              other = __shfl_down(l, delta, 64);
              l = (l == -2 || other == -2) ? -2 : max(l, other);
            }
            if (t == 0)
            {
                total_work[0] = r;
                pair_jump1[0] = a;
                pair_jump2[0] = b;
                pair_jumpmax[0] = m;
                last_work[0] = l;
            }
          }
        }
        else
        #endif // GAMEDEV_ROCM && __gfx942__
        {
        // reduce into separate arrays so pair_rate stays intact for serial partner sampling
        for (int j = threadIdx.x; j < N_K; j += blockDim.x)
        {
            total_work[j] = pair_rate[j];
            // -2 propagates invalid input; -1 denotes no positive contribution
            last_work[j] = (!isfinite(pair_rate[j]) || pair_rate[j] < 0.0)
                ? -2 : (pair_rate[j] > 0.0 ? j : -1);
        }
        __syncthreads();
        for (int count = N_K; count > 1; count = (count + 1) / 2)
        {
            int half = (count + 1) / 2;
            for (int j = threadIdx.x; j < count / 2; j += blockDim.x)
            {
                total_work[j] += total_work[j + half];
                pair_jump1[j] += pair_jump1[j + half];
                pair_jump2[j] += pair_jump2[j + half];
                pair_jumpmax[j] = fmax(pair_jumpmax[j], pair_jumpmax[j + half]);
                int left = last_work[j], right = last_work[j + half];
                last_work[j] = (left == -2 || right == -2) ? -2
                    : (left > right ? left : right);
            }
            __syncthreads();
        }

        }

        if (threadIdx.x == 0)
        {
            real total_rate = total_work[0];
            real total_jump1 = pair_jump1[0];
            real total_jump2 = pair_jump2[0];
            real total_jumpmax = pair_jumpmax[0];
            int last_positive = last_work[0];
            if (last_positive == -2)
            {
                dev_col_error[idx_old_i] = 1;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            if (keep_running && (!isfinite(total_rate) || total_rate < 0.0))
            {
                dev_col_error[idx_old_i] = 2;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            if (keep_running && total_rate == 0.0)
            {
                time_i = bath_end;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            if (keep_running)
            {
                real wait = -log(_get_col_uniform(&rngstate)) / total_rate;
                real remaining = bath_end - time_i;
                if (!isfinite(wait) || !(wait > 0.0))
                {
                    dev_col_error[idx_old_i] = 3;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else if (wait >= remaining)
                {
                    hazard_i += total_rate*remaining;
                    jump1_i += total_jump1*remaining;
                    jump2_i += total_jump2*remaining;
                    jumpmax_i = fmax(jumpmax_i, total_jumpmax);
                    time_i = bath_end;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else
                {
                    real event_time = time_i + wait;
                    if (!(event_time > time_i))
                    {
                        dev_col_error[idx_old_i] = 4;
                        dev_col_complete[idx_old_i] = 1;
                        keep_running = false;
                    }
                    else
                    {
                        hazard_i += total_rate*wait;
                        jump1_i += total_jump1*wait;
                        jump2_i += total_jump2*wait;
                        jumpmax_i = fmax(jumpmax_i, total_jumpmax);
                        // select the partner by walking cumulative pair rates, falling back to the last positive slot
                        // on roundoff
                        real target = _get_col_uniform(&rngstate)*total_rate;
                        real cumulative = 0.0;
                        int idx_slot = -1;
                        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
                        {
                            if (pair_rate[idx_neighbor] <= 0.0) continue;
                            cumulative += pair_rate[idx_neighbor];
                            if (cumulative > target)
                            {
                                idx_slot = idx_neighbor;
                                break;
                            }
                        }
                        if (idx_slot < 0) idx_slot = last_positive;
                        if (idx_slot < 0)
                        {
                            dev_col_error[idx_old_i] = 5;
                            dev_col_complete[idx_old_i] = 1;
                            keep_running = false;
                        }
                        else
                        {
                            int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_slot)];
                            int idx_old_j = _get_col_idx_old(neighbor);
                            int image_j = _get_col_image(neighbor);
                            real size_j = dev_size_old[idx_old_j];
                            real vrel = 0.0;
                            if constexpr (COAG_KERNEL == CUSTOM_KERNEL)
                            {
                                #ifdef COL_QUERY_ENV_CACHE
                                vrel = _cached_pair_velocity(environment[idx_old_i], size_i, size_j);
                                #else  // !COL_QUERY_ENV_CACHE
                                vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j, image_j
                                    #ifdef IMPORTGAS
                                    , dev_gas_dens
                                    #endif // IMPORTGAS
                                );
                                #endif // COL_QUERY_ENV_CACHE
                            }
                            // apply the outcome while conserving represented mass through the grain count
                            real mass_before = numr_i*size_i*size_i*size_i;
                            bool high_speed = vrel >= V_FRAG;
                            real sample = high_speed ? _get_col_uniform(&rngstate) : 0.0;
                            int category;
                            real log_mass;
                            real size_new = _sample_erosion_outcome(size_i, size_j,
                                high_speed, sample, category, log_mass);
                            #ifdef COL_DIAGNOSTICS
                            _record_event_work(event_stats, category, log_mass);
                            #endif // COL_DIAGNOSTICS
                            numr_i = mass_before / (size_new*size_new*size_new);
                            size_i = size_new;
                            time_i = event_time;
                            event_count++;
                            accepted++;
                            real mass_after = numr_i*size_i*size_i*size_i;
                            real mass_scale = fmax(fabs(mass_before), fabs(mass_after));
                            if (!isfinite(size_i) || !isfinite(numr_i)
                                || !(size_i > 0.0) || !(numr_i > 0.0)
                                || fabs(mass_after - mass_before) > 2.0e-12*mass_scale)
                            {
                                dev_col_error[idx_old_i] = 6;
                                dev_col_complete[idx_old_i] = 1;
                                keep_running = false;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        #ifdef COL_DIAGNOSTICS
        work[idx_old_i] = event_stats;
        #endif // COL_DIAGNOSTICS
        dev_particle[idx_old_i].par_size = size_i;
        dev_particle[idx_old_i].par_numr = numr_i;
        dev_rngstate[idx_old_i] = rngstate;
        dev_col_time[idx_old_i] = time_i;
        dev_col_events[idx_old_i] = event_count;
        dev_col_hazard[idx_old_i] = hazard_i;
        dev_col_jump1_int[idx_old_i] = jump1_i;
        dev_col_jump2_int[idx_old_i] = jump2_i;
        dev_col_jumpmax_int[idx_old_i] = jumpmax_i;
        if (dev_col_error[idx_old_i]) atomicMax(error_flag, dev_col_error[idx_old_i]);
        if (dev_col_complete[idx_old_i] == 0)
        {
            unfinished_ids[atomicAdd(dev_col_unfinished, 1)] = idx_old_i;
        }
    }
}

// aggregate predicted moments and realized changes for post-bath validation
__global__
void col_audit_bin (const int *owner_ids, int owner_count, col_audit_accum *dev_col_bin, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const real *dev_col_rate,
    const real *dev_col_hazard, const real *dev_col_jump1_int,
    const real *dev_col_jump2_int, const real *dev_col_jumpmax_int,
    const int *dev_col_events, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active, const real *group_step)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_spatial = dev_col_spatial[idx];
    real duration = group_step[idx_spatial];
    int idx_start_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_size_old[idx], idx_spatial);
    int idx_end_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    int idx_start = dev_col_binmap[idx_start_raw];
    int idx_end = dev_col_binmap[idx_end_raw];
    real weight = dev_numr_old[idx]*_get_grain_mass(dev_size_old[idx]);
    real probability = -expm1(-dev_col_rate[idx]*duration);
    real expected_events = dev_col_hazard[idx];
    real events = static_cast<real>(dev_col_events[idx]);
    col_audit_accum *bin = dev_col_bin + idx_start;
    atomicAdd(&bin->owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0) || !isfinite(probability)
        || !isfinite(expected_events) || expected_events < 0.0 || !isfinite(events)
        || !isfinite(dev_col_jump1_int[idx]) || dev_col_jump1_int[idx] < 0.0
        || !isfinite(dev_col_jump2_int[idx]) || dev_col_jump2_int[idx] < 0.0
        || !isfinite(dev_col_jumpmax_int[idx]) || dev_col_jumpmax_int[idx] < 0.0
        || !isfinite(dev_particle[idx].par_size) || !(dev_particle[idx].par_size > 0.0))
    {
        atomicAdd(&bin->invalid_count, 1);
        return;
    }
    atomicAdd(&bin->mass, weight);
    atomicAdd(&bin->predicted_f, weight*probability);
    atomicAdd(&bin->predicted_e, weight*expected_events);
    atomicAdd(&bin->predicted_var_f, weight*weight*probability*(1.0 - probability));
    atomicAdd(&bin->predicted_var_e, weight*weight*expected_events);
    atomicAdd(&bin->predicted_g, weight*dev_col_jump1_int[idx]);
    atomicAdd(&bin->predicted_var_g, weight*weight*dev_col_jump2_int[idx]);
    _col_atomic_max(&bin->maximum_weight, weight);
    _col_atomic_max(&bin->maximum_g_jump, weight*dev_col_jumpmax_int[idx]);
    if (dev_col_events[idx] > 0) atomicAdd(&bin->touched, weight);
    atomicAdd(&bin->event_weight, weight*events);
    atomicAdd(&bin->growth, weight*fabs(log(dev_particle[idx].par_size / dev_size_old[idx])));
    atomicAdd(&dev_col_bin[idx_end].end_mass, weight);
}

// count raw spatial and logarithmic-size controller bins
inline
int _get_col_raw_count ()
{
    int count_x = (N_X > 1) ? COL_BIN_X : 1;
    int count_z = (N_Z > 1) ? COL_BIN_Z : 1;
    return count_x*COL_BIN_Y*count_z*COL_BIN_S;
}

// merge adjacent sparse size bins independently inside every spatial bin
inline
int _build_col_binmap (const std::vector<int> &raw_count, std::vector<int> &raw_to_merged)
{
    int spatial_count = _get_col_raw_count() / COL_BIN_S;
    raw_to_merged.assign(raw_count.size(), -1);
    int merged_count = 0;
    for (int idx_spatial = 0; idx_spatial < spatial_count; idx_spatial++)
    {
        int last_occupied = -1;
        for (int idx_size = 0; idx_size < COL_BIN_S; idx_size++)
        {
            if (raw_count[idx_spatial*COL_BIN_S + idx_size] > 0) last_occupied = idx_size;
        }
        if (last_occupied < 0)
        {
            int idx_merged = merged_count++;
            for (int idx_size = 0; idx_size < COL_BIN_S; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
            continue;
        }

        int idx_begin = 0;
        int idx_previous = -1;
        while (idx_begin <= last_occupied)
        {
            int idx_end = idx_begin;
            int count = 0;
            while (idx_end <= last_occupied && count < COL_BIN_MIN)
            {
                count += raw_count[idx_spatial*COL_BIN_S + idx_end++];
            }
            bool merge_tail = idx_end > last_occupied && count < COL_BIN_MIN
                && idx_previous >= 0;
            int idx_merged = merge_tail ? idx_previous : merged_count++;
            for (int idx_size = idx_begin; idx_size < idx_end; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
            idx_previous = idx_merged;
            idx_begin = idx_end;
        }
        if (last_occupied + 1 < COL_BIN_S)
        {
            int idx_merged = merged_count++;
            for (int idx_size = last_occupied + 1; idx_size < COL_BIN_S; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
        }
    }
    return merged_count;
}

// choose a bath duration from the mean absolute log-size change and compound-Poisson variance constraints
inline
real _choose_col_bath (const std::vector<col_rate_bin> &bin, int bin_count,
    real remaining, real limit_scale, int *binding = nullptr)
{
    real duration = std::min(remaining, COL_BATH_MAX);
    real tolerance = COL_BATH_EPS*limit_scale;
    int reason = 0;
    for (int b = 0; b < bin_count; ++b)
    {
        const auto &v = bin[b];
        if (v.invalid_count) throw std::runtime_error("invalid change-based rate moments");
        if (!(v.mass > 0)) continue;
        auto bound = change_limit(v.weighted_change / v.mass, v.weighted_second / v.mass,
                                tolerance, duration);
        if (bound.duration < duration)
        {
            duration = bound.duration;
            reason = bound.reason;
        }
    }
    if (binding) *binding = reason;
    return duration;
}

// compare realized bath changes with concentration bounds and adapt the next limit
inline
col_bath_result _finish_col_bath (const std::vector<col_audit_accum> &bin,
    int bin_count, col_bath_state &state)
{
    col_bath_result result;
    int active_count = 0;
    real total_mass = 0.0;
    for (int idx_bin = 0; idx_bin < bin_count; idx_bin++)
    {
        const col_audit_accum &value = bin[idx_bin];
        if (value.invalid_count != 0)
        {
            throw std::runtime_error("collision bath controller returned an invalid audit state");
        }
        if (value.mass > 0.0) active_count++;
        total_mass += value.mass;
    }
    if (!(total_mass > 0.0))
    {
        throw std::runtime_error("collision bath controller found no active represented mass");
    }
    real confidence_log = log(2.0*static_cast<real>(active_count) / COL_BATH_ALPHA);
    real mass_difference = 0.0;
    for (int idx_bin = 0; idx_bin < bin_count; idx_bin++)
    {
        const col_audit_accum &value = bin[idx_bin];
        mass_difference += std::abs(value.end_mass - value.mass);
        if (!(value.mass > 0.0)) continue;
        real mass_sq = value.mass*value.mass;
        real predicted_f = value.predicted_f / value.mass;
        real predicted_e = value.predicted_e / value.mass;
        real predicted_g = value.predicted_g / value.mass;
        real touched = value.touched / value.mass;
        real events = value.event_weight / value.mass;
        real growth = value.growth / value.mass;
        real weight_max = value.maximum_weight / value.mass;
        real jump_max = value.maximum_g_jump / value.mass;
        real upper_f = std::min(1.0, predicted_f
            + std::sqrt(2.0*value.predicted_var_f / mass_sq*confidence_log)
            + weight_max*confidence_log / 3.0);
        real upper_e = predicted_e
            + std::sqrt(2.0*value.predicted_var_e / mass_sq*confidence_log)
            + weight_max*confidence_log / 3.0;
        real upper_g = predicted_g
            + std::sqrt(2.0*value.predicted_var_g / mass_sq*confidence_log)
            + jump_max*confidence_log / 3.0;
        result.max_f = std::max(result.max_f, predicted_f);
        result.max_e = std::max(result.max_e, predicted_e);
        result.max_touched = std::max(result.max_touched, touched);
        result.max_events = std::max(result.max_events, events);
        result.max_g = std::max(result.max_g, growth);
        result.max_g_upper = std::max(result.max_g_upper, upper_g);
        // independent weight sums can put a fully touched fraction a few ulps above one
        // keep the raw diagnostic, but enforce its physical ceiling in the overshoot test
        result.activity_overshoot = result.activity_overshoot
            || std::min(touched, real(1.0)) > upper_f || events > upper_e;
        result.distribution_overshoot = result.distribution_overshoot
            || growth > std::max(COL_BATH_EPS, upper_g);
    }
    result.d_bath = mass_difference / (2.0*total_mass);
    result.distribution_overshoot = result.distribution_overshoot
        || result.d_bath > COL_BATH_EPS;

    bool issue = result.activity_overshoot || result.distribution_overshoot;
    bool at_floor = state.limit_scale
        <= 0.25*(1.0 + 8.0*std::numeric_limits<real>::epsilon());
    if (issue)
    {
        state.activity_streak++;
        state.quiet_streak = 0;
        state.floor_streak = at_floor ? state.floor_streak + 1 : 0;
    }
    else
    {
        state.activity_streak = 0;
        state.quiet_streak++;
        state.floor_streak = 0;
    }
    if (result.distribution_overshoot || state.activity_streak >= 2)
    {
        state.limit_scale = std::max(0.25, 0.5*state.limit_scale);
        state.activity_streak = 0;
    }
    else if (state.quiet_streak >= 3)
    {
        state.limit_scale = std::min(1.0, 1.25*state.limit_scale);
        state.quiet_streak = 0;
    }
    result.persistent_overshoot = state.floor_streak >= 2;
    return result;
}

// retain the adaptive schedule and its strongest controller diagnostics
inline
void _record_col_bath (col_controller_summary &summary, const col_bath_record &record)
{
    const col_bath_result &result = record.result;
    summary.bath_count++;
    summary.continuation_launches += record.continuation_launches;
    summary.activity_overshoots += result.activity_overshoot ? 1 : 0;
    summary.distribution_overshoots += result.distribution_overshoot ? 1 : 0;
    summary.persistent_overshoots += result.persistent_overshoot ? 1 : 0;
    summary.minimum_duration = std::min(summary.minimum_duration, record.duration);
    summary.maximum_duration = std::max(summary.maximum_duration, record.duration);
    summary.minimum_limit_scale = std::min(summary.minimum_limit_scale, record.limit_after);
    summary.maximum_f = std::max(summary.maximum_f, result.max_f);
    summary.maximum_e = std::max(summary.maximum_e, result.max_e);
    summary.maximum_touched = std::max(summary.maximum_touched, result.max_touched);
    summary.maximum_events = std::max(summary.maximum_events, result.max_events);
    summary.maximum_g = std::max(summary.maximum_g, result.max_g);
    summary.maximum_g_upper = std::max(summary.maximum_g_upper, result.max_g_upper);
    summary.maximum_d_bath = std::max(summary.maximum_d_bath, result.d_bath);
    summary.baths.push_back(record);
}

// archive one output interval without mixing controller state into particle checkpoints
inline
bool save_col_controller (const std::string &file_name, const col_controller_summary &summary)
{
    std::ofstream file(file_name);
    if (!file) return false;
    real minimum_duration = std::isfinite(summary.minimum_duration)
        ? summary.minimum_duration : 0.0;
    file << std::setprecision(17);
    file << "{\n"
         << "  \"schema\": 2,\n"
         << "  \"operator_count\": " << summary.operator_count << ",\n"
         << "  \"bath_count\": " << summary.bath_count << ",\n"
         << "  \"wave_count\": " << summary.wave_count << ",\n"
         << "  \"continuation_launches\": " << summary.continuation_launches << ",\n"
         << "  \"activity_overshoots\": " << summary.activity_overshoots << ",\n"
         << "  \"distribution_overshoots\": " << summary.distribution_overshoots << ",\n"
         << "  \"persistent_overshoots\": " << summary.persistent_overshoots << ",\n"
         << "  \"minimum_duration\": " << minimum_duration << ",\n"
         << "  \"maximum_duration\": " << summary.maximum_duration << ",\n"
         << "  \"minimum_limit_scale\": " << summary.minimum_limit_scale << ",\n"
         << "  \"maximum_f\": " << summary.maximum_f << ",\n"
         << "  \"maximum_e\": " << summary.maximum_e << ",\n"
         << "  \"maximum_touched\": " << summary.maximum_touched << ",\n"
         << "  \"maximum_events\": " << summary.maximum_events << ",\n"
         << "  \"maximum_g\": " << summary.maximum_g << ",\n"
         << "  \"maximum_g_upper\": " << summary.maximum_g_upper << ",\n"
         << "  \"maximum_d_bath\": " << summary.maximum_d_bath << ",\n"
         << "  \"baths\": [\n";
    for (std::size_t idx = 0; idx < summary.baths.size(); idx++)
    {
        const col_bath_record &record = summary.baths[idx];
        const col_bath_result &result = record.result;
        file << "    {\"operator\": " << record.operator_index
             << ", \"group\": " << record.group_index
             << ", \"bath\": " << record.bath_index
             << ", \"merged_bins\": " << record.merged_bins
             << ", \"continuation_launches\": " << record.continuation_launches
             << ", \"duration\": " << record.duration
             << ", \"limit_before\": " << record.limit_before
             << ", \"limit_after\": " << record.limit_after
             << ", \"max_f\": " << result.max_f
             << ", \"max_e\": " << result.max_e
             << ", \"max_touched\": " << result.max_touched
             << ", \"max_events\": " << result.max_events
             << ", \"max_g\": " << result.max_g
             << ", \"max_g_upper\": " << result.max_g_upper
             << ", \"d_bath\": " << result.d_bath
             << ", \"activity_overshoot\": "
             << (result.activity_overshoot ? "true" : "false")
             << ", \"distribution_overshoot\": "
             << (result.distribution_overshoot ? "true" : "false")
             << ", \"persistent_overshoot\": "
             << (result.persistent_overshoot ? "true" : "false")
             << "}" << ((idx + 1 < summary.baths.size()) ? "," : "") << "\n";
    }
    file << "  ]\n}\n";
    return static_cast<bool>(file);
}

// local collision scheduling, workspace, and host orchestration
#include <algorithm> // std::max, std::min, std::min_element
#include <cmath>     // std::abs, std::isfinite, std::log, std::sqrt, M_PI
#include <cstdint>   // std::uint64_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <utility>   // std::pair
#include <vector>    // std::vector

// schedule spatial groups on power-of-two subdivisions of one collision operator horizon
// integer endpoints avoid rounding drift between power-of-two timestep levels
struct local_schedule
{
    double horizon;                              // physical duration of the collision operator
    std::vector<std::pair<int, int>> links;      // undirected group pairs joined by any KNN dependency
    std::vector<int> level;                      // group step is horizon/2^level
    std::vector<std::uint64_t> step, next;       // group step and next endpoint in integer ticks
    std::uint64_t end;                           // horizon in ticks of the finest level 2^52

    local_schedule (double h, const std::vector<double>& requested,
                   const std::vector<unsigned int>& edges)
        : horizon(h), level(requested.size(), 0), step(requested.size()),
          next(requested.size(), 0)
    {
        if (!(h > 0) || !std::isfinite(h))
        {
            throw std::runtime_error("invalid local collision horizon");
        }
        const int n = static_cast<int>(level.size()), words = (n + 31) / 32;
        // choose the coarsest power-of-two level not exceeding each requested duration
        for (int c = 0; c < n; ++c)
        {
            if (!(requested[c] > 0) || !std::isfinite(requested[c]))
            {
                throw std::runtime_error("invalid local collision timestep");
            }
            double dt = h;
            while (dt > requested[c])
            {
                if (++level[c] > 52)
                {
                    throw std::runtime_error("local collision timestep exceeds time resolution");
                }
                dt *= 0.5;
            }
        }
        // symmetrize the bit-packed dependency matrix into an edge list
        for (int c = 0; c < n; ++c)
        {
            for (int d = c + 1; d < n; ++d)
            {
                if ((edges[c*words + d / 32] & (1u << (d % 32))) ||
                    (edges[d*words + c / 32] & (1u << (c % 32))))
                {
                    links.emplace_back(c, d);
                }
            }
        }
        // enforce symmetric compatibility on every directed KNN dependency: step ratio <= 16
        bool changed;
        do
        {
            changed = false;
            for (int c = 0; c < n; ++c)
            {
                for (int d = 0; d < n; ++d)
                {
                    if (!(edges[c*words + d / 32] & (1u << (d % 32)))) continue;
                    if (level[c] < level[d] - 4)
                    {
                        level[c] = level[d] - 4;
                        changed = true;
                    }
                    if (level[d] < level[c] - 4)
                    {
                        level[d] = level[c] - 4;
                        changed = true;
                    }
                }
            }
        } while (changed);
        // a fixed lattice permits later refinement without moving pending endpoints
        int finest = 52;
        end = std::uint64_t(1) << finest;
        for (int c = 0; c < n; ++c) step[c] = std::uint64_t(1) << (finest - level[c]);
    }
    // earliest pending endpoint, which is the next scheduler tick
    std::uint64_t time () const { return *std::min_element(next.begin(), next.end()); }
    double seconds (std::uint64_t tick) const { return horizon*(double(tick) / double(end)); }
    // groups whose pending endpoint equals this tick
    std::vector<int> due (std::uint64_t tick) const
    {
        std::vector<int> out;
        for (int c = 0; c < int(next.size()); ++c) if (next[c] == tick) out.push_back(c);
        return out;
    }
    // change only due groups because other groups already hold computed endpoints
    void adapt (std::uint64_t tick, const std::vector<int>& groups,
               const std::vector<double>& requested, const std::vector<bool>& passed)
    {
        const int n = level.size();
        std::vector<bool> active(n, false);
        for (int c : groups) active[c] = true;
        // propagate the coarsest permitted level from immutable pending neighbors
        std::vector<int> cap(n, 52), candidate = level;
        for (int c = 0; c < n; ++c) if (!active[c]) cap[c] = level[c];
        bool changed;
        do
        {
            changed = false;
            for (const auto &edge : links)
            {
                int c = edge.first, d = edge.second;
                if (active[c] && cap[c] > cap[d] + 4)
                {
                    cap[c] = cap[d] + 4;
                    changed = true;
                }
                if (active[d] && cap[d] > cap[c] + 4)
                {
                    cap[d] = cap[c] + 4;
                    changed = true;
                }
            }
        } while (changed);
        for (int c : groups)
        {
            if (!(requested[c] > 0) || !std::isfinite(requested[c]))
            {
                throw std::runtime_error("invalid adaptive collision timestep");
            }
            int want = 0;
            double h = horizon;
            while (h > requested[c])
            {
                if (++want > 52) throw std::runtime_error("adaptive collision time resolution exceeded");
                h *= 0.5;
            }
            // recover directly after a passing audit; alignment and neighbors still constrain the level
            candidate[c] = passed[c] ? want : std::max(want, level[c]);
            while (tick % (std::uint64_t(1) << (52 - candidate[c]))) ++candidate[c];
            candidate[c] = std::min(candidate[c], cap[c]);
        }
        // refine due groups until all directed dependencies satisfy ratio <= 16
        // the caps above ensure this never requires changing an in-flight endpoint
        do
        {
            changed = false;
            for (const auto &edge : links)
            {
                int c = edge.first, d = edge.second;
                if (active[c] && candidate[c] < candidate[d] - 4)
                {
                    candidate[c] = candidate[d] - 4;
                    changed = true;
                }
                if (active[d] && candidate[d] < candidate[c] - 4)
                {
                    candidate[d] = candidate[c] - 4;
                    changed = true;
                }
            }
        } while (changed);
        for (int c : groups)
        {
            level[c] = candidate[c];
            step[c] = std::uint64_t(1) << (52 - level[c]);
        }
    }
    // move each due group to its next endpoint
    void advance (const std::vector<int>& groups)
    {
        for (int c : groups) next[c] += step[c];
    }
};
#include <chrono>  // std::chrono clocks and durations
#include <numeric> // std::iota

// map the host workflow's runtime calls to the selected backend
#ifdef GAMEDEV_CUDA
#define LOCAL_CHECK CUDA_CHECK
#define LOCAL_KERNEL CUDA_KERNEL_CHECK
#define localMalloc cudaMalloc
#define localFree cudaFree
#define localCopy cudaMemcpy
#define localZero cudaMemset
#define localH2D cudaMemcpyHostToDevice
#define localD2H cudaMemcpyDeviceToHost
#else  // !GAMEDEV_CUDA
#define LOCAL_CHECK HIP_CHECK
#define LOCAL_KERNEL HIP_KERNEL_CHECK
#define localMalloc hipMalloc
#define localFree hipFree
#define localCopy hipMemcpy
#define localZero hipMemset
#define localH2D hipMemcpyHostToDevice
#define localD2H hipMemcpyDeviceToHost
#endif // GAMEDEV_CUDA

// spatial controller groups and the 32-bit words of one bit-packed group dependency row
constexpr int LOCAL_GROUPS = ((N_X > 1) ? COL_BIN_X : 1)*COL_BIN_Y*((N_Z > 1) ? COL_BIN_Z : 1);
constexpr int LOCAL_WORDS = (LOCAL_GROUPS + 31) / 32;

#ifdef COL_QUERY_ENV_CACHE
// cache every active owner's gas environment; absorbed particles at y=0 are skipped
__global__ void col_env_cache (query_environment *env, const swarm *particle)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N_P && particle[i].position.y>0.0) env[i] = _cache_query_environment(particle[i]);
}

#endif // COL_QUERY_ENV_CACHE

// build the group dependency graph once per geometry epoch: edge c->d when an owner in c has a neighbor in d
// duplicate edges are reduced within each owner before issuing atomics, so no per-neighbor data reaches the host
__global__ void col_dep_graph (unsigned int *edges, const int *spatial,
    const int *neighbors, const unsigned char *active)
{
    #ifdef GAMEDEV_ROCM
    // assign 32 lanes to each owner and OR-reduce their neighbor masks by shuffles
    if constexpr(TPB % 32 == 0)
    {
    const int lane = threadIdx.x % 32;
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) / 32;
    if (i >= N_P || !active[i]) return;
    unsigned int bits[LOCAL_WORDS] = {};
    for (int k = lane; k < N_K; k += 32)
    {
        int entry = neighbors[_get_col_offset(i, k)];
        if (entry < 0)continue;
        int j = _get_col_idx_old(entry), c = spatial[j];
        bits[c / 32] |= 1u << (c % 32);
    }
    for (int w = 0; w < LOCAL_WORDS; ++w)
    {
        unsigned int value = bits[w];
        for (int delta = 16; delta; delta /= 2)value |= __shfl_down(value, delta, 32);
        if (lane == 0 && value)atomicOr(edges + spatial[i]*LOCAL_WORDS + w, value);
    }
    }
    else
    #endif // GAMEDEV_ROCM
    {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_P || !active[i]) return;
    unsigned int bits[LOCAL_WORDS] = {};
    for (int k = 0; k < N_K; ++k)
    {
        int entry = neighbors[_get_col_offset(i, k)];
        if (entry < 0) continue;
        int j = _get_col_idx_old(entry), c = spatial[j];
        bits[c / 32] |= 1u << (c % 32);
    }
    for (int w = 0; w < LOCAL_WORDS; ++w)
    {
        if (bits[w]) atomicOr(edges + spatial[i]*LOCAL_WORDS + w, bits[w]);
    }
    }
}

// clear the path-integrated compensators of owners entering a new bath
__global__ void col_comp_zero (const int *ids, int count, real *hazard,
    real *jump1, real *jump2, real *jumpmax)
{
    int slot = blockIdx.x*blockDim.x + threadIdx.x;
    if (slot >= count) return;
    int i = ids[slot];
    hazard[i] = jump1[i] = jump2[i] = jumpmax[i] = 0;
}

// reduce per-owner diagnostics once per collision half-step with one block per event category
__global__ void col_event_sum (const event_work *work, event_work *sum)
{
    const int k = blockIdx.x, t = threadIdx.x;
    __shared__ unsigned long long counts[TPB];
    __shared__ real growth[TPB];
    unsigned long long n = 0;
    real g = 0;
    for (int i = t; i < N_P; i += TPB)
    {
        n += work[i].count[k];
        g += work[i].log_mass[k];
    }
    counts[t] = n;
    growth[t] = g;
    __syncthreads();
    for (int stride = TPB / 2; stride > 0; stride /= 2)
    {
        if (t < stride)
        {
            counts[t] += counts[t + stride];
            growth[t] += growth[t + stride];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        sum->count[k] = counts[0];
        sum->log_mass[k] = growth[0];
    }
}
static_assert(TPB > 0 && (TPB & (TPB - 1)) == 0, "event reduction needs power-of-two TPB");

// per-group scheduler statistics written to the COL_DIAGNOSTICS JSONL record
struct local_group_stats
{
    std::uint64_t updates = 0;
    int overshoots = 0, persistent = 0;
    real max_age = 0, max_growth = 0, max_activity = 0, max_requested_ratio = 0;
};

// persistent device scratch and host scheduler state for the local collision workflow
// scratch allocations survive all operator calls; particle IDs retain their RNG identity even when
// continuation queues are reordered by atomic append
struct local_workspace
{
    query_environment *environment = nullptr;
    cached_rate_moments *cached = nullptr;
    event_work *work = nullptr, *work_sum = nullptr;
    int *ids = nullptr, *queue_a = nullptr, *queue_b = nullptr, *error = nullptr;
    real *dt = nullptr, *change_rate = nullptr, *second_rate = nullptr;
    unsigned int *graph = nullptr;
    std::vector<std::vector<int>> owners;
    std::vector<unsigned int> edges;
    std::vector<col_bath_state> state;
    #ifdef COL_DIAGNOSTICS
    std::ofstream log;
    std::uint64_t operator_id = 0;
    #endif // COL_DIAGNOSTICS
    local_workspace (const std::string &path): owners(LOCAL_GROUPS), edges(LOCAL_GROUPS*LOCAL_WORDS),
        state(LOCAL_GROUPS)
    {
        #ifdef COL_QUERY_ENV_CACHE
        LOCAL_CHECK(localMalloc((void**)&environment, sizeof(query_environment)*N_P));
        #endif // COL_QUERY_ENV_CACHE
        LOCAL_CHECK(localMalloc((void**)&cached, sizeof(cached_rate_moments)*N_P));
        #ifdef COL_DIAGNOSTICS
        LOCAL_CHECK(localMalloc((void**)&work, sizeof(event_work)*N_P));
        LOCAL_CHECK(localMalloc((void**)&work_sum, sizeof(event_work)));
        #endif // COL_DIAGNOSTICS
        LOCAL_CHECK(localMalloc((void**)&ids, sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_a, sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_b, sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&error, sizeof(int)));
        LOCAL_CHECK(localMalloc((void**)&dt, sizeof(real)*LOCAL_GROUPS));
        LOCAL_CHECK(localMalloc((void**)&change_rate, sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&second_rate, sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&graph, sizeof(unsigned int)*edges.size()));
        #ifdef COL_DIAGNOSTICS
        auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        log.open(path+"collision_local_"+std::to_string(stamp)+".jsonl");
        if (!log) throw std::runtime_error("cannot open local collision diagnostics");
        log << std::setprecision(17);
        #endif // COL_DIAGNOSTICS
    }
    ~local_workspace ()
    {
        if (environment) localFree(environment);
        localFree(cached);
        #ifdef COL_DIAGNOSTICS
        localFree(work);
        localFree(work_sum);
        #endif // COL_DIAGNOSTICS
        localFree(ids);
        localFree(queue_a);
        localFree(queue_b);
        localFree(error);
        localFree(dt);
        localFree(graph);
        localFree(change_rate);
        localFree(second_rate);
    }
};

// =====================================================================================================================
// host function: evolve_local_collisions
// purpose: advance all owners through one fixed-position collision operator with local power-of-two bath durations
//
// per call:
//   1 cache gas environments and, once per geometry epoch, the owner-group lists and dependency graph
//   2 publish every reservoir, compute bath-start rates and merged size bins, and request per-group durations
//   3 repeatedly advance the earliest due groups: republish and rerate them, adapt their levels,
//     screen no-event owners, run chain continuations until none remain, and audit the completed baths
// =====================================================================================================================
inline void evolve_local_collisions (
    local_workspace & local,
    bool & local_geometry_valid,
    real duration,
    real total_dust_mass,
    real & clock_dyn,
    real & dt_col,
    int & count_col,
    int col_raw_count,
    swarm *dev_particle,
    curs *dev_rngstate,
    int *dev_col_spatial,
    int *dev_col_neighbor,
    int *dev_col_events,
    int *dev_col_count,
    int *dev_col_binmap,
    int *dev_col_error,
    int *dev_col_unfinished,
    unsigned char*dev_col_active,
    unsigned char*dev_col_complete,
    real *dev_size_old,
    real *dev_numr_old,
    real *dev_col_time,
    real *dev_col_rate,
    real *dev_col_hazard,
    real *dev_col_jump1_int,
    real *dev_col_jump2_int,
    real *dev_col_jumpmax_int,
    real *dev_col_measure,
    col_rate_bin *dev_col_ratebin,
    col_audit_accum *dev_col_audit
#ifdef IMPORTGAS
    , const real *dev_gas_dens
#endif // IMPORTGAS
#ifdef COL_DIAGNOSTICS
    , real clock_sim, col_controller_summary &col_summary
#endif // COL_DIAGNOSTICS
)
{
// called by evolve_collisions after production geometry/cache construction
// all launches below use the default stream; no publication occurs while an event chain or its audit can
// still be reading the previous reservoir
#ifdef COL_DIAGNOSTICS
using local_clock = std::chrono::steady_clock;
const auto local_begin = local_clock::now();
#endif // COL_DIAGNOSTICS
#ifdef COL_QUERY_ENV_CACHE
col_env_cache <<< NB_P, TPB >>> (local.environment, dev_particle);
LOCAL_KERNEL("col_env_cache");
#endif // COL_QUERY_ENV_CACHE
#ifdef COL_DIAGNOSTICS
LOCAL_CHECK(localZero(local.work, 0, sizeof(event_work)*N_P));
#endif // COL_DIAGNOSTICS
// rebuild owner lists and the group dependency graph only after positions change
if (!local_geometry_valid)
{
    std::vector<int> spatial(N_P);
    LOCAL_CHECK(localCopy(spatial.data(), dev_col_spatial, sizeof(int)*N_P, localD2H));
    for (auto &v : local.owners) v.clear();
    for (int i = 0; i < N_P; ++i) local.owners[spatial[i]].push_back(i);
    LOCAL_CHECK(localZero(local.graph, 0, sizeof(unsigned int)*local.edges.size()));
    col_dep_graph <<<
        #ifdef GAMEDEV_ROCM
        (N_P*(TPB % 32 == 0 ? 32 : 1) + TPB - 1) / TPB, TPB
        #else  // !GAMEDEV_ROCM
        NB_P, TPB
        #endif // GAMEDEV_ROCM
    >>> (local.graph, dev_col_spatial, dev_col_neighbor, dev_col_active);
    LOCAL_KERNEL("col_dep_graph");
    LOCAL_CHECK(localCopy(local.edges.data(), local.graph,
        sizeof(unsigned int)*local.edges.size(), localD2H));
    local_geometry_valid = true;
}

std::vector<int> ids(N_P), counts(col_raw_count), binmap(col_raw_count);
std::iota(ids.begin(), ids.end(), 0);
LOCAL_CHECK(localCopy(local.ids, ids.data(), sizeof(int)*N_P, localH2D));
std::vector<col_rate_bin> ratebin(col_raw_count);
std::vector<col_audit_accum> audit(col_raw_count);
const real lambda0 = N_P / static_cast<real>(N_K) / total_dust_mass;

// publish the listed owners' current sizes and counts as the frozen reservoir and reset their bath clocks
auto initialize = [&](int count)
{
    #ifdef COL_PARTNER_REFRESH
    // campaign hook: only at refresh boundaries, before snapshots and cached rates
    COL_PARTNER_REFRESH(dev_col_neighbor, local.queue_a, count);
    #endif // COL_PARTNER_REFRESH
    int blocks = (count + TPB - 1) / TPB;
    col_bath_init <<< blocks, TPB >>> (local.ids, count, dev_size_old, dev_numr_old,
        dev_col_time, dev_col_events, dev_col_complete, dev_particle);
    LOCAL_KERNEL("local_publish_and_init");
    col_comp_zero <<< blocks, TPB >>> (local.ids, count, dev_col_hazard, dev_col_jump1_int,
        dev_col_jump2_int, dev_col_jumpmax_int);
    LOCAL_KERNEL("col_comp_zero");
};
// recompute bath-start rates, merge sparse size bins, and copy mass-weighted rate moments to the host
auto rates_and_bins = [&](int count)
{
    // keep each refreshed group's bounds unchanged until its collision interval and audit finish
    col_size_zero <<< (moving_groups + TPB - 1) / TPB, TPB >>> ();
    LOCAL_KERNEL("col_size_zero");
    col_size_scan <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_particle, dev_col_spatial,
        dev_col_active);
    LOCAL_KERNEL("col_size_scan");
    col_size_bnds <<< (moving_groups + TPB - 1) / TPB, TPB >>> ();
    LOCAL_KERNEL("col_size_bnds");
    #ifdef COL_PERF_VAL
    auto rate_start = col_perf_start();
    #endif // COL_PERF_VAL
    col_bath_rate <<< count, COL_BATH_TPB >>> (local.ids, count, dev_col_rate, local.change_rate, local.second_rate,
        dev_particle, dev_col_neighbor, dev_col_measure, dev_col_active,
        dev_size_old, dev_numr_old,
        #ifdef IMPORTGAS
        dev_gas_dens,
        #endif // IMPORTGAS
        lambda0, local.environment, local.cached);
    LOCAL_KERNEL("local_rates");
    LOCAL_CHECK(localZero(dev_col_count, 0, sizeof(int)*col_raw_count));
    col_count_bin <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_count,
        dev_particle, dev_col_spatial, dev_col_active);
    LOCAL_KERNEL("local_count");
    LOCAL_CHECK(localCopy(counts.data(), dev_col_count, sizeof(int)*col_raw_count, localD2H));
    int merged = _build_col_binmap(counts, binmap);
    LOCAL_CHECK(localCopy(dev_col_binmap, binmap.data(), sizeof(int)*col_raw_count, localH2D));
    LOCAL_CHECK(localZero(dev_col_ratebin, 0, sizeof(col_rate_bin)*col_raw_count));
    col_rate_bins <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_ratebin,
        dev_particle, dev_col_rate, local.change_rate, local.second_rate, dev_col_spatial, dev_col_binmap,
        dev_col_active);
    LOCAL_KERNEL("local_rate_bins");
    LOCAL_CHECK(localCopy(ratebin.data(), dev_col_ratebin, sizeof(col_rate_bin)*merged, localD2H));
    #ifdef COL_PERF_VAL
    col_perf_rate_ms += col_perf_stop(rate_start);
    #endif // COL_PERF_VAL
    return merged;
};
// merged bins of group c occupy the contiguous range [binmap[c*COL_BIN_S], bin_end(c))
auto bin_end = [&](int c, int merged)
{
    return c + 1 < LOCAL_GROUPS ? binmap[(c + 1)*COL_BIN_S] : merged;
};
auto requested_step = [&](int c, int merged, int *binding = nullptr)
{
    int first = binmap[c*COL_BIN_S], last = bin_end(c, merged);
    std::vector<col_rate_bin> slice(ratebin.begin() + first, ratebin.begin() + last);
    return _choose_col_bath(slice, int(slice.size()), duration, local.state[c].limit_scale, binding);
};

// start every group from the same published reservoir at tick zero
initialize(N_P);
int merged = rates_and_bins(N_P);
std::vector<double> requested(LOCAL_GROUPS);
#ifdef COL_DIAGNOSTICS
std::vector<int> binding(LOCAL_GROUPS);
for (int c = 0; c < LOCAL_GROUPS; ++c) requested[c] = requested_step(c, merged, &binding[c]);
#else  // !COL_DIAGNOSTICS
for (int c = 0; c < LOCAL_GROUPS; ++c) requested[c] = requested_step(c, merged);
#endif // COL_DIAGNOSTICS
local_schedule schedule(duration, requested, local.edges);
std::vector<real> steps(LOCAL_GROUPS);
#ifdef COL_DIAGNOSTICS
std::vector<real> published(LOCAL_GROUPS, 0);
#endif // COL_DIAGNOSTICS
for (int c = 0; c < LOCAL_GROUPS; ++c) steps[c] = schedule.seconds(schedule.step[c]);
LOCAL_CHECK(localCopy(local.dt, steps.data(), sizeof(real)*LOCAL_GROUPS, localH2D));
std::vector<bool> passed(LOCAL_GROUPS, true);
std::vector<double> current_request = requested;
#ifdef COL_DIAGNOSTICS
std::vector<local_group_stats> stats(LOCAL_GROUPS);
std::vector<int> initial_level = schedule.level;
std::vector<std::uint64_t> coarsened(LOCAL_GROUPS, 0), refined(LOCAL_GROUPS, 0),
    constrained(LOCAL_GROUPS, 0);
std::vector<double> min_request = requested, max_request = requested;
std::uint64_t owner_updates = 0, chain_blocks = 0, waves = 0, launches = 0;
double chain_seconds = 0, audit_seconds = 0;
col_summary.operator_count++;
int operator_index = col_summary.operator_count;
int batch_index = 0;

#endif // COL_DIAGNOSTICS

while (schedule.time() < schedule.end)
{
    auto tick = schedule.time();
    auto groups = schedule.due(tick);
    real time = schedule.seconds(tick);
    ids.clear();
    for (int c : groups)
    {
        #ifdef COL_DIAGNOSTICS
        published[c] = time;
        #endif // COL_DIAGNOSTICS
        ids.insert(ids.end(), local.owners[c].begin(), local.owners[c].end());
    }
    int count = static_cast<int>(ids.size());
    if (count == 0)
    {
        schedule.advance(groups);
        continue;
    }
    // at tick zero the complete population was already initialized and rated
    if (tick != 0)
    {
        LOCAL_CHECK(localCopy(local.ids, ids.data(), sizeof(int)*count, localH2D));
        initialize(count);
        merged = rates_and_bins(count);
    }
    if (tick != 0)
    {
        for (int c : groups)
        {
            current_request[c] = requested_step(c, merged);
            #ifdef COL_DIAGNOSTICS
            min_request[c] = std::min(min_request[c], current_request[c]);
            max_request[c] = std::max(max_request[c], current_request[c]);
            #endif // COL_DIAGNOSTICS
        }
        #ifdef COL_DIAGNOSTICS
        auto previous = schedule.level;
        #endif // COL_DIAGNOSTICS
        schedule.adapt(tick, groups, current_request, passed);
        for (int c : groups)
        {
            #ifdef COL_DIAGNOSTICS
            coarsened[c] += schedule.level[c] < previous[c];
            refined[c] += schedule.level[c] > previous[c];
            #endif // COL_DIAGNOSTICS
            steps[c] = schedule.seconds(schedule.step[c]);
            #ifdef COL_DIAGNOSTICS
            constrained[c] += steps[c] > current_request[c];
            #endif // COL_DIAGNOSTICS
        }
        LOCAL_CHECK(localCopy(local.dt, steps.data(), sizeof(real)*LOCAL_GROUPS, localH2D));
    }
    #ifdef COL_DIAGNOSTICS
    for (int c : groups)
    {
        if (local.owners[c].empty()) continue;
        for (int d = 0; d < LOCAL_GROUPS; ++d)
        {
            if (local.edges[c*LOCAL_WORDS + d / 32] & (1u << (d % 32)))
            {
                stats[c].max_age = std::max(stats[c].max_age, time - published[d]);
            }
        }
        stats[c].max_requested_ratio = std::max(stats[c].max_requested_ratio,
            steps[c] / requested_step(c, merged));
    }

    #endif // COL_DIAGNOSTICS
    #ifdef COL_PERF_VAL
    auto event_start = col_perf_start();
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    const auto chain_begin = local_clock::now();
    #endif // COL_DIAGNOSTICS
    // screen owners with no event in the interval, then relaunch the chain on the shrinking unfinished queue
    LOCAL_CHECK(localZero(local.error, 0, sizeof(int)));
    LOCAL_CHECK(localZero(dev_col_unfinished, 0, sizeof(int)));
    col_skip_scan <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_rngstate,
        dev_col_active, dev_col_measure, dev_col_spatial, local.dt, local.cached,
        dev_col_time, dev_col_events, dev_col_complete, dev_col_hazard,
        dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, local.queue_a, dev_col_unfinished);
    LOCAL_KERNEL("cached_screen");
    int unfinished = 0, continuations = 0;
    LOCAL_CHECK(localCopy(&unfinished, dev_col_unfinished, sizeof(int), localD2H));
    const int *input = local.queue_a;
    int *output = local.queue_b;
    while (unfinished > 0)
    {
        if (++continuations > 1000000)
        {
            throw std::runtime_error("local collision continuation limit exceeded");
        }
        LOCAL_CHECK(localZero(dev_col_unfinished, 0, sizeof(int)));
        #ifdef COL_DIAGNOSTICS
        chain_blocks += unfinished;
        #endif // COL_DIAGNOSTICS
        col_chain_run <<< unfinished, COL_BATH_TPB >>> (input, unfinished,
            dev_particle, dev_rngstate, dev_col_error, dev_col_unfinished,
            dev_col_time, dev_col_events, dev_col_complete, dev_col_hazard,
            dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, dev_col_neighbor,
            dev_col_measure, dev_col_active, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            lambda0, local.dt, dev_col_spatial, output, local.error, local.work, local.environment, local.cached);
        LOCAL_KERNEL("local_chain");
        LOCAL_CHECK(localCopy(&unfinished, dev_col_unfinished, sizeof(int), localD2H));
        input = output;
        output = (output == local.queue_a) ? local.queue_b : local.queue_a;
    }
    int error = 0;
    LOCAL_CHECK(localCopy(&error, local.error, sizeof(int), localD2H));
    if (error) throw std::runtime_error("local collision chain error "+std::to_string(error));
    #ifdef COL_DIAGNOSTICS
    chain_seconds += std::chrono::duration<double>(local_clock::now() - chain_begin).count();
    #endif // COL_DIAGNOSTICS

    #ifdef COL_PERF_VAL
    col_perf_event_ms += col_perf_stop(event_start);
    auto audit_start = col_perf_start();
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    const auto audit_begin = local_clock::now();
    #endif // COL_DIAGNOSTICS
    // compare realized activity and size change with the predicted envelopes and adapt each group's safety factor
    LOCAL_CHECK(localZero(dev_col_audit, 0, sizeof(col_audit_accum)*col_raw_count));
    col_audit_bin <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_audit,
        dev_particle, dev_size_old, dev_numr_old, dev_col_rate, dev_col_hazard,
        dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, dev_col_events,
        dev_col_spatial, dev_col_binmap, dev_col_active, local.dt);
    LOCAL_KERNEL("local_audit");
    LOCAL_CHECK(localCopy(audit.data(), dev_col_audit,
        sizeof(col_audit_accum)*merged, localD2H));
    for (int c : groups)
    {
        if (local.owners[c].empty()) continue;
        int first = binmap[c*COL_BIN_S], last = bin_end(c, merged);
        real mass = 0;
        for (int b = first; b < last; ++b)
        {
            if (audit[b].invalid_count) throw std::runtime_error("invalid local audit state");
            mass += audit[b].mass;
        }
        if (!(mass > 0)) continue;
        #ifdef COL_DIAGNOSTICS
        real before = local.state[c].limit_scale;
        #endif // COL_DIAGNOSTICS
        std::vector<col_audit_accum> slice(audit.begin() + first, audit.begin() + last);
        auto result = _finish_col_bath(slice, int(slice.size()), local.state[c]);
        passed[c] = !(result.activity_overshoot || result.distribution_overshoot || result.persistent_overshoot);
        #ifdef COL_DIAGNOSTICS
        auto &s = stats[c];
        ++s.updates;
        s.overshoots += result.activity_overshoot || result.distribution_overshoot;
        s.persistent += result.persistent_overshoot;
        s.max_growth = std::max(s.max_growth, result.max_g);
        s.max_activity = std::max(s.max_activity, result.max_f);
        // audit feedback controls the next due update; pending neighbors stay fixed
        col_bath_record record;
        record.group_index = c;
        record.operator_index = operator_index;
        record.bath_index=++batch_index;
        record.merged_bins = last - first;
        record.duration = steps[c];
        record.limit_before = before;
        record.limit_after = local.state[c].limit_scale;
        record.result = result;
        _record_col_bath(col_summary, record);
        #endif // COL_DIAGNOSTICS
    }
    // count physical launches once per wave, not once per group in that wave
    #ifdef COL_DIAGNOSTICS
    col_summary.continuation_launches += continuations;
    audit_seconds += std::chrono::duration<double>(local_clock::now() - audit_begin).count();
    #endif // COL_DIAGNOSTICS
    #ifdef COL_PERF_VAL
    col_perf_audit_ms += col_perf_stop(audit_start);
    ++col_perf_batches;
    col_perf_launches += continuations;
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    ++col_summary.wave_count;
    owner_updates += count;
    launches += continuations;
    ++waves;
    #endif // COL_DIAGNOSTICS
    ++count_col;
    dt_col = duration;
    for (int c : groups) dt_col = std::min(dt_col, steps[c]);
    schedule.advance(groups);
    clock_dyn = schedule.seconds(schedule.time());
}
// every pending endpoint has reached the horizon; transport may read all particle states
clock_dyn = duration;
#ifdef COL_DIAGNOSTICS
local.log << "{\"schema\":1,\"method\":\"local_cached_collision\",\"operator\":" << ++local.operator_id
    << ",\"clock_sim\":" << clock_sim << ",\"duration\":" << duration
    << ",\"waves\":" << waves << ",\"owner_updates\":" << owner_updates
    << ",\"chain_blocks\":" << chain_blocks << ",\"chain_launches\":" << launches
    << ",\"chain_seconds\":" << chain_seconds << ",\"audit_seconds\":" << audit_seconds
    << ",\"scheduler_wall_seconds\":"
    << std::chrono::duration<double>(local_clock::now() - local_begin).count()
    << ",\"finest_ticks\":" << schedule.end << ",\"groups\":[";
for (int c = 0; c < LOCAL_GROUPS; ++c)
{
    if (c) local.log << ',';
    const auto &s = stats[c];
    local.log << "{\"id\":" << c << ",\"owners\":" << local.owners[c].size()
        << ",\"level\":" << schedule.level[c] << ",\"dt\":" << steps[c]
        << ",\"initial_level\":" << initial_level[c]
        << ",\"coarsened_updates\":" << coarsened[c]
        << ",\"refined_updates\":" << refined[c]
        << ",\"neighbor_constrained_updates\":" << constrained[c]
        << ",\"minimum_requested_dt\":" << min_request[c]
        << ",\"maximum_requested_dt\":" << max_request[c]
        << ",\"final_requested_dt\":" << current_request[c]
        << ",\"initial_requested_dt\":" << requested[c]
        << ",\"initial_binding_constraint\":" << binding[c]
        << ",\"updates\":" << s.updates << ",\"overshoots\":" << s.overshoots
        << ",\"persistent_overshoots\":" << s.persistent
        << ",\"max_snapshot_age_at_start\":" << s.max_age
        << ",\"max_requested_dt_ratio\":" << s.max_requested_ratio
        << ",\"max_growth\":" << s.max_growth
        << ",\"max_predicted_activity\":" << s.max_activity << '}';
}
local.log << "],\"event_counts\":[";
col_event_sum <<< EVENT_CATEGORIES, TPB >>> (local.work, local.work_sum);
LOCAL_KERNEL("col_event_sum");
event_work totals;
LOCAL_CHECK(localCopy(&totals, local.work_sum, sizeof(event_work), localD2H));
for (int k = 0; k < EVENT_CATEGORIES; ++k)
{
    if (k) local.log<<',';
    local.log << totals.count[k];
}
local.log << "],\"event_log_mass_sums\":[";
for (int k = 0; k < EVENT_CATEGORIES; ++k)
{
    if (k) local.log<<',';
    local.log << totals.log_mass[k];
}
local.log << "]}\n";
local.log.flush();
if (!local.log) throw std::runtime_error("cannot write local collision diagnostics");
#endif // COL_DIAGNOSTICS
}

#endif // COLLISION && !BERNOULLI

#endif // GAMEDEV_SWARM_COL_CHAIN_CUH
