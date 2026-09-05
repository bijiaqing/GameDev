#ifndef SWARM_HOST_CUH
#define SWARM_HOST_CUH

#include <algorithm>        // std::copy, std::lower_bound, std::max, std::minmax_element
#include <chrono>           // std::chrono::system_clock
#include <cmath>            // std::abs, std::acos, std::atan2, std::cos, std::erf, std::erfc, std::exp, std::log, std::pow, std::sin, std::sqrt
#include <cstddef>          // std::size_t
#include <cstdlib>          // std::exit, EXIT_FAILURE
#include <ctime>            // std::time_t, std::ctime
#include <cuda_runtime.h>   // cudaError_t, cudaGetErrorString, cudaGetLastError, cudaSuccess
#include <fstream>          // std::ofstream, std::ifstream
#include <iomanip>          // std::setw, std::setfill, std::setprecision
#include <iostream>         // std::cout, std::cerr, std::endl
#include <limits>           // std::numeric_limits
#include <random>           // std::mt19937
#include <stdexcept>        // std::runtime_error
#include <string>           // std::string, std::to_string
#include <vector>           // std::vector

#include <const_defs.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>

// =========================================================================================================================
// host-only mesh coordinates
// =========================================================================================================================

inline __host__
real _get_ycent (int iy) { return Y_MIN*std::pow(_get_dy(), static_cast<real>(iy) + 0.5); }

inline __host__
real _get_zcent (int iz)
{ return (N_Z > 1) ? Z_MIN + (static_cast<real>(iz) + 0.5)*_get_dz() : 0.5*M_PI; }

inline __host__
real _get_sy (real y) { return std::pow(y, _get_mesh_dim()) / _get_mesh_dim(); }

inline __host__
real _get_sz (real z) { return -std::cos(z); }

// =========================================================================================================================
// elementary random profiles
// =========================================================================================================================

// share one deterministic host generator across all initialization samplers
extern std::mt19937 rand_generator;

#ifdef MULTISIZE
// calculate the scale that makes containment-weighted representative masses sum to the target domain mass
inline __host__
real get_mass_norm (const real *randsize, const std::vector <real> &mass_bank, real total_dust_mass)
{
    long double weight_sum = 0.0;
    int mass_bin_count = static_cast<int>(mass_bank.size());

    for (int idx = 0; idx < N_P; idx++)
    {
        real domain_mass = _get_domain_mass(randsize[idx], mass_bank.data(), mass_bin_count);
        weight_sum += static_cast<long double>(_get_mass_weight(randsize[idx]))
                    * static_cast<long double>(domain_mass);
    }

    return static_cast<real>(
        static_cast<long double>(total_dust_mass)*static_cast<long double>(N_P) / weight_sum
    );
}
#endif // MULTISIZE

// sample a probability density proportional to p raised to power_idx
inline __host__
void rand_powerlaw (real *randsize, int count, real p_min, real p_max, real power_idx)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    real tmp_min = std::pow(p_min, power_idx + 1.0);
    real tmp_max = std::pow(p_max, power_idx + 1.0);

    // invert the cumulative distribution for dN proportional to p^power_idx dp
    for (int idx = 0; idx < count; idx++)
    {
        randsize[idx] = std::pow((tmp_max - tmp_min)*random(rand_generator) + tmp_min, 1.0 / (power_idx + 1.0));
    }
}

// =========================================================================================================================
// smoothed power-law profiles
// =========================================================================================================================

// calculate the same physical convolved dust surface-density profile used by the fluid initializer
inline static __host__
void initdens_calc (std::vector <real> &initdens)
{
    const real smooth = 0.05*R_0;
    const real R_src_min = Y_MIN + 2.0*smooth;
    const real R_src_max = Y_MAX - 2.0*smooth;
    const real kernel_std = 0.5*smooth;
    const real R_min = _get_init_Rmin();
    const real dR = (Y_MAX - R_min) / static_cast<real>(N_Y);
    const real kernel_norm = 1.0 / (std::sqrt(2.0*M_PI)*kernel_std);

    std::vector <real> conv_u(N_Y + 1);
    initdens.assign(N_Y + 1, 0.0);

    for (int idx_dst = 0; idx_dst <= N_Y; idx_dst++)
    {
        conv_u[idx_dst] = R_min + static_cast<real>(idx_dst)*dR;
    }

    for (int idx_src = 0; idx_src <= N_Y; idx_src++)
    {
        real R_src = conv_u[idx_src];
        if (R_src < R_src_min || R_src > R_src_max) continue;

        real sigma_d = METAL_Z*SIGMA_0*std::pow(R_src / R_0, IDX_P);

        for (int idx_dst = 0; idx_dst <= N_Y; idx_dst++)
        {
            real delta_R = conv_u[idx_dst] - R_src;
            real kernel_weight = kernel_norm*std::exp(-0.5*delta_R*delta_R / (kernel_std*kernel_std));
            initdens[idx_dst] += sigma_d*kernel_weight*dR;
        }
    }
}

// interpolate a tabulated convolved profile on its uniform axis
inline static __host__
real initdens_lerp (real R, const std::vector <real> &initdens)
{
    real R_min = _get_init_Rmin();
    if (R < R_min || R > Y_MAX) return 0.0;

    int bin_count = static_cast<int>(initdens.size()) - 1;
    real loc = (R - R_min)*static_cast<real>(bin_count) / (Y_MAX - R_min);
    int idx_bin = std::min(static_cast<int>(loc), bin_count - 1);
    real frac = loc - static_cast<real>(idx_bin);

    return (1.0 - frac)*initdens[idx_bin] + frac*initdens[idx_bin + 1];
}

// calculate the initialized dust scale height for one cylindrical radius and grain size
inline static __host__
real _get_init_Hd (real R, real size)
{
    real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
    real H_d = h_g*R;

    #ifdef DIFFUSION
    real alpha_z = _get_alpha(R, h_g) / SCHMIDT_Z;

    real stokes_mid = STOKES_0*(size / S_0);
    #ifndef CONST_ST
    stokes_mid /= std::pow(R / R_0, IDX_P);
    #endif // NOT CONST_ST
    H_d *= std::sqrt(alpha_z / stokes_mid);
    #endif // DIFFUSION

    return H_d;
}

// select the common log-size resolution for domain-mass and conditional-CDF tables
inline static __host__
int _get_mass_bin_count ()
{
    #ifdef MULTISIZE
    #ifdef IMPORTGAS
    return 1;
    #else  // ANALYTIC_GAS
    if (N_Z == 1 || INIT_SMIN == INIT_SMAX) return 1;
    return 128;
    #endif // IMPORTGAS
    #else  // MONOSIZE
    return 1;
    #endif // MULTISIZE
}

// return a radial resolution independent of the simulation polar mesh
inline static __host__
int _get_init_Rbin_count ()
{ return std::max(2048, 4*N_Y); }

struct init_Zspan
{
    real Z_lo[2];
    real Z_hi[2];
    int count;
};

// intersect one cylindrical line with the configured spherical radial-polar domain
inline static __host__
init_Zspan _get_init_Zspan (real R)
{
    init_Zspan span = {{0.0, 0.0}, {0.0, 0.0}, 0};
    if (R < 0.0 || R > Y_MAX) return span;

    real infinity = std::numeric_limits<real>::infinity();
    real Z_polar_lo = (Z_MAX >= M_PI) ? -infinity : R*std::cos(Z_MAX) / std::sin(Z_MAX);
    real Z_polar_hi = (Z_MIN <= 0.0) ? +infinity : R*std::cos(Z_MIN) / std::sin(Z_MIN);
    real Z_o = std::sqrt(std::max(Y_MAX*Y_MAX - R*R, 0.0));
    real Z_lo = std::max(Z_polar_lo, -Z_o);
    real Z_hi = std::min(Z_polar_hi, +Z_o);
    if (Z_hi <= Z_lo) return span;

    if (R >= Y_MIN)
    {
        span.Z_lo[0] = Z_lo;
        span.Z_hi[0] = Z_hi;
        span.count = 1;
        return span;
    }

    real Z_i = std::sqrt(std::max(Y_MIN*Y_MIN - R*R, 0.0));
    real Z_neg_hi = std::min(Z_hi, -Z_i);
    if (Z_neg_hi > Z_lo)
    {
        span.Z_lo[span.count] = Z_lo;
        span.Z_hi[span.count] = Z_neg_hi;
        span.count++;
    }

    real Z_pos_lo = std::max(Z_lo, +Z_i);
    if (Z_hi > Z_pos_lo)
    {
        span.Z_lo[span.count] = Z_pos_lo;
        span.Z_hi[span.count] = Z_hi;
        span.count++;
    }

    return span;
}

// integrate a normalized zero-mean Gaussian over one finite interval
inline static __host__
real _get_normal_mass (real Z_lo, real Z_hi, real H_d)
{
    if (Z_hi <= Z_lo) return 0.0;

    real inv_width = 1.0 / (std::sqrt(2.0)*H_d);
    if (Z_lo >= 0.0)
        return 0.5*(std::erfc(Z_lo*inv_width) - std::erfc(Z_hi*inv_width));
    if (Z_hi <= 0.0)
        return 0.5*(std::erfc(-Z_hi*inv_width) - std::erfc(-Z_lo*inv_width));

    return 0.5*(std::erf(Z_hi*inv_width) - std::erf(Z_lo*inv_width));
}

// approximate the inverse standard-normal cumulative distribution
inline static __host__
real _get_normal_quantile (real probability)
{
    constexpr real a1 = -3.969683028665376e+01;
    constexpr real a2 = +2.209460984245205e+02;
    constexpr real a3 = -2.759285104469687e+02;
    constexpr real a4 = +1.383577518672690e+02;
    constexpr real a5 = -3.066479806614716e+01;
    constexpr real a6 = +2.506628277459239e+00;
    constexpr real b1 = -5.447609879822406e+01;
    constexpr real b2 = +1.615858368580409e+02;
    constexpr real b3 = -1.556989798598866e+02;
    constexpr real b4 = +6.680131188771972e+01;
    constexpr real b5 = -1.328068155288572e+01;
    constexpr real c1 = -7.784894002430293e-03;
    constexpr real c2 = -3.223964580411365e-01;
    constexpr real c3 = -2.400758277161838e+00;
    constexpr real c4 = -2.549732539343734e+00;
    constexpr real c5 = +4.374664141464968e+00;
    constexpr real c6 = +2.938163982698783e+00;
    constexpr real d1 = +7.784695709041462e-03;
    constexpr real d2 = +3.224671290700398e-01;
    constexpr real d3 = +2.445134137142996e+00;
    constexpr real d4 = +3.754408661907416e+00;
    constexpr real prob_lo = 0.02425;
    constexpr real prob_hi = 1.0 - prob_lo;

    real prob_min = std::numeric_limits<real>::min();
    real prob_max = 1.0 - std::numeric_limits<real>::epsilon();
    real prob = std::max(prob_min, std::min(probability, prob_max));

    if (prob < prob_lo)
    {
        real q = std::sqrt(-2.0*std::log(prob));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
             / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }

    if (prob > prob_hi)
    {
        real q = std::sqrt(-2.0*std::log(1.0 - prob));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
               / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }

    real q = prob - 0.5;
    real r = q*q;
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q
         / (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
}

// sample one Gaussian restricted to a selected finite interval
inline static __host__
real _sample_normal_interval (real Z_lo, real Z_hi, real H_d, real frac)
{
    real inv_width = 1.0 / (std::sqrt(2.0)*H_d);

    if (Z_lo >= 0.0)
    {
        real prob_lo = 0.5*std::erfc(Z_lo*inv_width);
        real prob_hi = 0.5*std::erfc(Z_hi*inv_width);
        real tail_prob = prob_lo + (prob_hi - prob_lo)*frac;
        return -H_d*_get_normal_quantile(tail_prob);
    }

    if (Z_hi <= 0.0)
    {
        real prob_lo = 0.5*std::erfc(-Z_lo*inv_width);
        real prob_hi = 0.5*std::erfc(-Z_hi*inv_width);
        real cdf_prob = prob_lo + (prob_hi - prob_lo)*frac;
        return H_d*_get_normal_quantile(cdf_prob);
    }

    real prob_lo = 0.5*(1.0 + std::erf(Z_lo*inv_width));
    real prob_hi = 0.5*(1.0 + std::erf(Z_hi*inv_width));
    return H_d*_get_normal_quantile(prob_lo + (prob_hi - prob_lo)*frac);
}

// integrate the vertical Gaussian over every allowed interval at one cylindrical radius
inline static __host__
real _get_init_containment (real R, real size)
{
    if (N_Z == 1) return 1.0;

    init_Zspan span = _get_init_Zspan(R);
    if (span.count == 0) return 0.0;

    real H_d = _get_init_Hd(R, size);
    real containment = 0.0;
    for (int idx_span = 0; idx_span < span.count; idx_span++)
    {
        containment += _get_normal_mass(span.Z_lo[idx_span], span.Z_hi[idx_span], H_d);
    }

    return containment;
}

// sample the exact truncated vertical Gaussian at one cylindrical radius
inline static __host__
real _sample_init_Z (real R, real size, real frac)
{
    if (N_Z == 1) return 0.0;

    init_Zspan span = _get_init_Zspan(R);
    real H_d = _get_init_Hd(R, size);
    real span_mass[2] = {0.0, 0.0};
    real total_mass = 0.0;

    for (int idx_span = 0; idx_span < span.count; idx_span++)
    {
        span_mass[idx_span] = _get_normal_mass(span.Z_lo[idx_span], span.Z_hi[idx_span], H_d);
        total_mass += span_mass[idx_span];
    }

    if (total_mass <= 0.0) throw std::runtime_error("initialized dust profile has zero vertical mass");

    real target_mass = frac*total_mass;
    real mass_before = 0.0;
    for (int idx_span = 0; idx_span < span.count; idx_span++)
    {
        real mass_after = mass_before + span_mass[idx_span];
        if (target_mass <= mass_after || idx_span == span.count - 1)
        {
            real local_frac = (span_mass[idx_span] > 0.0)
                ? (target_mass - mass_before) / span_mass[idx_span] : 0.5;
            local_frac = std::max(0.0, std::min(local_frac, 1.0));
            real Z = _sample_normal_interval(
                span.Z_lo[idx_span], span.Z_hi[idx_span], H_d, local_frac
            );
            return std::max(span.Z_lo[idx_span], std::min(Z, span.Z_hi[idx_span]));
        }
        mass_before = mass_after;
    }

    throw std::runtime_error("failed to select an initialized vertical interval");
}

// evaluate the cylindrical radial mass density after exact vertical containment
inline static __host__
real _get_init_Rmass (real R, const std::vector <real> &initdens, real size)
{
    if (R <= 0.0 || R > Y_MAX) return 0.0;

    real sigma_d = initdens_lerp(R, initdens);
    if (sigma_d <= 0.0) return 0.0;

    return R*sigma_d*_get_init_containment(R, size);
}

// tabulate the continuous cylindrical-radius CDF and its finite-domain physical mass
inline static __host__
real disk_cdf_calc (std::vector <real> &cdf, const std::vector <real> &initdens, real size)
{
    int radial_bin_count = _get_init_Rbin_count();
    real R_min = _get_init_Rmin();
    real dR = (Y_MAX - R_min) / static_cast<real>(radial_bin_count);
    cdf.assign(radial_bin_count + 1, 0.0);

    real mass_lo = _get_init_Rmass(R_min, initdens, size);
    for (int idx_R = 0; idx_R < radial_bin_count; idx_R++)
    {
        real R_hi = R_min + static_cast<real>(idx_R + 1)*dR;
        real mass_hi = _get_init_Rmass(R_hi, initdens, size);
        cdf[idx_R + 1] = cdf[idx_R] + 0.5*(mass_lo + mass_hi)*dR;
        mass_lo = mass_hi;
    }

    real radial_mass = cdf.back();
    if (radial_mass <= 0.0) throw std::runtime_error("initialized dust profile has zero mass in the domain");
    for (real &value : cdf)
    {
        value /= radial_mass;
    }

    real azimuth_extent = static_cast<real>(N_X)*_get_vol_x();
    return azimuth_extent*radial_mass;
}

// tabulate the physical domain mass on the same logarithmic size axis used by spatial CDFs
inline __host__
void initmass_calc (std::vector <real> &mass_bank)
{
    std::vector <real> initdens;
    initdens_calc(initdens);

    int mass_bin_count = _get_mass_bin_count();
    mass_bank.resize(mass_bin_count);
    std::vector <real> cdf;

    #ifdef MULTISIZE
    #ifdef IMPORTGAS
    mass_bank[0] = disk_cdf_calc(cdf, initdens, S_0);
    #else  // ANALYTIC_GAS
    real log_size_min = std::log(INIT_SMIN);
    real dlog_size = (mass_bin_count > 1)
        ? (std::log(INIT_SMAX) - log_size_min) / static_cast<real>(mass_bin_count - 1)
        : 0.0;

    for (int idx_size = 0; idx_size < mass_bin_count; idx_size++)
    {
        real size = std::exp(log_size_min + static_cast<real>(idx_size)*dlog_size);
        mass_bank[idx_size] = disk_cdf_calc(cdf, initdens, size);
    }
    #endif // IMPORTGAS
    #else  // MONOSIZE
    mass_bank[0] = disk_cdf_calc(cdf, initdens, S_0);
    #endif // MULTISIZE
}

// integrate the physical mass spectrum over grain size to obtain the represented domain mass
inline __host__
real get_total_dust_mass (const std::vector <real> &mass_bank)
{
    #ifdef MULTISIZE
    if (mass_bank.size() > 1)
    {
        constexpr int size_quad_count = 1024;
        real sqrt_size_min = std::sqrt(INIT_SMIN);
        real sqrt_size_max = std::sqrt(INIT_SMAX);
        long double mass_sum = 0.0;

        for (int idx_quad = 0; idx_quad <= size_quad_count; idx_quad++)
        {
            real frac = static_cast<real>(idx_quad) / static_cast<real>(size_quad_count);
            real sqrt_size = sqrt_size_min + (sqrt_size_max - sqrt_size_min)*frac;
            real size = sqrt_size*sqrt_size;
            real domain_mass = _get_domain_mass(
                size, mass_bank.data(), static_cast<int>(mass_bank.size())
            );

            real quad_weight = (idx_quad == 0 || idx_quad == size_quad_count)
                ? 1.0 : ((idx_quad % 2 == 0) ? 2.0 : 4.0);
            mass_sum += static_cast<long double>(quad_weight)*static_cast<long double>(domain_mass);
        }

        return static_cast<real>(mass_sum / (3.0L*static_cast<long double>(size_quad_count)));
    }
    #endif // MULTISIZE

    return mass_bank[0];
}

// =========================================================================================================================
// disk position sampling
// =========================================================================================================================

// place collision-test particles on a reproducibly jittered two-dimensional annular grid
inline __host__
void rand_collision_test_pos (
    real *randposx, real *randposy, real *randposz, int count, unsigned int seed
)
{
    const real radial_span = Y_MAX - Y_MIN;
    const real azimuth_span = X_MAX - X_MIN;
    int radial_count = std::max(1, static_cast<int>(std::round(std::sqrt(
        static_cast<real>(count)*radial_span / (R_0*azimuth_span)
    ))));

    int azimuth_base = count / radial_count;
    int azimuth_extra = count % radial_count;
    std::mt19937 position_generator(seed);
    std::uniform_real_distribution <real> jitter(-0.25, 0.25);

    int idx = 0;
    for (int idx_radial = 0; idx_radial < radial_count; idx_radial++)
    {
        int azimuth_count = azimuth_base + static_cast<int>(idx_radial < azimuth_extra);
        for (int idx_azimuth = 0; idx_azimuth < azimuth_count; idx_azimuth++, idx++)
        {
            randposx[idx] = X_MIN + azimuth_span
                *(static_cast<real>(idx_azimuth) + 0.5 + jitter(position_generator))
                / static_cast<real>(azimuth_count);
            randposy[idx] = Y_MIN + radial_span
                *(static_cast<real>(idx_radial) + 0.5 + jitter(position_generator))
                / static_cast<real>(radial_count);
            randposz[idx] = 0.5*M_PI;
        }
    }
}

#ifndef IMPORTGAS
// invert one linearly interpolated radial CDF
inline static __host__
real _sample_init_R (const real *cdf_lo, const real *cdf_hi, real frac_size, real cdf_sample)
{
    int radial_bin_count = _get_init_Rbin_count();
    int idx_lo = 0;
    int idx_hi = radial_bin_count;
    while (idx_lo < idx_hi)
    {
        int idx_mid = idx_lo + (idx_hi - idx_lo) / 2;
        real prob_mid = (1.0 - frac_size)*cdf_lo[idx_mid] + frac_size*cdf_hi[idx_mid];
        if (prob_mid < cdf_sample) idx_lo = idx_mid + 1;
        else idx_hi = idx_mid;
    }

    int idx_R = std::max(0, idx_lo - 1);
    real prob_lo = (1.0 - frac_size)*cdf_lo[idx_R] + frac_size*cdf_hi[idx_R];
    real prob_hi = (1.0 - frac_size)*cdf_lo[idx_R + 1] + frac_size*cdf_hi[idx_R + 1];
    real frac_R = (prob_hi > prob_lo) ? (cdf_sample - prob_lo) / (prob_hi - prob_lo) : 0.5;
    frac_R = std::max(0.0, std::min(frac_R, 1.0));

    real R_min = _get_init_Rmin();
    real dR = (Y_MAX - R_min) / static_cast<real>(radial_bin_count);
    return R_min + (static_cast<real>(idx_R) + frac_R)*dR;
}

// sample one continuous cylindrical position from a radial CDF and exact conditional Gaussian
inline static __host__
void _sample_disk_pos (real &x, real &y, real &z, real size, const real *cdf_lo, const real *cdf_hi,
    real frac_size, std::uniform_real_distribution <real> &random)
{
    real R = _sample_init_R(cdf_lo, cdf_hi, frac_size, random(rand_generator));
    real Z = _sample_init_Z(R, size, random(rand_generator));

    x = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);
    y = (N_Z > 1) ? std::sqrt(R*R + Z*Z) : R;
    z = (N_Z > 1) ? std::atan2(R, Z) : 0.5*M_PI;
}

// sample one-size dust continuously from the cylindrical density truncated by the spherical domain
inline __host__
void rand_disk_mono (real *randposx, real *randposy, real *randposz, real size, int count)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    std::vector <real> initdens;
    initdens_calc(initdens);

    std::vector <real> cdf;
    disk_cdf_calc(cdf, initdens, size);

    for (int idx = 0; idx < count; idx++)
    {
        _sample_disk_pos(
            randposx[idx], randposy[idx], randposz[idx], size,
            cdf.data(), cdf.data(), 0.0, random
        );
    }
}

#if defined(MULTISIZE) && defined(DIFFUSION)
// sample polydisperse dust from radial CDFs conditioned on each previously assigned grain size
inline __host__
void rand_disk_poly (real *randposx, real *randposy, real *randposz, const real *randsize, int count)
{
    auto [size_min_ptr, size_max_ptr] = std::minmax_element(randsize, randsize + count);
    real size_min = *size_min_ptr;
    real size_max = *size_max_ptr;

    if (N_Z == 1 || size_min == size_max)
    {
        rand_disk_mono(randposx, randposy, randposz, size_min, count);
        return;
    }

    // tabulate conditional CDFs uniformly in log size and interpolate their normalized probabilities
    int size_bin_count = _get_mass_bin_count();
    real log_size_min = std::log(INIT_SMIN);
    real log_size_max = std::log(INIT_SMAX);
    int radial_bin_count = _get_init_Rbin_count();
    real dlog_size = (log_size_max - log_size_min) / static_cast<real>(size_bin_count - 1);

    std::vector <real> initdens;
    initdens_calc(initdens);

    std::vector <real> cdf;
    std::vector <real> cdf_bank(static_cast<size_t>(size_bin_count)*static_cast<size_t>(radial_bin_count + 1));
    for (int idx_size = 0; idx_size < size_bin_count; idx_size++)
    {
        real size = std::exp(log_size_min + static_cast<real>(idx_size)*dlog_size);
        disk_cdf_calc(cdf, initdens, size);
        std::copy(cdf.begin(), cdf.end(),
            cdf_bank.begin() + static_cast<size_t>(idx_size)*static_cast<size_t>(radial_bin_count + 1));
    }

    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int idx = 0; idx < count; idx++)
    {
        real loc_size = (std::log(randsize[idx]) - log_size_min) / dlog_size;
        loc_size = std::max(0.0, std::min(loc_size, static_cast<real>(size_bin_count - 1)));
        int size_lo = std::min(static_cast<int>(loc_size), size_bin_count - 2);
        real frac_size = loc_size - static_cast<real>(size_lo);
        const real *cdf_lo = cdf_bank.data()
            + static_cast<size_t>(size_lo)*static_cast<size_t>(radial_bin_count + 1);
        const real *cdf_hi = cdf_lo + radial_bin_count + 1;

        _sample_disk_pos(
            randposx[idx], randposy[idx], randposz[idx], randsize[idx],
            cdf_lo, cdf_hi, frac_size, random
        );
    }
}
#endif // MULTISIZE && DIFFUSION
#endif // !IMPORTGAS

// =========================================================================================================================
// imported dust distribution
// =========================================================================================================================

#ifdef IMPORTGAS
// sample positions from imported gas density times dust-to-gas ratio and exact cell measure
inline __host__
void rand_from_file (real *randposx, real *randposy, real *randposz, int count, const real *gas_dens, const real *epsilon)
{
    std::uniform_real_distribution<real> random(0.0, 1.0);
    
    real mesh_dim = _get_mesh_dim();
    
    // integrate the imported dust density with the same exact disk cell measure
    std::vector <real> cell_mass(N_G);
    real total_mass = 0.0;
    
    for (int iz = 0; iz < N_Z; iz++)
    {
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real vol_y = _get_vol_y(iy);

            for (int ix = 0; ix < N_X; ix++)
            {
                real vol_x = _get_vol_x();
                real cell_measure = vol_x*vol_y*vol_z;

                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                if (!std::isfinite(gas_dens[idx_cell]) || gas_dens[idx_cell] < 0.0)
                    throw std::runtime_error("invalid imported gas density at cell " + std::to_string(idx_cell));
                if (!std::isfinite(epsilon[idx_cell]) || epsilon[idx_cell] < 0.0)
                    throw std::runtime_error("invalid imported dust-to-gas ratio at cell " + std::to_string(idx_cell));

                cell_mass[idx_cell] = gas_dens[idx_cell]*epsilon[idx_cell]*cell_measure;
                if (!std::isfinite(cell_mass[idx_cell]))
                    throw std::runtime_error("nonfinite imported dust mass at cell " + std::to_string(idx_cell));
                total_mass += cell_mass[idx_cell];
            }
        }
    }

    if (!std::isfinite(total_mass) || total_mass <= 0.0)
        throw std::runtime_error("imported dust profile has zero or nonfinite total mass");
    
    // build the cell-mass cumulative distribution
    std::vector <real> cdf(N_G + 1);
    cdf[0] = 0.0;
    
    for (int idx = 0; idx < N_G; idx++)
    {
        cdf[idx + 1] = cdf[idx] + cell_mass[idx];
    }
    
    // normalize the CDF to unit total probability
    for (int idx = 0; idx <= N_G; idx++)
    {
        cdf[idx] /= total_mass;
    }

    std::vector<real> y_face_s(N_Y + 1);
    for (int iy = 0; iy <= N_Y; iy++)
    {
        y_face_s[iy] = _get_sy(_get_yface(iy));
    }

    std::vector<real> z_face_s(N_Z + 1);
    for (int iz = 0; iz <= N_Z; iz++)
    {
        z_face_s[iz] = _get_sz(_get_zface(iz));
    }
    
    // select cells by inverse transform sampling
    for (int idx = 0; idx < count; idx++)
    {
        real cdf_sample = random(rand_generator);
        
        // locate the cell containing the sampled cumulative probability
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), cdf_sample);
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        
        int ix = idx_cell % N_X;
        int iy = (idx_cell / N_X) % N_Y;
        int iz = idx_cell / (N_X*N_Y);
        
        // sample logarithmic radial cells uniformly in the exact radial volume coordinate
        real s_y0 = y_face_s[iy];
        real s_y1 = y_face_s[iy + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);

        // sample active azimuth uniformly within the selected cell and lock an inactive azimuth
        randposx[idx] = (N_X > 1) ? X_MIN + _get_dx()*(static_cast<real>(ix) + random(rand_generator)) : 0.5*(X_MIN + X_MAX);
        randposy[idx] = std::pow(mesh_dim*s_y, 1.0 / mesh_dim);
        if (N_Z > 1)
        {
            real s_z0 = z_face_s[iz];
            real s_z1 = z_face_s[iz + 1];
            real s_z = s_z0 + (s_z1 - s_z0)*random(rand_generator);
            
            randposz[idx] = std::acos(-s_z);
        }
        else
        {
            randposz[idx] = 0.5*M_PI;
        }
    }
}
#endif // IMPORTGAS

// =========================================================================================================================
// cuda error handling
// =========================================================================================================================

inline __host__
void cuda_fail (cudaError_t status, const char *operation, const char *file, int line)
{
    std::cerr
    << "CUDA error at " << file << ":" << line
    << " during " << operation << ": " << cudaGetErrorString(status)
    << " (" << static_cast<int>(status) << ")" 
    << std::endl;
    
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(OPERATION)                                                       \
do {                                                                                \
    cudaError_t cuda_status_ = (OPERATION);                                         \
    if (cuda_status_ != cudaSuccess)                                                \
    { cuda_fail(cuda_status_, #OPERATION, __FILE__, __LINE__); }                    \
} while (0)

#define CUDA_KERNEL_CHECK(KERNEL_NAME)                                              \
do {                                                                                \
    cudaError_t cuda_status_ = cudaGetLastError();                                  \
    if (cuda_status_ != cudaSuccess)                                                \
    { cuda_fail(cuda_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }  \
} while (0)

// =========================================================================================================================
// output timing
// =========================================================================================================================

// calculate a nonnegative integer power without floating-point roundoff
inline __host__
real int_pow (int base, int power)
{
    real result = 1.0;

    for (int idx_pow = 0; idx_pow < power; idx_pow++)
    {
        result *= static_cast<real>(base);
    }
    
    return result;
}

// calculate the physical duration between the preceding and requested output frames
inline __host__
real _get_dt_out (int idx_file)
{
    #ifdef LOGTIMING
    if (idx_file == 1)
    {
        return DT_OUT*(int_pow(LOG_BASE, idx_file) - 0.0);
    }
    else
    {
        return DT_OUT*(int_pow(LOG_BASE, idx_file) - int_pow(LOG_BASE, idx_file - 1));
    }
    #else  // LOGOUTPUT or LINEAR
    return DT_OUT;
    #endif // LOGTIMING
}

// =========================================================================================================================
// binary file I/O
// =========================================================================================================================

constexpr std::size_t binary_chunk_bytes = 64ULL*1024ULL*1024ULL;

// write a contiguous host array without format conversion
template <typename DataType> __host__ inline
bool save_host_binary (const std::string &file_name, const DataType *data, std::size_t count)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;

    const std::size_t chunk_max = std::max<std::size_t>(1, binary_chunk_bytes / sizeof(DataType));
    for (std::size_t offset = 0; offset < count; offset += chunk_max)
    {
        std::size_t chunk = std::min(chunk_max, count - offset);
        file.write(reinterpret_cast<const char*>(data + offset), sizeof(DataType)*chunk);
        if (!file) return false;
    }

    return true;
}

// read a contiguous host array without format conversion
template <typename DataType> __host__ inline
bool load_host_binary (const std::string &file_name, DataType *data, std::size_t count)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;

    const std::streamoff expected = static_cast<std::streamoff>(sizeof(DataType)*count);
    file.seekg(0, std::ios::end);
    if (file.tellg() != expected) return false;
    file.seekg(0, std::ios::beg);

    const std::size_t chunk_max = std::max<std::size_t>(1, binary_chunk_bytes / sizeof(DataType));
    for (std::size_t offset = 0; offset < count; offset += chunk_max)
    {
        std::size_t chunk = std::min(chunk_max, count - offset);
        file.read(reinterpret_cast<char*>(data + offset), sizeof(DataType)*chunk);
        if (!file) return false;
    }

    return true;
}

// write a device array through a bounded host buffer
template <typename DataType> __host__ inline
bool save_device_binary (const std::string &file_name, const DataType *dev_data, std::size_t count)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;

    const std::size_t chunk_max = std::max<std::size_t>(1, binary_chunk_bytes / sizeof(DataType));
    std::vector<DataType> buffer(std::min(count, chunk_max));

    for (std::size_t offset = 0; offset < count; offset += chunk_max)
    {
        std::size_t chunk = std::min(chunk_max, count - offset);
        CUDA_CHECK(cudaMemcpy(buffer.data(), dev_data + offset, sizeof(DataType)*chunk, cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<const char*>(buffer.data()), sizeof(DataType)*chunk);
        if (!file) return false;
    }

    return true;
}

// read a device array through a bounded host buffer
template <typename DataType> __host__ inline
bool load_device_binary (const std::string &file_name, DataType *dev_data, std::size_t count)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;

    const std::streamoff expected = static_cast<std::streamoff>(sizeof(DataType)*count);
    file.seekg(0, std::ios::end);
    if (file.tellg() != expected) return false;
    file.seekg(0, std::ios::beg);

    const std::size_t chunk_max = std::max<std::size_t>(1, binary_chunk_bytes / sizeof(DataType));
    std::vector<DataType> buffer(std::min(count, chunk_max));

    for (std::size_t offset = 0; offset < count; offset += chunk_max)
    {
        std::size_t chunk = std::min(chunk_max, count - offset);
        file.read(reinterpret_cast<char*>(buffer.data()), sizeof(DataType)*chunk);
        if (!file) return false;
        CUDA_CHECK(cudaMemcpy(dev_data + offset, buffer.data(), sizeof(DataType)*chunk, cudaMemcpyHostToDevice));
    }

    return true;
}

// convert internal angular variables to linear azimuthal and polar velocities before file output
inline __host__
void save_sam_as_velocity (swarm *particle)
{
    for (int idx = 0; idx < N_P; idx++)
    {
        real y = particle[idx].position.y;
        real z = particle[idx].position.z;
        real R = _get_cyl_R(y, z);

        if (N_X == 1) particle[idx].position.x = 0.5*(X_MIN + X_MAX);
        if (N_Z == 1) particle[idx].position.z = 0.5*M_PI;
        particle[idx].velocity.x = (R > 0.0) ? particle[idx].velocity.x / R : 0.0;
        particle[idx].velocity.z = (N_Z > 1 && y > 0.0) ? particle[idx].velocity.z / y : 0.0;
    }
}

// convert linear azimuthal and polar file velocities to the internal angular variables
inline __host__
void load_velocity_as_sam (swarm *particle)
{
    for (int idx = 0; idx < N_P; idx++)
    {
        if (N_X == 1) particle[idx].position.x = 0.5*(X_MIN + X_MAX);
        if (N_Z == 1) particle[idx].position.z = 0.5*M_PI;

        real y = particle[idx].position.y;
        real z = particle[idx].position.z;
        real R = _get_cyl_R(y, z);

        particle[idx].velocity.x *= R;
        particle[idx].velocity.z = (N_Z > 1) ? particle[idx].velocity.z*y : 0.0;
    }
}

// =========================================================================================================================
// file naming, loading, and metadata
// =========================================================================================================================

// format a frame index with the zero padding used by binary output files
inline __host__
std::string frame_num (int idx_file)
{
    std::string num_str = std::to_string(idx_file);
    int width = std::max(5, static_cast<int>(std::to_string(SAVE_MAX).length()));
    if (num_str.length() < width) num_str.insert(0, width - num_str.length(), '0');
    return num_str;
}

// report completion time for one output frame
inline __host__
void msg_output (int idx_file)
{
    std::time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    int width = std::max(3, static_cast<int>(std::to_string(SAVE_MAX).length()));
    std::cout   
    << std::endl << std::setfill('0')
    << std::setw(width) << idx_file << "/"
    << std::setw(width) << SAVE_MAX << " finished on " << std::ctime(&time_now)
    << std::endl;
}

#ifdef LOGOUTPUT
// test whether a linear frame index is an integer power selected for logarithmic output
inline __host__
bool is_log_power (int idx_file)
{
    // include the zeroth power at idx_file = 1
    if (LOG_BASE == 2)
    {
        // use the bit test for the common base-two case
        return (idx_file > 0) && ((idx_file & (idx_file - 1)) == 0);
    }
    else
    {
        // repeatedly remove factors for an arbitrary integer base
        if (idx_file <= 0) return false;

        while (idx_file % LOG_BASE == 0)
        {
            idx_file /= LOG_BASE;
        }

        return (idx_file == 1);
    }
}
#endif // LOGOUTPUT

#ifdef IMPORTGAS
// load one dust-to-gas ratio frame from the imported dataset
inline __host__
bool load_epsilon (const std::string &path, int idx_file, real *epsilon)
{
    std::string file_name = path + "epsilon_" + frame_num(idx_file) + ".dat";
    return load_host_binary(file_name, epsilon, N_G);
}

// load density and all three linear velocity components for one gas frame
inline __host__
bool load_gas_data (const std::string &path, int idx_file, real *gas_dens, real *gas_velx, real *gas_vely, real *gas_velz)
{
    std::string file_name;
    bool success = true;
    
    file_name = path + "gasdens_" + frame_num(idx_file) + ".dat";
    success &= load_host_binary(file_name, gas_dens, N_G);
    
    file_name = path + "gasvelx_" + frame_num(idx_file) + ".dat";
    success &= load_host_binary(file_name, gas_velx, N_G);
    
    file_name = path + "gasvely_" + frame_num(idx_file) + ".dat";
    success &= load_host_binary(file_name, gas_vely, N_G);
    
    file_name = path + "gasvelz_" + frame_num(idx_file) + ".dat";
    success &= load_host_binary(file_name, gas_velz, N_G);
    
    return success;
}
#endif // IMPORTGAS

// write the active physical, numerical, grid, and binary-layout configuration
inline __host__
bool save_variable (
    const std::string &file_name, real total_dust_mass,
    unsigned int position_seed, unsigned int collision_seed
)
{
    std::ofstream file(file_name);
    if (!file) return false;
    
    file << "[PARAMETERS]"                                                                  << std::endl;
    file                                                                                    << std::endl;

    // gas parameters
    file << "SIGMA_0     = " << std::scientific     << std::setprecision(8) << SIGMA_0      << std::endl;
    file << "METAL_Z     = " << std::scientific     << std::setprecision(8) << METAL_Z      << std::endl;
    file << "ASPR_0      = " << std::defaultfloat   << std::setprecision(8) << ASPR_0       << std::endl;
    file << "IDX_P       = " << std::defaultfloat   << std::setprecision(8) << IDX_P        << std::endl;
    file << "IDX_Q       = " << std::defaultfloat   << std::setprecision(8) << IDX_Q        << std::endl;
    #if defined(DIFFUSION) || defined(COLLISION)
    #ifdef CONST_NU
    file << "NU          = " << std::scientific     << std::setprecision(8) << NU           << std::endl;
    #else  // CONST_ALPHA
    file << "ALPHA       = " << std::scientific     << std::setprecision(8) << ALPHA        << std::endl;
    #endif // CONST_NU
    #endif // DIFFUSION || COLLISION
    #ifdef VISC_FLOW
    file << "VISC_FLOW   = " << std::defaultfloat << 1                                      << std::endl;
    #endif // VISC_FLOW
    #ifdef COLLISION
    #ifdef CODE_UNIT
    file << "REYNOLDS_0        = " << std::scientific     << std::setprecision(8) << REYNOLDS_0         << std::endl;
    #else  // PHYSICAL_UNIT
    file << "M_MOL       = " << std::scientific     << std::setprecision(8) << M_MOL        << std::endl;
    file << "X_SEC       = " << std::scientific     << std::setprecision(8) << X_SEC        << std::endl;
    #endif // CODE_UNIT
    #endif // COLLISION
    file                                                                                    << std::endl;
    
    // dust parameters
    file << "STOKES_0    = " << std::scientific     << std::setprecision(8) << STOKES_0     << std::endl;
    file << "TOTAL_DUST_MASS = " << std::scientific << std::setprecision(8) << total_dust_mass << std::endl;
    file << "POSITION_SEED = " << position_seed                                             << std::endl;
    file << "COLLISION_SEED = " << collision_seed                                           << std::endl;
    file << "RHO_0       = " << std::scientific     << std::setprecision(8) << RHO_0        << std::endl;
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific     << std::setprecision(8) << BETA_0       << std::endl;
    file << "KAPPA_0     = " << std::scientific     << std::setprecision(8) << KAPPA_0      << std::endl;
    file << "T_BETA      = " << std::scientific     << std::setprecision(8) << T_BETA       << std::endl;
    #ifdef PR_EFFECT
    file << "C_LIGHT     = " << std::scientific     << std::setprecision(8) << C_LIGHT      << std::endl;
    #endif // PR_EFFECT
    #endif // RADIATION
    #ifdef DIFFUSION
    file << "SCHMIDT_X   = " << std::scientific     << std::setprecision(8) << SCHMIDT_X    << std::endl;
    file << "SCHMIDT_R   = " << std::scientific     << std::setprecision(8) << SCHMIDT_R    << std::endl;
    #endif // DIFFUSION
    #if defined(DIFFUSION) || defined(COLLISION)
    file << "SCHMIDT_Z   = " << std::scientific     << std::setprecision(8) << SCHMIDT_Z    << std::endl;
    #endif // DIFFUSION || COLLISION

    #ifdef COLLISION
    file << "LAMBDA_0    = " << std::scientific     << std::setprecision(8)
         << N_P / static_cast<real>(N_K) / total_dust_mass << std::endl;
    file << "V_FRAG      = " << std::scientific     << std::setprecision(8) << V_FRAG       << std::endl;
    file << "COAG_KERNEL = " << std::defaultfloat   << std::setprecision(8) << COAG_KERNEL  << std::endl;
    file << "N_K         = " << std::defaultfloat   << std::setprecision(8) << N_K          << std::endl;
    file << "H_SEARCH    = " << std::defaultfloat   << std::setprecision(8) << H_SEARCH     << std::endl;
    #ifndef BERNOULLI
    file << "COLLISION_INTEGRATOR = frozen_bath"                                         << std::endl;
    file << "COL_BATH_TPB  = " << COL_BATH_TPB                                            << std::endl;
    file << "COL_EVENT_CAP  = " << COL_EVENT_CAP                                          << std::endl;
    file << "COL_BATH_MAX   = rate_adaptive"                                           << std::endl;
    file << "COL_BATH_EPS   = " << std::scientific << std::setprecision(8) << COL_BATH_EPS << std::endl;
    file << "COL_BATH_ALPHA = " << std::scientific << std::setprecision(8) << COL_BATH_ALPHA << std::endl;
    file << "COL_CONTROLLER_AUDIT = path_integrated"                                  << std::endl;
    file << "COL_CONTROLLER_FAILURE = two_consecutive_minimum_scale_overshoots"        << std::endl;
    file << "COL_CONTROLLER_BINS = " << COL_BIN_X << " " << COL_BIN_Y << " "
         << COL_BIN_Z << " " << COL_BIN_S                                           << std::endl;
    file << "COL_BIN_MIN    = " << COL_BIN_MIN                                            << std::endl;
    file << "COL_SIZE_RANGE_FACTORS = " << std::scientific << std::setprecision(8)
         << COL_SIZE_MIN_FACTOR << " " << COL_SIZE_MAX_FACTOR                        << std::endl;
    #else  // BERNOULLI
    #ifdef KNN_CACHE
    file << "COLLISION_INTEGRATOR = bernoulli_cache"                                     << std::endl;
    #else  // DIRECT_BERNOULLI
    file << "COLLISION_INTEGRATOR = bernoulli_direct"                                    << std::endl;
    #endif // KNN_CACHE
    file << "CFL_COL     = " << std::defaultfloat   << std::setprecision(8) << CFL_COL      << std::endl;
    #endif // FROZEN_BATH / BERNOULLI
    #if !defined(BERNOULLI) || defined(KNN_CACHE)
    file << "COLLISION_NEIGHBORS = cached"                                              << std::endl;
    #else  // DIRECT_BERNOULLI
    file << "COLLISION_NEIGHBORS = direct"                                              << std::endl;
    #endif // FROZEN_BATH || KNN_CACHE
    file << "RNG_STREAM_POLICY = shared_per_particle"                                    << std::endl;
    #ifdef COLLISION_KDTREE
    file << "COLLISION_SEARCH = kdtree"                                                   << std::endl;
    #else  // COLLISION_MORTON
    file << "COLLISION_SEARCH = morton"                                                   << std::endl;
    file << "MORTON_LEAF_TARGET = " << MORTON_LEAF_TARGET                                 << std::endl;
    file << "MORTON_MAX_LEVEL   = " << MORTON_MAX_LEVEL                                   << std::endl;
    #endif // COLLISION_KDTREE
    #endif // COLLISION
    file                                                                                    << std::endl;

    // mesh domain
    file << "N_P         = " << std::scientific     << std::setprecision(8) << N_P          << std::endl;
    file                                                                                    << std::endl;
    file << "N_X         = " << std::defaultfloat   << std::setprecision(8) << N_X          << std::endl;
    file << "X_MIN       = " << std::defaultfloat   << std::setprecision(8) << X_MIN        << std::endl;
    file << "X_MAX       = " << std::defaultfloat   << std::setprecision(8) << X_MAX        << std::endl;
    file                                                                                    << std::endl;
    file << "N_Y         = " << std::defaultfloat   << std::setprecision(8) << N_Y          << std::endl;
    file << "Y_MIN       = " << std::defaultfloat   << std::setprecision(8) << Y_MIN        << std::endl;
    file << "Y_MAX       = " << std::defaultfloat   << std::setprecision(8) << Y_MAX        << std::endl;
    file                                                                                    << std::endl;
    file << "N_Z         = " << std::defaultfloat   << std::setprecision(8) << N_Z          << std::endl;
    file << "Z_MIN       = " << std::defaultfloat   << std::setprecision(8) << Z_MIN        << std::endl;
    file << "Z_MAX       = " << std::defaultfloat   << std::setprecision(8) << Z_MAX        << std::endl;
    file                                                                                    << std::endl;
    file << "N_G         = " << std::scientific     << std::setprecision(8) << N_G          << std::endl;
    file                                                                                    << std::endl;

    if (N_X == 1 && N_Z == 1) file << "GEOMETRY    = radial"                              << std::endl;
    if (N_X >  1 && N_Z == 1) file << "GEOMETRY    = radial_azimuthal"                    << std::endl;
    if (N_X == 1 && N_Z >  1) file << "GEOMETRY    = radial_polar"                        << std::endl;
    if (N_X >  1 && N_Z >  1) file << "GEOMETRY    = full_3d"                             << std::endl;
    file << "DUST_DENSITY_KIND = " << ((N_Z == 1) ? "surface" : "volume")                << std::endl;
    #ifdef IMPORTGAS
    file << "GAS_DENSITY_KIND  = " << ((N_Z == 1) ? "surface" : "volume")                << std::endl;
    file << "IMPORTED_EPSILON_NORMALIZATION = shape_only"                                  << std::endl;
    #endif // IMPORTGAS
    if (N_X == 1) file << "INACTIVE_X  = " << 0.5*(X_MIN + X_MAX)                          << std::endl;
    if (N_Z == 1) file << "INACTIVE_Z  = " << 0.5*M_PI                                    << std::endl;
    if (N_X == 1 && N_Z == 1) file << "DYNAMICAL_CLOSURE = midplane"                     << std::endl;
    file                                                                                    << std::endl;

    // initialization parameters
    #ifdef MULTISIZE
    file << "INIT_SMIN   = " << std::scientific     << std::setprecision(8) << INIT_SMIN    << std::endl;
    file << "INIT_SMAX   = " << std::scientific     << std::setprecision(8) << INIT_SMAX    << std::endl;
    #endif // MULTISIZE
    file                                                                                    << std::endl;

    // timestep and output parameters
    file << "SAVE_MAX    = " << std::defaultfloat   << std::setprecision(8) << SAVE_MAX     << std::endl;
    #if defined(LOGTIMING) || defined(LOGOUTPUT)
    file << "LOG_BASE    = " << std::defaultfloat   << std::setprecision(8) << LOG_BASE     << std::endl;
    #else  // LINEAR
    file << "LIN_BASE    = " << std::defaultfloat   << std::setprecision(8) << LIN_BASE     << std::endl;
    #endif // LOGOUTPUT or LOGTIMING
    file << "DT_OUT      = " << std::scientific     << std::setprecision(8) << DT_OUT       << std::endl;
    #ifdef TRANSPORT
    file << "DT_MAX      = " << std::scientific     << std::setprecision(8) << DT_MAX       << std::endl;
    file << "CFL_DYN     = " << std::defaultfloat   << std::setprecision(8) << CFL_DYN      << std::endl;
    #endif // TRANSPORT
    file                                                                                    << std::endl;

    // swarm structure with linear physical velocities as a configparser-compatible NumPy dtype, in Python write as:
    // dtype = np.dtype([(name, dtype) for name, dtype in config['SWARM_DTYPE'].items()])
    file << "[SWARM_DTYPE]"                                                                 << std::endl;
    file << "position_x = f8"                                                               << std::endl;
    file << "position_y = f8"                                                               << std::endl;
    file << "position_z = f8"                                                               << std::endl;
    file << "velocity_x = f8"                                                               << std::endl;
    file << "velocity_y = f8"                                                               << std::endl;
    file << "velocity_z = f8"                                                               << std::endl;
    #ifdef MULTISIZE
    file << "par_size   = f8"                                                               << std::endl;
    file << "par_numr   = f8"                                                               << std::endl;
    #endif // MULTISIZE
    
    return file.good();
}

// =========================================================================================================================
// main-loop file transfers
// =========================================================================================================================

// save the particle state used by the coagulation-distribution analysis
inline __host__
bool save_particle_data (const std::string &path, int idx_file, swarm *particle, const swarm *dev_particle)
{
    CUDA_CHECK(cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost));
    save_sam_as_velocity(particle);

    std::string file_name = path + "particle_" + frame_num(idx_file) + ".dat";
    if (!save_host_binary(file_name, particle, N_P))
    {
        std::cerr << "Error: Failed to save file: " << file_name << std::endl;
        return false;
    }

    return true;
}

// load the complete particle checkpoint and its stochastic state
inline __host__
bool load_particle_data (const std::string &path, int idx_file, swarm *particle, swarm *dev_particle
    #if defined(COLLISION) || defined(DIFFUSION)
    , curs *dev_rngstate
    #endif // COLLISION || DIFFUSION
)
{
    std::string file_name = path + "particle_" + frame_num(idx_file) + ".dat";
    if (!load_host_binary(file_name, particle, N_P))
    {
        std::cerr << "Error: Failed to load file: " << file_name << std::endl;
        return false;
    }

    load_velocity_as_sam(particle);
    CUDA_CHECK(cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_P, cudaMemcpyHostToDevice));

    #if defined(COLLISION) || defined(DIFFUSION)
    file_name = path + "rngstate_" + frame_num(idx_file) + ".dat";
    if (!load_device_binary(file_name, dev_rngstate, N_P))
    {
        std::cerr << "Error: Failed to load file: " << file_name << std::endl;
        return false;
    }
    #endif // COLLISION || DIFFUSION

    return true;
}

#if defined(COLLISION) || defined(DIFFUSION)
#define SAVE_PARTICLE_TO_FILE(idx_file)                                                     \
do {                                                                                        \
    if (!save_particle_data(PATH, idx_file, particle, dev_particle)) return 1;              \
} while(0)
#define LOAD_PARTICLE_TO_VRAM(idx_file)                                                     \
do {                                                                                        \
    if (!load_particle_data(PATH, idx_file, particle, dev_particle, dev_rngstate)) return 1; \
} while(0)
#else  // NO COLLISION OR DIFFUSION
#define SAVE_PARTICLE_TO_FILE(idx_file)                                                     \
do {                                                                                        \
    if (!save_particle_data(PATH, idx_file, particle, dev_particle)) return 1;              \
} while(0)
#define LOAD_PARTICLE_TO_VRAM(idx_file)                                                     \
do {                                                                                        \
    if (!load_particle_data(PATH, idx_file, particle, dev_particle)) return 1;              \
} while(0)
#endif // COLLISION || DIFFUSION

#ifdef IMPORTGAS
#define LOAD_GAS_DATA_TO_VRAM(idx_file)                                                     \
do {                                                                                        \
    if (!load_gas_data(PATH, idx_file, gas_dens, gas_velx, gas_vely, gas_velz))             \
    {                                                                                       \
        std::cerr << "Error: Failed to load gas data files for frame " << idx_file << std::endl; \
        return 1;                                                                           \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(dev_gas_dens, gas_dens, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_velx, gas_velx, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_vely, gas_vely, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_velz, gas_velz, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
} while(0)

#define LOAD_GAS_NEXT_TO_VRAM(idx_file)                                                     \
do {                                                                                        \
    if (!load_gas_data(PATH, idx_file, gas_dens, gas_velx, gas_vely, gas_velz))             \
    {                                                                                       \
        std::cerr << "Error: Failed to load gas data files for frame " << idx_file << std::endl; \
        return 1;                                                                           \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(dev_gas_dens_next, gas_dens, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_velx_next, gas_velx, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_vely_next, gas_vely, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_velz_next, gas_velz, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
} while(0)
#endif // IMPORTGAS

#ifdef SAVE_DENS
#define SAVE_DUSTDENS_TO_FILE(idx_file)                                                     \
do {                                                                                        \
    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);                                         \
    CUDA_KERNEL_CHECK("dustdens_init");                                                     \
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, total_dust_mass);          \
    CUDA_KERNEL_CHECK("dustdens_depo");                                                     \
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);                                         \
    CUDA_KERNEL_CHECK("dustdens_calc");                                                     \
    CUDA_CHECK(cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost));           \
    std::string file_name = PATH + "dustdens_" + frame_num(idx_file) + ".dat";              \
    if (!save_host_binary(file_name, dustdens, N_G))                                        \
    {                                                                                       \
        std::cerr << "Error: Failed to save file: " << file_name << std::endl;              \
        return 1;                                                                           \
    }                                                                                       \
} while(0)
#endif // SAVE_DENS

#ifdef RADIATION
#define SAVE_OPTDEPTH_TO_FILE(idx_file, do_avg)                                             \
do {                                                                                        \
    optdepth_init <<< NB_G, TPB >>> (dev_optdepth);                                         \
    CUDA_KERNEL_CHECK("optdepth_init");                                                     \
    optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, total_dust_mass);          \
    CUDA_KERNEL_CHECK("optdepth_depo");                                                     \
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);                                         \
    CUDA_KERNEL_CHECK("optdepth_calc");                                                     \
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);                                         \
    CUDA_KERNEL_CHECK("optdepth_csum");                                                     \
    if (do_avg)                                                                             \
    {                                                                                       \
        optdepth_mean <<< NB_X, TPB >>> (dev_optdepth);                                     \
        CUDA_KERNEL_CHECK("optdepth_mean");                                                 \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost));           \
    std::string file_name = PATH + "optdepth_" + frame_num(idx_file) + ".dat";              \
    if (!save_host_binary(file_name, optdepth, N_G))                                        \
    {                                                                                       \
        std::cerr << "Error: Failed to save file: " << file_name << std::endl;              \
        return 1;                                                                           \
    }                                                                                       \
} while(0)
#endif // RADIATION

// =========================================================================================================================
// main-loop console output
// =========================================================================================================================

#ifdef TRANSPORT
#define PRINT_TITLE_TRANSPORT()             \
std::cout                                   \
<< std::setw(10) << "count_dyn" << " "      \
<< std::setw(10) << "dt_dyn"    << " ";
#define PRINT_VALUE_TRANSPORT()             \
std::cout                                   \
<< std::defaultfloat                        \
<< std::setw(10) << count_dyn   << " "      \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << dt_dyn      << " ";
#else  // NO TRANSPORT
#define PRINT_TITLE_TRANSPORT()
#define PRINT_VALUE_TRANSPORT()
#endif // TRANSPORT

#if defined(COLLISION) && defined(TRANSPORT)
#define PRINT_TITLE_COLLISION()             \
std::cout                                   \
<< std::setw(10) << "clock_dyn" << " "      \
<< std::setw(10) << "count_col" << " "      \
<< std::setw(10) << "dt_col"    << " ";
#define PRINT_VALUE_COLLISION()             \
std::cout                                   \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << clock_dyn   << " "      \
<< std::defaultfloat                        \
<< std::setw(10) << count_col   << " "      \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << dt_col      << " ";
#elif defined(COLLISION)
#define PRINT_TITLE_COLLISION()             \
std::cout                                   \
<< std::setw(10) << "count_col" << " "      \
<< std::setw(10) << "dt_col"    << " ";
#define PRINT_VALUE_COLLISION()             \
std::cout                                   \
<< std::defaultfloat                        \
<< std::setw(10) << count_col   << " "      \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << dt_col      << " ";
#else  // NO COLLISION
#define PRINT_TITLE_COLLISION()
#define PRINT_VALUE_COLLISION()
#endif // COLLISION

#define PRINT_TITLE_TO_SCREEN()             \
std::cout << std::setfill(' ')              \
<< std::setw(10) << "idx"       << " "      \
<< std::setw(10) << "clock_sim" << " "      \
<< std::setw(10) << "clock_out" << " ";     \
PRINT_TITLE_TRANSPORT();                    \
PRINT_TITLE_COLLISION();                    \
std::cout << std::endl;

#define PRINT_VALUE_TO_SCREEN()             \
std::cout << std::setfill(' ')              \
<< std::defaultfloat                        \
<< std::setw(10) << idx_file    << " "      \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << clock_sim   << " "      \
<< std::scientific << std::setprecision(3)  \
<< std::setw(10) << clock_out   << " ";     \
PRINT_VALUE_TRANSPORT();                    \
PRINT_VALUE_COLLISION();                    \
std::cout << std::endl;

// =========================================================================================================================

#endif // SWARM_HOST_CUH
