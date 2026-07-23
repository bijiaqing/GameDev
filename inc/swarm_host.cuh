#ifndef SWARM_HOST_CUH
#define SWARM_HOST_CUH

#include <algorithm>        // std::copy, std::lower_bound, std::max, std::max_element, std::minmax_element
#include <chrono>           // std::chrono::system_clock
#include <cmath>            // std::abs, std::acos, std::cos, std::exp, std::log, std::pow, std::sin, std::sqrt
#include <cstdlib>          // std::exit, EXIT_FAILURE
#include <ctime>            // std::time_t, std::ctime
#include <cuda_runtime.h>   // cudaError_t, cudaGetErrorString, cudaGetLastError, cudaSuccess
#include <fstream>          // std::ofstream, std::ifstream
#include <iomanip>          // std::setw, std::setfill, std::setprecision
#include <iostream>         // std::cout, std::cerr, std::endl
#include <limits>           // std::numeric_limits
#include <random>           // std::mt19937
#include <stdexcept>        // std::domain_error, std::runtime_error
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
real _get_zcent (int iz) { return Z_MIN + (static_cast<real>(iz) + 0.5)*_get_dz(); }

inline __host__
real _get_s_y (real y) { return std::pow(y, _get_mesh_dim()) / _get_mesh_dim(); }

inline __host__
real _get_s_z (real z) { return -std::cos(z); }

// =========================================================================================================================
// elementary random profiles
// =========================================================================================================================

// share one deterministic host generator across all initialization samplers
extern std::mt19937 rand_generator;

#ifdef MULTISIZE
// calculate the mass scale that makes the sampled representative masses sum to the target dust mass
inline __host__
real get_mass_norm (const real *grain_size, real dust_mass)
{
    long double weight_sum = 0.0;

    for (int idx = 0; idx < N_P; idx++)
    {
        weight_sum += static_cast<long double>(_get_mass_weight(grain_size[idx]));
    }

    return static_cast<real>(
        static_cast<long double>(dust_mass)*static_cast<long double>(N_P) / weight_sum
    );
}
#endif // MULTISIZE

// sample a Gaussian distribution truncated to the parameter interval by rejection
inline __host__
void rand_gaussian (real *profile, int number, real p_min, real p_max, real mu, real std)
{
    std::normal_distribution <real> random(mu, std);

    for (int i = 0; i < number; i++)
    {
        real value;
        
        do
        {
            value = random(rand_generator);
        } 
        while (value < p_min || value > p_max);
        
        profile[i] = value;
    }
}

// sample a probability density proportional to p raised to idx_pow
inline __host__
void rand_powerlaw (real *profile, int number, real p_min, real p_max, real idx_pow)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    real tmp_min = std::pow(p_min, idx_pow + 1.0);
    real tmp_max = std::pow(p_max, idx_pow + 1.0);

    // invert the cumulative distribution for dN proportional to p^idx_pow dp
    for (int i = 0; i < number; i++)
    {
        profile[i] = std::pow((tmp_max - tmp_min)*random(rand_generator) + tmp_min, 1.0/(idx_pow + 1.0));
    }
}

// =========================================================================================================================
// smoothed power-law profiles
// =========================================================================================================================

// evaluate an unnormalized Gaussian convolution kernel
inline static __host__
real _get_gaussian (real x, real mu, real std)
{
    return std::exp(-(x - mu)*(x - mu)/(2.0*std*std));
}

// evaluate the compactly supported power-law profile before edge smoothing
inline static __host__
real _get_tapered_pow (real x, real x_min, real x_max, real idx_pow)
{
    if (x >= x_min && x <= x_max)
    {
        return std::pow(x / x_min, idx_pow);
    }
    else
    {
        return 0.0;
    }
}

// convolve a tapered power law with a Gaussian on a uniform numerical grid
inline static __host__
void _get_convpow_profile (std::vector <real> &x_axis, std::vector <real> &y_axis,
    real x_min, real x_max, real idx_pow, real smooth, int bins)
{
    real p_min = x_min + 2.0*smooth;
    real p_max = x_max - 2.0*smooth;
    real dx = (x_max - x_min) / static_cast<real>(bins);

    x_axis.resize(bins + 1);
    y_axis.assign(bins + 1, 0.0);

    for (int i = 0; i <= bins; i++) x_axis[i] = x_min + static_cast<real>(i)*dx;

    // accumulate every source bin into every destination bin
    for (int j = 0; j <= bins; j++)
    {
        for (int k = 0; k <= bins; k++)
        {
            y_axis[k] += _get_tapered_pow(x_axis[j], p_min, p_max, idx_pow)*_get_gaussian(x_axis[k], x_axis[j], 0.5*smooth);
        }
    }
}

// calculate the same physical convolved dust surface-density profile used by the fluid initializer
inline static __host__
void _get_initdens_profile (std::vector <real> &radial_axis, std::vector <real> &sigma_profile)
{
    const real smooth = 0.05*R_0;
    const real src_min = Y_MIN + 2.0*smooth;
    const real src_max = Y_MAX - 2.0*smooth;
    const real sigma_kernel = 0.5*smooth;
    const real du = (Y_MAX - Y_MIN) / static_cast<real>(N_Y);
    const real norm = 1.0 / (std::sqrt(2.0*M_PI)*sigma_kernel);

    radial_axis.resize(N_Y + 1);
    sigma_profile.assign(N_Y + 1, 0.0);

    for (int iu = 0; iu <= N_Y; iu++)
    {
        radial_axis[iu] = Y_MIN + static_cast<real>(iu)*du;
    }

    for (int is = 0; is <= N_Y; is++)
    {
        real R_src = radial_axis[is];
        if (R_src < src_min || R_src > src_max) continue;

        real sigma_d = METAL_Z*SIGMA_0*std::pow(R_src / R_0, IDX_P);

        for (int iu = 0; iu <= N_Y; iu++)
        {
            real delta_R = radial_axis[iu] - R_src;
            real kernel = norm*std::exp(-0.5*delta_R*delta_R/(sigma_kernel*sigma_kernel));
            sigma_profile[iu] += sigma_d*kernel*du;
        }
    }
}

// sample the numerically convolved power law by inverse-CDF interpolation
inline __host__
void rand_convpow (real *profile, int number, real x_min, real x_max, real idx_pow, real smooth, int bins)
{
    std::vector <real> x_axis;
    std::vector <real> y_axis;
    real dx = (x_max - x_min) / static_cast<real>(bins);

    _get_convpow_profile(x_axis, y_axis, x_min, x_max, idx_pow, smooth, bins);

    std::uniform_real_distribution <real> random(0.0, 1.0);
    std::vector <real> cdf(bins + 1);
    cdf[0] = 0.0;
    
    for (int bin_idx = 1; bin_idx <= bins; bin_idx++)
    {
        cdf[bin_idx] = cdf[bin_idx - 1] + y_axis[bin_idx]*dx;
    }
    
    real cdf_total = cdf[bins];
    
    for (int bin_idx = 0; bin_idx <= bins; bin_idx++)
    {
        cdf[bin_idx] /= cdf_total;
    }

    for (int sample_idx = 0; sample_idx < number; sample_idx++)
    {
        real u_sample = random(rand_generator);
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), u_sample);
        int bin_lower = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        
        // interpolate within the selected CDF bin
        real bin_frac = (u_sample - cdf[bin_lower]) / (cdf[bin_lower + 1] - cdf[bin_lower]);
        profile[sample_idx] = x_axis[bin_lower] + bin_frac*dx;
    }
}

// interpolate a tabulated convolved profile on its uniform axis
inline static __host__
real _interp_convpow_profile (real x, const std::vector <real> &profile, real x_min, real x_max)
{
    if (x < x_min || x > x_max) return 0.0;

    int bins = static_cast<int>(profile.size()) - 1;
    real loc = (x - x_min)*static_cast<real>(bins) / (x_max - x_min);
    int idx = std::min(static_cast<int>(loc), bins - 1);
    real frac = loc - static_cast<real>(idx);

    return (1.0 - frac)*profile[idx] + frac*profile[idx + 1];
}

// evaluate the physical initialized dust density for one cylindrical position and grain size
inline static __host__
real _get_init_density (real sigma, real R, real Z, real size)
{
    if (N_Z == 1 || sigma <= 0.0) return sigma;

    real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
    real H_d = h_g*R;

    #ifdef DIFFUSION
    #ifdef CONST_NU
    real omega = std::sqrt(G*M_S / (R*R*R));
    real alpha_z = NU / (h_g*h_g*R*R*omega*SCHMIDT_Z);
    #else  // CONST_ALPHA
    real alpha_z = ALPHA / SCHMIDT_Z;
    #endif // CONST_NU

    real stokes_mid = STOKES_0*(size / S_0);
    #ifndef CONST_ST
    stokes_mid /= std::pow(R / R_0, IDX_P);
    #endif // NOT CONST_ST
    H_d *= std::sqrt(alpha_z / stokes_mid);
    #endif // DIFFUSION

    return sigma*std::exp(-0.5*Z*Z/(H_d*H_d)) / (std::sqrt(2.0*M_PI)*H_d);
}

// integrate the initialized reference-size profile over the represented spherical mesh
inline __host__
real get_dust_mass ()
{
    std::vector <real> radial_axis;
    std::vector <real> sigma_profile;
    _get_initdens_profile(radial_axis, sigma_profile);

    real vol_x = (N_X > 1) ? X_MAX - X_MIN : 2.0*M_PI;
    real dust_mass = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real yc = _get_ycent(iy);
            real R = yc*std::sin(zc);
            real Z = yc*std::cos(zc);
            real sigma = _interp_convpow_profile(R, sigma_profile, Y_MIN, Y_MAX);
            real density = _get_init_density(sigma, R, Z, S_0);
            real vol_y = _get_vol_y(iy);

            dust_mass += density*vol_x*vol_y*vol_z;
        }
    }

    return dust_mass;
}

// =========================================================================================================================
// disk position sampling
// =========================================================================================================================

#ifndef IMPORTGAS
// sample one-size dust from the joint spherical disk distribution with the exact cell measure
// draw y and z together because R = y sin(z) and Z = y cos(z) jointly determine radial and vertical dust density
// when diffusion is enabled use size to calculate the Stokes-dependent scale height before drawing the shared cell
inline __host__
void rand_disk_mono (real *pos_x, real *pos_y, real *pos_z, real size, int number)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    std::vector <real> radial_axis;
    std::vector <real> sigma_profile;
    _get_initdens_profile(radial_axis, sigma_profile);

    real mesh_dim = _get_mesh_dim();

    std::vector <real> cell_mass(N_Y*N_Z);
    std::vector <real> cdf(N_Y*N_Z + 1, 0.0);

    // integrate the local dust profile over each radial-polar cell
    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real yc = _get_ycent(iy);
            real R = yc*std::sin(zc);
            real Z = yc*std::cos(zc);
            real sigma = _interp_convpow_profile(R, sigma_profile, Y_MIN, Y_MAX);
            real density = _get_init_density(sigma, R, Z, size);

            real vol_y = _get_vol_y(iy);
            int idx = iy + iz*N_Y;
            cell_mass[idx] = density*vol_y*vol_z;
            cdf[idx + 1] = cdf[idx] + cell_mass[idx];
        }
    }

    // normalize the cell-mass CDF before inverse sampling
    real total_mass = cdf.back();
    for (real &value : cdf) value /= total_mass;

    std::vector<real> y_face_s(N_Y + 1);
    for (int iy = 0; iy <= N_Y; iy++) y_face_s[iy] = _get_s_y(_get_yedge(iy));

    std::vector<real> z_face_s(N_Z + 1);
    for (int iz = 0; iz <= N_Z; iz++) z_face_s[iz] = _get_s_z(_get_zedge(iz));

    for (int i = 0; i < number; i++)
    {
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), random(rand_generator));
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        int iy = idx_cell % N_Y;
        int iz = idx_cell / N_Y;

        pos_x[i] = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);

        // sample uniformly in the exact radial and polar volume coordinates inside the chosen cell
        real s_y0 = y_face_s[iy];
        real s_y1 = y_face_s[iy + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);
        pos_y[i] = std::pow(mesh_dim*s_y, 1.0 / mesh_dim);

        if (N_Z > 1)
        {
            real s_z0 = z_face_s[iz];
            real s_z1 = z_face_s[iz + 1];
            real s_z = s_z0 + (s_z1 - s_z0)*random(rand_generator);
            pos_z[i] = std::acos(-s_z);
        }
        else
        {
            pos_z[i] = 0.5*M_PI;
        }
    }
}

#if defined(MULTISIZE) && defined(DIFFUSION)
// precompute one normalized spatial CDF for a selected grain size
inline static __host__
void _get_disk_cdf (std::vector <real> &cdf, const std::vector <real> &sigma_profile, real size)
{
    int cells = N_Y*N_Z;
    real log_zero = -std::numeric_limits<real>::infinity();
    std::vector <real> log_mass(cells, log_zero);
    cdf.assign(cells + 1, 0.0);

    for (int iz = 0; iz < N_Z; iz++)
    {
        real zc = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real yc = _get_ycent(iy);
            real R = yc*std::sin(zc);
            real Z = yc*std::cos(zc);
            real sigma = _interp_convpow_profile(R, sigma_profile, Y_MIN, Y_MAX);
            real log_density = (sigma > 0.0) ? std::log(sigma) : log_zero;

            if (N_Z > 1 && sigma > 0.0)
            {
                real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
                real H_g = h_g*R;

                #ifdef CONST_NU
                real omega = std::sqrt(G*M_S / (R*R*R));
                real alpha_z = NU / (h_g*h_g*R*R*omega*SCHMIDT_Z);
                #else  // CONST_ALPHA
                real alpha_z = ALPHA / SCHMIDT_Z;
                #endif // CONST_NU

                real stokes_mid = STOKES_0*(size / S_0);
                #ifndef CONST_ST
                stokes_mid /= std::pow(R / R_0, IDX_P);
                #endif // NOT CONST_ST

                real H_d = H_g*std::sqrt(alpha_z / stokes_mid);
                log_density -= 0.5*Z*Z/(H_d*H_d);
                log_density -= std::log(H_d);
            }

            real vol_y = _get_vol_y(iy);
            int idx = iy + iz*N_Y;
            if (sigma > 0.0) log_mass[idx] = log_density + std::log(vol_y) + std::log(vol_z);
        }
    }

    // remove the largest logarithm before exponentiation to preserve highly settled distributions
    real max_log_mass = *std::max_element(log_mass.begin(), log_mass.end());
    for (int idx = 0; idx < cells; idx++)
    {
        cdf[idx + 1] = cdf[idx] + std::exp(log_mass[idx] - max_log_mass);
    }

    real total_mass = cdf[cells];
    for (real &value : cdf) value /= total_mass;
}

// sample polydisperse dust from the joint y-z distribution conditioned on each previously assigned grain size
// interpolate log-size CDFs because size changes the Stokes number and therefore the coupled vertical distribution
inline __host__
void rand_disk_poly (real *pos_x, real *pos_y, real *pos_z, const real *par_size, int number)
{
    auto [size_min_ptr, size_max_ptr] = std::minmax_element(par_size, par_size + number);
    real size_min = *size_min_ptr;
    real size_max = *size_max_ptr;

    if (N_Z == 1 || size_min == size_max)
    {
        rand_disk_mono(pos_x, pos_y, pos_z, size_min, number);
        return;
    }

    // tabulate conditional CDFs uniformly in log size and interpolate their normalized probabilities
    int size_bins = std::max(2, std::min(128, std::max(N_Y, N_Z)));
    int cells = N_Y*N_Z;
    real log_size_min = std::log(size_min);
    real log_size_max = std::log(size_max);
    real dlog_size = (log_size_max - log_size_min) / static_cast<real>(size_bins - 1);

    std::vector <real> radial_axis;
    std::vector <real> sigma_profile;
    _get_initdens_profile(radial_axis, sigma_profile);

    std::vector <real> cdf;
    std::vector <real> cdf_bank(static_cast<size_t>(size_bins)*static_cast<size_t>(cells + 1));
    for (int is = 0; is < size_bins; is++)
    {
        real size = std::exp(log_size_min + static_cast<real>(is)*dlog_size);
        _get_disk_cdf(cdf, sigma_profile, size);
        std::copy(cdf.begin(), cdf.end(), cdf_bank.begin() + static_cast<size_t>(is)*static_cast<size_t>(cells + 1));
    }

    real mesh_dim = _get_mesh_dim();
    std::uniform_real_distribution <real> random(0.0, 1.0);

    std::vector<real> y_face_s(N_Y + 1);
    for (int iy = 0; iy <= N_Y; iy++) y_face_s[iy] = _get_s_y(_get_yedge(iy));

    std::vector<real> z_face_s(N_Z + 1);
    for (int iz = 0; iz <= N_Z; iz++) z_face_s[iz] = _get_s_z(_get_zedge(iz));

    for (int i = 0; i < number; i++)
    {
        real loc_size = (std::log(par_size[i]) - log_size_min) / dlog_size;
        int size_lo = std::min(static_cast<int>(loc_size), size_bins - 2);
        real frac_size = loc_size - static_cast<real>(size_lo);
        const real *cdf_lo = cdf_bank.data() + static_cast<size_t>(size_lo)*static_cast<size_t>(cells + 1);
        const real *cdf_hi = cdf_lo + cells + 1;

        real sample = random(rand_generator);
        int idx_lo = 0;
        int idx_hi = cells;
        while (idx_lo < idx_hi)
        {
            int idx_mid = idx_lo + (idx_hi - idx_lo)/2;
            real prob_mid = (1.0 - frac_size)*cdf_lo[idx_mid] + frac_size*cdf_hi[idx_mid];
            if (prob_mid < sample) idx_lo = idx_mid + 1;
            else idx_hi = idx_mid;
        }

        int idx_cell = std::max(0, idx_lo - 1);
        int iy = idx_cell % N_Y;
        int iz = idx_cell / N_Y;

        pos_x[i] = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);

        real s_y0 = y_face_s[iy];
        real s_y1 = y_face_s[iy + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);
        pos_y[i] = std::pow(mesh_dim*s_y, 1.0 / mesh_dim);

        real s_z0 = z_face_s[iz];
        real s_z1 = z_face_s[iz + 1];
        real s_z = s_z0 + (s_z1 - s_z0)*random(rand_generator);
        pos_z[i] = std::acos(-s_z);
    }
}
#endif // MULTISIZE && DIFFUSION
#endif // !IMPORTGAS

// =========================================================================================================================
// collision-test random profiles
// =========================================================================================================================

#ifdef COLLISION
// solve the negative real Lambert-W branch with Newton iteration
inline static __host__
real _get_lambertW_m1 (real z, int max_iter = 50, real tol = 1e-12)
{
    if (z < -1.0 / std::exp(1.0) || z >= 0.0) 
    {
        throw std::domain_error("lambertWm1: z out of domain");
    }

    // initialize from the asymptotic form of the negative branch
    double val = std::log(-z);

    for (int i = 0; i < max_iter; ++i)
    {
        double exp_val = std::exp(val);
        double d_val = (val*exp_val - z) / (exp_val*(val + 1.0));
        
        val -= d_val;

        if (std::abs(d_val) < tol*(1.0 + std::abs(val))) 
        {
            return val;
        }
    }

    throw std::runtime_error("lambertWm1: did not converge");
}

// sample the analytic initial distribution used by the linear-kernel collision test
inline __host__
void rand_gamma_k2 (real *profile, int number)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = -(_get_lambertW_m1((random(rand_generator) - 1.0) / std::exp(1.0)) + 1.0);
    }
}
#endif // COLLISION

// =========================================================================================================================
// imported dust distribution
// =========================================================================================================================

#ifdef IMPORTGAS
// sample positions from imported gas density times dust-to-gas ratio and exact cell measure
inline __host__
void rand_from_file (real *pos_x, real *pos_y, real *pos_z, int number, const real *gas_dens, const real *epsilon)
{
    std::uniform_real_distribution<real> random(0.0, 1.0);
    
    real mesh_dim = _get_mesh_dim();
    
    // integrate the imported dust density with the same exact disk cell measure
    std::vector <real> dust_mass(N_G);
    real total_mass = 0.0;
    
    for (int idx_z = 0; idx_z < N_Z; idx_z++)
    {
        real vol_z = _get_vol_z(idx_z);
        
        for (int idx_y = 0; idx_y < N_Y; idx_y++)
        {
            real vol_y = _get_vol_y(idx_y);
            
            for (int idx_x = 0; idx_x < N_X; idx_x++)
            {
                real vol_x = _get_vol_x();
                real cell_volume = vol_x*vol_y*vol_z;
                
                int idx = idx_x + idx_y*N_X + idx_z*N_X*N_Y;
                dust_mass[idx] = gas_dens[idx]*epsilon[idx]*cell_volume;
                total_mass += dust_mass[idx];
            }
        }
    }
    
    // build the cell-mass cumulative distribution
    std::vector <real> cdf(N_G + 1);
    cdf[0] = 0.0;
    
    for (int idx = 0; idx < N_G; idx++)
    {
        cdf[idx + 1] = cdf[idx] + dust_mass[idx];
    }
    
    // normalize the CDF to unit total probability
    for (int idx = 0; idx <= N_G; idx++)
    {
        cdf[idx] /= total_mass;
    }

    std::vector<real> y_face_s(N_Y + 1);
    for (int idx_y = 0; idx_y <= N_Y; idx_y++) y_face_s[idx_y] = _get_s_y(_get_yedge(idx_y));

    std::vector<real> z_face_s(N_Z + 1);
    for (int idx_z = 0; idx_z <= N_Z; idx_z++) z_face_s[idx_z] = _get_s_z(_get_zedge(idx_z));
    
    // select cells by inverse transform sampling
    for (int i = 0; i < number; i++)
    {
        real u_sample = random(rand_generator);
        
        // locate the cell containing the sampled cumulative probability
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), u_sample);
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        
        int idx_x = idx_cell % N_X;
        int idx_y = (idx_cell / N_X) % N_Y;
        int idx_z = idx_cell / (N_X * N_Y);
        
        // sample logarithmic radial cells uniformly in the exact radial volume coordinate
        real s_y0 = y_face_s[idx_y];
        real s_y1 = y_face_s[idx_y + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);

        // sample azimuth uniformly and polar angle uniformly in cos(z)
        pos_x[i] = X_MIN + dx*(static_cast<real>(idx_x) + random(rand_generator));
        pos_y[i] = std::pow(mesh_dim*s_y, 1.0 / mesh_dim);
        if (N_Z > 1)
        {
            real s_z0 = z_face_s[idx_z];
            real s_z1 = z_face_s[idx_z + 1];
            real s_z = s_z0 + (s_z1 - s_z0)*random(rand_generator);
            pos_z[i] = std::acos(-s_z);
        }
        else
        {
            pos_z[i] = 0.5*M_PI;
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
    << " (" << static_cast<int>(status) << ")\n";
    
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
real int_pow (int base, int exp)
{
    int result = 1;

    for (int i = 0; i < exp; i++)
    {
        result *= base;
    }
    
    return static_cast<real>(result);
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

// write a contiguous host array without format conversion
template <typename DataType> __host__ inline
bool save_binary (const std::string &file_name, DataType *data, int number)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

// read a contiguous host array without format conversion
template <typename DataType> __host__ inline
bool load_binary (const std::string &file_name, DataType *data, int number)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

// convert internal angular variables to linear azimuthal and polar velocities before file output
inline __host__
void save_sam_as_velocity (swarm *particle)
{
    for (int idx = 0; idx < N_P; idx++)
    {
        real y = particle[idx].position.y;
        real z = particle[idx].position.z;
        real R = y*std::sin(z);

        particle[idx].velocity.x = (R > 0.0) ? particle[idx].velocity.x / R : 0.0;
        particle[idx].velocity.z = (y > 0.0) ? particle[idx].velocity.z / y : 0.0;
    }
}

// convert linear azimuthal and polar file velocities to the internal angular variables
inline __host__
void load_velocity_as_sam (swarm *particle)
{
    for (int idx = 0; idx < N_P; idx++)
    {
        real y = particle[idx].position.y;
        real z = particle[idx].position.z;
        real R = y*std::sin(z);

        particle[idx].velocity.x *= R;
        particle[idx].velocity.z *= y;
    }
}

// =========================================================================================================================
// file naming, loading, and metadata
// =========================================================================================================================

// format a frame index with the zero padding used by binary output files
inline __host__
std::string frame_num (int number)
{
    std::string str = std::to_string(number);
    int length = std::max(5, static_cast<int>(std::to_string(SAVE_MAX).length()));
    if (str.length() < length) str.insert(0, length - str.length(), '0');
    return str;
}

// report completion time for one output frame
inline __host__
void msg_output (int idx_file)
{
    std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    int length = std::max(3, static_cast<int>(std::to_string(SAVE_MAX).length()));
    std::cout   
    << std::endl << std::setfill('0')
    << std::setw(length) << idx_file << "/" 
    << std::setw(length) << SAVE_MAX << " finished on " << std::ctime(&end_time)
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
        if (idx_file <= 0)
        {
            return false;
        }

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
    std::string fname = path + "epsilon_" + frame_num(idx_file) + ".dat";
    return load_binary(fname, epsilon, N_G);
}

// load density and all three linear velocity components for one gas frame
inline __host__
bool load_gas_data (const std::string &path, int idx_file, real *gas_dens, real *gas_velx, real *gas_vely, real *gas_velz)
{
    std::string fname;
    bool success = true;
    
    fname = path + "gasdens_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gas_dens, N_G);
    
    fname = path + "gasvelx_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gas_velx, N_G);
    
    fname = path + "gasvely_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gas_vely, N_G);
    
    fname = path + "gasvelz_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gas_velz, N_G);
    
    return success;
}
#endif // IMPORTGAS

// write the active physical, numerical, grid, and binary-layout configuration
inline __host__
bool save_variable (const std::string &file_name, real dust_mass)
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
    #ifdef VISC_ACCRETION
    file << "VISC_ACCRETION = " << std::defaultfloat << 1                                << std::endl;
    #endif // VISC_ACCRETION
    #ifdef COLLISION
    #ifdef CODE_UNIT
    file << "RE_0        = " << std::scientific     << std::setprecision(8) << RE_0         << std::endl;
    #else  // PHYSICAL_UNIT
    file << "M_MOL       = " << std::scientific     << std::setprecision(8) << M_MOL        << std::endl;
    file << "X_SEC       = " << std::scientific     << std::setprecision(8) << X_SEC        << std::endl;
    #endif // CODE_UNIT
    #endif // COLLISION
    file                                                                                    << std::endl;
    
    // dust parameters
    file << "STOKES_0    = " << std::scientific     << std::setprecision(8) << STOKES_0     << std::endl;
    file << "DUST_MASS   = " << std::scientific     << std::setprecision(8) << dust_mass    << std::endl;
    file << "RHO_0       = " << std::scientific     << std::setprecision(8) << RHO_0        << std::endl;
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific     << std::setprecision(8) << BETA_0       << std::endl;
    file << "KAPPA_0     = " << std::scientific     << std::setprecision(8) << KAPPA_0      << std::endl;
    file << "T_BETA      = " << std::scientific     << std::setprecision(8) << T_BETA       << std::endl;
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
         << N_P / (N_K - 1.0) / dust_mass << std::endl;
    file << "V_FRAG      = " << std::scientific     << std::setprecision(8) << V_FRAG       << std::endl;
    file << "COAG_KERNEL = " << std::defaultfloat   << std::setprecision(8) << COAG_KERNEL  << std::endl;
    file << "N_K         = " << std::defaultfloat   << std::setprecision(8) << N_K          << std::endl;
    file << "H_SEARCH    = " << std::defaultfloat   << std::setprecision(8) << H_SEARCH     << std::endl;
    file << "CFL_COL     = " << std::defaultfloat   << std::setprecision(8) << CFL_COL      << std::endl;
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

#define SAVE_PARTICLE_TO_FILE(idx)                                                          \
do {                                                                                        \
    CUDA_CHECK(cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost));          \
    save_sam_as_velocity(particle);                                                         \
    std::string fname = PATH + "particle_" + frame_num(idx) + ".dat";                       \
    save_binary(fname, particle, N_P);                                                      \
} while(0)

#define LOAD_PARTICLE_TO_VRAM(idx)                                                          \
do {                                                                                        \
    std::string fname = PATH + "particle_" + frame_num(idx) + ".dat";                       \
    if (!load_binary(fname, particle, N_P))                                                 \
    {                                                                                       \
        std::cerr << "Error: Failed to load file: " << fname << std::endl;                  \
        return 1;                                                                           \
    }                                                                                       \
    load_velocity_as_sam(particle);                                                         \
    if (N_Z == 1)                                                                           \
    {                                                                                       \
        for (int idx_particle = 0; idx_particle < N_P; idx_particle++)                     \
        {                                                                                   \
            particle[idx_particle].position.z = 0.5*M_PI;                                  \
            particle[idx_particle].velocity.z = 0.0;                                       \
        }                                                                                   \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_P, cudaMemcpyHostToDevice));          \
} while(0)

#ifdef IMPORTGAS
#define LOAD_GAS_DATA_TO_VRAM(idx)                                                          \
do {                                                                                        \
    if (!load_gas_data(PATH, idx, gas_dens, gas_velx, gas_vely, gas_velz))                      \
    {                                                                                       \
        std::cerr << "Error: Failed to load gas data files for frame " << idx << std::endl; \
        return 1;                                                                           \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(dev_gas_dens, gas_dens, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_velx, gas_velx, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_vely, gas_vely, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
    CUDA_CHECK(cudaMemcpy(dev_gas_velz, gas_velz, sizeof(real)*N_G, cudaMemcpyHostToDevice));             \
} while(0)

#define LOAD_GAS_NEXT_TO_VRAM(idx)                                                          \
do {                                                                                        \
    if (!load_gas_data(PATH, idx, gas_dens, gas_velx, gas_vely, gas_velz))                      \
    {                                                                                       \
        std::cerr << "Error: Failed to load gas data files for frame " << idx << std::endl; \
        return 1;                                                                           \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpy(dev_gas_dens_next, gas_dens, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_velx_next, gas_velx, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_vely_next, gas_vely, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
    CUDA_CHECK(cudaMemcpy(dev_gas_velz_next, gas_velz, sizeof(real)*N_G, cudaMemcpyHostToDevice));        \
} while(0)
#endif // IMPORTGAS

#ifdef SAVE_DENS
#define SAVE_DUSTDENS_TO_FILE(idx)                                                          \
do {                                                                                        \
    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);                                         \
    CUDA_KERNEL_CHECK("dustdens_init");                                                     \
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, dust_mass);                \
    CUDA_KERNEL_CHECK("dustdens_depo");                                                     \
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);                                         \
    CUDA_KERNEL_CHECK("dustdens_calc");                                                     \
    CUDA_CHECK(cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost));           \
    std::string fname = PATH + "dustdens_" + frame_num(idx) + ".dat";                       \
    save_binary(fname, dustdens, N_G);                                                      \
} while(0)
#endif // SAVE_DENS

#ifdef RADIATION
#define SAVE_OPTDEPTH_TO_FILE(idx, do_avg)                                                  \
do {                                                                                        \
    optdepth_init <<< NB_G, TPB >>> (dev_optdepth);                                         \
    CUDA_KERNEL_CHECK("optdepth_init");                                                     \
    optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, dust_mass);                \
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
    std::string fname = PATH + "optdepth_" + frame_num(idx) + ".dat";                       \
    save_binary(fname, optdepth, N_G);                                                      \
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
