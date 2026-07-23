#ifndef HOST_RAND_CUH
#define HOST_RAND_CUH

#include <algorithm>        // for std::copy, std::lower_bound, std::max_element, std::minmax_element
#include <cmath>            // for std::abs, std::acos, std::cos, std::exp, std::log, std::pow, std::sin, std::sqrt
#include <limits>           // for std::numeric_limits
#include <random>           // for std::mt19937
#include <stdexcept>        // for std::domain_error, std::runtime_error
#include <vector>           // for std::vector

#include <const.cuh>
#include <paramgrid.cuh>

// =========================================================================================================================
// elementary random profiles
// =========================================================================================================================

// share one deterministic host generator across all initialization samplers
extern std::mt19937 rand_generator;

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

#ifndef IMPORTGAS
// sample one-size dust from the joint spherical disk distribution with the exact cell measure
// draw y and z together because R = y sin(z) and Z = y cos(z) jointly determine surface density and stratification
// when diffusion is enabled use size to calculate the Stokes-dependent scale height before drawing the shared cell
inline __host__
void rand_disk_mono (real *pos_x, real *pos_y, real *pos_z, real size, int number)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    std::vector <real> radial_axis;
    std::vector <real> sigma_profile;
    _get_convpow_profile(radial_axis, sigma_profile, Y_MIN, Y_MAX, IDX_P, 0.05*R_0, N_Y);

    real dy = _get_dy();
    real dz = _get_dz();
    real pow_y = 1.0 + static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1);
    real dy_pow = std::pow(dy, pow_y);

    std::vector <real> cell_mass(N_Y*N_Z);
    std::vector <real> cdf(N_Y*N_Z + 1, 0.0);

    // integrate the local dust profile over each radial-polar cell
    for (int iz = 0; iz < N_Z; iz++)
    {
        real z0 = (N_Z > 1) ? Z_MIN + static_cast<real>(iz)*dz : 0.5*(Z_MIN + Z_MAX);
        real z1 = (N_Z > 1) ? z0 + dz : z0;
        real zc = (N_Z > 1) ? 0.5*(z0 + z1) : z0;
        real vol_z = (N_Z > 1) ? std::cos(z0) - std::cos(z1) : 1.0;

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*std::pow(dy, static_cast<real>(iy));
            real y1 = y0*dy;
            real yc = std::sqrt(y0*y1);
            real R = yc*std::sin(zc);
            real Z = yc*std::cos(zc);
            real sigma = _interp_convpow_profile(R, sigma_profile, Y_MIN, Y_MAX);
            real density = sigma;

            if (N_Z > 1 && sigma > 0.0)
            {
                real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
                real H_g = h_g*R;
                real H_d = H_g;

                #ifdef DIFFUSION
                #ifndef CONST_NU
                real alpha_z = ALPHA / SCHMIDT_Z;
                #else  // CONST_NU
                real omega = std::sqrt(G*M_S / (R*R*R));
                real alpha_z = NU / (h_g*h_g*R*R*omega*SCHMIDT_Z);
                #endif // NOT CONST_NU

                real stokes_mid = ST_0*(size / S_0);
                #ifndef CONST_ST
                stokes_mid /= std::pow(R / R_0, IDX_P);
                #endif // NOT CONST_ST

                H_d *= std::sqrt(alpha_z / (alpha_z + stokes_mid));
                #endif // DIFFUSION

                real gas_strat = std::exp((R / yc - 1.0) / (h_g*h_g));
                real settle_exp = std::exp(-0.5*Z*Z*(1.0/(H_d*H_d) - 1.0/(H_g*H_g)));
                density = sigma*gas_strat*settle_exp / H_d;
            }

            real vol_y = std::pow(y0, pow_y)*(dy_pow - 1.0) / pow_y;
            int idx = iy + iz*N_Y;
            cell_mass[idx] = density*vol_y*vol_z;
            cdf[idx + 1] = cdf[idx] + cell_mass[idx];
        }
    }

    // normalize the cell-mass CDF before inverse sampling
    real total_mass = cdf.back();
    for (real &value : cdf) value /= total_mass;

    for (int i = 0; i < number; i++)
    {
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), random(rand_generator));
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        int iy = idx_cell % N_Y;
        int iz = idx_cell / N_Y;

        pos_x[i] = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);

        // sample uniformly in the exact radial and polar volume coordinates inside the chosen cell
        real y0 = Y_MIN*std::pow(dy, static_cast<real>(iy));
        real y_pow = std::pow(y0, pow_y)*(1.0 + (dy_pow - 1.0)*random(rand_generator));
        pos_y[i] = std::pow(y_pow, 1.0 / pow_y);

        if (N_Z > 1)
        {
            real z0 = Z_MIN + static_cast<real>(iz)*dz;
            real cos_z = std::cos(z0) + (std::cos(z0 + dz) - std::cos(z0))*random(rand_generator);
            pos_z[i] = std::acos(cos_z);
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
    real dy = _get_dy();
    real dz = _get_dz();
    real pow_y = 1.0 + static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1);
    real dy_pow = std::pow(dy, pow_y);

    int cells = N_Y*N_Z;
    real log_zero = -std::numeric_limits<real>::infinity();
    std::vector <real> log_mass(cells, log_zero);
    cdf.assign(cells + 1, 0.0);

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z0 = (N_Z > 1) ? Z_MIN + static_cast<real>(iz)*dz : 0.5*(Z_MIN + Z_MAX);
        real z1 = (N_Z > 1) ? z0 + dz : z0;
        real zc = (N_Z > 1) ? 0.5*(z0 + z1) : z0;
        real vol_z = (N_Z > 1) ? std::cos(z0) - std::cos(z1) : 1.0;

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*std::pow(dy, static_cast<real>(iy));
            real y1 = y0*dy;
            real yc = std::sqrt(y0*y1);
            real R = yc*std::sin(zc);
            real Z = yc*std::cos(zc);
            real sigma = _interp_convpow_profile(R, sigma_profile, Y_MIN, Y_MAX);
            real log_density = (sigma > 0.0) ? std::log(sigma) : log_zero;

            if (N_Z > 1 && sigma > 0.0)
            {
                real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
                real H_g = h_g*R;

                #ifndef CONST_NU
                real alpha_z = ALPHA / SCHMIDT_Z;
                #else  // CONST_NU
                real omega = std::sqrt(G*M_S / (R*R*R));
                real alpha_z = NU / (h_g*h_g*R*R*omega*SCHMIDT_Z);
                #endif // NOT CONST_NU

                real stokes_mid = ST_0*(size / S_0);
                #ifndef CONST_ST
                stokes_mid /= std::pow(R / R_0, IDX_P);
                #endif // NOT CONST_ST

                real H_d = H_g*std::sqrt(alpha_z / (alpha_z + stokes_mid));
                log_density += (R / yc - 1.0) / (h_g*h_g);
                log_density -= 0.5*Z*Z*(1.0/(H_d*H_d) - 1.0/(H_g*H_g));
                log_density -= std::log(H_d);
            }

            real vol_y = std::pow(y0, pow_y)*(dy_pow - 1.0) / pow_y;
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
    _get_convpow_profile(radial_axis, sigma_profile, Y_MIN, Y_MAX, IDX_P, 0.05*R_0, N_Y);

    std::vector <real> cdf;
    std::vector <real> cdf_bank(static_cast<size_t>(size_bins)*static_cast<size_t>(cells + 1));
    for (int is = 0; is < size_bins; is++)
    {
        real size = std::exp(log_size_min + static_cast<real>(is)*dlog_size);
        _get_disk_cdf(cdf, sigma_profile, size);
        std::copy(cdf.begin(), cdf.end(), cdf_bank.begin() + static_cast<size_t>(is)*static_cast<size_t>(cells + 1));
    }

    real dy = _get_dy();
    real dz = _get_dz();
    real pow_y = 1.0 + static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1);
    real dy_pow = std::pow(dy, pow_y);
    std::uniform_real_distribution <real> random(0.0, 1.0);

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

        real y0 = Y_MIN*std::pow(dy, static_cast<real>(iy));
        real y_pow = std::pow(y0, pow_y)*(1.0 + (dy_pow - 1.0)*random(rand_generator));
        pos_y[i] = std::pow(y_pow, 1.0 / pow_y);

        real z0 = Z_MIN + static_cast<real>(iz)*dz;
        real cos_z = std::cos(z0) + (std::cos(z0 + dz) - std::cos(z0))*random(rand_generator);
        pos_z[i] = std::acos(cos_z);
    }
}
#endif // MULTISIZE and DIFFUSION
#endif // NOT IMPORTGAS

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
    
    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();
    
    real idx_dim = static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1) + 1.0;
    real dy_pow = std::pow(dy, idx_dim);
    
    // integrate the imported dust density with the same cell measure as _get_grid_volume
    std::vector <real> dust_mass(N_G);
    real total_mass = 0.0;
    
    for (int idx_z = 0; idx_z < N_Z; idx_z++)
    {
        real z0 = Z_MIN + dz*static_cast<real>(idx_z);
        real vol_z = (N_Z > 1) ? (std::cos(z0) - std::cos(z0 + dz)) : 1.0;
        
        for (int idx_y = 0; idx_y < N_Y; idx_y++)
        {
            real y0 = Y_MIN*std::pow(dy, static_cast<real>(idx_y));
            real vol_y = std::pow(y0, idx_dim)*(dy_pow - 1.0) / idx_dim;
            
            for (int idx_x = 0; idx_x < N_X; idx_x++)
            {
                real vol_x = (N_X > 1) ? dx : 1.0;
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
        real y0 = Y_MIN*std::pow(dy, static_cast<real>(idx_y));
        real y0_pow = std::pow(y0, idx_dim);
        real y_pow = y0_pow*(1.0 + (dy_pow - 1.0)*random(rand_generator));

        // sample azimuth uniformly and polar angle uniformly in cos(z)
        pos_x[i] = X_MIN + dx*(static_cast<real>(idx_x) + random(rand_generator));
        pos_y[i] = std::pow(y_pow, 1.0 / idx_dim);
        if (N_Z > 1)
        {
            real z0 = Z_MIN + dz*static_cast<real>(idx_z);
            real cos_z = std::cos(z0) + (std::cos(z0 + dz) - std::cos(z0))*random(rand_generator);
            pos_z[i] = std::acos(cos_z);
        }
        else
        {
            pos_z[i] = 0.5*M_PI;
        }
    }
}
#endif // IMPORTGAS

// =========================================================================================================================

#endif // NOT HOST_RAND_CUH
