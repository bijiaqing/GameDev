#ifndef GRAFFITI_HOST_CUH 
#define GRAFFITI_HOST_CUH

#include <algorithm>        // for std::lower_bound
#include <chrono>           // for std::chrono::system_clock
#include <ctime>            // for std::time_t, std::time, std::ctime
#include <cmath>            // for std::abs, std::exp, std::log, std::pow, std::sqrt
#include <fstream>          // for std::ofstream, std::ifstream
#include <iomanip>          // for std::setw, std::setfill, std::setprecision
#include <iostream>         // for std::cout, std::endl
#include <random>           // for std::mt19937
#include <stdexcept>        // for std::domain_error, std::runtime_error
#include <string>           // for std::string
#include <vector>           // for std::vector

#include <const.cuh>

// =========================================================================================================================
// Random profile generation
// Note: rand_generator must be defined in exactly one .cu file (e.g., main.cu)
// =========================================================================================================================

extern std::mt19937 rand_generator;

inline __host__
void rand_uniform (real *profile, int number, real p_min, real p_max)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = p_min + (p_max - p_min)*random(rand_generator);
    }
}

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

inline __host__
void rand_pow_law (real *profile, int number, real p_min, real p_max, real idx_pow)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    real tmp_min = std::pow(p_min, idx_pow + 1.0);
    real tmp_max = std::pow(p_max, idx_pow + 1.0);

    // check https://mathworld.wolfram.com/RandomNumber.html for derivations
    // NOTE: this is the probability distribution function dN(x) ~ x^n*dx
    for (int i = 0; i < number; i++)
    {
        profile[i] = std::pow((tmp_max - tmp_min)*random(rand_generator) + tmp_min, 1.0/(idx_pow + 1.0));
    }
}

// ========================================================================================================================
// rand_convpow: generate random numbers following a convolved power-law distribution using inverse transform sampling

inline static __host__
real _get_gaussian (real x, real mu, real std)
{
    return std::exp(-(x - mu)*(x - mu)/(2.0*std*std));
}

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

inline __host__
void rand_convpow (real *profile, int number, real x_min, real x_max, real idx_pow, real smooth, int bins)
{
    real p_min = x_min + 2.0*smooth;
    real p_max = x_max - 2.0*smooth;
    
    std::vector <real> x_axis(bins + 1);
    std::vector <real> y_axis(bins + 1, 0.0);
    
    real dx = (x_max - x_min) / static_cast<real>(bins);
    
    for (int i = 0; i < bins + 1; i++)
    {
        x_axis[i] = x_min + i*dx;
    }

    for (int j = 0; j < bins + 1; j++)
    {
        for (int k = 0; k < bins + 1; k++)
        {
            y_axis[k] += _get_tapered_pow(x_axis[j], p_min, p_max, idx_pow)*_get_gaussian(x_axis[k], x_axis[j], 0.5*smooth);
        }
    }

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
        
        // interpolate between x_axis[bin_lower] and x_axis[bin_lower+1]
        real bin_frac = (u_sample - cdf[bin_lower]) / (cdf[bin_lower + 1] - cdf[bin_lower]);
        profile[sample_idx] = x_axis[bin_lower] + bin_frac*dx;
    }
}

// =========================================================================================================================
// Random profile generation for special cases (e.g., Lambert W function for linear coagulation kernel)

#ifdef COLLISION
inline static __host__
real _get_lambertW_m1 (real z, int max_iter = 50, real tol = 1e-12)
{
    if (z < -1.0 / std::exp(1.0) || z >= 0.0) 
    {
        throw std::domain_error("lambertWm1: z out of domain");
    }

    // initial guess (asymptotic for k = -1)
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

inline __host__
void rand_4_linear (real *profile, int number) // initial distribution for linear kernel tests
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = -(_get_lambertW_m1((random(rand_generator) - 1.0) / std::exp(1.0)) + 1.0);
    }
}
#endif // COLLISION

// =========================================================================================================================
// rand_from_file: generate random positions following dust mass distribution from gas density and dust-to-gas ratio

#ifdef IMPORTGAS
inline __host__
void rand_from_file (real *pos_x, real *pos_y, real *pos_z, int number, const real *gas_dens, const real *epsilon)
{
    std::uniform_real_distribution<real> random(0.0, 1.0);
    
    real dx =         (X_MAX - X_MIN)     / static_cast<real>(N_X);
    real dy = std::pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real dz =         (Z_MAX - Z_MIN)     / static_cast<real>(N_Z);
    
    real idx_dim = static_cast<real>(N_X > 1) + static_cast<real>(N_Z > 1) + 1.0;
    real dy_pow = std::pow(dy, idx_dim);
    
    // compute dust mass in each cell using same volume calculation as _get_grid_volume
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
    
    // build cumulative distribution function
    std::vector <real> cdf(N_G + 1);
    cdf[0] = 0.0;
    
    for (int idx = 0; idx < N_G; idx++)
    {
        cdf[idx + 1] = cdf[idx] + dust_mass[idx];
    }
    
    // normalize CDF to [0, 1]
    for (int idx = 0; idx <= N_G; idx++)
    {
        cdf[idx] /= total_mass;
    }
    
    // sample particles using inverse transform sampling
    for (int i = 0; i < number; i++)
    {
        real u_sample = random(rand_generator);
        
        // binary search to find cell
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), u_sample);
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        
        int idx_x = idx_cell % N_X;
        int idx_y = (idx_cell / N_X) % N_Y;
        int idx_z = idx_cell / (N_X * N_Y);
        
        // y: radius (logarithmic spacing) - sample uniformly in r^idx_dim within cell
        real y0 = Y_MIN*std::pow(dy, static_cast<real>(idx_y));
        real y0_pow = std::pow(y0, idx_dim);
        real y_pow = y0_pow*(1.0 + (dy_pow - 1.0)*random(rand_generator));

        // randomly position particle within the cell
        pos_x[i] = X_MIN + dx*(static_cast<real>(idx_x) + random(rand_generator));
        pos_y[i] = std::pow(y_pow, 1.0 / idx_dim);
        pos_z[i] = Z_MIN + dz*(static_cast<real>(idx_z) + random(rand_generator));
    }
}
#endif // IMPORTGAS

// =========================================================================================================================
// Get dt_out based on output index and output mode
// =========================================================================================================================

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
// Binary file I/O templates
// =========================================================================================================================

template <typename DataType> __host__ inline
bool save_binary (const std::string &file_name, DataType *data, int number)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

template <typename DataType> __host__ inline
bool load_binary (const std::string &file_name, DataType *data, int number)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

// =========================================================================================================================
// File I/O helpers
// =========================================================================================================================

inline __host__
std::string frame_num (int number)
{
    std::string str = std::to_string(number);
    int length = std::max(5, static_cast<int>(std::to_string(SAVE_MAX).length()));
    if (str.length() < length) str.insert(0, length - str.length(), '0');
    return str;
}

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
inline __host__
bool is_log_power (int idx_file)
{
    // Check if n is an integer power of LOG_BASE (including 0th power: n=1)
    if (LOG_BASE == 2)
    {
        // Fast bit-manipulation for powers of 2
        return (idx_file > 0) && ((idx_file & (idx_file - 1)) == 0);
    }
    else
    {
        // General case for any base
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
inline __host__
bool load_epsilon (const std::string &path, int idx_file, real *epsilon)
{
    std::string fname = path + "epsilon_" + frame_num(idx_file) + ".dat";
    return load_binary(fname, epsilon, N_G);
}

inline __host__
bool load_gas_data (const std::string &path, int idx_file, real *gasdens, real *gasvelx, real *gasvely, real *gasvelz)
{
    std::string fname;
    bool success = true;
    
    fname = path + "gasdens_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gasdens, N_G);
    
    fname = path + "gasvelx_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gasvelx, N_G);
    
    fname = path + "gasvely_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gasvely, N_G);
    
    fname = path + "gasvelz_" + frame_num(idx_file) + ".dat";
    success &= load_binary(fname, gasvelz, N_G);
    
    return success;
}
#endif // IMPORTGAS

inline __host__
bool save_variable (const std::string &file_name)
{
    std::ofstream file(file_name);
    if (!file) return false;
    
    file << "[PARAMETERS]"                                                                  << std::endl;
    file                                                                                    << std::endl;

    // Gas parameters
    file << "SIGMA_0     = " << std::scientific     << std::setprecision(8) << SIGMA_0      << std::endl;
    file << "ASPR_0      = " << std::defaultfloat   << std::setprecision(8) << ASPR_0       << std::endl;
    file << "IDX_P       = " << std::defaultfloat   << std::setprecision(8) << IDX_P        << std::endl;
    file << "IDX_Q       = " << std::defaultfloat   << std::setprecision(8) << IDX_Q        << std::endl;
    #if (defined(TRANSPORT) && defined(DIFFUSION)) || defined(COLLISION)
    #ifndef CONST_NU
    file << "ALPHA       = " << std::scientific     << std::setprecision(8) << ALPHA        << std::endl;
    #else  // CONST_NU
    file << "NU          = " << std::scientific     << std::setprecision(8) << NU           << std::endl;
    #endif // NOT CONST_NU
    #endif // DIFFUSION or COLLISION
    #ifdef COLLISION
    #ifndef CODE_UNIT
    file << "M_MOL       = " << std::scientific     << std::setprecision(8) << M_MOL        << std::endl;
    file << "X_SEC       = " << std::scientific     << std::setprecision(8) << X_SEC        << std::endl;
    #else  // CODE_UNIT
    file << "RE_0        = " << std::scientific     << std::setprecision(8) << RE_0         << std::endl;
    #endif // NOT CODE_UNIT
    #endif // COLLISION
    file                                                                                    << std::endl;
    
    // Dust parameters
    file << "ST_0        = " << std::scientific     << std::setprecision(8) << ST_0         << std::endl;
    file << "M_D         = " << std::scientific     << std::setprecision(8) << M_D          << std::endl;
    file << "RHO_0       = " << std::scientific     << std::setprecision(8) << RHO_0        << std::endl;
    #if (defined(TRANSPORT) && defined(RADIATION))
    file << "BETA_0      = " << std::scientific     << std::setprecision(8) << BETA_0       << std::endl;
    file << "KAPPA_0     = " << std::scientific     << std::setprecision(8) << KAPPA_0      << std::endl;
    #endif // RADIATION
    #if (defined(TRANSPORT) && defined(DIFFUSION))
    file << "SC_X        = " << std::scientific     << std::setprecision(8) << SC_X         << std::endl;
    file << "SC_R        = " << std::scientific     << std::setprecision(8) << SC_R         << std::endl;
    #endif // DIFFUSION
    #if (defined(TRANSPORT) && defined(DIFFUSION)) || defined(COLLISION)
    file << "SC_Z        = " << std::scientific     << std::setprecision(8) << SC_Z         << std::endl;
    #endif // DIFFUSION or COLLISION

    #ifdef COLLISION
    file << "LAMBDA_0    = " << std::scientific     << std::setprecision(8) << LAMBDA_0     << std::endl;
    file << "V_FRAG      = " << std::scientific     << std::setprecision(8) << V_FRAG       << std::endl;
    file << "COAG_KERNEL = " << std::defaultfloat   << std::setprecision(8) << COAG_KERNEL  << std::endl;
    file << "N_K         = " << std::defaultfloat   << std::setprecision(8) << N_K          << std::endl;
    #endif // COLLISION
    file                                                                                    << std::endl;

    // Mesh domain
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

    // Initialization parameters
    #ifndef IMPORTGAS
    file << "INIT_XMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_XMIN    << std::endl;
    file << "INIT_XMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_XMAX    << std::endl;
    file << "INIT_YMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_YMIN    << std::endl;
    file << "INIT_YMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_YMAX    << std::endl;
    file << "INIT_ZMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_ZMIN    << std::endl;
    file << "INIT_ZMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_ZMAX    << std::endl;
    #endif // NOT IMPORTGAS
    file << "INIT_SMIN   = " << std::scientific     << std::setprecision(8) << INIT_SMIN    << std::endl;
    file << "INIT_SMAX   = " << std::scientific     << std::setprecision(8) << INIT_SMAX    << std::endl;
    file                                                                                    << std::endl;

    // Time step and output
    file << "SAVE_MAX    = " << std::defaultfloat   << std::setprecision(8) << SAVE_MAX     << std::endl;
    #if defined(LOGTIMING) || defined(LOGOUTPUT)
    file << "LOG_BASE    = " << std::defaultfloat   << std::setprecision(8) << LOG_BASE     << std::endl;
    #else  // LINEAR
    file << "LIN_BASE    = " << std::defaultfloat   << std::setprecision(8) << LIN_BASE     << std::endl;
    #endif // LOGOUTPUT or LOGTIMING
    file << "DT_OUT      = " << std::scientific     << std::setprecision(8) << DT_OUT       << std::endl;
    #ifdef TRANSPORT
    file << "DT_DYN      = " << std::scientific     << std::setprecision(8) << DT_DYN       << std::endl;
    #endif // TRANSPORT
    file << "DT_MIN      = " << std::scientific     << std::setprecision(8) << DT_MIN       << std::endl;
    file                                                                                    << std::endl;

    // Swarm structure as numpy dtype (configparser-compatible), in python, write as:
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
    #ifdef COLLISION
    file << "col_rate   = f8"                                                               << std::endl;
    file << "max_dist   = f8"                                                               << std::endl;
    #endif // COLLISION
    
    return file.good();
}

// =========================================================================================================================
// File I/O macros for main evolution loop
// =========================================================================================================================

#define SAVE_PARTICLE_TO_FILE(idx)                                                          \
do {                                                                                        \
    cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost);          \
    fname = PATH + "particle_" + frame_num(idx) + ".dat";                                   \
    save_binary(fname, particle, N_P);                                                      \
} while(0)

#define LOAD_PARTICLE_TO_VRAM(idx)                                                          \
do {                                                                                        \
    fname = PATH + "particle_" + frame_num(idx) + ".dat";                                   \
    if (!load_binary(fname, particle, N_P))                                                 \
    {                                                                                       \
        std::cerr << "Error: Failed to load file: " << fname << std::endl;                  \
        return 1;                                                                           \
    }                                                                                       \
    cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_P, cudaMemcpyHostToDevice);          \
} while(0)

#ifdef IMPORTGAS
#define LOAD_GAS_DATA_TO_VRAM(idx)                                                          \
do {                                                                                        \
    if (!load_gas_data(PATH, idx, gasdens, gasvelx, gasvely, gasvelz))                      \
    {                                                                                       \
        std::cerr << "Error: Failed to load gas data files for frame " << idx << std::endl; \
        return 1;                                                                           \
    }                                                                                       \
    cudaMemcpy(dev_gasdens, gasdens, sizeof(real)*N_G, cudaMemcpyHostToDevice);             \
    cudaMemcpy(dev_gasvelx, gasvelx, sizeof(real)*N_G, cudaMemcpyHostToDevice);             \
    cudaMemcpy(dev_gasvely, gasvely, sizeof(real)*N_G, cudaMemcpyHostToDevice);             \
    cudaMemcpy(dev_gasvelz, gasvelz, sizeof(real)*N_G, cudaMemcpyHostToDevice);             \
} while(0)
#endif // IMPORTGAS

#ifdef SAVE_DENS
#define SAVE_DUSTDENS_TO_FILE(idx)                                                          \
do {                                                                                        \
    dustdens_init <<< NB_A, TPB >>> (dev_dustdens);                                         \
    dustdens_scat <<< NB_P, TPB >>> (dev_dustdens, dev_particle);                           \
    dustdens_calc <<< NB_A, TPB >>> (dev_dustdens);                                         \
    cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost);           \
    fname = PATH + "dustdens_" + frame_num(idx) + ".dat";                                   \
    save_binary(fname, dustdens, N_G);                                                      \
} while(0)
#endif // SAVE_DENS

#if defined(TRANSPORT) && defined(RADIATION)
#define SAVE_OPTDEPTH_TO_FILE(idx, do_avg)                                                  \
do {                                                                                        \
    optdepth_init <<< NB_A, TPB >>> (dev_optdepth);                                         \
    optdepth_scat <<< NB_P, TPB >>> (dev_optdepth, dev_particle);                           \
    optdepth_calc <<< NB_A, TPB >>> (dev_optdepth);                                         \
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);                                         \
    if (do_avg) optdepth_mean <<< NB_X, TPB >>> (dev_optdepth);                             \
    cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost);           \
    fname = PATH + "optdepth_" + frame_num(idx) + ".dat";                                   \
    save_binary(fname, optdepth, N_G);                                                      \
} while(0)
#endif // TRANSPORT && RADIATION

// =========================================================================================================================
// Console output macros (access local variables directly, no parameters needed)
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
#else // NO TRANSPORT
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

#endif // NOT GRAFFITI_HOST_CUH
