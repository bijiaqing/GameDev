#ifndef SWARM_HOST_CUH
#define SWARM_HOST_CUH

#include <algorithm>        // std::copy, std::lower_bound, std::max, std::max_element, std::minmax_element
#include <chrono>           // std::chrono::system_clock
#include <cmath>            // std::abs, std::acos, std::cos, std::exp, std::log, std::pow, std::sin, std::sqrt
#include <cstddef>          // std::size_t
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
real _get_sy (real y) { return std::pow(y, _get_mesh_dim()) / _get_mesh_dim(); }

inline __host__
real _get_sz (real z) { return -std::cos(z); }

// =========================================================================================================================
// elementary random profiles
// =========================================================================================================================

// share one deterministic host generator across all initialization samplers
extern std::mt19937 rand_generator;

#ifdef MULTISIZE
// calculate the mass scale that makes the sampled representative masses sum to the target dust mass
inline __host__
real get_mass_norm (const real *randsize, real total_dust_mass)
{
    long double weight_sum = 0.0;

    for (int idx = 0; idx < N_P; idx++)
    {
        weight_sum += static_cast<long double>(_get_mass_weight(randsize[idx]));
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

// evaluate the physical initialized dust density for one cylindrical position and grain size
inline static __host__
real _get_init_rhod (real sigma_d, real R, real Z, real size)
{
    if (N_Z == 1 || sigma_d <= 0.0) return sigma_d;

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

    return sigma_d*std::exp(-0.5*Z*Z / (H_d*H_d)) / (std::sqrt(2.0*M_PI)*H_d);
}

// integrate the initialized profile to obtain the total dust mass represented by the simulation domain
inline __host__
real get_total_dust_mass ()
{
    std::vector <real> initdens;
    initdens_calc(initdens);

    real vol_x = _get_vol_x();
    real total_dust_mass = 0.0;

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y = _get_ycent(iy);
            real R = y*std::sin(z);
            real Z = y*std::cos(z);

            real sigma_d = initdens_lerp(R, initdens);
            real rhod = _get_init_rhod(sigma_d, R, Z, S_0);
            real vol_y = _get_vol_y(iy);

            total_dust_mass += rhod*vol_x*vol_y*vol_z;
        }
    }

    return total_dust_mass;
}

// =========================================================================================================================
// disk position sampling
// =========================================================================================================================

#ifndef IMPORTGAS
// sample one-size dust from the joint spherical disk distribution with the exact cell measure
// draw y and z together because R = y sin(z) and Z = y cos(z) jointly determine radial and vertical dust density
// when diffusion is enabled use size to calculate the Stokes-dependent scale height before drawing the shared cell
inline __host__
void rand_disk_mono (real *randposx, real *randposy, real *randposz, real size, int count)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    std::vector <real> initdens;
    initdens_calc(initdens);

    real mesh_dim = _get_mesh_dim();

    std::vector <real> cell_mass(N_Y*N_Z);
    std::vector <real> cdf(N_Y*N_Z + 1, 0.0);

    // integrate the local dust profile over each radial-polar cell
    for (int iz = 0; iz < N_Z; iz++)
    {
        real z = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y = _get_ycent(iy);
            real R = y*std::sin(z);
            real Z = y*std::cos(z);

            real sigma_d = initdens_lerp(R, initdens);
            real rhod = _get_init_rhod(sigma_d, R, Z, size);

            real vol_y = _get_vol_y(iy);
            int idx_cell = iy + iz*N_Y;

            cell_mass[idx_cell] = rhod*vol_y*vol_z;
            cdf[idx_cell + 1] = cdf[idx_cell] + cell_mass[idx_cell];
        }
    }

    // normalize the cell-mass CDF before inverse sampling
    real total_mass = cdf.back();
    for (real &value : cdf)
    {
        value /= total_mass;
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

    for (int idx = 0; idx < count; idx++)
    {
        real cdf_sample = random(rand_generator);
        auto cdf_iter = std::lower_bound(cdf.begin(), cdf.end(), cdf_sample);
        int idx_cell = std::max(0, static_cast<int>(cdf_iter - cdf.begin()) - 1);
        int iy = idx_cell % N_Y;
        int iz = idx_cell / N_Y;

        randposx[idx] = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);

        // sample uniformly in the exact radial and polar volume coordinates inside the chosen cell
        real s_y0 = y_face_s[iy];
        real s_y1 = y_face_s[iy + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);

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

#if defined(MULTISIZE) && defined(DIFFUSION)
// precompute one normalized spatial CDF for a selected grain size
inline static __host__
void disk_cdf_calc (std::vector <real> &cdf, const std::vector <real> &initdens, real size)
{
    int cell_count = N_Y*N_Z;
    real log_zero = -std::numeric_limits<real>::infinity();
    std::vector <real> log_mass(cell_count, log_zero);
    cdf.assign(cell_count + 1, 0.0);

    for (int iz = 0; iz < N_Z; iz++)
    {
        real z = _get_zcent(iz);
        real vol_z = _get_vol_z(iz);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y = _get_ycent(iy);
            real R = y*std::sin(z);
            real Z = y*std::cos(z);
            real sigma_d = initdens_lerp(R, initdens);
            real log_rhod = (sigma_d > 0.0) ? std::log(sigma_d) : log_zero;

            if (N_Z > 1 && sigma_d > 0.0)
            {
                real h_g = ASPR_0*std::pow(R / R_0, 0.5*(IDX_Q + 1.0));
                real H_g = h_g*R;

                real alpha_z = _get_alpha(R, h_g) / SCHMIDT_Z;

                real stokes_mid = STOKES_0*(size / S_0);
                #ifndef CONST_ST
                stokes_mid /= std::pow(R / R_0, IDX_P);
                #endif // NOT CONST_ST

                real H_d = H_g*std::sqrt(alpha_z / stokes_mid);

                log_rhod -= 0.5*Z*Z / (H_d*H_d);
                log_rhod -= std::log(H_d);
            }

            real vol_y = _get_vol_y(iy);
            int idx_cell = iy + iz*N_Y;
            if (sigma_d > 0.0) log_mass[idx_cell] = log_rhod + std::log(vol_y) + std::log(vol_z);
        }
    }

    // remove the largest logarithm before exponentiation to preserve highly settled distributions
    real max_log_mass = *std::max_element(log_mass.begin(), log_mass.end());
    for (int idx = 0; idx < cell_count; idx++)
    {
        cdf[idx + 1] = cdf[idx] + std::exp(log_mass[idx] - max_log_mass);
    }

    real total_mass = cdf[cell_count];
    for (real &value : cdf) 
    {
        value /= total_mass;
    }
}

// sample polydisperse dust from the joint y-z distribution conditioned on each previously assigned grain size
// interpolate log-size CDFs because size changes the Stokes number and therefore the coupled vertical distribution
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
    int size_bin_count = std::max(2, std::min(128, std::max(N_Y, N_Z)));
    int cell_count = N_Y*N_Z;
    real log_size_min = std::log(size_min);
    real log_size_max = std::log(size_max);
    real dlog_size = (log_size_max - log_size_min) / static_cast<real>(size_bin_count - 1);

    std::vector <real> initdens;
    initdens_calc(initdens);

    std::vector <real> cdf;
    std::vector <real> cdf_bank(static_cast<size_t>(size_bin_count)*static_cast<size_t>(cell_count + 1));
    for (int idx_size = 0; idx_size < size_bin_count; idx_size++)
    {
        real size = std::exp(log_size_min + static_cast<real>(idx_size)*dlog_size);
        disk_cdf_calc(cdf, initdens, size);
        std::copy(cdf.begin(), cdf.end(), cdf_bank.begin() + static_cast<size_t>(idx_size)*static_cast<size_t>(cell_count + 1));
    }

    real mesh_dim = _get_mesh_dim();
    std::uniform_real_distribution <real> random(0.0, 1.0);

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

    for (int idx = 0; idx < count; idx++)
    {
        real loc_size = (std::log(randsize[idx]) - log_size_min) / dlog_size;
        int size_lo = std::min(static_cast<int>(loc_size), size_bin_count - 2);
        real frac_size = loc_size - static_cast<real>(size_lo);
        const real *cdf_lo = cdf_bank.data() + static_cast<size_t>(size_lo)*static_cast<size_t>(cell_count + 1);
        const real *cdf_hi = cdf_lo + cell_count + 1;

        real cdf_sample = random(rand_generator);
        int idx_lo = 0;
        int idx_hi = cell_count;
        while (idx_lo < idx_hi)
        {
            int idx_mid = idx_lo + (idx_hi - idx_lo) / 2;
            real prob_mid = (1.0 - frac_size)*cdf_lo[idx_mid] + frac_size*cdf_hi[idx_mid];
            if (prob_mid < cdf_sample) idx_lo = idx_mid + 1;
            else idx_hi = idx_mid;
        }

        int idx_cell = std::max(0, idx_lo - 1);
        int iy = idx_cell % N_Y;
        int iz = idx_cell / N_Y;

        randposx[idx] = (N_X > 1) ? X_MIN + (X_MAX - X_MIN)*random(rand_generator) : 0.5*(X_MIN + X_MAX);

        real s_y0 = y_face_s[iy];
        real s_y1 = y_face_s[iy + 1];
        real s_y = s_y0 + (s_y1 - s_y0)*random(rand_generator);
        
        randposy[idx] = std::pow(mesh_dim*s_y, 1.0 / mesh_dim);

        real s_z0 = z_face_s[iz];
        real s_z1 = z_face_s[iz + 1];
        real s_z = s_z0 + (s_z1 - s_z0)*random(rand_generator);
        
        randposz[idx] = std::acos(-s_z);
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

    for (int idx_iter = 0; idx_iter < max_iter; ++idx_iter)
    {
        double exp_val = std::exp(val);
        double d_val = (val*exp_val - z) / (exp_val*(val + 1.0));
        
        val -= d_val;

        if (std::abs(d_val) < tol*(1.0 + std::abs(val))) return val;
    }

    throw std::runtime_error("lambertWm1: did not converge");
}

// sample the analytic initial distribution used by the linear-kernel collision test
inline __host__
void rand_gamma_k2 (real *randsize, int count)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int idx = 0; idx < count; idx++)
    {
        randsize[idx] = -(_get_lambertW_m1((random(rand_generator) - 1.0) / std::exp(1.0)) + 1.0);
    }
}
#endif // COLLISION

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
                cell_mass[idx_cell] = gas_dens[idx_cell]*epsilon[idx_cell]*cell_measure;
                total_mass += cell_mass[idx_cell];
            }
        }
    }
    
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
bool save_variable (const std::string &file_name, real total_dust_mass)
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
         << N_P / (N_K - 1.0) / total_dust_mass << std::endl;
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

// save the complete particle checkpoint and its stochastic state
inline __host__
bool save_particle_data (const std::string &path, int idx_file, swarm *particle, const swarm *dev_particle
    #if defined(COLLISION) || defined(DIFFUSION)
    , const curs *dev_rngstate
    #endif // COLLISION || DIFFUSION
)
{
    CUDA_CHECK(cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost));
    save_sam_as_velocity(particle);

    std::string file_name = path + "particle_" + frame_num(idx_file) + ".dat";
    if (!save_host_binary(file_name, particle, N_P))
    {
        std::cerr << "Error: Failed to save file: " << file_name << std::endl;
        return false;
    }

    #if defined(COLLISION) || defined(DIFFUSION)
    file_name = path + "rngstate_" + frame_num(idx_file) + ".dat";
    if (!save_device_binary(file_name, dev_rngstate, N_P))
    {
        std::cerr << "Error: Failed to save file: " << file_name << std::endl;
        return false;
    }
    #endif // COLLISION || DIFFUSION

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
    if (N_Z == 1)
    {
        for (int idx = 0; idx < N_P; idx++)
        {
            particle[idx].position.z = 0.5*M_PI;
            particle[idx].velocity.z = 0.0;
        }
    }
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
    if (!save_particle_data(PATH, idx_file, particle, dev_particle, dev_rngstate)) return 1; \
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
