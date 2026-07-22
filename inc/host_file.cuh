#ifndef HOST_FILE_CUH
#define HOST_FILE_CUH

#include <algorithm>        // for std::max
#include <chrono>           // for std::chrono::system_clock
#include <cstdlib>          // for std::exit, EXIT_FAILURE
#include <ctime>            // for std::time_t, std::ctime
#include <fstream>          // for std::ofstream, std::ifstream
#include <iomanip>          // for std::setw, std::setfill, std::setprecision
#include <iostream>         // for std::cout, std::cerr, std::endl
#include <string>           // for std::string, std::to_string

#include <cuda_runtime.h>   // for cudaError_t, cudaGetErrorString, cudaGetLastError, cudaSuccess

#include <const.cuh>

// =========================================================================================================================
// cuda error handling

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
bool save_variable (const std::string &file_name)
{
    std::ofstream file(file_name);
    if (!file) return false;
    
    file << "[PARAMETERS]"                                                                  << std::endl;
    file                                                                                    << std::endl;

    // gas parameters
    file << "SIGMA_0     = " << std::scientific     << std::setprecision(8) << SIGMA_0      << std::endl;
    file << "ASPR_0      = " << std::defaultfloat   << std::setprecision(8) << ASPR_0       << std::endl;
    file << "IDX_P       = " << std::defaultfloat   << std::setprecision(8) << IDX_P        << std::endl;
    file << "IDX_Q       = " << std::defaultfloat   << std::setprecision(8) << IDX_Q        << std::endl;
    #if defined(DIFFUSION) || defined(COLLISION)
    #ifndef CONST_NU
    file << "ALPHA       = " << std::scientific     << std::setprecision(8) << ALPHA        << std::endl;
    #else  // CONST_NU
    file << "NU          = " << std::scientific     << std::setprecision(8) << NU           << std::endl;
    #endif // NOT CONST_NU
    #endif // DIFFUSION || COLLISION
    #ifdef COLLISION
    #ifndef CODE_UNIT
    file << "M_MOL       = " << std::scientific     << std::setprecision(8) << M_MOL        << std::endl;
    file << "X_SEC       = " << std::scientific     << std::setprecision(8) << X_SEC        << std::endl;
    #else  // CODE_UNIT
    file << "RE_0        = " << std::scientific     << std::setprecision(8) << RE_0         << std::endl;
    #endif // NOT CODE_UNIT
    #endif // COLLISION
    file                                                                                    << std::endl;
    
    // dust parameters
    file << "ST_0        = " << std::scientific     << std::setprecision(8) << ST_0         << std::endl;
    file << "M_D         = " << std::scientific     << std::setprecision(8) << M_D          << std::endl;
    file << "RHO_0       = " << std::scientific     << std::setprecision(8) << RHO_0        << std::endl;
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific     << std::setprecision(8) << BETA_0       << std::endl;
    file << "KAPPA_0     = " << std::scientific     << std::setprecision(8) << KAPPA_0      << std::endl;
    #endif // RADIATION
    #ifdef DIFFUSION
    file << "SCHMIDT_X   = " << std::scientific     << std::setprecision(8) << SCHMIDT_X    << std::endl;
    file << "SCHMIDT_R   = " << std::scientific     << std::setprecision(8) << SCHMIDT_R    << std::endl;
    #endif // DIFFUSION
    #if defined(DIFFUSION) || defined(COLLISION)
    file << "SCHMIDT_Z   = " << std::scientific     << std::setprecision(8) << SCHMIDT_Z    << std::endl;
    #endif // DIFFUSION || COLLISION

    #ifdef COLLISION
    file << "LAMBDA_0    = " << std::scientific     << std::setprecision(8) << LAMBDA_0     << std::endl;
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
    file << "INIT_SMIN   = " << std::scientific     << std::setprecision(8) << INIT_SMIN    << std::endl;
    file << "INIT_SMAX   = " << std::scientific     << std::setprecision(8) << INIT_SMAX    << std::endl;
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
    file << "DT_DYN      = " << std::scientific     << std::setprecision(8) << DT_DYN       << std::endl;
    file << "CFL_DYN     = " << std::defaultfloat   << std::setprecision(8) << CFL_DYN      << std::endl;
    #endif // TRANSPORT
    file << "DT_MIN      = " << std::scientific     << std::setprecision(8) << DT_MIN       << std::endl;
    file                                                                                    << std::endl;

    // swarm structure as a configparser-compatible NumPy dtype, in Python write as:
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
    dustdens_scat <<< NB_P, TPB >>> (dev_dustdens, dev_particle);                           \
    CUDA_KERNEL_CHECK("dustdens_scat");                                                     \
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
    optdepth_scat <<< NB_P, TPB >>> (dev_optdepth, dev_particle);                           \
    CUDA_KERNEL_CHECK("optdepth_scat");                                                     \
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

#endif // NOT HOST_FILE_CUH
