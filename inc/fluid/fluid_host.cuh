#ifndef FLUID_HOST_CUH
#define FLUID_HOST_CUH

#include <algorithm>           // std::fill, std::max, std::min, std::swap
#include <chrono>              // std::chrono::system_clock
#include <cmath>               // std::abs, std::cos, std::exp, std::fmin, std::fmax, std::isfinite, ...
#include <cstddef>             // std::size_t
#include <cstdlib>             // std::exit, EXIT_FAILURE
#include <ctime>               // std::ctime, std::time_t
#include <fstream>             // std::ifstream, std::ofstream
#include <iomanip>             // std::defaultfloat, std::scientific, std::setfill, std::setprecision, std::setw
#include <iostream>            // std::cerr, std::cout, std::endl
#include <string>              // std::string, std::to_string
#include <vector>              // std::vector

#include <gpu_compat.cuh>
#include <thrust/device_ptr.h> // thrust::device_ptr
#include <thrust/extrema.h>    // thrust::max_element
#include <thrust/reduce.h>     // thrust::reduce

#include <const_defs.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// host-only finite-volume coordinates

inline __host__
real _get_sy (real y) { return std::pow(y, _get_mesh_dim()) / _get_mesh_dim(); }

inline __host__
real _get_sz (real z) { return -std::cos(z); }

// =========================================================================================================================
// cuda error handling

inline __host__
void cuda_fail (gpuError_t status, const char *operation, const char *file, int line)
{
    std::cerr
    << GPU_BACKEND_NAME " error at " << file << ":" << line
    << " during " << operation << ": " << gpuGetErrorString(status)
    << " (" << static_cast<int>(status) << ")\n";
    
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(OPERATION)                                                       \
do {                                                                                \
    gpuError_t cuda_status_ = (OPERATION);                                         \
    if (cuda_status_ != gpuSuccess)                                                \
    { cuda_fail(cuda_status_, #OPERATION, __FILE__, __LINE__); }                    \
} while (0)

#ifdef GPU_SYNC_TRACE
#define CUDA_KERNEL_CHECK(KERNEL_NAME)                                                \
do {                                                                                  \
    gpuError_t cuda_status_ = gpuGetLastError();                                    \
    if (cuda_status_ != gpuSuccess)                                                  \
    { cuda_fail(cuda_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }    \
    cuda_status_ = gpuDeviceSynchronize();                                           \
    if (cuda_status_ != gpuSuccess)                                                  \
    { cuda_fail(cuda_status_, KERNEL_NAME " kernel execution", __FILE__, __LINE__); } \
    std::cout << "  [" GPU_BACKEND_NAME "] completed " << KERNEL_NAME << std::endl;                   \
} while (0)
#else  // !GPU_SYNC_TRACE
#define CUDA_KERNEL_CHECK(KERNEL_NAME)                                              \
do {                                                                                \
    gpuError_t cuda_status_ = gpuGetLastError();                                  \
    if (cuda_status_ != gpuSuccess)                                                \
    { cuda_fail(cuda_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }  \
} while (0)
#endif // GPU_SYNC_TRACE

// =========================================================================================================================
#ifdef GAMEDEV_ROCM
struct lds_usage
{
    std::size_t device_limit;
    std::size_t kernel_static;
    std::size_t kernel_dynamic_limit;
    std::size_t requested_dynamic;
    std::size_t total;
    bool kernel_limit_reported;
};

// query the active device and compiled kernel before requesting dynamic LDS
inline __host__
lds_usage get_lds_usage (const void *kernel, std::size_t requested_dynamic)
{
    int idx_device = 0;
    int device_limit = 0;
    hipFuncAttributes attributes = {};
    CUDA_CHECK(hipGetDevice(&idx_device));
    CUDA_CHECK(hipDeviceGetAttribute(
        &device_limit, hipDeviceAttributeMaxSharedMemoryPerBlock, idx_device
    ));
    CUDA_CHECK(hipFuncGetAttributes(&attributes, kernel));

    std::size_t kernel_static = attributes.sharedSizeBytes;
    bool kernel_limit_reported = attributes.maxDynamicSharedSizeBytes > 0;
    std::size_t kernel_dynamic_limit = kernel_limit_reported ?
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes) :
        static_cast<std::size_t>(device_limit) - std::min(
            static_cast<std::size_t>(device_limit), kernel_static
        );

    return {
        static_cast<std::size_t>(device_limit),
        kernel_static,
        kernel_dynamic_limit,
        requested_dynamic,
        kernel_static + requested_dynamic,
        kernel_limit_reported,
    };
}

// reject an unsupported LDS request before launching the kernel
inline __host__
lds_usage require_lds (const void *kernel, std::size_t requested_dynamic, const char *kernel_name)
{
    lds_usage usage = get_lds_usage(kernel, requested_dynamic);
    bool dynamic_supported = usage.requested_dynamic <= usage.kernel_dynamic_limit;
    bool total_supported = usage.total <= usage.device_limit;
    if (dynamic_supported && total_supported) return usage;

    std::cerr
    << "Error: " << kernel_name << " requires " << usage.requested_dynamic
    << " dynamic LDS bytes and " << usage.total << " total LDS bytes, but the active AMD device permits "
    << usage.kernel_dynamic_limit << " dynamic bytes for this kernel and " << usage.device_limit
    << " total bytes per block" << std::endl;
    std::exit(EXIT_FAILURE);
}

#endif

// geometry-aware PPM interpolation weights

// solve four-cell interpolation weights that reproduce cubic data at one interior face
inline __host__
void _ppm_cubic_face_weights (const std::vector<real> &face_s, int idx_face, real *face_weight)
{
    real face = face_s[idx_face];
    real scale = std::max(face - face_s[idx_face - 2], face_s[idx_face + 2] - face);

    // assemble moment constraints that make the face interpolation exact for cubic data
    real aug_matrix[4][5] = {};

    // impose exactness for polynomial moments from degree zero through three
    for (int degree = 0; degree < 4; degree++)
    {
        // evaluate each monomial cell average over the four-cell stencil
        for (int idx_stencil = 0; idx_stencil < 4; idx_stencil++)
        {
            int idx_cell = idx_face - 2 + idx_stencil;
            real t_lower = (face_s[idx_cell]     - face) / scale;
            real t_upper = (face_s[idx_cell + 1] - face) / scale;

            aug_matrix[degree][idx_stencil]  = std::pow(t_upper, degree + 1) - std::pow(t_lower, degree + 1);
            aug_matrix[degree][idx_stencil] /= static_cast<real>(degree + 1)*(t_upper - t_lower);
        }
        aug_matrix[degree][4] = (degree == 0) ? 1.0 : 0.0;
    }

    // solve the interpolation weights by Gauss-Jordan elimination with partial pivoting
    for (int idx_pivot = 0; idx_pivot < 4; idx_pivot++)
    {
        int idx_pivot_row = idx_pivot;

        // select the largest available pivot in the current column
        for (int idx_row = idx_pivot + 1; idx_row < 4; idx_row++)
        {
            if (std::abs(aug_matrix[idx_row][idx_pivot]) > std::abs(aug_matrix[idx_pivot_row][idx_pivot]))
            {
                idx_pivot_row = idx_row;
            }
        }

        // move the selected pivot row into the current row
        for (int idx_aug = idx_pivot; idx_aug < 5; idx_aug++)
        {
            std::swap(aug_matrix[idx_pivot][idx_aug], aug_matrix[idx_pivot_row][idx_aug]);
        }

        real pivot = aug_matrix[idx_pivot][idx_pivot];

        // normalize the current pivot row
        for (int idx_aug = idx_pivot; idx_aug < 5; idx_aug++)
        {
            aug_matrix[idx_pivot][idx_aug] /= pivot;
        }

        // eliminate the current column from every other row
        for (int idx_row = 0; idx_row < 4; idx_row++)
        {
            if (idx_row == idx_pivot) continue;

            real factor = aug_matrix[idx_row][idx_pivot];

            // update the remaining augmented entries in the target row
            for (int idx_aug = idx_pivot; idx_aug < 5; idx_aug++)
            {
                aug_matrix[idx_row][idx_aug] -= factor*aug_matrix[idx_pivot][idx_aug];
            }
        }
    }

    // copy the solved interpolation weights
    for (int idx_stencil = 0; idx_stencil < 4; idx_stencil++)
    {
        face_weight[idx_stencil] = aug_matrix[idx_stencil][4];
    }
}

// build face interpolation weights for one nonuniform volume coordinate
inline __host__
void _ppm_nonuniform_weights (const std::vector<real> &face_s, real *face_weight)
{
    int cell_count = static_cast<int>(face_s.size()) - 1;
    std::fill(face_weight, face_weight + 4*(cell_count + 1), 0.0);

    // assign cubic interior weights and linear boundary-adjacent weights to every internal face
    for (int idx_face = 1; idx_face < cell_count; idx_face++)
    {
        real *weight = face_weight + 4*idx_face;
        if (idx_face >= 2 && idx_face <= cell_count - 2)
        {
            _ppm_cubic_face_weights(face_s, idx_face, weight);
            continue;
        }

        real center_L = 0.5*(face_s[idx_face - 1] + face_s[idx_face]);
        real center_R = 0.5*(face_s[idx_face] + face_s[idx_face + 1]);

        weight[0] = (center_R - face_s[idx_face]) / (center_R - center_L);
        weight[1] = (face_s[idx_face] - center_L) / (center_R - center_L);
    }
}

// build radial and polar PPM weights in their finite-volume coordinates
inline __host__
void ppm_geometry_weights_calc (real *ppm_weight_y, real *ppm_weight_z)
{
    std::vector<real> y_face_s(N_Y + 1);

    // map radial faces to the volume coordinate used by the radial finite-volume operator
    for (int iy = 0; iy <= N_Y; iy++)
    {
        y_face_s[iy] = _get_sy(_get_yface(iy));
    }

    _ppm_nonuniform_weights(y_face_s, ppm_weight_y);

    std::vector<real> z_face_s(N_Z + 1);

    // map polar faces to the spherical volume coordinate s=-cos(z)
    for (int iz = 0; iz <= N_Z; iz++)
    {
        z_face_s[iz] = _get_sz(_get_zface(iz));
    }

    _ppm_nonuniform_weights(z_face_s, ppm_weight_z);
}

// =========================================================================================================================
// convolved initial surface-density profile

inline __host__
void initdens_calc (real *initdens)
{
    const real smooth = 0.05*R_0;
    const real R_src_min = Y_MIN + 2.0*smooth;
    const real R_src_max = Y_MAX - 2.0*smooth;
    const real kernel_std = 0.5*smooth;

    const int bin_count = N_Y;
    const real R_min = _get_init_Rmin();
    const real dR = (Y_MAX - R_min) / static_cast<real>(bin_count);

    std::vector<real> conv_u(bin_count + 1);
    std::fill(initdens, initdens + bin_count + 1, 0.0);

    for (int idx_dst = 0; idx_dst <= bin_count; idx_dst++)
    {
        conv_u[idx_dst] = R_min + static_cast<real>(idx_dst)*dR;
    }

    // normalize the Gaussian over physical cylindrical radius
    const real kernel_norm = 1.0 / (std::sqrt(2.0*M_PI)*kernel_std);

    for (int idx_src = 0; idx_src <= bin_count; idx_src++)
    {
        real R_src = conv_u[idx_src];
        if (R_src < R_src_min || R_src > R_src_max) continue;

        real sigma_g = SIGMA_0*std::pow(R_src / R_0, IDX_P);
        real sigma_d = METAL_Z*sigma_g;

        for (int idx_dst = 0; idx_dst <= bin_count; idx_dst++)
        {
            real delta_R = conv_u[idx_dst] - R_src;
            real kernel_weight = kernel_norm*std::exp(-delta_R*delta_R / (2.0*kernel_std*kernel_std));
            initdens[idx_dst] += sigma_d*kernel_weight*dR;
        }
    }
}

// =========================================================================================================================
// reduce cellwise CFL rates and optionally report the limiting cell

inline __host__
real get_dt_cfl (const real *dev_cfl_rate, const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    bool verbose = true)
{
    thrust::device_ptr <const real> cfl_rate_ptr(dev_cfl_rate);
    auto max_cfl_rate_ptr = thrust::max_element(cfl_rate_ptr, cfl_rate_ptr + N_G);
    real max_cfl_rate = *max_cfl_rate_ptr;

    if (!std::isfinite(max_cfl_rate))
    {
        int idx_bad = static_cast<int>(max_cfl_rate_ptr - cfl_rate_ptr);

        int ix_bad = idx_bad % N_X;
        int iy_bad = (idx_bad / N_X) % N_Y;
        int iz_bad = idx_bad / (N_X * N_Y);

        std::cerr
        << "Error: non-finite dust state detected by CFL validation at cell ("
        << ix_bad << "," << iy_bad << "," << iz_bad << ")"
        << std::endl;

        std::exit(EXIT_FAILURE);
    }

    if (max_cfl_rate <= 0.0) return DT_MAX;

    real dt_cfl = std::fmin(CFL_DYN / max_cfl_rate, DT_MAX);
    if (!verbose) return dt_cfl;

    // reconstruct physical velocities in the residual FARGO frame for diagnostics
    int idx_cfl_max = static_cast<int>(max_cfl_rate_ptr - cfl_rate_ptr);
    int ix = idx_cfl_max % N_X;
    int iy = (idx_cfl_max / N_X) % N_Y;
    int iz = idx_cfl_max / (N_X*N_Y);

    real lx, vy, lz;
    CUDA_CHECK(gpuMemcpy(&lx, dev_dustvelx + idx_cfl_max, sizeof(real), gpuMemcpyDeviceToHost));
    CUDA_CHECK(gpuMemcpy(&vy, dev_dustvely + idx_cfl_max, sizeof(real), gpuMemcpyDeviceToHost));
    CUDA_CHECK(gpuMemcpy(&lz, dev_dustvelz + idx_cfl_max, sizeof(real), gpuMemcpyDeviceToHost));

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*std::sin(z);

    int idx_ring = iy*N_X + iz*N_X*N_Y;
    thrust::device_ptr <const real> lx_ptr(dev_dustvelx + idx_ring);
    real lx_avg = thrust::reduce(lx_ptr, lx_ptr + N_X, 0.0) / static_cast<real>(N_X);
    real vx_res = (lx - lx_avg) / std::fmax(R, 1.0e-30);
    real vz = lz / y;

    std::cout
    << std::setfill(' ')
    << "  [CFL] cell=("
    << std::setw(4) << ix << ","
    << std::setw(4) << iy << ","
    << std::setw(4) << iz << ")"
    << "  R="    << std::scientific << std::setprecision(3) << R
    << "  dvx="  << std::setw(8) << vx_res
    << "  vy="   << std::setw(8) << vy
    << "  vz="   << std::setw(8) << vz
    << "  rate=" << std::setw(8) << max_cfl_rate
    << "  dt="   << std::setw(8) << CFL_DYN / max_cfl_rate
    << std::endl;

    return dt_cfl;
}

// =========================================================================================================================
// simulation progress output

inline __host__
void msg_output (int idx_file)
{
    std::time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    int width = std::max(3, (int)std::to_string(SAVE_MAX).length());

    std::cout
    << std::endl
    << std::setfill('0')
    << std::setw(width) << idx_file << "/"
    << std::setw(width) << SAVE_MAX
    << " finished on " << std::ctime(&time_now)
    << std::endl;
}

inline __host__
void msg_step_title ()
{
    std::cout
    << std::setfill(' ')
    << std::setw(10) << "idx_from"   << " "
    << std::setw(12) << "dt"         << " "
    << std::setw(12) << "clock_out"  << " "
    << std::setw(12) << "clock_sim"
    << std::endl;
}

inline __host__
void msg_step (int idx_from, real dt, real clock_out, real clock_sim)
{
    std::cout
    << std::setfill(' ') << std::defaultfloat
    << std::setw(10) << idx_from   << " "
    << std::scientific << std::setprecision(4)
    << std::setw(12) << dt         << " "
    << std::setw(12) << clock_out  << " "
    << std::setw(12) << clock_sim
    << std::endl;
}

// =========================================================================================================================
// binary field conversion and file I/O

inline __host__
std::string frame_num (int idx_file)
{
    std::string num_str = std::to_string(idx_file);
    int width = std::max(5, (int)std::to_string(SAVE_MAX).length());

    if ((int)num_str.length() < width)
    {
        num_str.insert(0, width - num_str.length(), '0');
    }

    return num_str;
}

constexpr std::size_t binary_chunk_bytes = 64ULL*1024ULL*1024ULL;

// write a contiguous host field in bounded chunks
template <typename DataType> inline __host__
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

// read one exact-size host field in bounded chunks
template <typename DataType> inline __host__
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

// convert internal angular variables to linear velocities before file output
inline __host__
void save_sam_as_velocity (real *dustvelx, real *dustvelz)
{
    for (int iz = 0; iz < N_Z; iz++)
    {
        const real z = _get_zcent(iz);
        const real sin_z = std::sin(z);

        for (int iy = 0; iy < N_Y; iy++)
        {
            const real y = _get_ycent(iy);
            const real R = y*sin_z;

            for (int ix = 0; ix < N_X; ix++)
            {
                const int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dustvelx[idx_cell] = (R > 0.0) ? dustvelx[idx_cell] / R : 0.0;
                dustvelz[idx_cell] /= y;
            }
        }
    }
}

// convert linear file velocities to internal angular variables after loading
inline __host__
void load_velocity_as_sam (real *dustvelx, real *dustvelz)
{
    for (int iz = 0; iz < N_Z; iz++)
    {
        const real z = _get_zcent(iz);
        const real sin_z = std::sin(z);

        for (int iy = 0; iy < N_Y; iy++)
        {
            const real y = _get_ycent(iy);
            const real R = y*sin_z;

            for (int ix = 0; ix < N_X; ix++)
            {
                const int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dustvelx[idx_cell] *= R;
                dustvelz[idx_cell] *= y;
            }
        }
    }
}

#ifdef RADIATION
#define SAVE_OPTDEPTH_TO_FILE(idx_file)                                                        \
do {                                                                                           \
    CUDA_CHECK(gpuMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, gpuMemcpyDeviceToHost));  \
    if (!save_host_binary(PATH + "optdepth_" + frame_num(idx_file) + ".dat", optdepth, N_G)) \
    { std::cerr << "Error: failed to save optdepth frame " << idx_file << std::endl; }         \
} while(0)
#endif // RADIATION

#define SAVE_DUSTDENS_TO_FILE(idx_file)                                                        \
do {                                                                                           \
    CUDA_CHECK(gpuMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, gpuMemcpyDeviceToHost));  \
    if (!save_host_binary(PATH + "dustdens_" + frame_num(idx_file) + ".dat", dustdens, N_G)) \
    { std::cerr << "Error: failed to save dustdens frame " << idx_file << std::endl; }         \
} while(0)

#define SAVE_DUST_VEL_TO_FILE(idx_file)                                                        \
do {                                                                                           \
    CUDA_CHECK(gpuMemcpy(dustvelx, dev_dustvelx, sizeof(real)*N_G, gpuMemcpyDeviceToHost));  \
    CUDA_CHECK(gpuMemcpy(dustvely, dev_dustvely, sizeof(real)*N_G, gpuMemcpyDeviceToHost));  \
    CUDA_CHECK(gpuMemcpy(dustvelz, dev_dustvelz, sizeof(real)*N_G, gpuMemcpyDeviceToHost));  \
    save_sam_as_velocity(dustvelx, dustvelz);                                                  \
    if (!save_host_binary(PATH + "dustvelx_" + frame_num(idx_file) + ".dat", dustvelx, N_G)) \
    { std::cerr << "Error: failed to save dustvelx frame " << idx_file << std::endl; }         \
    if (!save_host_binary(PATH + "dustvely_" + frame_num(idx_file) + ".dat", dustvely, N_G)) \
    { std::cerr << "Error: failed to save dustvely frame " << idx_file << std::endl; }         \
    if (!save_host_binary(PATH + "dustvelz_" + frame_num(idx_file) + ".dat", dustvelz, N_G)) \
    { std::cerr << "Error: failed to save dustvelz frame " << idx_file << std::endl; }         \
} while(0)

#define LOAD_DUSTDATA_TO_VRAM(idx_file)                                                        \
do {                                                                                           \
    if (!load_host_binary(PATH + "dustdens_" + frame_num(idx_file) + ".dat", dustdens, N_G)) \
    { std::cerr << "Error: failed to load dustdens frame " << idx_file << std::endl; return 1; } \
    if (!load_host_binary(PATH + "dustvelx_" + frame_num(idx_file) + ".dat", dustvelx, N_G)) \
    { std::cerr << "Error: failed to load dustvelx frame " << idx_file << std::endl; return 1; } \
    if (!load_host_binary(PATH + "dustvely_" + frame_num(idx_file) + ".dat", dustvely, N_G)) \
    { std::cerr << "Error: failed to load dustvely frame " << idx_file << std::endl; return 1; } \
    if (!load_host_binary(PATH + "dustvelz_" + frame_num(idx_file) + ".dat", dustvelz, N_G)) \
    { std::cerr << "Error: failed to load dustvelz frame " << idx_file << std::endl; return 1; } \
    load_velocity_as_sam(dustvelx, dustvelz);                                                  \
    CUDA_CHECK(gpuMemcpy(dev_dustdens, dustdens, sizeof(real)*N_G, gpuMemcpyHostToDevice));  \
    CUDA_CHECK(gpuMemcpy(dev_dustvelx, dustvelx, sizeof(real)*N_G, gpuMemcpyHostToDevice));  \
    CUDA_CHECK(gpuMemcpy(dev_dustvely, dustvely, sizeof(real)*N_G, gpuMemcpyHostToDevice));  \
    CUDA_CHECK(gpuMemcpy(dev_dustvelz, dustvelz, sizeof(real)*N_G, gpuMemcpyHostToDevice));  \
} while(0)

// =========================================================================================================================
// runtime field transfers

inline __host__
bool save_variable (const std::string &file_name)
{
    std::ofstream file(file_name);
    if (!file) return false;

    file << "[PARAMETERS]"                                                             << "\n";
    file                                                                               << "\n";

    file << "SIGMA_0     = " << std::scientific   << std::setprecision(8) << SIGMA_0   << "\n";
    file << "ASPR_0      = " << std::defaultfloat << std::setprecision(8) << ASPR_0    << "\n";
    file << "IDX_P       = " << std::defaultfloat << std::setprecision(8) << IDX_P     << "\n";
    file << "IDX_Q       = " << std::defaultfloat << std::setprecision(8) << IDX_Q     << "\n";
    #ifdef DIFFUSION
    #ifndef CONST_NU  // CONST_ALPHA
    file << "ALPHA       = " << std::scientific   << std::setprecision(8) << ALPHA     << "\n";
    #else             // CONST_NU
    file << "NU          = " << std::scientific   << std::setprecision(8) << NU        << "\n";
    #endif // CONST_NU
    #endif // DIFFUSION
    #ifdef VISC_FLOW
    file << "VISC_FLOW   = " << std::defaultfloat << 1                                << "\n";
    #endif // VISC_FLOW
    file                                                                               << "\n";

    file << "STOKES_0    = " << std::scientific   << std::setprecision(8) << STOKES_0  << "\n";
    file << "METAL_Z     = " << std::scientific   << std::setprecision(8) << METAL_Z   << "\n";
    file << "RHO_VAC     = " << std::scientific   << std::setprecision(8) << RHO_VAC   << "\n";
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific   << std::setprecision(8) << BETA_0    << "\n";
    file << "KAPPA_0     = " << std::scientific   << std::setprecision(8) << KAPPA_0   << "\n";
    file << "T_BETA      = " << std::scientific   << std::setprecision(8) << T_BETA    << "\n";
    #endif // RADIATION
    #ifdef DIFFUSION
    file << "SCHMIDT_Y   = " << std::scientific   << std::setprecision(8) << SCHMIDT_Y << "\n";
    file << "SCHMIDT_X   = " << std::scientific   << std::setprecision(8) << SCHMIDT_X << "\n";
    file << "SCHMIDT_Z   = " << std::scientific   << std::setprecision(8) << SCHMIDT_Z << "\n";
    file << "POS_LIMIT   = " << std::scientific   << std::setprecision(8) << POS_LIMIT << "\n";
    #endif // DIFFUSION
    file                                                                               << "\n";

    file << "N_X         = " << std::defaultfloat << std::setprecision(8) << N_X       << "\n";
    file << "X_MIN       = " << std::defaultfloat << std::setprecision(8) << X_MIN     << "\n";
    file << "X_MAX       = " << std::defaultfloat << std::setprecision(8) << X_MAX     << "\n";
    file                                                                               << "\n";
    file << "N_Y         = " << std::defaultfloat << std::setprecision(8) << N_Y       << "\n";
    file << "Y_MIN       = " << std::scientific   << std::setprecision(8) << Y_MIN     << "\n";
    file << "Y_MAX       = " << std::scientific   << std::setprecision(8) << Y_MAX     << "\n";
    file                                                                               << "\n";
    file << "N_Z         = " << std::defaultfloat << std::setprecision(8) << N_Z       << "\n";
    file << "Z_MIN       = " << std::scientific   << std::setprecision(8) << Z_MIN     << "\n";
    file << "Z_MAX       = " << std::scientific   << std::setprecision(8) << Z_MAX     << "\n";
    file                                                                               << "\n";

    file << "SAVE_MAX    = " << std::defaultfloat << std::setprecision(8) << SAVE_MAX  << "\n";
    file << "DT_OUT      = " << std::scientific   << std::setprecision(8) << DT_OUT    << "\n";
    file << "DT_MAX      = " << std::scientific   << std::setprecision(8) << DT_MAX    << "\n";
    file << "CFL_DYN     = " << std::scientific   << std::setprecision(8) << CFL_DYN   << "\n";
    file                                                                               << "\n";

    return file.good();
}

// =========================================================================================================================


#ifndef HIP_CHECK
#define HIP_CHECK CUDA_CHECK
#define HIP_KERNEL_CHECK CUDA_KERNEL_CHECK
#endif
#endif // FLUID_HOST_CUH
