#ifndef FLUID_HOST_CUH
#define FLUID_HOST_CUH

#include <algorithm>           // std::copy, std::fill, std::max, std::swap
#include <chrono>              // std::chrono::system_clock
#include <cmath>               // std::abs, std::cos, std::exp, std::fmin, std::fmax, std::isfinite, ...
#include <cstdlib>             // std::exit, EXIT_FAILURE
#include <ctime>               // std::ctime, std::time_t
#include <fstream>             // std::ifstream, std::ofstream
#include <iomanip>             // std::defaultfloat, std::scientific, std::setfill, std::setprecision, std::setw
#include <iostream>            // std::cerr, std::cout, std::endl
#include <string>              // std::string, std::to_string
#include <vector>              // std::vector

#include <cuda_runtime.h>      // cudaGetErrorString, cudaGetLastError, cudaMemcpy
#include <thrust/device_ptr.h> // thrust::device_ptr
#include <thrust/extrema.h>    // thrust::max_element
#include <thrust/reduce.h>     // thrust::reduce

#include <const.cuh>
#include <param_grid.cuh>

// =========================================================================================================================
// host-only finite-volume coordinates

inline __host__
real _get_s_y (real y) { return std::pow(y, _get_mesh_dim()) / _get_mesh_dim(); }

inline __host__
real _get_s_z (real z) { return -std::cos(z); }

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
// precompute geometry-aware PPM face interpolation weights from cell averages

// solve four-cell interpolation weights that reproduce cubic data at one interior face
inline __host__
void _ppm_cubic_face_weights (const std::vector<real> &face_s, int iface, real *face_weight)
{
    real face = face_s[iface];
    real scale = std::max(face - face_s[iface - 2], face_s[iface + 2] - face);

    // assemble moment constraints that make the face interpolation exact for cubic data
    real aug_matrix[4][5] = {};

    // impose exactness for polynomial moments from degree zero through three
    for (int n = 0; n < 4; n++)
    {
        // evaluate each monomial cell average over the four-cell stencil
        for (int j = 0; j < 4; j++)
        {
            int icell = iface - 2 + j;
            real t_lower = (face_s[icell]     - face) / scale;
            real t_upper = (face_s[icell + 1] - face) / scale;

            aug_matrix[n][j]  = std::pow(t_upper, n + 1) - std::pow(t_lower, n + 1);
            aug_matrix[n][j] /= static_cast<real>(n + 1)*(t_upper - t_lower);
        }
        aug_matrix[n][4] = (n == 0) ? 1.0 : 0.0;
    }

    // solve the interpolation weights by Gauss-Jordan elimination with partial pivoting
    for (int col = 0; col < 4; col++)
    {
        int pivot_row = col;

        // select the largest available pivot in the current column
        for (int row = col + 1; row < 4; row++)
        {
            if (std::abs(aug_matrix[row][col]) > std::abs(aug_matrix[pivot_row][col]))
            {
                pivot_row = row;
            }
        }

        // move the selected pivot row into the current row
        for (int k = col; k < 5; k++)
        {
            std::swap(aug_matrix[col][k], aug_matrix[pivot_row][k]);
        }

        real pivot = aug_matrix[col][col];

        // normalize the current pivot row
        for (int k = col; k < 5; k++)
        {
            aug_matrix[col][k] /= pivot;
        }

        // eliminate the current column from every other row
        for (int row = 0; row < 4; row++)
        {
            if (row == col) continue;

            real factor = aug_matrix[row][col];

            // update the remaining augmented entries in the target row
            for (int k = col; k < 5; k++)
            {
                aug_matrix[row][k] -= factor*aug_matrix[col][k];
            }
        }
    }

    // copy the solved interpolation weights
    for (int j = 0; j < 4; j++)
    {
        face_weight[j] = aug_matrix[j][4];
    }
}

// build face interpolation weights for one nonuniform volume coordinate
inline __host__
void _ppm_nonuniform_weights (const std::vector<real> &face_s, real *face_weight)
{
    int cell_count = static_cast<int>(face_s.size()) - 1;
    std::fill(face_weight, face_weight + 4*(cell_count + 1), 0.0);

    // assign cubic interior weights and linear boundary-adjacent weights to every internal face
    for (int iface = 1; iface < cell_count; iface++)
    {
        real *weight = face_weight + 4*iface;
        if (iface >= 2 && iface <= cell_count - 2)
        {
            _ppm_cubic_face_weights(face_s, iface, weight);
            continue;
        }

        real center_L = 0.5*(face_s[iface - 1] + face_s[iface]);
        real center_R = 0.5*(face_s[iface] + face_s[iface + 1]);

        weight[0] = (center_R - face_s[iface]) / (center_R - center_L);
        weight[1] = (face_s[iface] - center_L) / (center_R - center_L);
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
        y_face_s[iy] = _get_s_y(_get_yedge(iy));
    }

    _ppm_nonuniform_weights(y_face_s, ppm_weight_y);

    std::vector<real> z_face_s(N_Z + 1);

    // map polar faces to the spherical volume coordinate minus cosine theta
    for (int iz = 0; iz <= N_Z; iz++)
    {
        z_face_s[iz] = _get_s_z(_get_zedge(iz));
    }

    _ppm_nonuniform_weights(z_face_s, ppm_weight_z);
}

// =========================================================================================================================
// calculate the power-law surface density after convolution with a Gaussian kernel 

inline __host__
void convpow_calc (real *initdens)
{
    const real u_s   = 0.05*R_0;
    const real u_min = Y_MIN + 2.0*u_s;
    const real u_max = Y_MAX - 2.0*u_s;
    const real sig_u = 0.5*u_s;

    const int n_bin = N_Y;
    const real du = (Y_MAX - Y_MIN) / static_cast<real>(n_bin);

    std::vector<real> u_axis(n_bin + 1);
    std::vector<real> v_axis(n_bin + 1, 0.0);

    for (int i = 0; i <= n_bin; i++)
    {
        u_axis[i] = Y_MIN + i*du;
    }

    // normalization factor for the Gaussian kernel
    const real norm = 1.0 / (std::sqrt(2.0*M_PI)*sig_u);

    for (int j = 0; j <= n_bin; j++)
    {
        real u_j = u_axis[j];
        if (u_j < u_min || u_j > u_max) continue;

        real sigma_g = SIGMA_0*std::pow(u_j / R_0, IDX_P);
        real sigma_d = METAL_Z*sigma_g;

        for (int k = 0; k <= n_bin; k++)
        {
            real delta_u = u_axis[k] - u_j;
            real kernel = norm*std::exp(-delta_u*delta_u / (2.0*sig_u*sig_u));
            v_axis[k] += sigma_d*kernel*du;
        }
    }

    std::copy(v_axis.begin(), v_axis.end(), initdens);
}

// =========================================================================================================================
// obtain the CFL time step based on the maximum CFL rate across all cells, 
// and print information about the cell with the maximum rate if verbose is true

inline __host__
real get_dt_cfl (const real *dev_cfl_rates, const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz,
    bool verbose = true)
{
    thrust::device_ptr <const real> ptr_cfl(dev_cfl_rates);
    auto max_it = thrust::max_element(ptr_cfl, ptr_cfl + N_G);
    real max_rate = *max_it;

    if (!std::isfinite(max_rate))
    {
        int idx_bad = static_cast<int>(max_it - ptr_cfl);

        int ix_bad = idx_bad % N_X;
        int iy_bad = (idx_bad / N_X) % N_Y;
        int iz_bad = idx_bad / (N_X * N_Y);

        std::cerr
        << "Error: non-finite dust state detected by CFL validation at cell ("
        << ix_bad << "," << iy_bad << "," << iz_bad << ")"
        << std::endl;

        std::exit(EXIT_FAILURE);
    }

    if (max_rate <= 0.0) return DT_MAX;

    real dt_cfl = std::fmin(CFL_DYN / max_rate, DT_MAX);
    if (!verbose) return dt_cfl;

    // print the cell with the maximum CFL rate and its corresponding velocity components
    int idx_max = static_cast<int>(max_it - ptr_cfl);
    int ix = idx_max % N_X;
    int iy = (idx_max / N_X) % N_Y;
    int iz = idx_max / (N_X*N_Y);

    real lx, vy, lz;
    CUDA_CHECK(cudaMemcpy(&lx, dev_dustvelx + idx_max, sizeof(real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&vy, dev_dustvely + idx_max, sizeof(real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lz, dev_dustvelz + idx_max, sizeof(real), cudaMemcpyDeviceToHost));

    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*std::sin(z);

    int idx_ring = iy*N_X + iz*N_X*N_Y;
    thrust::device_ptr <const real> ptr_lx(dev_dustvelx + idx_ring);
    real lx_avg = thrust::reduce(ptr_lx, ptr_lx + N_X, 0.0) / static_cast<real>(N_X);
    real vel_x_res = (lx - lx_avg) / std::fmax(R, 1.0e-30);
    real vel_z = lz / y;

    std::cout
    << std::setfill(' ')
    << "  [CFL] cell=("
    << std::setw(4) << ix << ","
    << std::setw(4) << iy << ","
    << std::setw(4) << iz << ")"
    << "  R="    << std::scientific << std::setprecision(3) << R
    << "  dvx="  << std::setw(8) << vel_x_res
    << "  vy="   << std::setw(8) << vy
    << "  vz="   << std::setw(8) << vel_z
    << "  rate=" << std::setw(8) << max_rate
    << "  dt="   << std::setw(8) << CFL_DYN / max_rate
    << std::endl;

    return dt_cfl;
}

// =========================================================================================================================
// diagnostic message output functions for simulation progress

inline __host__
void msg_output (int idx_file)
{
    std::time_t t_curr = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    int len = std::max(3, (int)std::to_string(SAVE_MAX).length());

    std::cout
    << std::endl
    << std::setfill('0')
    << std::setw(len) << idx_file << "/"
    << std::setw(len) << SAVE_MAX
    << " finished on " << std::ctime(&t_curr)
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
// save and load binary data to/from files

inline __host__
std::string frame_num (int idx_file)
{
    std::string num_str = std::to_string(idx_file);
    int num_len = std::max(5, (int)std::to_string(SAVE_MAX).length());

    if ((int)num_str.length() < num_len)
    {
        num_str.insert(0, num_len - num_str.length(), '0');
    }

    return num_str;
}

template <typename T> inline __host__
bool save_binary (const std::string &file_name, T *data, int count)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<char*>(data), sizeof(T)*count);
    return file.good();
}

template <typename T> inline __host__
bool load_binary (const std::string &file_name, T *data, int count)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(data), sizeof(T)*count);
    return file.good();
}

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
#define SAVE_OPTDEPTH_TO_FILE(IDX)                                                              \
do {                                                                                            \
    CUDA_CHECK(cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    if (!save_binary(PATH + "optdepth_" + frame_num(IDX) + ".dat", optdepth, N_G))              \
    { std::cerr << "Error: failed to save optdepth frame " << IDX << "\n"; }                    \
} while(0)
#endif

#define SAVE_DUSTDENS_TO_FILE(IDX)                                                              \
do {                                                                                            \
    CUDA_CHECK(cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    if (!save_binary(PATH + "dustdens_" + frame_num(IDX) + ".dat", dustdens, N_G))              \
    { std::cerr << "Error: failed to save dustdens frame " << IDX << "\n"; }                    \
} while(0)

#define SAVE_DUST_VEL_TO_FILE(IDX)                                                              \
do {                                                                                            \
    CUDA_CHECK(cudaMemcpy(dustvelx, dev_dustvelx, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    CUDA_CHECK(cudaMemcpy(dustvely, dev_dustvely, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    CUDA_CHECK(cudaMemcpy(dustvelz, dev_dustvelz, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    save_sam_as_velocity(dustvelx, dustvelz);                                                   \
    if (!save_binary(PATH + "dustvelx_" + frame_num(IDX) + ".dat", dustvelx, N_G))              \
    { std::cerr << "Error: failed to save dustvelx frame " << IDX << "\n"; }                    \
    if (!save_binary(PATH + "dustvely_" + frame_num(IDX) + ".dat", dustvely, N_G))              \
    { std::cerr << "Error: failed to save dustvely frame " << IDX << "\n"; }                    \
    if (!save_binary(PATH + "dustvelz_" + frame_num(IDX) + ".dat", dustvelz, N_G))              \
    { std::cerr << "Error: failed to save dustvelz frame " << IDX << "\n"; }                    \
} while(0)

#define LOAD_DUSTDATA_TO_VRAM(IDX)                                                              \
do {                                                                                            \
    if (!load_binary(PATH + "dustdens_" + frame_num(IDX) + ".dat", dustdens, N_G))              \
    { std::cerr << "Error: failed to load dustdens frame " << IDX << "\n"; return 1; }          \
    if (!load_binary(PATH + "dustvelx_" + frame_num(IDX) + ".dat", dustvelx, N_G))              \
    { std::cerr << "Error: failed to load dustvelx frame " << IDX << "\n"; return 1; }          \
    if (!load_binary(PATH + "dustvely_" + frame_num(IDX) + ".dat", dustvely, N_G))              \
    { std::cerr << "Error: failed to load dustvely frame " << IDX << "\n"; return 1; }          \
    if (!load_binary(PATH + "dustvelz_" + frame_num(IDX) + ".dat", dustvelz, N_G))              \
    { std::cerr << "Error: failed to load dustvelz frame " << IDX << "\n"; return 1; }          \
    load_velocity_as_sam(dustvelx, dustvelz);                                                   \
    CUDA_CHECK(cudaMemcpy(dev_dustdens, dustdens, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvelx, dustvelx, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvely, dustvely, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvelz, dustvelz, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
} while(0)

// =========================================================================================================================
// saving variables to file 

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
    #ifdef VISC_ACCRETION
    file << "VISC_ACCRETION = " << std::defaultfloat << 1                            << "\n";
    #endif // VISC_ACCRETION
    file                                                                               << "\n";

    file << "STOKES_0    = " << std::scientific   << std::setprecision(8) << STOKES_0  << "\n";
    file << "METAL_Z     = " << std::scientific   << std::setprecision(8) << METAL_Z   << "\n";
    file << "RHO_VAC     = " << std::scientific   << std::setprecision(8) << RHO_VAC   << "\n";
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific   << std::setprecision(8) << BETA_0    << "\n";
    file << "KAPPA_0     = " << std::scientific   << std::setprecision(8) << KAPPA_0   << "\n";
    file << "T_BETA      = " << std::scientific   << std::setprecision(8) << T_BETA    << "\n";
    #endif
    #ifdef DIFFUSION
    file << "SCHMIDT_Y   = " << std::scientific   << std::setprecision(8) << SCHMIDT_Y << "\n";
    file << "SCHMIDT_X   = " << std::scientific   << std::setprecision(8) << SCHMIDT_X << "\n";
    file << "SCHMIDT_Z   = " << std::scientific   << std::setprecision(8) << SCHMIDT_Z << "\n";
    file << "POS_LIMIT   = " << std::scientific   << std::setprecision(8) << POS_LIMIT << "\n";
    #endif
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

#endif
