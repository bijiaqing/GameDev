#ifndef GRAFFITI_HOST_CUH
#define GRAFFITI_HOST_CUH

#include <algorithm>            // std::fill, std::max/min, std::swap
#include <chrono>               // std::chrono::system_clock
#include <cmath>                // std::abs, trigonometric, exponential, and power functions
#include <cstdlib>              // std::exit, EXIT_FAILURE
#include <ctime>                // std::time_t, std::ctime
#include <fstream>              // std::ifstream, std::ofstream
#include <iomanip>              // stream formatters: setfill, setw, setprecision
#include <iostream>             // std::cout, std::cerr, std::endl
#include <sstream>              // std::ostringstream
#include <string>               // std::string, std::to_string
#include <vector>               // std::vector

#include <cuda_runtime.h>       // CUDA error types/checks and cudaMemcpy
#include <thrust/device_ptr.h>  // thrust::device_ptr
#include <thrust/extrema.h>     // thrust::max_element
#include <thrust/reduce.h>      // thrust::reduce

#include <const.cuh>
#include <helpers.cuh>

// =========================================================================================================================
// CUDA runtime error handling
// =========================================================================================================================

// Report CUDA runtime failures with the originating expression and source location, 
// then stop before stale or partially updated device fields can be used by later operators

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

// Call immediately after a kernel launch. cudaGetLastError catches invalid launch parameters,
// resource overuse, and other launch-time failures. Later checked synchronization/copy calls
// catch asynchronous execution failures

#define CUDA_KERNEL_CHECK(KERNEL_NAME)                                              \
do {                                                                                \
    cudaError_t cuda_status_ = cudaGetLastError();                                  \
    if (cuda_status_ != cudaSuccess)                                                \
    { cuda_fail(cuda_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }  \
} while (0)

// =========================================================================================================================
// Conservative nonuniform PPM reconstruction weights
// =========================================================================================================================

// Solve for the four weights that evaluate a cubic polynomial at face `iface` from the
// volume averages in cells iface-2 ... iface+1. `face_s` is a monotone volume coordinate,
// so each stored finite-volume value is an ordinary average over ds
// The local normalization keeps the small Vandermonde-like system well conditioned on logarithmic and polar meshes

inline __host__
void _ppm_cubic_face_weights (const std::vector<real> &face_s, int iface, real *weight)
{
    real face = face_s[iface];
    real scale = std::max(face - face_s[iface - 2], face_s[iface + 2] - face);

    // A[n][j] is the cell average of t^n in stencil cell j, with
    // t=(s-s_face)/scale. Solve A*w=e_0, which makes sum_j w_j*qbar_j=p(s_face)
    real aug[4][5] = {};
    for (int n = 0; n < 4; n++)
    {
        for (int j = 0; j < 4; j++)
        {
            int icell = iface - 2 + j;
            real t_lo = (face_s[icell]     - face) / scale;
            real t_hi = (face_s[icell + 1] - face) / scale;
            aug[n][j] = (std::pow(t_hi, n + 1) - std::pow(t_lo, n + 1)) / (static_cast<real>(n + 1)*(t_hi - t_lo));
        }
        aug[n][4] = (n == 0) ? 1.0 : 0.0;
    }

    // Four-by-four Gaussian elimination with partial pivoting; this runs only once at startup
    for (int col = 0; col < 4; col++)
    {
        int pivot_row = col;

        for (int row = col + 1; row < 4; row++)
        {
            if (std::abs(aug[row][col]) > std::abs(aug[pivot_row][col])) 
            {
                pivot_row = row;
            }
        }
            
        for (int k = col; k < 5; k++) 
        {
            std::swap(aug[col][k], aug[pivot_row][k]);
        }

        real pivot = aug[col][col];

        for (int k = col; k < 5; k++)
        {
            aug[col][k] /= pivot;
        }

        for (int row = 0; row < 4; row++)
        {
            if (row == col) continue;
            real factor = aug[row][col];
            for (int k = col; k < 5; k++)
            {
                aug[row][k] -= factor*aug[col][k];
            }
        }
    }

    for (int j = 0; j < 4; j++)
    {
        weight[j] = aug[j][4];
    }
}

// Fill four coefficients per face. Interior faces use conservative cubic reconstruction;
// the first and last interior faces use a geometry-aware linear reconstruction between the
// two adjacent volume centroids. Boundary faces themselves remain first-order outflow states

inline __host__
void _ppm_nonuniform_weights (const std::vector<real> &face_s, real *weight)
{
    int n_cells = static_cast<int>(face_s.size()) - 1;
    std::fill(weight, weight + 4*(n_cells + 1), 0.0);

    for (int iface = 1; iface < n_cells; iface++)
    {
        real *face_weight = weight + 4*iface;
        if (iface >= 2 && iface <= n_cells - 2)
        {
            _ppm_cubic_face_weights(face_s, iface, face_weight);
            continue;
        }

        real center_L = 0.5*(face_s[iface-1] + face_s[iface]);
        real center_R = 0.5*(face_s[iface] + face_s[iface+1]);

        face_weight[0] = (center_R - face_s[iface]) / (center_R - center_L);
        face_weight[1] = (face_s[iface] - center_L) / (center_R - center_L);
    }
}

// Radial volume coordinate: s_r=r^d/d. Polar volume coordinate: s_theta=-cos(theta)
// These make r^(d-1)dr and sin(theta)dtheta ordinary ds measures, respectively

inline __host__
void ppm_geometry_weights_calc (real *weight_y, real *weight_z)
{
    real pow_y = (N_Z > 1) ? 3.0 : 2.0;
    real dy = std::pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    std::vector<real> y_face_s(N_Y + 1);
    
    for (int iy = 0; iy <= N_Y; iy++)
    {
        real y0 = Y_MIN*std::pow(dy, static_cast<real>(iy));
        y_face_s[iy] = std::pow(y0, pow_y) / pow_y;
    }
    
    _ppm_nonuniform_weights(y_face_s, weight_y);

    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
    std::vector<real> z_face_s(N_Z + 1);
    
    for (int iz = 0; iz <= N_Z; iz++)
    {
        z_face_s[iz] = -std::cos(Z_MIN + dz*static_cast<real>(iz));
    }
    
    _ppm_nonuniform_weights(z_face_s, weight_z);
}

// =========================================================================================================================
// Convolved power-law surface density profile
// =========================================================================================================================

// Computes the radial dust surface-density profile on N_Y+1 uniformly spaced cylindrical-R
// nodes.  f_rho_initial interpolates these nodes at each cell's R=r*sin(theta).
// Before convolution, Sigma_d(R)=METAL_Z*Sigma_g(R) inside the tapered radial support, where
// Sigma_g(R)=SIGMA_0*(R/R_0)^IDX_P. The Gaussian is normalized and integrated with the
// uniform-grid quadrature factor, so the amplitude is independent of N_Y 
// No global post-convolution normalization is applied: METAL_Z sets the input ratio, 
// while edge smoothing is allowed to alter the realized local ratio and total dust mass
// Conversion to a midplane volume density occurs exactly once in f_rho_initial

inline __host__
void convpow_calc (real *profile)
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

    const real norm = 1.0 / (std::sqrt(2.0*M_PI)*sig_u);

    // Convolve the unconvolved metallicity-scaled gas surface density with a normalized Gaussian
    // The du factor is the source-coordinate quadrature measure.
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

    std::copy(v_axis.begin(), v_axis.end(), profile);
}

// =========================================================================================================================
// Host CFL reduction and diagnostics
// =========================================================================================================================

inline __host__
real get_dt_cfl (const real *dev_cfl_rate, const real *dev_dustvelx, const real *dev_dustvely, const real *dev_dustvelz, 
    bool verbose = true)
{
    thrust::device_ptr <const real> ptr_cfl(dev_cfl_rate);
    auto max_it = thrust::max_element(ptr_cfl, ptr_cfl + N_G);
    real max_rate = *max_it;

    if (!std::isfinite(max_rate))
    {
        int idx_bad = static_cast<int>(max_it - ptr_cfl);
        int ix_bad = idx_bad % N_X;
        int iy_bad = (idx_bad / N_X) % N_Y;
        int iz_bad = idx_bad / (N_X*N_Y);
        std::cerr << "Error: non-finite dust state detected by CFL validation at cell ("
                  << ix_bad << "," << iy_bad << "," << iz_bad << ").\n";
        std::exit(EXIT_FAILURE);
    }

    if (max_rate <= 0.0) return DT_MAX;

    real dt_cfl = std::fmin(CFL_NUM / max_rate, DT_MAX);
    if (!verbose) return dt_cfl;

    int idx_max = static_cast<int>(max_it - ptr_cfl);
    int ix = idx_max % N_X;
    int iy = (idx_max / N_X) % N_Y;
    int iz = idx_max / (N_X * N_Y);

    real lx, vy, lz;
    CUDA_CHECK(cudaMemcpy(&lx, dev_dustvelx + idx_max, sizeof(real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&vy, dev_dustvely + idx_max, sizeof(real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lz, dev_dustvelz + idx_max, sizeof(real), cudaMemcpyDeviceToHost));

    real yc = Y_MIN*std::pow(_get_dy(), iy + 0.5);
    real zc = (N_Z > 1) ? Z_MIN + (iz + 0.5)*_get_dz() : 0.5*(Z_MIN + Z_MAX);
    real Rc = yc*std::sin(zc);

    // FARGO residual azimuthal velocity (what actually enters the CFL)
    // Reconstruct the arithmetic mean of the limiting cell's ring to match cfl_rate_calc/f_advection_x
    int idx_base = iy*N_X + iz*N_X*N_Y;
    thrust::device_ptr <const real> ptr_lx(dev_dustvelx + idx_base);
    real lx_avg = thrust::reduce(ptr_lx, ptr_lx + N_X, 0.0) / static_cast<real>(N_X);
    real vx_res = (N_X > 1) ? ((lx - lx_avg) / std::fmax(Rc, 1.0e-30)) : 0.0;
    real vz = lz / yc;

    std::cout 
    << std::setfill(' ')
    << "  [CFL] cell=("
    << std::setw(4) << ix << "," 
    << std::setw(4) << iy << "," 
    << std::setw(3) << iz << ")"
    << "  Rc="   << std::scientific << std::setprecision(3) << Rc
    << "  vy="   << std::setw(11) << vy
    << "  vz="   << std::setw(11) << vz
    << "  dvx="  << std::setw(11) << vx_res
    << "  rate=" << std::setw(11) << max_rate
    << "  dt="   << std::setw(11) << CFL_NUM / max_rate
    << std::endl;

    return dt_cfl;
}

// =========================================================================================================================
// Console progress message
// =========================================================================================================================

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
// Frame numbering
// =========================================================================================================================

inline __host__
std::string frame_num (int number)
{
    std::string num_str = std::to_string(number);
    int num_len = std::max(5, (int)std::to_string(SAVE_MAX).length());

    if ((int)num_str.length() < num_len)
    {
        num_str.insert(0, num_len - num_str.length(), '0');
    }
    
    return num_str;
}

// =========================================================================================================================
// Binary file I/O
// =========================================================================================================================

template <typename T> inline __host__
bool save_binary (const std::string &fname, T *data, int number)
{
    std::ofstream file(fname, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<char*>(data), sizeof(T)*number);
    return file.good();
}

template <typename T> inline __host__
bool load_binary (const std::string &fname, T *data, int number)
{
    std::ifstream file(fname, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(data), sizeof(T)*number);
    return file.good();
}

// =========================================================================================================================
// Field save/load macros
// =========================================================================================================================

#ifdef RADIATION
#define SAVE_OPTDEPTH_TO_FILE(IDX)                                                              \
do {                                                                                            \
    CUDA_CHECK(cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    if (!save_binary(PATH + "optdepth_" + frame_num(IDX) + ".dat", optdepth, N_G))              \
    { std::cerr << "Error: failed to save optdepth frame " << IDX << "\n"; }                    \
} while(0)
#endif

#define SAVE_DUSTDATA_TO_FILE(IDX)                                                              \
do {                                                                                            \
    CUDA_CHECK(cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    if (!save_binary(PATH + "dustdens_" + frame_num(IDX) + ".dat", dustdens, N_G))              \
    { std::cerr << "Error: failed to save dustdens frame " << IDX << "\n"; }                    \
    CUDA_CHECK(cudaMemcpy(dustvelx, dev_dustvelx, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    CUDA_CHECK(cudaMemcpy(dustvely, dev_dustvely, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
    CUDA_CHECK(cudaMemcpy(dustvelz, dev_dustvelz, sizeof(real)*N_G, cudaMemcpyDeviceToHost));   \
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
    CUDA_CHECK(cudaMemcpy(dev_dustdens, dustdens, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvelx, dustvelx, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvely, dustvely, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
    CUDA_CHECK(cudaMemcpy(dev_dustvelz, dustvelz, sizeof(real)*N_G, cudaMemcpyHostToDevice));   \
} while(0)

// =========================================================================================================================
// Save simulation parameters
// =========================================================================================================================

// Exact, machine-readable identity for every physical/numerical setting that must remain
// unchanged across a restart.  Hexadecimal floating-point output preserves all bits and avoids
// tolerance choices during comparison.  SAVE_MAX is deliberately excluded so a run can be
// extended; DT_OUT is included because restart time is reconstructed from the frame number.
inline __host__
std::string restart_config_signature ()
{
    std::ostringstream sig;
    sig << std::hexfloat
        << "schema=1"
        << ";real_bytes=" << sizeof(real)
        << ";G=" << G << ";M_S=" << M_S << ";R_0=" << R_0
        << ";N_X=" << N_X << ";X_MIN=" << X_MIN << ";X_MAX=" << X_MAX
        << ";N_Y=" << N_Y << ";Y_MIN=" << Y_MIN << ";Y_MAX=" << Y_MAX
        << ";N_Z=" << N_Z << ";Z_MIN=" << Z_MIN << ";Z_MAX=" << Z_MAX
        << ";SIGMA_0=" << SIGMA_0 << ";ASPR_0=" << ASPR_0
        << ";IDX_P=" << IDX_P << ";IDX_Q=" << IDX_Q
        << ";METAL_Z=" << METAL_Z << ";ST_0=" << ST_0
        << ";DT_OUT=" << DT_OUT << ";DT_MAX=" << DT_MAX << ";OUTPUT_TIME_TOL=" << OUTPUT_TIME_TOL
        << ";CFL_NUM=" << CFL_NUM << ";RHO_VAC=" << RHO_VAC
        << ";TPB=" << TPB;

    #ifdef RADIATION
    sig << ";RADIATION=1;BETA_0=" << BETA_0 << ";KAPPA_0=" << KAPPA_0 << ";T_BETA=" << T_BETA;
    #else
    sig << ";RADIATION=0";
    #endif

    #ifdef DIFFUSION
    sig << ";DIFFUSION=1;SC_X=" << SC_X << ";SC_Y=" << SC_Y << ";SC_Z=" << SC_Z
        << ";POS_LIMIT=" << POS_LIMIT;
        #ifdef CONST_NU
        sig << ";CONST_NU=1;NU=" << NU;
        #else
        sig << ";CONST_NU=0;ALPHA=" << ALPHA;
        #endif
    #else
    sig << ";DIFFUSION=0";
    #endif

    #ifdef HALFDISK
    sig << ";HALFDISK=1";
    #else
    sig << ";HALFDISK=0";
    #endif

    return sig.str();
}

inline __host__
bool validate_restart_config (const std::string &fname)
{
    std::ifstream file(fname);
    if (!file)
    {
        std::cerr << "Error: cannot open restart configuration file " << fname << "\n";
        return false;
    }

    const std::string prefix = "RESTART_CONFIG = ";
    std::string line, saved_signature;
    while (std::getline(file, line))
    {
        if (line.rfind(prefix, 0) == 0)
        {
            saved_signature = line.substr(prefix.size());
            break;
        }
    }

    if (saved_signature.empty())
    {
        std::cerr << "Error: restart configuration signature is missing from " << fname
                  << ". Legacy outputs must be restarted with their original code.\n";
        return false;
    }

    std::string current_signature = restart_config_signature();
    if (saved_signature != current_signature)
    {
        std::cerr << "Error: restart configuration does not match the current build.\n"
                  << "Saved:  " << saved_signature << "\n"
                  << "Current: " << current_signature << "\n";
        return false;
    }

    return true;
}

inline __host__
bool save_variable (const std::string &fname)
{
    std::ofstream file(fname);
    if (!file) return false;

    file << "[PARAMETERS]\n\n";

    file << "RESTART_CONFIG = " << restart_config_signature() << "\n\n";

    #ifdef RADIATION
    file << "RADIATION    = 1\n";
    #else
    file << "RADIATION    = 0\n";
    #endif
    #ifdef DIFFUSION
    file << "DIFFUSION    = 1\n";
    #else
    file << "DIFFUSION    = 0\n";
    #endif
    #ifdef CONST_NU
    file << "CONST_NU     = 1\n";
    #else
    file << "CONST_NU     = 0\n";
    #endif
    #ifdef HALFDISK
    file << "HALFDISK     = 1\n\n";
    #else
    file << "HALFDISK     = 0\n\n";
    #endif

    // Gas parameters
    file << "SIGMA_0     = " << std::scientific   << std::setprecision(8) << SIGMA_0  << "\n";
    file << "ASPR_0      = " << std::defaultfloat << std::setprecision(8) << ASPR_0   << "\n";
    file << "IDX_P       = " << std::defaultfloat << std::setprecision(8) << IDX_P    << "\n";
    file << "IDX_Q       = " << std::defaultfloat << std::setprecision(8) << IDX_Q    << "\n";
    #ifdef DIFFUSION
    #ifndef CONST_NU
    file << "ALPHA       = " << std::scientific   << std::setprecision(8) << ALPHA    << "\n";
    #else
    file << "NU          = " << std::scientific   << std::setprecision(8) << NU       << "\n";
    #endif
    #endif
    file                                                                              << "\n";

    // Dust parameters
    file << "ST_0        = " << std::scientific   << std::setprecision(8) << ST_0     << "\n";
    file << "METAL_Z     = " << std::scientific   << std::setprecision(8) << METAL_Z  << "\n";
    file << "RHO_VAC     = " << std::scientific   << std::setprecision(8) << RHO_VAC  << "\n";
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific   << std::setprecision(8) << BETA_0   << "\n";
    file << "KAPPA_0     = " << std::scientific   << std::setprecision(8) << KAPPA_0  << "\n";
    file << "T_BETA      = " << std::scientific   << std::setprecision(8) << T_BETA   << "\n";
    #endif
    #ifdef DIFFUSION
    file << "SC_Y        = " << std::scientific   << std::setprecision(8) << SC_Y     << "\n";
    file << "SC_X        = " << std::scientific   << std::setprecision(8) << SC_X     << "\n";
    file << "SC_Z        = " << std::scientific   << std::setprecision(8) << SC_Z     << "\n";
    file << "POS_LIMIT   = " << std::scientific   << std::setprecision(8) << POS_LIMIT << "\n";
    #endif
    file                                                                              << "\n";

    // Mesh domain
    file << "N_X         = " << std::defaultfloat << std::setprecision(8) << N_X      << "\n";
    file << "X_MIN       = " << std::defaultfloat << std::setprecision(8) << X_MIN    << "\n";
    file << "X_MAX       = " << std::defaultfloat << std::setprecision(8) << X_MAX    << "\n";
    file                                                                              << "\n";
    file << "N_Y         = " << std::defaultfloat << std::setprecision(8) << N_Y      << "\n";
    file << "Y_MIN       = " << std::scientific   << std::setprecision(8) << Y_MIN    << "\n";
    file << "Y_MAX       = " << std::scientific   << std::setprecision(8) << Y_MAX    << "\n";
    file                                                                              << "\n";
    file << "N_Z         = " << std::defaultfloat << std::setprecision(8) << N_Z      << "\n";
    file << "Z_MIN       = " << std::scientific   << std::setprecision(8) << Z_MIN    << "\n";
    file << "Z_MAX       = " << std::scientific   << std::setprecision(8) << Z_MAX    << "\n";
    file                                                                              << "\n";
    file << "N_G         = " << std::scientific   << std::setprecision(8) << N_G      << "\n";
    file                                                                              << "\n";

    // Time step and output
    file << "SAVE_MAX    = " << std::defaultfloat << std::setprecision(8) << SAVE_MAX << "\n";
    file << "DT_OUT      = " << std::scientific   << std::setprecision(8) << DT_OUT   << "\n";
    file << "CFL_NUM     = " << std::scientific   << std::setprecision(8) << CFL_NUM  << "\n";
    file << "OUTPUT_TIME_TOL = " << std::scientific << std::setprecision(8) << OUTPUT_TIME_TOL << "\n";
    file << "DT_MAX      = " << std::scientific   << std::setprecision(8) << DT_MAX   << "\n";
    file                                                                              << "\n";

    return file.good();
}

// =========================================================================================================================

#endif // GRAFFITI_HOST_CUH
