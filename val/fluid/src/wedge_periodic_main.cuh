// include fragment: the shared wedge-periodic main program, included once by each wedge test's fluid_runtime.cu
#include <cmath>      // exp, fmin, fmod, pow, sin, sqrt
#include <cstddef>    // std::size_t
#include <cstdlib>    // std::exit, EXIT_FAILURE
#include <filesystem> // std::filesystem::create_directories
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <gpu.cuh>

#include <fluid_kern.cuh>

// test-only driver for periodic azimuthal transport and diffusion on a non-2pi wedge
// the transport pulse crosses the seam under an integer-plus-fractional FARGO shift, while the diffusion mode exercises
// the cyclic Crank-Nicolson corner coupling; both references use the configured wedge period rather than a full-disk
// assumption

namespace
{
const std::string output_path = PATH_OUT;
constexpr real TRANSPORT_RATE = 0.25;
constexpr real TRANSPORT_TIME = 1.0;
constexpr real DIFFUSION_TIME = 0.2;
constexpr real BUMP_LOWER = 0.45;
constexpr real BUMP_UPPER = 0.75;
constexpr real MODE_PHASE = 0.17;

template <typename Function>
real gauss8 (Function function, real lower, real upper)
{
    static constexpr real node[4] = {
        0.18343464249564980494, 0.52553240991632898582,
        0.79666647741362673959, 0.96028985649753623168
    };
    static constexpr real weight[4] = {
        0.36268378337836198297, 0.31370664587788728734,
        0.22238103445337447054, 0.10122853629037625915
    };

    real midpoint = 0.5*(lower + upper);
    real radius = 0.5*(upper - lower);
    real sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        real offset = radius*node[i];
        sum += weight[i]*(function(midpoint - offset) + function(midpoint + offset));
    }
    return radius*sum;
}

real wrap_x (real x)
{
    real period = X_MAX - X_MIN;
    real wrapped = fmod(x - X_MIN, period);
    if (wrapped < 0.0) wrapped += period;
    return X_MIN + wrapped;
}

real compact_bump_periodic (real x)
{
    real value = wrap_x(x);
    if (value <= BUMP_LOWER || value >= BUMP_UPPER) return 0.0;
    real center = 0.5*(BUMP_LOWER + BUMP_UPPER);
    real half_width = 0.5*(BUMP_UPPER - BUMP_LOWER);
    real coordinate = (value - center) / half_width;
    return exp(1.0 - 1.0 / (1.0 - coordinate*coordinate));
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open wedge-periodic output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write wedge-periodic output");
}

const char *case_name ()
{
    #ifdef VERIFY_X_WEDGE_TRANSPORT
    return "x_wedge_transport_2d";
    #else  // VERIFY_X_WEDGE_DIFFUSION
    return "x_wedge_diffusion_2d";
    #endif // VERIFY_X_WEDGE_TRANSPORT
}
}

int main ()
{
    std::filesystem::create_directories(output_path);

    std::vector<real> dustdens(N_G);
    std::vector<real> dustmomx(N_G);
    std::vector<real> dustmomy(N_G);
    std::vector<real> dustmomz(N_G);
    std::vector<real> dustdens_initial(N_G);

    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real wave_number = 2.0*M_PI / (X_MAX - X_MIN);
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y_i = Y_MIN*pow(dy, static_cast<real>(iy));
        real y_o = y_i*dy;
        real R = sqrt(y_i*y_o);

        for (int ix = 0; ix < N_X; ix++)
        {
            real x_i = X_MIN + static_cast<real>(ix)*dx;
            real x_o = x_i + dx;
            int idx_cell = ix + iy*N_X;

            #ifdef VERIFY_X_WEDGE_TRANSPORT
            // the nonzero floor keeps the pulse away from the vacuum closure while the compact component crosses the
            // seam
            real density = 1.0 + 0.5*gauss8(compact_bump_periodic, x_i, x_o) / dx;
            dustdens[idx_cell] = density;
            dustmomx[idx_cell] = density*R*R*TRANSPORT_RATE;
            dustmomy[idx_cell] = 0.10*density;
            dustmomz[idx_cell] = -0.05*density;
            #else  // VERIFY_X_WEDGE_DIFFUSION
            // one wedge-periodic Fourier mode has an exact radius-dependent exponential decay under azimuthal diffusion
            real mode_avg = (sin(wave_number*(x_o - MODE_PHASE))
                - sin(wave_number*(x_i - MODE_PHASE))) / (wave_number*dx);
            real density = 1.0 + 0.1*mode_avg;
            dustdens[idx_cell] = density;
            dustmomx[idx_cell] = 0.70*density;
            dustmomy[idx_cell] = -0.15*density;
            dustmomz[idx_cell] = 0.11*density;
            #endif // VERIFY_X_WEDGE_TRANSPORT
        }
    }
    dustdens_initial = dustdens;

    real *dev_dustdens = nullptr;
    real *dev_dustmomx = nullptr;
    real *dev_dustmomy = nullptr;
    real *dev_dustmomz = nullptr;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_dustdens),
        sizeof(*dev_dustdens)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_dustmomx),
        sizeof(*dev_dustmomx)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_dustmomy),
        sizeof(*dev_dustmomy)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_dustmomz),
        sizeof(*dev_dustmomz)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_dustdens, dustdens.data(), sizeof(*(dev_dustdens))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_dustmomx, dustmomx.data(), sizeof(*(dev_dustmomx))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_dustmomy, dustmomy.data(), sizeof(*(dev_dustmomy))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_dustmomz, dustmomz.data(), sizeof(*(dev_dustmomz))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    #if defined(FLUID_BLOCK_SWEEP) && defined(VERIFY_X_WEDGE_TRANSPORT)
    real *dev_adv_work = nullptr;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_adv_work),
        sizeof(*dev_adv_work)*(static_cast<std::size_t>(BLOCK_ADV_FIELDS)*N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate block advection workspace" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    #endif // FLUID_BLOCK_SWEEP && VERIFY_X_WEDGE_TRANSPORT

    #ifdef VERIFY_X_WEDGE_TRANSPORT
    real final_time = TRANSPORT_TIME;
    // ten-thirds cells makes the standard power-of-two grids take exactly 2, 4, 8, and 16 equal steps
    real dt_nominal = (10.0 / 3.0)*dx / TRANSPORT_RATE;
    #else  // VERIFY_X_WEDGE_DIFFUSION
    real final_time = DIFFUSION_TIME;
    real dt_nominal = 0.25*Y_MIN*dx;
    #endif // VERIFY_X_WEDGE_TRANSPORT

    real clock = 0.0;
    int steps = 0;
    while (clock < final_time)
    {
        real dt = fmin(dt_nominal, final_time - clock);
            #ifdef VERIFY_X_WEDGE_TRANSPORT
            #ifdef FLUID_BLOCK_SWEEP
            advection_xbl <<< N_Y*N_Z, TPB_BLOCK >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_adv_work, dt
            );
            #else  // !FLUID_BLOCK_SWEEP
            advection_xth <<< NB_X, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
            );
            #endif // FLUID_BLOCK_SWEEP
            #else  // VERIFY_X_WEDGE_DIFFUSION
            #ifdef FLUID_BLOCK_SWEEP
            diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
            );
            #else  // !FLUID_BLOCK_SWEEP
            diffusion_xth <<< NB_X, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
            );
            #endif // FLUID_BLOCK_SWEEP
            #endif // VERIFY_X_WEDGE_TRANSPORT
        if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
        {
            std::cerr << "wedge-periodic operator" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
        {
            std::cerr << "wedge-periodic operator" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        clock += dt;
        steps++;
    }

    if (gpuError_t status = gpuMemcpy(dustdens.data(), dev_dustdens, sizeof(*(dustdens.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "download density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dustmomx.data(), dev_dustmomx, sizeof(*(dustmomx.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "download x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dustmomy.data(), dev_dustmomy, sizeof(*(dustmomy.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "download y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dustmomz.data(), dev_dustmomz, sizeof(*(dustmomz.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "download z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    write_binary("dustdens_initial", dustdens_initial);
    write_binary("dustdens_final", dustdens);
    write_binary("dustmomx_final", dustmomx);
    write_binary("dustmomy_final", dustmomy);
    write_binary("dustmomz_final", dustmomz);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"" << case_name() << "\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"x_min\": " << X_MIN << ",\n"
         << "  \"x_max\": " << X_MAX << ",\n"
         << "  \"y_min\": " << Y_MIN << ",\n"
         << "  \"y_max\": " << Y_MAX << ",\n"
         << "  \"time\": " << clock << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"transport_rate\": " << TRANSPORT_RATE << ",\n"
         << "  \"diffusivity\": " << VERIFY_D << ",\n"
         << "  \"mode_phase\": " << MODE_PHASE << "\n"
         << "}\n";

    if (gpuError_t status = gpuFree(dev_dustdens); status != gpuSuccess)
    {
        std::cerr << "free density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_dustmomx); status != gpuSuccess)
    {
        std::cerr << "free x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_dustmomy); status != gpuSuccess)
    {
        std::cerr << "free y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_dustmomz); status != gpuSuccess)
    {
        std::cerr << "free z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    #if defined(FLUID_BLOCK_SWEEP) && defined(VERIFY_X_WEDGE_TRANSPORT)
    if (gpuError_t status = gpuFree(dev_adv_work); status != gpuSuccess)
    {
        std::cerr << "free block advection workspace" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    #endif // FLUID_BLOCK_SWEEP && VERIFY_X_WEDGE_TRANSPORT

    std::cout << "fluid " << case_name() << " completed at N=" << VERIFY_RES
              << " in " << steps << " step(s)." << std::endl;
    return 0;
}
