#include <cmath>      // exp, fmin, pow, sin, sqrt
#include <cstddef>    // std::size_t
#include <filesystem> // std::filesystem::create_directories
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include "device_api.cuh"

#include <fluid_kern.cuh>

// test-only long-duration composition of periodic FARGO transport and azimuthal diffusion
// four complete revolutions repeatedly exercise the periodic seam and conservative momentum fluxes without introducing
// radial boundaries or source terms, whose effects would obscure the exact translated-and-damped Fourier reference

namespace
{
const std::string output_path = PATH_OUT;
constexpr real ANGULAR_RATE = 1.0;
constexpr real RADIAL_SPEED = 0.13;
constexpr real POLAR_MOMENT = -0.09;
constexpr real MODE_PHASE = 0.17;
constexpr real SHIFT_CELLS = 3.25;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open long-ring output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write long-ring output");
}

void run_diffusion_x (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    #ifdef FLUID_BLOCK_SWEEP
    diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
    );
    #else // !FLUID_BLOCK_SWEEP
    diffusion_xth <<< NB_X, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
    );
    #endif // FLUID_BLOCK_SWEEP
    qav_kernel_check("long-ring azimuthal diffusion");
}

void run_advection_x (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real *dev_adv_work, real dt)
{
    #ifdef FLUID_BLOCK_SWEEP
    advection_xbl <<< N_Y*N_Z, TPB_BLOCK >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_adv_work, dt
    );
    #else // !FLUID_BLOCK_SWEEP
    (void)dev_adv_work;
    advection_xth <<< NB_X, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
    );
    #endif // FLUID_BLOCK_SWEEP
    qav_kernel_check("long-ring FARGO advection");
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
    std::vector<real> dustmomx_initial(N_G);
    std::vector<real> dustmomy_initial(N_G);
    std::vector<real> dustmomz_initial(N_G);

    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y_i = Y_MIN*pow(dy, static_cast<real>(iy));
        real y_o = y_i*dy;
        real R = sqrt(y_i*y_o);

        for (int ix = 0; ix < N_X; ix++)
        {
            real x_i = X_MIN + static_cast<real>(ix)*dx;
            real x_o = x_i + dx;
            real mode_avg = (sin(VERIFY_M*(x_o - MODE_PHASE))
                - sin(VERIFY_M*(x_i - MODE_PHASE))) / (VERIFY_M*dx);
            real density = R*R*(1.0 + VERIFY_EPS*mode_avg);
            int idx_cell = ix + iy*N_X;

            dustdens[idx_cell] = density;
            dustmomx[idx_cell] = density*R*R*ANGULAR_RATE;
            dustmomy[idx_cell] = density*RADIAL_SPEED;
            dustmomz[idx_cell] = density*POLAR_MOMENT;
        }
    }
    dustdens_initial = dustdens;
    dustmomx_initial = dustmomx;
    dustmomy_initial = dustmomy;
    dustmomz_initial = dustmomz;

    real *dev_dustdens = nullptr;
    real *dev_dustmomx = nullptr;
    real *dev_dustmomy = nullptr;
    real *dev_dustmomz = nullptr;
    qav_malloc(&dev_dustdens, N_G, "allocate density");
    qav_malloc(&dev_dustmomx, N_G, "allocate x momentum");
    qav_malloc(&dev_dustmomy, N_G, "allocate y momentum");
    qav_malloc(&dev_dustmomz, N_G, "allocate z momentum");
    qav_copy_h2d(dev_dustdens, dustdens.data(), N_G, "upload density");
    qav_copy_h2d(dev_dustmomx, dustmomx.data(), N_G, "upload x momentum");
    qav_copy_h2d(dev_dustmomy, dustmomy.data(), N_G, "upload y momentum");
    qav_copy_h2d(dev_dustmomz, dustmomz.data(), N_G, "upload z momentum");

    real *dev_adv_work = nullptr;
    #ifdef FLUID_BLOCK_SWEEP
    qav_malloc(&dev_adv_work, static_cast<std::size_t>(BLOCK_ADV_FIELDS)*N_G,
        "allocate block advection workspace");
    #endif // FLUID_BLOCK_SWEEP

    // a three-and-one-quarter-cell FARGO displacement combines a nonzero integer shift with a quarter-cell residual
    real dt_nominal = SHIFT_CELLS*dx / ANGULAR_RATE;
    real clock = 0.0;
    int steps = 0;
    while (clock < VERIFY_TEND)
    {
        real dt = fmin(dt_nominal, VERIFY_TEND - clock);
        run_diffusion_x(dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt);
        run_advection_x(dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_adv_work, dt);
        run_diffusion_x(dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt);
        clock += dt;
        steps++;
    }

    qav_copy_d2h(dustdens.data(), dev_dustdens, N_G, "download density");
    qav_copy_d2h(dustmomx.data(), dev_dustmomx, N_G, "download x momentum");
    qav_copy_d2h(dustmomy.data(), dev_dustmomy, N_G, "download y momentum");
    qav_copy_d2h(dustmomz.data(), dev_dustmomz, N_G, "download z momentum");
    write_binary("dustdens_initial", dustdens_initial);
    write_binary("dustmomx_initial", dustmomx_initial);
    write_binary("dustmomy_initial", dustmomy_initial);
    write_binary("dustmomz_initial", dustmomz_initial);
    write_binary("dustdens_final", dustdens);
    write_binary("dustmomx_final", dustmomx);
    write_binary("dustmomy_final", dustmomy);
    write_binary("dustmomz_final", dustmomz);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"ring_long_2d\",\n"
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
         << "  \"angular_rate\": " << ANGULAR_RATE << ",\n"
         << "  \"radial_speed\": " << RADIAL_SPEED << ",\n"
         << "  \"polar_moment\": " << POLAR_MOMENT << ",\n"
         << "  \"mode\": " << VERIFY_M << ",\n"
         << "  \"mode_phase\": " << MODE_PHASE << ",\n"
         << "  \"diffusivity\": " << NU << ",\n"
         << "  \"shift_cells\": " << SHIFT_CELLS << "\n"
         << "}\n";

    qav_free(dev_dustdens, "free density");
    qav_free(dev_dustmomx, "free x momentum");
    qav_free(dev_dustmomy, "free y momentum");
    qav_free(dev_dustmomz, "free z momentum");
    #ifdef FLUID_BLOCK_SWEEP
    qav_free(dev_adv_work, "free block advection workspace");
    #endif // FLUID_BLOCK_SWEEP

    std::cout << "fluid ring_long_2d completed through " << clock/(2.0*M_PI)
              << " revolution(s) at N=" << VERIFY_RES << " in " << steps << " step(s)." << std::endl;
    return 0;
}
