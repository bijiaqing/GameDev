// include fragment: the shared polar-boundary main program, included once by each polar test's fluid_runtime.cu
#include <cmath>      // cos, exp, fmin, pow, sin, sqrt
#include <cstddef>    // std::size_t
#include <filesystem> // std::filesystem::create_directories
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <gpu.cuh>

#include <fluid_host.cuh>
#include <fluid_kern.cuh>

// test-only driver for the production polar advection kernel and its two boundary policies
// the outflow case follows a translated compact pulse beyond the outer polar face; the HALF_DISK case follows an exact
// reflection-symmetric compression whose polar velocity vanishes at the midplane, so its continuum wall flux is zero

namespace
{
const std::string output_path = PATH_OUT;
constexpr real OUTFLOW_RATE = 0.15;
constexpr real REFLECT_RATE = 0.20;
constexpr real FINAL_TIME = 1.0;

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

real compact_bump (real value, real lower, real upper)
{
    if (value <= lower || value >= upper) return 0.0;
    real center = 0.5*(lower + upper);
    real half_width = 0.5*(upper - lower);
    real coordinate = (value - center) / half_width;
    return std::exp(1.0 - 1.0 / (1.0 - coordinate*coordinate));
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open polar-boundary output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write polar-boundary output");
}

const char *case_name ()
{
    #ifdef VERIFY_Z_OUTFLOW
    return "z_outflow_3d";
    #else  // VERIFY_Z_REFLECT
    return "z_reflect_3d";
    #endif // VERIFY_Z_OUTFLOW
}
}

int main ()
{
    std::filesystem::create_directories(output_path);

    std::vector<real> dustdens(N_G);
    std::vector<real> dustmomx(N_G, 0.0);
    std::vector<real> dustmomy(N_G, 0.0);
    std::vector<real> dustmomz(N_G);
    std::vector<real> dustdens_initial(N_G);
    std::vector<real> ppm_weight_y(4*(N_Y + 1));
    std::vector<real> ppm_weight_z(4*(N_Z + 1));
    ppm_geometry_weights_calc(ppm_weight_y.data(), ppm_weight_z.data());

    real dy = std::pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
    for (int iz = 0; iz < N_Z; iz++)
    {
        real z_i = Z_MIN + static_cast<real>(iz)*dz;
        real z_o = z_i + dz;
        real vol_z = std::cos(z_i) - std::cos(z_o);

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y_i = Y_MIN*std::pow(dy, static_cast<real>(iy));
            real y_o = y_i*dy;
            real y = std::sqrt(y_i*y_o);
            real area_z = 0.5*(y_o*y_o - y_i*y_i);
            real vol_y = (y_o*y_o*y_o - y_i*y_i*y_i) / 3.0;
            real geom_z = area_z / vol_y;

            #ifdef VERIFY_Z_OUTFLOW
            real rho_int = gauss8([&](real z)
            {
                return compact_bump(z, 2.45, 2.75);
            }, z_i, z_o);
            real lz = y*OUTFLOW_RATE / geom_z;
            real mz_int = lz*rho_int;
            #else  // VERIFY_Z_REFLECT
            // q=rho*sin(z) is compressed toward the symmetry plane by w=a*(pi/2-z); specific polar momentum is
            // proportional to w, so it is constant along the pressureless characteristics used by the production kernel
            real rho_int = gauss8([&](real z)
            {
                return compact_bump(z, 0.70, M_PI - 0.70);
            }, z_i, z_o);
            real mz_int = gauss8([&](real z)
            {
                real q = compact_bump(z, 0.70, M_PI - 0.70);
                real lz = y*REFLECT_RATE*(0.5*M_PI - z) / geom_z;
                return q*lz;
            }, z_i, z_o);
            #endif // VERIFY_Z_OUTFLOW

            for (int ix = 0; ix < N_X; ix++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                dustdens[idx_cell] = rho_int / vol_z;
                dustmomz[idx_cell] = mz_int / vol_z;
            }
        }
    }
    dustdens_initial = dustdens;

    real *dev_dustdens = nullptr;
    real *dev_dustmomx = nullptr;
    real *dev_dustmomy = nullptr;
    real *dev_dustmomz = nullptr;
    real *dev_ppm_weight_z = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustdens), sizeof(*dev_dustdens)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomx), sizeof(*dev_dustmomx)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomy), sizeof(*dev_dustmomy)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomz), sizeof(*dev_dustmomz)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_ppm_weight_z), sizeof(*dev_ppm_weight_z)*(ppm_weight_z.size())));
    GPU_CHECK(gpuMemcpy(dev_dustdens, dustdens.data(), sizeof(*(dev_dustdens))*(N_G), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomx, dustmomx.data(), sizeof(*(dev_dustmomx))*(N_G), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomy, dustmomy.data(), sizeof(*(dev_dustmomy))*(N_G), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomz, dustmomz.data(), sizeof(*(dev_dustmomz))*(N_G), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dev_ppm_weight_z, ppm_weight_z.data(), sizeof(*(dev_ppm_weight_z))*(ppm_weight_z.size()),
        gpuMemcpyHostToDevice));

    #ifdef FLUID_BLOCK_SWEEP
    real *dev_adv_work = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_adv_work),
        sizeof(*dev_adv_work)*(static_cast<std::size_t>(BLOCK_ADV_FIELDS)*N_G)));
        #endif // FLUID_BLOCK_SWEEP

    real max_rate;
    #ifdef VERIFY_Z_OUTFLOW
    max_rate = OUTFLOW_RATE;
    #else  // VERIFY_Z_REFLECT
    max_rate = REFLECT_RATE*(0.5*M_PI - Z_MIN);
    #endif // VERIFY_Z_OUTFLOW
    real dt_nominal = 0.25*dz / max_rate;
    real clock = 0.0;
    int steps = 0;
    while (clock < FINAL_TIME)
    {
        real dt = std::fmin(dt_nominal, FINAL_TIME - clock);
        #ifdef FLUID_BLOCK_SWEEP
        advection_zbl <<< N_X*N_Y, TPB_BLOCK >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_ppm_weight_z, dev_adv_work, dt
        );
        #else  // !FLUID_BLOCK_SWEEP
        advection_zth <<< NB_Z, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_ppm_weight_z, dt
        );
        #endif // FLUID_BLOCK_SWEEP
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
        clock += dt;
        steps++;
    }

    GPU_CHECK(gpuMemcpy(dustdens.data(), dev_dustdens, sizeof(*(dustdens.data()))*(N_G), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(dustmomx.data(), dev_dustmomx, sizeof(*(dustmomx.data()))*(N_G), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(dustmomy.data(), dev_dustmomy, sizeof(*(dustmomy.data()))*(N_G), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(dustmomz.data(), dev_dustmomz, sizeof(*(dustmomz.data()))*(N_G), gpuMemcpyDeviceToHost));
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
         << "  \"y_min\": " << Y_MIN << ",\n"
         << "  \"y_max\": " << Y_MAX << ",\n"
         << "  \"z_min\": " << Z_MIN << ",\n"
         << "  \"z_max\": " << Z_MAX << ",\n"
         << "  \"time\": " << clock << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"outflow_rate\": " << OUTFLOW_RATE << ",\n"
         << "  \"reflect_rate\": " << REFLECT_RATE << "\n"
         << "}\n";

    GPU_CHECK(gpuFree(dev_dustdens));
    GPU_CHECK(gpuFree(dev_dustmomx));
    GPU_CHECK(gpuFree(dev_dustmomy));
    GPU_CHECK(gpuFree(dev_dustmomz));
    GPU_CHECK(gpuFree(dev_ppm_weight_z));
    #ifdef FLUID_BLOCK_SWEEP
    GPU_CHECK(gpuFree(dev_adv_work));
    #endif // FLUID_BLOCK_SWEEP

    std::cout << "fluid " << case_name() << " completed at N=" << VERIFY_RES
              << " in " << steps << " step(s)." << std::endl;
    return 0;
}
