// include fragment: the attenuation test's main program, included once by its fluid_runtime.cu
#include <cmath>      // exp, pow
#include <cstdlib>    // std::exit, EXIT_FAILURE
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <gpu.cuh>

#include <fluid_kern.cuh>

// test-only driver for the coupled optical-depth and radiation-source path
// density is frozen so the analytical answer contains only the radial quadrature and one exact drag-force update;
// transport and diffusion are intentionally absent because their errors would obscure whether attenuation reaches
// source_update correctly

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open attenuation output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write attenuation output");
}
}

int main ()
{
    const real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    const real dt = 0.2;
    std::vector<real> rhod(N_G), lx(N_G, 0.0), vy(N_G, 0.0), lz(N_G, 0.0);
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y = Y_MIN*pow(dy, static_cast<real>(iy) + 0.5);
        real h_g = ASPR_0*pow(y / R_0, 0.5*(IDX_Q + 1.0));
        // choose Sigma_d so the production 2D well-mixed reconstruction gives rho_d,mid proportional to y^VERIFY_POWER
        real sigma_d = sqrt(2.0*M_PI)*h_g*y*pow(y / R_0, static_cast<real>(VERIFY_POWER));
        for (int ix = 0; ix < N_X; ix++) rhod[ix + iy*N_X] = sigma_d;
    }

    real *dev_rhod, *dev_lx, *dev_vy, *dev_lz, *dev_optdepth;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_rhod),
        sizeof(*dev_rhod)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate attenuation density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_lx), sizeof(*dev_lx)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate attenuation lx" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_vy), sizeof(*dev_vy)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate attenuation vy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_lz), sizeof(*dev_lz)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate attenuation lz" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_optdepth),
        sizeof(*dev_optdepth)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate attenuation optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_rhod, rhod.data(), sizeof(*(dev_rhod))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload attenuation density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_lx, lx.data(), sizeof(*(dev_lx))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload attenuation lx" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_vy, vy.data(), sizeof(*(dev_vy))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload attenuation vy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_lz, lz.data(), sizeof(*(dev_lz))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload attenuation lz" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // exercise the production cell optical-depth increments and radial inclusive scan without host-side reconstruction
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth, dev_rhod);
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "attenuation optical-depth increments" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "attenuation optical-depth increments" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    optdepth_csum <<< (N_X*N_Z) / TPB + 1, TPB >>> (dev_optdepth);
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "attenuation optical-depth prefix sum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "attenuation optical-depth prefix sum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // a unit taper removes startup ramping so the one-step attenuated force has a closed analytical expression
    source_update <<< NB_G, TPB >>> (dev_lx, dev_vy, dev_lz, dev_rhod, dev_optdepth, 1.0, dt);
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "attenuated frozen source response" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "attenuated frozen source response" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real> optdepth(N_G);
    if (gpuError_t status = gpuMemcpy(optdepth.data(), dev_optdepth, sizeof(*(optdepth.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy attenuation optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(lx.data(), dev_lx, sizeof(*(lx.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy attenuation lx" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(vy.data(), dev_vy, sizeof(*(vy.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy attenuation vy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(lz.data(), dev_lz, sizeof(*(lz.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy attenuation lz" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    write_binary("density", rhod);
    write_binary("optdepth", optdepth);
    write_binary("lx", lx);
    write_binary("vy", vy);
    write_binary("lz", lz);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"attenuation_2d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"power\": " << static_cast<real>(VERIFY_POWER) << ",\n"
         << "  \"well_mixed_2d\": true,\n"
         << "  \"radiation_taper\": 1.0\n"
         << "}\n";

    if (gpuError_t status = gpuFree(dev_rhod); status != gpuSuccess)
    {
        std::cerr << "free attenuation density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_lx); status != gpuSuccess)
    {
        std::cerr << "free attenuation lx" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_vy); status != gpuSuccess)
    {
        std::cerr << "free attenuation vy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_lz); status != gpuSuccess)
    {
        std::cerr << "free attenuation lz" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_optdepth); status != gpuSuccess)
    {
        std::cerr << "free attenuation optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "fluid attenuation case completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
