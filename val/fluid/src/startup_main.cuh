// include fragment: the startup-balance probe's main program, included once by its fluid_runtime.cu
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
#include <param_grid.cuh>

// test-only probe of the production initializer's resolved-polar density balance
// advection and diffusion start from identical state copies, so their summed finite-difference tendencies approximate
// the instantaneous continuum residual without mixing in later momentum relaxation or operator-splitting effects

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open startup output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write startup output");
}
}

int main ()
{
    std::filesystem::create_directories(output_path);

    std::vector<real> initdens(N_Y + 1);
    std::vector<real> dustdens_initial(N_G);
    std::vector<real> dustdens_advection(N_G);
    std::vector<real> dustdens_diffusion(N_G);
    std::vector<real> ppm_weight_y(4*(N_Y + 1));
    std::vector<real> ppm_weight_z(4*(N_Z + 1));
    initdens_calc(initdens.data());
    ppm_geometry_weights_calc(ppm_weight_y.data(), ppm_weight_z.data());

    real *dev_initdens = nullptr;
    real *dev_dustdens = nullptr;
    real *dev_dustvelx = nullptr;
    real *dev_dustvely = nullptr;
    real *dev_dustvelz = nullptr;
    real *dev_dustmomx = nullptr;
    real *dev_dustmomy = nullptr;
    real *dev_dustmomz = nullptr;
    real *dev_state_initial = nullptr;
    real *dev_ppm_weight_z = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_initdens), sizeof(*dev_initdens)*(initdens.size())));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustdens), sizeof(*dev_dustdens)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustvelx), sizeof(*dev_dustvelx)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustvely), sizeof(*dev_dustvely)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustvelz), sizeof(*dev_dustvelz)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomx), sizeof(*dev_dustmomx)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomy), sizeof(*dev_dustmomy)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_dustmomz), sizeof(*dev_dustmomz)*(N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_state_initial), sizeof(*dev_state_initial)*(4*N_G)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_ppm_weight_z), sizeof(*dev_ppm_weight_z)*(ppm_weight_z.size())));

    #ifdef FLUID_BLOCK_SWEEP
    real *dev_adv_work = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&dev_adv_work),
        sizeof(*dev_adv_work)*(static_cast<std::size_t>(BLOCK_ADV_FIELDS)*N_G)));
        #endif // FLUID_BLOCK_SWEEP

    GPU_CHECK(gpuMemcpy(dev_initdens, initdens.data(), sizeof(*(dev_initdens))*(initdens.size()),
        gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dev_ppm_weight_z, ppm_weight_z.data(), sizeof(*(dev_ppm_weight_z))*(ppm_weight_z.size()),
        gpuMemcpyHostToDevice));

    // initialize density, its balancing polar velocity, and the conserved momenta through the production routines
    init_rho_calc <<< NB_G, TPB >>> (dev_dustdens, dev_initdens);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    init_vel_calc <<< NB_G, TPB >>> (dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    momentum_setv <<< NB_G, TPB >>> (
        dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz,
        dev_dustmomx, dev_dustmomy, dev_dustmomz
    );
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    // retain one immutable device copy so both directional operators receive bit-identical input
    GPU_CHECK(gpuMemcpy(dev_state_initial + 0*N_G, dev_dustdens, sizeof(*(dev_state_initial + 0*N_G))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_state_initial + 1*N_G, dev_dustmomx, sizeof(*(dev_state_initial + 1*N_G))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_state_initial + 2*N_G, dev_dustmomy, sizeof(*(dev_state_initial + 2*N_G))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_state_initial + 3*N_G, dev_dustmomz, sizeof(*(dev_state_initial + 3*N_G))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dustdens_initial.data(), dev_dustdens, sizeof(*(dustdens_initial.data()))*(N_G),
        gpuMemcpyDeviceToHost));

    // scale the probe interval with polar spacing; temporal differencing then remains at least as accurate as the
    // expected second-order spatial cancellation while retaining enough change to stay above roundoff
    real probe_dt = 0.25*_get_dz();

    #ifdef FLUID_BLOCK_SWEEP
    advection_zbl <<< N_X*N_Y, TPB_BLOCK >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
        dev_ppm_weight_z, dev_adv_work, probe_dt
    );
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    #else  // !FLUID_BLOCK_SWEEP
    advection_zth <<< NB_Z, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_ppm_weight_z, probe_dt
    );
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    #endif // FLUID_BLOCK_SWEEP
    GPU_CHECK(gpuMemcpy(dustdens_advection.data(), dev_dustdens, sizeof(*(dustdens_advection.data()))*(N_G),
        gpuMemcpyDeviceToHost));

    // restore the same initial state before evaluating the diffusion contribution
    GPU_CHECK(gpuMemcpy(dev_dustdens, dev_state_initial + 0*N_G, sizeof(*(dev_dustdens))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomx, dev_state_initial + 1*N_G, sizeof(*(dev_dustmomx))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomy, dev_state_initial + 2*N_G, sizeof(*(dev_dustmomy))*(N_G),
        gpuMemcpyDeviceToDevice));
    GPU_CHECK(gpuMemcpy(dev_dustmomz, dev_state_initial + 3*N_G, sizeof(*(dev_dustmomz))*(N_G),
        gpuMemcpyDeviceToDevice));

    #ifdef FLUID_BLOCK_SWEEP
    diffusion_zbl <<< N_X*N_Y, TPB_BLOCK, sizeof(real)*6*N_Z >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, probe_dt
    );
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    #else  // !FLUID_BLOCK_SWEEP
    diffusion_zth <<< NB_Z, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, probe_dt
    );
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    #endif // FLUID_BLOCK_SWEEP
    GPU_CHECK(gpuMemcpy(dustdens_diffusion.data(), dev_dustdens, sizeof(*(dustdens_diffusion.data()))*(N_G),
        gpuMemcpyDeviceToHost));

    write_binary("dustdens_initial", dustdens_initial);
    write_binary("dustdens_advection", dustdens_advection);
    write_binary("dustdens_diffusion", dustdens_diffusion);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"startup_3d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"y_min\": " << Y_MIN << ",\n"
         << "  \"y_max\": " << Y_MAX << ",\n"
         << "  \"z_min\": " << Z_MIN << ",\n"
         << "  \"z_max\": " << Z_MAX << ",\n"
         << "  \"probe_dt\": " << probe_dt << "\n"
         << "}\n";

    GPU_CHECK(gpuFree(dev_initdens));
    GPU_CHECK(gpuFree(dev_dustdens));
    GPU_CHECK(gpuFree(dev_dustvelx));
    GPU_CHECK(gpuFree(dev_dustvely));
    GPU_CHECK(gpuFree(dev_dustvelz));
    GPU_CHECK(gpuFree(dev_dustmomx));
    GPU_CHECK(gpuFree(dev_dustmomy));
    GPU_CHECK(gpuFree(dev_dustmomz));
    GPU_CHECK(gpuFree(dev_state_initial));
    GPU_CHECK(gpuFree(dev_ppm_weight_z));
    #ifdef FLUID_BLOCK_SWEEP
    GPU_CHECK(gpuFree(dev_adv_work));
    #endif // FLUID_BLOCK_SWEEP

    std::cout << "fluid polar startup probe completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
