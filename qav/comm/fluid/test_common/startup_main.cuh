#include <cstddef>    // std::size_t
#include <filesystem> // std::filesystem::create_directories
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include "device_api.cuh"

#include <fluid_host.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// test-only probe of the production initializer's resolved-polar density balance
// advection and diffusion start from identical state copies, so their summed finite-difference tendencies approximate the
// instantaneous continuum residual without mixing in later momentum relaxation or operator-splitting effects

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
    qav_malloc(&dev_initdens, initdens.size(), "allocate convolved density");
    qav_malloc(&dev_dustdens, N_G, "allocate working density");
    qav_malloc(&dev_dustvelx, N_G, "allocate x velocity");
    qav_malloc(&dev_dustvely, N_G, "allocate y velocity");
    qav_malloc(&dev_dustvelz, N_G, "allocate z velocity");
    qav_malloc(&dev_dustmomx, N_G, "allocate x momentum");
    qav_malloc(&dev_dustmomy, N_G, "allocate y momentum");
    qav_malloc(&dev_dustmomz, N_G, "allocate z momentum");
    qav_malloc(&dev_state_initial, 4*N_G, "allocate initial conserved state");
    qav_malloc(&dev_ppm_weight_z, ppm_weight_z.size(), "allocate polar PPM weights");

    #ifdef FLUID_BLOCK_SWEEP
    real *dev_adv_work = nullptr;
    qav_malloc(&dev_adv_work, static_cast<std::size_t>(BLOCK_ADV_FIELDS)*N_G, "allocate block advection workspace");
    #endif // FLUID_BLOCK_SWEEP

    qav_copy_h2d(dev_initdens, initdens.data(), initdens.size(), "upload convolved density");
    qav_copy_h2d(dev_ppm_weight_z, ppm_weight_z.data(), ppm_weight_z.size(), "upload polar PPM weights");

    // initialize density, its balancing polar velocity, and the conserved momenta through the production routines
    init_rho_calc <<< NB_G, TPB >>> (dev_dustdens, dev_initdens);
    qav_kernel_check("production density initialization");
    init_vel_calc <<< NB_G, TPB >>> (dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens);
    qav_kernel_check("production velocity initialization");
    momentum_setv <<< NB_G, TPB >>> (
        dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz,
        dev_dustmomx, dev_dustmomy, dev_dustmomz
    );
    qav_kernel_check("production momentum initialization");

    // retain one immutable device copy so both directional operators receive bit-identical input
    qav_copy_d2d(dev_state_initial + 0*N_G, dev_dustdens, N_G, "save initial density");
    qav_copy_d2d(dev_state_initial + 1*N_G, dev_dustmomx, N_G, "save initial x momentum");
    qav_copy_d2d(dev_state_initial + 2*N_G, dev_dustmomy, N_G, "save initial y momentum");
    qav_copy_d2d(dev_state_initial + 3*N_G, dev_dustmomz, N_G, "save initial z momentum");
    qav_copy_d2h(dustdens_initial.data(), dev_dustdens, N_G, "copy initial density");

    // scale the probe interval with polar spacing; temporal differencing then remains at least as accurate as the expected
    // second-order spatial cancellation while retaining enough change to stay above roundoff
    real probe_dt = 0.25*_get_dz();

    #ifdef FLUID_BLOCK_SWEEP
    advection_zbl <<< N_X*N_Y, TPB_BLOCK >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
        dev_ppm_weight_z, dev_adv_work, probe_dt
    );
    qav_kernel_check("polar advection tendency");
    #else // !FLUID_BLOCK_SWEEP
    advection_zth <<< NB_Z, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_ppm_weight_z, probe_dt
    );
    qav_kernel_check("polar advection tendency");
    #endif // FLUID_BLOCK_SWEEP
    qav_copy_d2h(dustdens_advection.data(), dev_dustdens, N_G, "copy advected density");

    // restore the same initial state before evaluating the diffusion contribution
    qav_copy_d2d(dev_dustdens, dev_state_initial + 0*N_G, N_G, "restore initial density");
    qav_copy_d2d(dev_dustmomx, dev_state_initial + 1*N_G, N_G, "restore initial x momentum");
    qav_copy_d2d(dev_dustmomy, dev_state_initial + 2*N_G, N_G, "restore initial y momentum");
    qav_copy_d2d(dev_dustmomz, dev_state_initial + 3*N_G, N_G, "restore initial z momentum");

    #ifdef FLUID_BLOCK_SWEEP
    diffusion_zbl <<< N_X*N_Y, TPB_BLOCK, sizeof(real)*6*N_Z >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, probe_dt
    );
    qav_kernel_check("polar diffusion tendency");
    #else // !FLUID_BLOCK_SWEEP
    diffusion_zth <<< NB_Z, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, probe_dt
    );
    qav_kernel_check("polar diffusion tendency");
    #endif // FLUID_BLOCK_SWEEP
    qav_copy_d2h(dustdens_diffusion.data(), dev_dustdens, N_G, "copy diffused density");

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

    qav_free(dev_initdens, "free convolved density");
    qav_free(dev_dustdens, "free working density");
    qav_free(dev_dustvelx, "free x velocity");
    qav_free(dev_dustvely, "free y velocity");
    qav_free(dev_dustvelz, "free z velocity");
    qav_free(dev_dustmomx, "free x momentum");
    qav_free(dev_dustmomy, "free y momentum");
    qav_free(dev_dustmomz, "free z momentum");
    qav_free(dev_state_initial, "free initial conserved state");
    qav_free(dev_ppm_weight_z, "free polar PPM weights");
    #ifdef FLUID_BLOCK_SWEEP
    qav_free(dev_adv_work, "free block advection workspace");
    #endif // FLUID_BLOCK_SWEEP

    std::cout << "fluid polar startup probe completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
