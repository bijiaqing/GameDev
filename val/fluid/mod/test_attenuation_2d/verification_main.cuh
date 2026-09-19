#include <cmath>      // exp, pow
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <device_api.cuh>
#include <fluid_kern.cuh>

// test-only driver for the coupled optical-depth and radiation-source path
// density is frozen so the analytical answer contains only the radial quadrature and one exact drag-force update; transport
// and diffusion are intentionally absent because their errors would obscure whether attenuation reaches source_update correctly

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
    const real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
    const real dt = 0.2;
    std::vector<real> rhod(N_G), lx(N_G, 0.0), vy(N_G, 0.0), lz(N_G, 0.0);
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y = Y_MIN*pow(dy, static_cast<real>(iy) + 0.5);
        real h_g = ASPR_0*pow(y/R_0, 0.5*(IDX_Q + 1.0));
        // choose Sigma_d so the production 2D well-mixed reconstruction gives rho_d,mid proportional to y^VERIFY_POWER
        real sigma_d = sqrt(2.0*M_PI)*h_g*y*pow(y/R_0, static_cast<real>(VERIFY_POWER));
        for (int ix = 0; ix < N_X; ix++) rhod[ix + iy*N_X] = sigma_d;
    }

    real *dev_rhod, *dev_lx, *dev_vy, *dev_lz, *dev_optdepth;
    val_malloc(&dev_rhod, N_G, "allocate attenuation density");
    val_malloc(&dev_lx, N_G, "allocate attenuation lx");
    val_malloc(&dev_vy, N_G, "allocate attenuation vy");
    val_malloc(&dev_lz, N_G, "allocate attenuation lz");
    val_malloc(&dev_optdepth, N_G, "allocate attenuation optical depth");
    val_copy_h2d(dev_rhod, rhod.data(), N_G, "upload attenuation density");
    val_copy_h2d(dev_lx, lx.data(), N_G, "upload attenuation lx");
    val_copy_h2d(dev_vy, vy.data(), N_G, "upload attenuation vy");
    val_copy_h2d(dev_lz, lz.data(), N_G, "upload attenuation lz");

    // exercise the production cell optical-depth increments and radial inclusive scan without host-side reconstruction
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth, dev_rhod);
    val_kernel_check("attenuation optical-depth increments");
    optdepth_csum <<< (N_X*N_Z)/TPB + 1, TPB >>> (dev_optdepth);
    val_kernel_check("attenuation optical-depth prefix sum");
    // a unit taper removes startup ramping so the one-step attenuated force has a closed analytical expression
    source_update <<< NB_G, TPB >>> (dev_lx, dev_vy, dev_lz, dev_rhod, dev_optdepth, 1.0, dt);
    val_kernel_check("attenuated frozen source response");

    std::vector<real> optdepth(N_G);
    val_copy_d2h(optdepth.data(), dev_optdepth, N_G, "copy attenuation optical depth");
    val_copy_d2h(lx.data(), dev_lx, N_G, "copy attenuation lx");
    val_copy_d2h(vy.data(), dev_vy, N_G, "copy attenuation vy");
    val_copy_d2h(lz.data(), dev_lz, N_G, "copy attenuation lz");
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

    val_free(dev_rhod, "free attenuation density");
    val_free(dev_lx, "free attenuation lx");
    val_free(dev_vy, "free attenuation vy");
    val_free(dev_lz, "free attenuation lz");
    val_free(dev_optdepth, "free attenuation optical depth");
    std::cout << "fluid attenuation case completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
