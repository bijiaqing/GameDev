#include <cmath>      // cos, fmax, sin, sqrt
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
#include <param_phys.cuh>

// test-only driver for the complete production initialization path
// the density and velocity kernels remain production objects; the diagnostic kernel only exposes intermediate physical
// targets so the independent Python reference can distinguish a helper error from a projection or initialization error

namespace
{
const std::string output_path = PATH_OUT;
constexpr int DIAG_FIELDS = 7;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open initialization output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write initialization output");
}
}

__global__
void initstate_get (real *dev_diagnostic)
{
    int idx_cell = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_cell >= N_G) return;

    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);
    real y = _get_ycent(iy);
    real z = _get_zcent(iz);
    real R = y*sin(z);
    real Z = y*cos(z);
    real h_g = _get_hg(R);
    real omega = _get_omegaK(R);
    real v_K = R*omega;
    real eta = _get_eta(R, Z, h_g);

    dev_diagnostic[0*N_G + idx_cell] = h_g;
    dev_diagnostic[1*N_G + idx_cell] = _get_gas_strat(R, Z, h_g);
    dev_diagnostic[2*N_G + idx_cell] = _get_rhog(R, Z, h_g);
    dev_diagnostic[3*N_G + idx_cell] = _get_stokes(R, Z, h_g);
    dev_diagnostic[4*N_G + idx_cell] = eta;
    dev_diagnostic[5*N_G + idx_cell] = _get_visc_vel(R, Z, h_g);
    dev_diagnostic[6*N_G + idx_cell] = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
}

int main ()
{
    std::filesystem::create_directories(output_path);

    std::vector<real> initdens(N_Y + 1);
    std::vector<real> dustdens(N_G);
    std::vector<real> dustvelx(N_G);
    std::vector<real> dustvely(N_G);
    std::vector<real> dustvelz(N_G);
    std::vector<real> diagnostic(DIAG_FIELDS*N_G);
    initdens_calc(initdens.data());

    real *dev_initdens = nullptr;
    real *dev_dustdens = nullptr;
    real *dev_dustvelx = nullptr;
    real *dev_dustvely = nullptr;
    real *dev_dustvelz = nullptr;
    real *dev_diagnostic = nullptr;
    qav_malloc(&dev_initdens, initdens.size(), "allocate convolved density");
    qav_malloc(&dev_dustdens, N_G, "allocate initialized density");
    qav_malloc(&dev_dustvelx, N_G, "allocate initialized x velocity");
    qav_malloc(&dev_dustvely, N_G, "allocate initialized y velocity");
    qav_malloc(&dev_dustvelz, N_G, "allocate initialized z velocity");
    qav_malloc(&dev_diagnostic, diagnostic.size(), "allocate initialization diagnostics");
    qav_copy_h2d(dev_initdens, initdens.data(), initdens.size(), "upload convolved density");

    // run the same ordered density and velocity initialization used by fluid_runtime
    init_rho_calc <<< NB_G, TPB >>> (dev_dustdens, dev_initdens);
    qav_kernel_check("production density initialization");
    init_vel_calc <<< NB_G, TPB >>> (dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens);
    qav_kernel_check("production velocity initialization");
    initstate_get <<< NB_G, TPB >>> (dev_diagnostic);
    qav_kernel_check("initialization physical diagnostics");

    qav_copy_d2h(dustdens.data(), dev_dustdens, N_G, "copy initialized density");
    qav_copy_d2h(dustvelx.data(), dev_dustvelx, N_G, "copy initialized x velocity");
    qav_copy_d2h(dustvely.data(), dev_dustvely, N_G, "copy initialized y velocity");
    qav_copy_d2h(dustvelz.data(), dev_dustvelz, N_G, "copy initialized z velocity");
    qav_copy_d2h(diagnostic.data(), dev_diagnostic, diagnostic.size(), "copy initialization diagnostics");
    write_binary("initdens", initdens);
    write_binary("dustdens", dustdens);
    write_binary("dustvelx", dustvelx);
    write_binary("dustvely", dustvely);
    write_binary("dustvelz", dustvelz);
    write_binary("diagnostic", diagnostic);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"initstate_" << (N_Z == 1 ? "2d" : "3d") << "\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"x_min\": " << X_MIN << ",\n"
         << "  \"x_max\": " << X_MAX << ",\n"
         << "  \"y_min\": " << Y_MIN << ",\n"
         << "  \"y_max\": " << Y_MAX << ",\n"
         << "  \"z_min\": " << Z_MIN << ",\n"
         << "  \"z_max\": " << Z_MAX << ",\n"
         << "  \"diagnostic_fields\": " << DIAG_FIELDS << "\n"
         << "}\n";

    qav_free(dev_initdens, "free convolved density");
    qav_free(dev_dustdens, "free initialized density");
    qav_free(dev_dustvelx, "free initialized x velocity");
    qav_free(dev_dustvely, "free initialized y velocity");
    qav_free(dev_dustvelz, "free initialized z velocity");
    qav_free(dev_diagnostic, "free initialization diagnostics");
    std::cout << "fluid production initialization completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
