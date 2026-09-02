#include <algorithm>  // std::max, std::min
#include <array>      // std::array
#include <cmath>      // fabs, isfinite
#include <cstdlib>    // EXIT_FAILURE
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cerr, std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <device_api.cuh>
#include <fluid_kern.cuh>

// test-only driver for the production positivity-controlled cyclic CN solver
// advance identical initial data once with automatic internal subcycling and once with the equivalent explicit sequence
// of smaller kernel calls to distinguish the CN formula from the subcycling control path

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open diffusion positivity output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write diffusion positivity output");
}

void diffuse (real *rhod, real *mx, real *my, real *mz, real dt)
{
    // select the same thread- or block-line production kernel requested by FLUID_SWEEP
#ifdef FLUID_BLOCK_SWEEP
    diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (rhod, mx, my, mz, dt);
#else
    diffusion_xth <<< NB_X, TPB >>> (rhod, mx, my, mz, dt);
#endif
    val_kernel_check("azimuthal diffusion positivity step");
}
}

int main ()
{
    const real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    const real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
    const real R = sqrt(Y_MIN*Y_MIN*dy);
    const real dt = 1.0;
    // reproduce the production positivity estimate outside the kernel solely to construct an independent manual call sequence
    const int sub_count = std::max(
        1, static_cast<int>(ceil(dt*VERIFY_D / (R*R*dx*dx) / POS_LIMIT))
    );

    // the large Fourier amplitude forces more than one positivity substep while retaining strictly positive initial density
    std::vector<real> rhod_initial(N_G), mx_initial(N_G), my_initial(N_G), mz_initial(N_G);
    for (int ix = 0; ix < N_X; ix++)
    {
        real x0 = X_MIN + static_cast<real>(ix)*dx;
        real x1 = x0 + dx;
        real mode_avg = (sin(VERIFY_M*x1) - sin(VERIFY_M*x0)) / (VERIFY_M*dx);
        rhod_initial[ix] = VERIFY_Q0 + 0.9*mode_avg;
        mx_initial[ix] = 0.7*rhod_initial[ix];
        my_initial[ix] = -0.15*rhod_initial[ix];
        mz_initial[ix] = 0.11*rhod_initial[ix];
    }

    real *rhod_auto, *mx_auto, *my_auto, *mz_auto;
    real *rhod_manual, *mx_manual, *my_manual, *mz_manual;
    val_malloc(&rhod_auto, N_G, "allocate automatic density");
    val_malloc(&mx_auto, N_G, "allocate automatic x momentum");
    val_malloc(&my_auto, N_G, "allocate automatic y momentum");
    val_malloc(&mz_auto, N_G, "allocate automatic z momentum");
    val_malloc(&rhod_manual, N_G, "allocate manual density");
    val_malloc(&mx_manual, N_G, "allocate manual x momentum");
    val_malloc(&my_manual, N_G, "allocate manual y momentum");
    val_malloc(&mz_manual, N_G, "allocate manual z momentum");

    for (auto fields : {
        std::array<real *, 4>{rhod_auto, mx_auto, my_auto, mz_auto},
        std::array<real *, 4>{rhod_manual, mx_manual, my_manual, mz_manual}
    })
    {
        val_copy_h2d(fields[0], rhod_initial.data(), N_G, "upload density");
        val_copy_h2d(fields[1], mx_initial.data(), N_G, "upload x momentum");
        val_copy_h2d(fields[2], my_initial.data(), N_G, "upload y momentum");
        val_copy_h2d(fields[3], mz_initial.data(), N_G, "upload z momentum");
    }

    // exercise the kernel's internal substep-count decision
    diffuse(rhod_auto, mx_auto, my_auto, mz_auto, dt);

    std::vector<real> rhod_work(N_G), minima(sub_count);
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        // bypass automatic splitting by supplying one already-admissible substep at a time
        diffuse(rhod_manual, mx_manual, my_manual, mz_manual, dt/static_cast<real>(sub_count));
        val_copy_d2h(rhod_work.data(), rhod_manual, N_G, "copy manual density");
        minima[idx_sub] = *std::min_element(rhod_work.begin(), rhod_work.end());
    }

    std::vector<real> rhod_auto_host(N_G), rhod_manual_host(N_G);
    val_copy_d2h(rhod_auto_host.data(), rhod_auto, N_G, "copy automatic density");
    val_copy_d2h(rhod_manual_host.data(), rhod_manual, N_G, "copy final manual density");
    write_binary("density_initial", rhod_initial);
    write_binary("density_auto", rhod_auto_host);
    write_binary("density_manual", rhod_manual_host);
    write_binary("substep_minimum", minima);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"diffusion_poslimit\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"radius\": " << R << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"diffusivity\": " << VERIFY_D << ",\n"
         << "  \"mode\": " << VERIFY_M << ",\n"
         << "  \"sub_count\": " << sub_count << ",\n"
         << "  \"constant_diffusivity\": true,\n"
         << "  \"automatic_subcycling\": true\n"
         << "}\n";

    val_free(rhod_auto, "free automatic density");
    val_free(mx_auto, "free automatic x momentum");
    val_free(my_auto, "free automatic y momentum");
    val_free(mz_auto, "free automatic z momentum");
    val_free(rhod_manual, "free manual density");
    val_free(mx_manual, "free manual x momentum");
    val_free(my_manual, "free manual y momentum");
    val_free(mz_manual, "free manual z momentum");

    std::cout << "fluid diffusion positivity case completed at N=" << VERIFY_RES
              << " with " << sub_count << " CN substeps" << std::endl;
    return 0;
}
