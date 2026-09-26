// include fragment: the positivity-limited diffusion test's main program, included once by its fluid_runtime.cu
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

#include <gpu.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

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
    #else  // !FLUID_BLOCK_SWEEP
    diffusion_xth <<< NB_X, TPB >>> (rhod, mx, my, mz, dt);
    #endif // FLUID_BLOCK_SWEEP
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "azimuthal diffusion positivity step" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "azimuthal diffusion positivity step" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#if defined(TEST_DIRECTION_X) || defined(TEST_DIRECTION_Y) || defined(TEST_DIRECTION_Z)

real front_dt ()
{
    #ifdef TEST_DIRECTION_X
    real dx_len = _get_ycent(0)*sin(_get_zcent(0))*_get_dx();
    return 0.89*dx_len*dx_len / VERIFY_D;
    #elif defined(TEST_DIRECTION_Y)
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real vol_y = _get_vol_y(iy);
        real dr_i = _get_ycent(iy)*(_get_dy() - 1.0) / _get_dy();
        real dr_o = _get_ycent(iy)*(_get_dy() - 1.0);
        real cn_i = (iy > 0) ? 0.5*_get_area_y(iy)*VERIFY_D / (dr_i*vol_y) : 0.0;
        real cn_o = (iy < N_Y - 1) ? 0.5*_get_area_y(iy + 1)*VERIFY_D / (dr_o*vol_y) : 0.0;
        max_cn_sum = std::max(max_cn_sum, cn_i + cn_o);
    }
    return 0.89 / max_cn_sum;
    #else  // !(TEST_DIRECTION_X || TEST_DIRECTION_Y)
    real max_cn_sum = 0.0;
    for (int iy = 0; iy < N_Y; iy++)
    {
        real y = _get_ycent(iy);
        real dz_len = y*_get_dz();
        for (int iz = 0; iz < N_Z; iz++)
        {
            real vol_z = _get_vol_z(iz);
            real cn_i = (iz > 0)
                ? 0.5*sin(_get_zface(iz))*VERIFY_D / (y*dz_len*vol_z) : 0.0;
            real cn_o = (iz < N_Z - 1)
                ? 0.5*sin(_get_zface(iz + 1))*VERIFY_D / (y*dz_len*vol_z) : 0.0;
            max_cn_sum = std::max(max_cn_sum, cn_i + cn_o);
        }
    }
    return 0.89 / max_cn_sum;
    #endif // TEST_DIRECTION_X / TEST_DIRECTION_Y
}

void diffuse_front (real *rhod, real *mx, real *my, real *mz, real dt)
{
    #ifdef TEST_DIRECTION_X
    #ifdef FLUID_BLOCK_SWEEP
    diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (rhod, mx, my, mz, dt);
    #else  // !FLUID_BLOCK_SWEEP
    diffusion_xth <<< NB_X, TPB >>> (rhod, mx, my, mz, dt);
    #endif // FLUID_BLOCK_SWEEP
    #elif defined(TEST_DIRECTION_Y)
    #ifdef FLUID_BLOCK_SWEEP
    diffusion_ybl <<< N_X*N_Z, TPB_BLOCK, sizeof(real)*6*N_Y >>> (rhod, mx, my, mz, dt);
    #else  // !FLUID_BLOCK_SWEEP
    diffusion_yth <<< NB_Y, TPB >>> (rhod, mx, my, mz, dt);
    #endif // FLUID_BLOCK_SWEEP
    #else  // !(TEST_DIRECTION_X || TEST_DIRECTION_Y)
    #ifdef FLUID_BLOCK_SWEEP
    diffusion_zbl <<< N_X*N_Y, TPB_BLOCK, sizeof(real)*6*N_Z >>> (rhod, mx, my, mz, dt);
    #else  // !FLUID_BLOCK_SWEEP
    diffusion_zth <<< NB_Z, TPB >>> (rhod, mx, my, mz, dt);
    #endif // FLUID_BLOCK_SWEEP
    #endif // TEST_DIRECTION_X / TEST_DIRECTION_Y
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "directional diffusion donor-limit step" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "directional diffusion donor-limit step" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int line_index (int ix, int iy, int iz)
{
    #ifdef TEST_DIRECTION_X
    return ix;
    #elif defined(TEST_DIRECTION_Y)
    return iy;
    #else  // !(TEST_DIRECTION_X || TEST_DIRECTION_Y)
    return iz;
    #endif // TEST_DIRECTION_X / TEST_DIRECTION_Y
}

int run_front ()
{
    static_assert(VERIFY_RES == 8, "the donor-limit front uses the reviewed eight-cell reproducer");

    std::vector<real> rhod_initial(N_G), mx_initial(N_G), my_initial(N_G), mz_initial(N_G);
    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                int idx_line = line_index(ix, iy, iz);
                real rhod = (idx_line == 3) ? 20.0 : 1.0;
                real qx = -0.4 + 0.2*static_cast<real>(idx_line);
                real qy =  0.6 - 0.12*static_cast<real>(idx_line);
                real qz =  0.25*static_cast<real>(idx_line % 3 - 1);

                rhod_initial[idx_cell] = rhod;
                mx_initial[idx_cell] = rhod*qx;
                my_initial[idx_cell] = rhod*qy;
                mz_initial[idx_cell] = rhod*qz;
            }
        }
    }

    real *rhod, *mx, *my, *mz;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&rhod), sizeof(*rhod)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate front density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mx), sizeof(*mx)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate front x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&my), sizeof(*my)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate front y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mz), sizeof(*mz)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate front z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(rhod, rhod_initial.data(), sizeof(*(rhod))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload front density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(mx, mx_initial.data(), sizeof(*(mx))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload front x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(my, my_initial.data(), sizeof(*(my))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload front y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(mz, mz_initial.data(), sizeof(*(mz))*(N_G),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload front z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    real dt = front_dt();
    diffuse_front(rhod, mx, my, mz, dt);

    std::vector<real> rhod_final(N_G), mx_final(N_G), my_final(N_G), mz_final(N_G);
    if (gpuError_t status = gpuMemcpy(rhod_final.data(), rhod, sizeof(*(rhod_final.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy front density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(mx_final.data(), mx, sizeof(*(mx_final.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy front x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(my_final.data(), my, sizeof(*(my_final.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy front y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(mz_final.data(), mz, sizeof(*(mz_final.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy front z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    write_binary("density_initial", rhod_initial);
    write_binary("momentum_x_initial", mx_initial);
    write_binary("momentum_y_initial", my_initial);
    write_binary("momentum_z_initial", mz_initial);
    write_binary("density_final", rhod_final);
    write_binary("momentum_x_final", mx_final);
    write_binary("momentum_y_final", my_final);
    write_binary("momentum_z_final", mz_final);

    #ifdef TEST_DIRECTION_X
    const char *direction = "x";
    #elif defined(TEST_DIRECTION_Y)
    const char *direction = "y";
    #else  // !(TEST_DIRECTION_X || TEST_DIRECTION_Y)
    const char *direction = "z";
    #endif // TEST_DIRECTION_X / TEST_DIRECTION_Y
    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"diffusion_poslimit\",\n"
         << "  \"mode\": \"donor_front\",\n"
         << "  \"direction\": \"" << direction << "\",\n"
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
         << "  \"dt\": " << dt << ",\n"
         << "  \"diffusivity\": " << VERIFY_D << "\n"
         << "}\n";

    if (gpuError_t status = gpuFree(rhod); status != gpuSuccess)
    {
        std::cerr << "free front density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mx); status != gpuSuccess)
    {
        std::cerr << "free front x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(my); status != gpuSuccess)
    {
        std::cerr << "free front y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mz); status != gpuSuccess)
    {
        std::cerr << "free front z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "fluid " << direction << " diffusion donor-limit case completed at N="
              << VERIFY_RES << std::endl;
    return 0;
}

#endif // TEST_DIRECTION_X || TEST_DIRECTION_Y || TEST_DIRECTION_Z
}

int main ()
{
    #if defined(TEST_DIRECTION_X) || defined(TEST_DIRECTION_Y) || defined(TEST_DIRECTION_Z)
    return run_front();
    #else  // !(TEST_DIRECTION_X || TEST_DIRECTION_Y || TEST_DIRECTION_Z)
    const real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    const real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    const real R = sqrt(Y_MIN*Y_MIN*dy);
    const real dt = 1.0;
    // reproduce the production positivity estimate outside the kernel solely to construct an independent manual call
    // sequence
    const int sub_count = std::max(
        1, static_cast<int>(ceil(dt*VERIFY_D / (R*R*dx*dx) / POS_LIMIT))
    );

    // the large Fourier amplitude forces more than one positivity substep while retaining strictly positive initial
    // density
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
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&rhod_auto),
        sizeof(*rhod_auto)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate automatic density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mx_auto),
        sizeof(*mx_auto)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate automatic x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&my_auto),
        sizeof(*my_auto)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate automatic y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mz_auto),
        sizeof(*mz_auto)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate automatic z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&rhod_manual),
        sizeof(*rhod_manual)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate manual density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mx_manual),
        sizeof(*mx_manual)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate manual x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&my_manual),
        sizeof(*my_manual)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate manual y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&mz_manual),
        sizeof(*mz_manual)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate manual z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (auto fields : {
        std::array<real *, 4>{rhod_auto, mx_auto, my_auto, mz_auto},
        std::array<real *, 4>{rhod_manual, mx_manual, my_manual, mz_manual}
    })
    {
        if (gpuError_t status = gpuMemcpy(fields[0], rhod_initial.data(), sizeof(*(fields[0]))*(N_G),
            gpuMemcpyHostToDevice); status != gpuSuccess)
        {
            std::cerr << "upload density" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (gpuError_t status = gpuMemcpy(fields[1], mx_initial.data(), sizeof(*(fields[1]))*(N_G),
            gpuMemcpyHostToDevice); status != gpuSuccess)
        {
            std::cerr << "upload x momentum" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (gpuError_t status = gpuMemcpy(fields[2], my_initial.data(), sizeof(*(fields[2]))*(N_G),
            gpuMemcpyHostToDevice); status != gpuSuccess)
        {
            std::cerr << "upload y momentum" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (gpuError_t status = gpuMemcpy(fields[3], mz_initial.data(), sizeof(*(fields[3]))*(N_G),
            gpuMemcpyHostToDevice); status != gpuSuccess)
        {
            std::cerr << "upload z momentum" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // exercise the kernel's internal substep-count decision
    diffuse(rhod_auto, mx_auto, my_auto, mz_auto, dt);

    std::vector<real> rhod_work(N_G), minima(sub_count);
    for (int idx_sub = 0; idx_sub < sub_count; idx_sub++)
    {
        // bypass automatic splitting by supplying one already-admissible substep at a time
        diffuse(rhod_manual, mx_manual, my_manual, mz_manual, dt / static_cast<real>(sub_count));
        if (gpuError_t status = gpuMemcpy(rhod_work.data(), rhod_manual, sizeof(*(rhod_work.data()))*(N_G),
            gpuMemcpyDeviceToHost); status != gpuSuccess)
        {
            std::cerr << "copy manual density" << ": " << gpuGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
        minima[idx_sub] = *std::min_element(rhod_work.begin(), rhod_work.end());
    }

    std::vector<real> rhod_auto_host(N_G), rhod_manual_host(N_G);
    if (gpuError_t status = gpuMemcpy(rhod_auto_host.data(), rhod_auto, sizeof(*(rhod_auto_host.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy automatic density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(rhod_manual_host.data(), rhod_manual, sizeof(*(rhod_manual_host.data()))*(N_G),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy final manual density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
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

    if (gpuError_t status = gpuFree(rhod_auto); status != gpuSuccess)
    {
        std::cerr << "free automatic density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mx_auto); status != gpuSuccess)
    {
        std::cerr << "free automatic x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(my_auto); status != gpuSuccess)
    {
        std::cerr << "free automatic y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mz_auto); status != gpuSuccess)
    {
        std::cerr << "free automatic z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(rhod_manual); status != gpuSuccess)
    {
        std::cerr << "free manual density" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mx_manual); status != gpuSuccess)
    {
        std::cerr << "free manual x momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(my_manual); status != gpuSuccess)
    {
        std::cerr << "free manual y momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(mz_manual); status != gpuSuccess)
    {
        std::cerr << "free manual z momentum" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "fluid diffusion positivity case completed at N=" << VERIFY_RES
              << " with " << sub_count << " CN substeps" << std::endl;
    return 0;
    #endif // TEST_DIRECTION_X || TEST_DIRECTION_Y || TEST_DIRECTION_Z
}
