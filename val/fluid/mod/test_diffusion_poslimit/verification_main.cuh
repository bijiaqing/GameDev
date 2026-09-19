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
#else
    diffusion_xth <<< NB_X, TPB >>> (rhod, mx, my, mz, dt);
#endif
    val_kernel_check("azimuthal diffusion positivity step");
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
#else
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
#endif
}

void diffuse_front (real *rhod, real *mx, real *my, real *mz, real dt)
{
#ifdef TEST_DIRECTION_X
#ifdef FLUID_BLOCK_SWEEP
    diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (rhod, mx, my, mz, dt);
#else
    diffusion_xth <<< NB_X, TPB >>> (rhod, mx, my, mz, dt);
#endif
#elif defined(TEST_DIRECTION_Y)
#ifdef FLUID_BLOCK_SWEEP
    diffusion_ybl <<< N_X*N_Z, TPB_BLOCK, sizeof(real)*6*N_Y >>> (rhod, mx, my, mz, dt);
#else
    diffusion_yth <<< NB_Y, TPB >>> (rhod, mx, my, mz, dt);
#endif
#else
#ifdef FLUID_BLOCK_SWEEP
    diffusion_zbl <<< N_X*N_Y, TPB_BLOCK, sizeof(real)*6*N_Z >>> (rhod, mx, my, mz, dt);
#else
    diffusion_zth <<< NB_Z, TPB >>> (rhod, mx, my, mz, dt);
#endif
#endif
    val_kernel_check("directional diffusion donor-limit step");
}

int line_index (int ix, int iy, int iz)
{
#ifdef TEST_DIRECTION_X
    return ix;
#elif defined(TEST_DIRECTION_Y)
    return iy;
#else
    return iz;
#endif
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
    val_malloc(&rhod, N_G, "allocate front density");
    val_malloc(&mx, N_G, "allocate front x momentum");
    val_malloc(&my, N_G, "allocate front y momentum");
    val_malloc(&mz, N_G, "allocate front z momentum");
    val_copy_h2d(rhod, rhod_initial.data(), N_G, "upload front density");
    val_copy_h2d(mx, mx_initial.data(), N_G, "upload front x momentum");
    val_copy_h2d(my, my_initial.data(), N_G, "upload front y momentum");
    val_copy_h2d(mz, mz_initial.data(), N_G, "upload front z momentum");

    real dt = front_dt();
    diffuse_front(rhod, mx, my, mz, dt);

    std::vector<real> rhod_final(N_G), mx_final(N_G), my_final(N_G), mz_final(N_G);
    val_copy_d2h(rhod_final.data(), rhod, N_G, "copy front density");
    val_copy_d2h(mx_final.data(), mx, N_G, "copy front x momentum");
    val_copy_d2h(my_final.data(), my, N_G, "copy front y momentum");
    val_copy_d2h(mz_final.data(), mz, N_G, "copy front z momentum");

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
#else
    const char *direction = "z";
#endif
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

    val_free(rhod, "free front density");
    val_free(mx, "free front x momentum");
    val_free(my, "free front y momentum");
    val_free(mz, "free front z momentum");

    std::cout << "fluid " << direction << " diffusion donor-limit case completed at N="
              << VERIFY_RES << std::endl;
    return 0;
}

#endif
}

int main ()
{
#if defined(TEST_DIRECTION_X) || defined(TEST_DIRECTION_Y) || defined(TEST_DIRECTION_Z)
    return run_front();
#else
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
#endif
}
