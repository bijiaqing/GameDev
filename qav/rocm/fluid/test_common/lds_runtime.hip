#include <algorithm>    // std::max
#include <cmath>        // std::abs, std::isfinite
#include <cstddef>      // std::size_t
#include <cstdlib>      // EXIT_FAILURE, EXIT_SUCCESS
#include <fstream>      // std::ofstream
#include <iomanip>      // std::boolalpha, std::setprecision
#include <iostream>     // std::cerr, std::cout, std::endl
#include <string>       // std::string
#include <vector>       // std::vector

#include <hip/hip_runtime.h> // hipDeviceSynchronize, hipFree, hipGetLastError, hipMalloc, hipMemcpy

#include <fluid_host.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// ROCm qualification driver rather than a physical evolution model
// request the exact dynamic LDS used by one production block-line diffusion kernel, verify that a uniform state remains
// stationary, and separately confirm that an unsupported allocation is rejected before launch

const std::string PATH = PATH_OUT;

int main ()
{
    #ifdef VERIFY_LDS_REJECT
    // deliberately exceed the supported launch allocation; successful continuation means the production guard failed
    require_lds(
        reinterpret_cast<const void*>(diffusion_ybl), sizeof(real)*6*N_Y, "diffusion_ybl"
    );
    std::cerr << "Error: unsupported LDS request was not rejected" << std::endl;
    return EXIT_FAILURE;
    #else
    // use a spatially uniform conserved state whose zero diffusive flux exposes launch, indexing, or synchronization errors
    std::vector<real> rhod(N_G, 1.0);
    std::vector<real> mx(N_G, 2.0);
    std::vector<real> my(N_G, -0.5);
    std::vector<real> mz(N_G, 0.25);

    real *dev_rhod, *dev_mx, *dev_my, *dev_mz;
    HIP_CHECK(hipMalloc((void**)&dev_rhod, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_mx, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_my, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_mz, sizeof(real)*N_G));
    HIP_CHECK(hipMemcpy(dev_rhod, rhod.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_mx, mx.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_my, my.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_mz, mz.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));

    const void *kernel = nullptr;
    const char *kernel_name = nullptr;
    std::size_t dynamic_lds = 0;
    int block_count = 0;
    #ifdef VERIFY_LDS_X
    kernel = reinterpret_cast<const void*>(diffusion_xbl);
    kernel_name = "diffusion_xbl";
    dynamic_lds = sizeof(real)*4*N_X;
    block_count = N_Y*N_Z;
    #elif defined(VERIFY_LDS_Y)
    kernel = reinterpret_cast<const void*>(diffusion_ybl);
    kernel_name = "diffusion_ybl";
    dynamic_lds = sizeof(real)*6*N_Y;
    block_count = N_X*N_Z;
    #elif defined(VERIFY_LDS_Z)
    kernel = reinterpret_cast<const void*>(diffusion_zbl);
    kernel_name = "diffusion_zbl";
    dynamic_lds = sizeof(real)*6*N_Z;
    block_count = N_X*N_Y;
    #endif

    // query and validate the device-specific LDS limit through the same host guard used by production
    lds_usage usage = require_lds(kernel, dynamic_lds, kernel_name);
    constexpr real dt = 1.0e-8;
    #ifdef VERIFY_LDS_X
    diffusion_xbl <<< block_count, TPB_BLOCK, dynamic_lds >>> (dev_rhod, dev_mx, dev_my, dev_mz, dt);
    #elif defined(VERIFY_LDS_Y)
    diffusion_ybl <<< block_count, TPB_BLOCK, dynamic_lds >>> (dev_rhod, dev_mx, dev_my, dev_mz, dt);
    #elif defined(VERIFY_LDS_Z)
    diffusion_zbl <<< block_count, TPB_BLOCK, dynamic_lds >>> (dev_rhod, dev_mx, dev_my, dev_mz, dt);
    #endif
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(rhod.data(), dev_rhod, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mx.data(), dev_mx, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(my.data(), dev_my, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mz.data(), dev_mz, sizeof(real)*N_G, hipMemcpyDeviceToHost));

    bool finite = true;
    real max_abs_error = 0.0;
    long double mass_initial = 0.0;
    long double mass_final = 0.0;
    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            long double measure = static_cast<long double>(_get_dx())*
                static_cast<long double>(_get_vol_y(iy))*static_cast<long double>(_get_vol_z(iz));
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx = ix + iy*N_X + iz*N_X*N_Y;
                finite = finite && std::isfinite(rhod[idx]) && std::isfinite(mx[idx])
                    && std::isfinite(my[idx]) && std::isfinite(mz[idx]);
                max_abs_error = std::max(max_abs_error, std::abs(rhod[idx] - 1.0));
                max_abs_error = std::max(max_abs_error, std::abs(mx[idx] - 2.0));
                max_abs_error = std::max(max_abs_error, std::abs(my[idx] + 0.5));
                max_abs_error = std::max(max_abs_error, std::abs(mz[idx] - 0.25));
                mass_initial += measure;
                mass_final += static_cast<long double>(rhod[idx])*measure;
            }
        }
    }
    real mass_relative_error = static_cast<real>(
        std::abs(mass_final - mass_initial) / std::abs(mass_initial)
    );
    bool passed = finite && max_abs_error <= 1.0e-11 && mass_relative_error <= 1.0e-12;

    std::ofstream file(PATH + "metrics.json");
    file << std::boolalpha << std::setprecision(17)
    << "{\n"
    << "  \"kernel\": \"" << kernel_name << "\",\n"
    << "  \"grid\": [" << N_X << ", " << N_Y << ", " << N_Z << "],\n"
    << "  \"device_lds_limit\": " << usage.device_limit << ",\n"
    << "  \"kernel_static_lds\": " << usage.kernel_static << ",\n"
    << "  \"kernel_dynamic_limit\": " << usage.kernel_dynamic_limit << ",\n"
    << "  \"kernel_limit_reported\": " << usage.kernel_limit_reported << ",\n"
    << "  \"requested_dynamic_lds\": " << usage.requested_dynamic << ",\n"
    << "  \"total_lds\": " << usage.total << ",\n"
    << "  \"launch_succeeded\": true,\n"
    << "  \"finite\": " << finite << ",\n"
    << "  \"max_abs_stationary_error\": " << max_abs_error << ",\n"
    << "  \"mass_relative_error\": " << mass_relative_error << ",\n"
    << "  \"passed\": " << passed << "\n"
    << "}\n";

    HIP_CHECK(hipFree(dev_rhod));
    HIP_CHECK(hipFree(dev_mx));
    HIP_CHECK(hipFree(dev_my));
    HIP_CHECK(hipFree(dev_mz));

    std::cout << kernel_name << " large-LDS launch: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    #endif // VERIFY_LDS_REJECT
}
