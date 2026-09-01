#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <_collision.cuh>
#include <device_api.cuh>

namespace
{
constexpr int turbulence_count = 16;
constexpr int result_count = 22;
const std::string output_path = PATH_OUT;

// evaluate every physical-rate branch at fixed inputs without running a stochastic collision history
__global__ void collision_physics (real *result, const swarm *particle, const real *size, const real *number)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr real R = 1.0;
    constexpr real size_ratio = 0.3;
    real h_g = _get_hg(R);
    real sigma_g = _get_sigma_g(R);
    real alpha = _get_alpha(R, h_g);
    real re_inv_sqrt = _get_re_inv_sqrt(R, alpha, sigma_g);
    result[0] = re_inv_sqrt;

    // the first six points lie inside the six Ormel-Cuzzi regimes; the remaining pairs straddle every branch boundary
    real stokes_large[turbulence_count];
    stokes_large[0] = 0.02*re_inv_sqrt;
    stokes_large[1] = 0.40*re_inv_sqrt;
    stokes_large[2] = 2.00*re_inv_sqrt;
    stokes_large[3] = sqrt(5.0*re_inv_sqrt*0.2);
    stokes_large[4] = 0.5;
    stokes_large[5] = 2.0;

    real boundary[5] = {
        0.2*re_inv_sqrt,
        re_inv_sqrt / 1.6,
        5.0*re_inv_sqrt,
        0.2,
        1.0
    };
    for (int idx = 0; idx < 5; idx++)
    {
        stokes_large[6 + 2*idx] = boundary[idx]*(1.0 - 1.0e-6);
        stokes_large[7 + 2*idx] = boundary[idx]*(1.0 + 1.0e-6);
    }
    for (int idx = 0; idx < turbulence_count; idx++)
        result[1 + idx] = _get_vrel_t(R, stokes_large[idx], size_ratio*stokes_large[idx], h_g, sigma_g);

    #ifdef CODE_UNIT
    result[17] = 0.0;
    result[18] = 0.0;
    #else  // PHYSICAL_UNIT
    result[17] = _get_vrel_b(R, size[0], size[1], h_g);
    result[18] = _get_vrel_b(R, 1.0e-12, 1.0e-12, h_g);
    #endif // CODE_UNIT

    result[19] = _get_vrel_pair(particle, size[0], size[1], 0, 1);
    result[20] = _get_col_rate_ij<CUSTOM_KERNEL>(particle, size, number, 0, 1, 0.3);

    real3 velocity_i = _get_cart_vel(particle[0]);
    real3 velocity_j = _get_cart_vel(particle[1]);
    real dvx = velocity_i.x - velocity_j.x;
    real dvy = velocity_i.y - velocity_j.y;
    real dvz = velocity_i.z - velocity_j.z;
    result[21] = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open collision-physics output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write collision-physics output");
}
}

int main ()
{
    std::array<swarm, N_P> particle{};
    std::array<real, N_P> size = {0.5, 1.75};
    std::array<real, N_P> number = {3.0, 7.0};

    particle[0].position = make_double3(0.3, 1.0, 0.5*M_PI);
    particle[1].position = make_double3(-0.4, 1.0, 0.5*M_PI);
    particle[0].velocity = make_double3(0.2, -0.03, 0.0);
    particle[1].velocity = make_double3(-0.1, 0.04, 0.0);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].par_size = size[idx];
        particle[idx].par_numr = number[idx];
    }

    swarm *dev_particle;
    real *dev_size;
    real *dev_number;
    real *dev_result;
    qav_malloc(&dev_particle, N_P, "allocate collision-physics particles");
    qav_malloc(&dev_size, N_P, "allocate collision-physics sizes");
    qav_malloc(&dev_number, N_P, "allocate collision-physics numbers");
    qav_malloc(&dev_result, result_count, "allocate collision-physics results");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload collision-physics particles");
    qav_copy_h2d(dev_size, size.data(), N_P, "upload collision-physics sizes");
    qav_copy_h2d(dev_number, number.data(), N_P, "upload collision-physics numbers");

    collision_physics <<< 1, 1 >>> (dev_result, dev_particle, dev_size, dev_number);
    qav_kernel_check("collision_physics");

    std::vector<real> result(result_count);
    qav_copy_d2h(result.data(), dev_result, result_count, "copy collision-physics results");
    write_binary("collision_physics", result);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    if (!meta) throw std::runtime_error("cannot open collision-physics metadata");
    meta << std::setprecision(17)
         << "{\n"
         #ifdef CODE_UNIT
         << "  \"case\": \"colphys_code\",\n"
         << "  \"code_unit\": true,\n"
         #else  // PHYSICAL_UNIT
         << "  \"case\": \"colphys_cgs\",\n"
         << "  \"code_unit\": false,\n"
         #endif // CODE_UNIT
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"result_count\": " << result_count << ",\n"
         << "  \"turbulence_count\": " << turbulence_count << ",\n"
         << "  \"position_x\": [0.3, -0.4],\n"
         << "  \"velocity_lx\": [0.2, -0.1],\n"
         << "  \"velocity_y\": [-0.03, 0.04],\n"
         << "  \"size\": [0.5, 1.75],\n"
         << "  \"number\": [3.0, 7.0]\n"
         << "}\n";
    if (!meta) throw std::runtime_error("cannot write collision-physics metadata");

    qav_free(dev_result, "free collision-physics results");
    qav_free(dev_number, "free collision-physics numbers");
    qav_free(dev_size, "free collision-physics sizes");
    qav_free(dev_particle, "free collision-physics particles");
    std::cout << "swarm physical collision kernel completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
