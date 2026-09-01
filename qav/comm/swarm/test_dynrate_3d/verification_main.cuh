#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <device_api.cuh>
#include <param_grid.cuh>
#include <swarm_kern.cuh>

// test-only driver for the production dynamics-rate kernel
// deterministic particles span the spherical mesh and several decades of angular and linear motion; Python reconstructs
// every enabled rate candidate rather than using another device implementation as the reference

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open dynamics-rate output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write dynamics-rate output");
}
}

int main ()
{
    static_assert(N_X > 1 && N_Y > 1 && N_Z > 1, "dynamics-rate qualification requires all mesh directions");

    const real dy = _get_dy();
    const real dz = _get_dz();
    const real motion[] = {0.0, 1.0e-4, -3.0e-3, 8.0e-2, -1.0, 15.0, -80.0, 250.0};
    std::vector<swarm> particle(N_P);
    std::vector<real> state(6*N_P);
    for (int idx = 0; idx < N_P - 1; idx++)
    {
        int ix = (3*idx + 1) % N_X;
        int iy = (7*idx + 2) % N_Y;
        int iz = (5*idx + 3) % N_Z;
        real x = X_MIN + (static_cast<real>(ix) + 0.37)*_get_dx();
        real y = Y_MIN*pow(dy, static_cast<real>(iy) + 0.43);
        real z = Z_MIN + (static_cast<real>(iz) + 0.41)*dz;
        real R = y*sin(z);
        real lx = sqrt(R)*motion[idx % 8];
        real vy = motion[(idx + 3) % 8];
        real lz = y*motion[(idx + 5) % 8];

        particle[idx].position = make_double3(x, y, z);
        particle[idx].velocity = make_double3(lx, vy, lz);
    }
    // the inactive sentinel proves that global maximum reductions may safely retain zero-rate representatives
    particle[N_P - 1].position = make_double3(0.0, 0.0, 0.0);
    particle[N_P - 1].velocity = make_double3(0.0, 0.0, 0.0);

    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx] = particle[idx].position.x;
        state[N_P + idx] = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
    }

    swarm *dev_particle;
    real *dev_dyn_rate;
    qav_malloc(&dev_particle, N_P, "allocate dynamics-rate particles");
    qav_malloc(&dev_dyn_rate, N_P, "allocate dynamics rates");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload dynamics-rate particles");
    dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle);
    qav_kernel_check("complete analytic dynamics-rate calculation");

    std::vector<real> dyn_rate(N_P);
    qav_copy_d2h(dyn_rate.data(), dev_dyn_rate, N_P, "copy dynamics rates");
    write_binary("state", state);
    write_binary("dyn_rate", dyn_rate);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"dynrate_3d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"x_min\": " << X_MIN << ",\n"
         << "  \"x_max\": " << X_MAX << ",\n"
         << "  \"y_min\": " << Y_MIN << ",\n"
         << "  \"y_max\": " << Y_MAX << ",\n"
         << "  \"z_min\": " << Z_MIN << ",\n"
         << "  \"z_max\": " << Z_MAX << ",\n"
         << "  \"cfl_dyn\": " << CFL_DYN << ",\n"
         << "  \"gms\": " << G*M_S << ",\n"
         << "  \"aspr_0\": " << ASPR_0 << ",\n"
         << "  \"idx_p\": " << IDX_P << ",\n"
         << "  \"idx_q\": " << IDX_Q << ",\n"
         << "  \"nu\": " << NU << ",\n"
         << "  \"schmidt_x\": " << SCHMIDT_X << ",\n"
         << "  \"schmidt_R\": " << SCHMIDT_R << ",\n"
         << "  \"schmidt_Z\": " << SCHMIDT_Z << ",\n"
         << "  \"viscous_flow\": true,\n"
         << "  \"density_diffusion\": true\n"
         << "}\n";

    qav_free(dev_particle, "free dynamics-rate particles");
    qav_free(dev_dyn_rate, "free dynamics rates");
    std::cout << "swarm dynamics-rate case completed with " << N_P << " particles" << std::endl;
    return 0;
}
