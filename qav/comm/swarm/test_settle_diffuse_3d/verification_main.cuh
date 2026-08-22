#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <device_api.cuh>
#include <swarm_kern.cuh>

// test-only driver for linear settling combined with production stochastic diffusion
// the deterministic map supplies vZ=-gamma Z, while diffusion_pos supplies the vertical random increment and performs the
// production coordinate and velocity-basis transformations

namespace
{
const std::string output_path = PATH_OUT;
constexpr real gamma_settle = 0.7;
constexpr real height_initial = 0.25;
constexpr real time_end = 1.0;
constexpr int random_seed = 2718;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open settling-diffusion output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write settling-diffusion output");
}
}

// apply the Euler drift of the documented discrete Ornstein-Uhlenbeck recurrence
// preserve the physical cylindrical velocity while the local spherical basis changes with position
__global__
void settling_step (swarm *dev_particle, real gamma, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    real R = y*sin(z);
    real Z = y*cos(z);

    real lx = dev_particle[idx].velocity.x;
    real vy = dev_particle[idx].velocity.y;
    real lz = dev_particle[idx].velocity.z;
    real vx = lx / R;
    real vz = lz / y;
    real vR = vy*sin(z) + vz*cos(z);
    real vZ = vy*cos(z) - vz*sin(z);

    real Z_new = (1.0 - gamma*dt)*Z;
    real y_new = sqrt(R*R + Z_new*Z_new);
    real z_new = atan2(R, Z_new);
    real sinz_new = R / y_new;
    real cosz_new = Z_new / y_new;

    dev_particle[idx].position.y = y_new;
    dev_particle[idx].position.z = z_new;
    dev_particle[idx].velocity.x = R*vx;
    dev_particle[idx].velocity.y = vR*sinz_new + vZ*cosz_new;
    dev_particle[idx].velocity.z = (vR*cosz_new - vZ*sinz_new)*y_new;
}

int main ()
{
    constexpr real radius_initial = 1.0;
    constexpr real velocity_R = 0.17;
    constexpr real velocity_x = 0.31;
    constexpr real velocity_Z = -0.23;
    real y_initial = sqrt(radius_initial*radius_initial + height_initial*height_initial);
    real z_initial = atan2(radius_initial, height_initial);

    std::vector<swarm> particle(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        real x = X_MIN + (static_cast<real>(idx) + 0.5)*(X_MAX - X_MIN) / static_cast<real>(N_P);
        real vy = velocity_R*sin(z_initial) + velocity_Z*cos(z_initial);
        real lz = (velocity_R*cos(z_initial) - velocity_Z*sin(z_initial))*y_initial;
        particle[idx].position = make_double3(x, y_initial, z_initial);
        particle[idx].velocity = make_double3(radius_initial*velocity_x, vy, lz);
    }

    swarm *dev_particle;
    curs *dev_rngstate;
    qav_malloc(&dev_particle, N_P, "allocate settling-diffusion particles");
    qav_malloc(&dev_rngstate, N_P, "allocate settling-diffusion random states");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload settling-diffusion particles");
    rngstate_init <<< NB_P, TPB >>> (dev_rngstate, random_seed);
    qav_kernel_check("initialize settling-diffusion random states");

    real dt = time_end / static_cast<real>(VERIFY_RES);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        settling_step <<< NB_P, TPB >>> (dev_particle, gamma_settle, dt);
        diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    }
    qav_kernel_check("settling-diffusion recurrence");
    qav_copy_d2h(particle.data(), dev_particle, N_P, "copy settling-diffusion particles");

    std::vector<real> state(6*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx] = particle[idx].position.x;
        state[N_P + idx] = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
    }
    write_binary("state", state);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"settle_diffuse_3d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"steps\": " << VERIFY_RES << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"gamma\": " << gamma_settle << ",\n"
         << "  \"diffusivity\": " << NU << ",\n"
         << "  \"height_initial\": " << height_initial << ",\n"
         << "  \"radius_initial\": " << radius_initial << ",\n"
         << "  \"velocity_R\": " << velocity_R << ",\n"
         << "  \"velocity_x\": " << velocity_x << ",\n"
         << "  \"velocity_Z\": " << velocity_Z << ",\n"
         << "  \"random_seed\": " << random_seed << ",\n"
         << "  \"local_linear_settling\": true,\n"
         << "  \"production_vertical_diffusion\": true,\n"
         << "  \"radial_diffusion_suppressed\": true,\n"
         << "  \"azimuthal_diffusion_suppressed\": true\n"
         << "}\n";

    qav_free(dev_rngstate, "free settling-diffusion random states");
    qav_free(dev_particle, "free settling-diffusion particles");
    std::cout << "swarm settling-diffusion completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
