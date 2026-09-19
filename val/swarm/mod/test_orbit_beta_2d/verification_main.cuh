#include <cstdlib>
#include <cmath>      // atan2, cos, sin, sqrt
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <_transport.cuh>
#include <gpu.cuh>
#include <swarm_kern.cuh>

// test-only driver for an unattenuated reduced-gravity orbit
// zero optical depth and unit taper make the production radiation kernel evolve the exact potential -(1-beta)GM/y

namespace
{
const std::string output_path = PATH_OUT;

real eccentric_anomaly (real mean_anomaly, real eccentricity)
{
    // solve Kepler's equation only to construct the exact initial state
    real anomaly = mean_anomaly;
    for (int iteration = 0; iteration < 20; iteration++)
    {
        real residual = anomaly - eccentricity*sin(anomaly) - mean_anomaly;
        anomaly -= residual / (1.0 - eccentricity*cos(anomaly));
    }
    return anomaly;
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open reduced-gravity orbit output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write reduced-gravity orbit output");
}
}

int main ()
{
    constexpr real semimajor = 1.0;
    constexpr real eccentricity = 0.2;
    constexpr real time_end = 1.3;
    constexpr real beta_taper = 1.0;
    const real mean_initial[N_P] = {0.1, 0.7, 1.4, 2.2};
    const real size_initial[N_P] = {S_0, S_0, 2.0*S_0, 2.0*S_0};

    std::vector<swarm> particle(N_P);
    std::vector<real> mean_anomaly(N_P);
    // encode exact reduced-gravity Kepler states in the production spherical variables
    for (int idx = 0; idx < N_P; idx++)
    {
        real E = eccentric_anomaly(mean_initial[idx], eccentricity);
        real beta = BETA_0 / (size_initial[idx] / S_0);
        real mu_eff = (1.0 - beta)*G*M_S;
        real angular_momentum = sqrt(mu_eff*semimajor*(1.0 - eccentricity*eccentricity));
        real x = atan2(sqrt(1.0 - eccentricity*eccentricity)*sin(E), cos(E) - eccentricity);
        real y = semimajor*(1.0 - eccentricity*cos(E));
        real vy = sqrt(mu_eff/semimajor)*eccentricity*sin(E)/(1.0 - eccentricity*cos(E));
        particle[idx].position = make_double3(x, y, 0.5*M_PI);
        particle[idx].velocity = make_double3(angular_momentum, vy, 0.0);
        particle[idx].par_size = size_initial[idx];
        particle[idx].par_numr = 1.0;
        mean_anomaly[idx] = mean_initial[idx];
    }

    swarm *dev_particle;
    real *dev_optdepth;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_particle), sizeof(*dev_particle)*(N_P)); status != gpuSuccess)
    {
        std::cerr << "allocate reduced-gravity particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(*dev_optdepth)*(N_G)); status != gpuSuccess)
    {
        std::cerr << "allocate zero optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_particle, particle.data(), sizeof(*(dev_particle))*(N_P), gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload reduced-gravity particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::vector<real> optdepth(N_G, 0.0);
    if (gpuError_t status = gpuMemcpy(dev_optdepth, optdepth.data(), sizeof(*(dev_optdepth))*(N_G), gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload zero optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // refine the split transport timestep while preserving the same orbit and end time
    const real dt = time_end/static_cast<real>(VERIFY_RES);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt);
        ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, beta_taper, dt);
    }
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "reduced-gravity radiation transport" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "reduced-gravity radiation transport" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(particle.data(), dev_particle, sizeof(*(particle.data()))*(N_P), gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy reduced-gravity particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

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
    write_binary("mean_anomaly", mean_anomaly);
    write_binary("size", std::vector<real>(size_initial, size_initial + N_P));

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"orbit_beta_2d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"semimajor\": " << semimajor << ",\n"
         << "  \"eccentricity\": " << eccentricity << ",\n"
         << "  \"beta\": " << BETA_0 << ",\n"
         << "  \"size_ref\": " << S_0 << ",\n"
         << "  \"zero_drag_specialization\": " << (VAL_ZERO_DRAG_ACTIVE ? "true" : "false") << ",\n"
         << "  \"zero_optical_depth\": true,\n"
         << "  \"unit_radiation_taper\": true\n"
         << "}\n";

    if (gpuError_t status = gpuFree(dev_optdepth); status != gpuSuccess)
    {
        std::cerr << "free zero optical depth" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuFree(dev_particle); status != gpuSuccess)
    {
        std::cerr << "free reduced-gravity particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "swarm reduced-gravity orbit completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
