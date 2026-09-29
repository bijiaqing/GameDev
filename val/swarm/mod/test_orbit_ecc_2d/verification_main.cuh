// include fragment: the eccentric orbit test's main program, included once by its swarm_runtime.cu
#include <cmath>      // atan2, cos, sin, sqrt
#include <cstdlib>    // std::exit, EXIT_FAILURE
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <gpu.cuh>

#include <_transport.cuh>
#include <swarm_kern.cuh>

// test-only driver for non-circular, drag-free production transport
// four particles start at different orbital phases to exercise radial and azimuthal coupling, while the model-local
// transport specialization removes gas drag so the final state can be compared directly with Kepler's equation

namespace
{
const std::string output_path = PATH_OUT;

real eccentric_anomaly (real mean_anomaly, real eccentricity)
{
    // solve Kepler's equation only to construct initial conditions; the GPU evolution never calls this host reference
    // routine
    real anomaly = mean_anomaly;
    for (int iteration = 0; iteration < 20; iteration++)
    {
        real residual = anomaly - eccentricity*std::sin(anomaly) - mean_anomaly;
        anomaly -= residual / (1.0 - eccentricity*std::cos(anomaly));
    }
    return anomaly;
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open eccentric-orbit output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write eccentric-orbit output");
}
}

int main ()
{
    constexpr real semimajor = 1.0;
    constexpr real eccentricity = 0.2;
    constexpr real time_end = 1.3;
    const real mean_initial[N_P] = {0.1, 0.7, 1.4, 2.2};
    const real angular_momentum = std::sqrt(G*M_S*semimajor*(1.0 - eccentricity*eccentricity));

    std::vector<swarm> particle(N_P);
    std::vector<real> mean_anomaly(N_P);
    // store the production swarm representation: azimuth, spherical radius, polar angle, lx, vy, and lz
    for (int idx = 0; idx < N_P; idx++)
    {
        real E = eccentric_anomaly(mean_initial[idx], eccentricity);
        real x = std::atan2(std::sqrt(1.0 - eccentricity*eccentricity)*std::sin(E), std::cos(E) - eccentricity);
        real y = semimajor*(1.0 - eccentricity*std::cos(E));
        real vy = std::sqrt(G*M_S / semimajor)*eccentricity*std::sin(E) / (1.0 - eccentricity*std::cos(E));
        particle[idx].position = make_double3(x, y, 0.5*M_PI);
        particle[idx].velocity = make_double3(angular_momentum, vy, 0.0);
        mean_anomaly[idx] = mean_initial[idx];
    }

    swarm *dev_particle;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_particle),
        sizeof(*dev_particle)*(N_P)); status != gpuSuccess)
    {
        std::cerr << "allocate eccentric-orbit particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_particle, particle.data(), sizeof(*(dev_particle))*(N_P),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload eccentric-orbit particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    real dt = time_end / static_cast<real>(VERIFY_RES);
    // treat VERIFY_RES as a temporal refinement count rather than a particle or mesh resolution
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
    }
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "eccentric Kepler transport" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "eccentric Kepler transport" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(particle.data(), dev_particle, sizeof(*(particle.data()))*(N_P),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy eccentric-orbit particles" << ": " << gpuGetErrorString(status) << std::endl;
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

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"orbit_ecc_2d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"semimajor\": " << semimajor << ",\n"
         << "  \"eccentricity\": " << eccentricity << ",\n"
         << "  \"zero_drag_specialization\": " << (VAL_ZERO_DRAG_ACTIVE ? "true" : "false") << "\n"
         << "}\n";

    if (gpuError_t status = gpuFree(dev_particle); status != gpuSuccess)
    {
        std::cerr << "free eccentric-orbit particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "swarm eccentric orbit completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
