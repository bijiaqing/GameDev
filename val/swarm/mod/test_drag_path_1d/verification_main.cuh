// include fragment: the drag-path test's main program, included once by its swarm_runtime.cu
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

// test-only driver for multi-step drag trajectories
// three controlled stopping times span stiff, intermediate, and weak relaxation; the model-local transport
// specialization supplies constant coefficients so both velocity and integrated displacement have closed analytical
// solutions

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open drag-path output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write drag-path output");
}
}

int main ()
{
    constexpr real time_end = 1.0;
    constexpr real y_initial = 0.8;
    constexpr real vy_initial = 0.25;
    const real stopping_time[N_P] = {0.02, 0.2, 2.0};
    std::vector<swarm> particle(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(0.0, y_initial, 0.5*M_PI);
        particle[idx].velocity = make_double3(0.0, vy_initial, 0.0);
        // par_size is deliberately repurposed as the exact stopping time by the test-local _transport.cuh
        particle[idx].par_size = stopping_time[idx];
        particle[idx].par_numr = 1.0;
    }

    swarm *dev_particle;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&dev_particle),
        sizeof(*dev_particle)*(N_P)); status != gpuSuccess)
    {
        std::cerr << "allocate drag-path particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(dev_particle, particle.data(), sizeof(*(dev_particle))*(N_P),
        gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "upload drag-path particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    real dt = time_end / static_cast<real>(VERIFY_RES);
    // refine only the integration step while keeping the same physical end time and particle ensemble
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
    }
    if (gpuError_t status = gpuGetLastError(); status != gpuSuccess)
    {
        std::cerr << "constant drag trajectory" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuDeviceSynchronize(); status != gpuSuccess)
    {
        std::cerr << "constant drag trajectory" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(particle.data(), dev_particle, sizeof(*(particle.data()))*(N_P),
        gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "copy drag-path particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real> state(6*N_P), stopping(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx] = particle[idx].position.x;
        state[N_P + idx] = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
        stopping[idx] = particle[idx].par_size;
    }
    write_binary("state", state);
    write_binary("stopping_time", stopping);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"drag_path_1d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"y_initial\": " << y_initial << ",\n"
         << "  \"vy_initial\": " << vy_initial << ",\n"
         << "  \"gas_velocity\": " << VAL_DRAG_GAS_VY << ",\n"
         << "  \"force\": " << VAL_DRAG_FORCE_Y << ",\n"
         << "  \"constant_drag_specialization\": " << (VAL_CONSTANT_DRAG_ACTIVE ? "true" : "false") << "\n"
         << "}\n";

    if (gpuError_t status = gpuFree(dev_particle); status != gpuSuccess)
    {
        std::cerr << "free drag-path particles" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "swarm constant drag path completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
