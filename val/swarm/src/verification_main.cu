#include <cmath>          // std::pow, std::sqrt
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cerr, std::cout, std::endl
#include <random>         // std::mt19937
#include <stdexcept>      // std::runtime_error
#include <string>         // std::string, std::to_string
#include <vector>         // std::vector

#include <gpu.cuh> // gpuMalloc, gpuMemcpy, gpuFree, kernel launches

#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef TEST_INITIAL_3D
#include <swarm_host.cuh>

std::mt19937 rand_generator;
#endif // TEST_INITIAL_3D

#if !defined(TEST_DIFFUSION_1D) && !defined(TEST_DIFFUSION_2D)  && !defined(TEST_DIFFUSION_3D) && !defined(TEST_DIFFUSION_WEDGE_2D)  && !defined(TEST_DIFFUSION_WEDGE_3D) && !defined(TEST_INITIAL_3D)  && !defined(TEST_PRDRAG_2D)
#error "shared swarm driver received an unsupported publication case"
#endif

namespace
{

// =========================================================================================================================
// shared GPU driver for retained swarm publication cases
//
// each branch launches the production operator wherever a closed-form or statistical reference exists; the host-only
// initialization branch records its continuous sampler and mass normalization for independent Python reconstruction
// =========================================================================================================================

const std::string output_path = PATH_OUT;

// fail at the GPU operation that produced the error
void gpu_check (gpuError_t status, const char *operation)
{
    if (status == gpuSuccess) return;
    std::cerr << operation << ": " << gpuGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

// expose asynchronous launch and execution failures at the tested operator boundary
void kernel_check (const char *kernel)
{
    gpu_check(gpuGetLastError(), kernel);
    gpu_check(gpuDeviceSynchronize(), kernel);
}

std::string suffix ()
{
    return "_N" + std::to_string(VERIFY_RES) + ".dat";
}

const char *case_name ()
{
#if defined(TEST_DIFFUSION_1D)
    return "diffusion_1d";
#elif defined(TEST_DIFFUSION_2D)
    return "diffusion_2d";
#elif defined(TEST_DIFFUSION_3D)
    return "diffusion_3d";
#elif defined(TEST_DIFFUSION_WEDGE_2D)
    return "diffusion_wedge_2d";
#elif defined(TEST_DIFFUSION_WEDGE_3D)
    return "diffusion_wedge_3d";
#elif defined(TEST_INITIAL_3D)
    return "initial_3d";
#else
    return "prdrag_2d";
#endif
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::string path = output_path + name + suffix();
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open output file: " + path);
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    file.close();
    if (!file) throw std::runtime_error("cannot write output file: " + path);
}

void write_state (const std::vector<swarm> &particle, const std::string &name = "state")
{
    std::vector<real> state(6*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx]         = particle[idx].position.x;
        state[N_P + idx]   = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
    }
    write_binary(name, state);

#ifdef MULTISIZE
    std::vector<real> species(2*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        species[idx] = particle[idx].par_size;
        species[N_P + idx] = particle[idx].par_numr;
    }
    write_binary("species", species);
#endif
}

void write_meta (real dt, real time)
{
    std::string path = output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json";
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot open metadata file: " + path);
    file << std::setprecision(17);
    file << "{\n";
    file << "  \"case\": \"" << case_name() << "\",\n";
    file << "  \"resolution\": " << VERIFY_RES << ",\n";
    file << "  \"np\": " << N_P << ",\n";
    file << "  \"nx\": " << N_X << ",\n";
    file << "  \"ny\": " << N_Y << ",\n";
    file << "  \"nz\": " << N_Z << ",\n";
    file << "  \"x_min\": " << X_MIN << ",\n";
    file << "  \"x_max\": " << X_MAX << ",\n";
    file << "  \"y_min\": " << Y_MIN << ",\n";
    file << "  \"y_max\": " << Y_MAX << ",\n";
    file << "  \"z_min\": " << Z_MIN << ",\n";
    file << "  \"z_max\": " << Z_MAX << ",\n";
    file << "  \"dt\": " << dt << ",\n";
    file << "  \"time\": " << time << "\n";
    file << "}\n";
    file.close();
    if (!file) throw std::runtime_error("cannot write metadata file: " + path);
}

void copy_state_from_device (std::vector<swarm> &particle, const swarm *dev_particle)
{
    gpu_check(gpuMemcpy(particle.data(), dev_particle, sizeof(swarm)*N_P, gpuMemcpyDeviceToHost), "copy state");
}

}

int main ()
{
#ifdef TEST_INITIAL_3D
    // sample the exact finite-domain distribution and verify mass normalization without a grid-dependent vertical histogram
    std::vector<real> mass_bank;
    initmass_calc(mass_bank);
    real total_dust_mass = get_total_dust_mass(mass_bank);

    std::vector<real> randposx(N_P);
    std::vector<real> randposy(N_P);
    std::vector<real> randposz(N_P);
    std::vector<real> randsize(N_P);
    real size_mid = std::sqrt(INIT_SMIN*INIT_SMAX);
    for (int idx = 0; idx < N_P; idx++)
    {
        if (idx < N_P / 3)
        {
            randsize[idx] = INIT_SMIN;
        }
        else if (idx < 2*N_P / 3)
        {
            randsize[idx] = size_mid;
        }
        else
        {
            randsize[idx] = INIT_SMAX;
        }
    }

    rand_generator.seed(0);
    rand_disk_poly(randposx.data(), randposy.data(), randposz.data(), randsize.data(), N_P);
    real mass_norm = get_mass_norm(randsize.data(), mass_bank, total_dust_mass);
    long double represented_mass = 0.0;
    for (int idx = 0; idx < N_P; idx++)
    {
        real domain_mass = _get_domain_mass(
            randsize[idx], mass_bank.data(), static_cast<int>(mass_bank.size())
        );
        represented_mass += static_cast<long double>(mass_norm)
                          * static_cast<long double>(_get_mass_weight(randsize[idx]))
                          * static_cast<long double>(domain_mass)
                          / static_cast<long double>(N_P);
    }

    std::vector<real> initial(4*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        initial[idx]         = randposx[idx];
        initial[N_P + idx]   = randposy[idx];
        initial[2*N_P + idx] = randposz[idx];
        initial[3*N_P + idx] = randsize[idx];
    }
    std::vector<real> mass_summary = {
        total_dust_mass, static_cast<real>(represented_mass), mass_norm,
    };
    write_binary("initial", initial);
    write_binary("mass_bank", mass_bank);
    write_binary("mass_summary", mass_summary);
    write_meta(0.0, 0.0);
#else
    std::vector<swarm> particle(N_P);
    swarm *dev_particle = nullptr;
    gpu_check(gpuMalloc(reinterpret_cast<void **>(&dev_particle), sizeof(swarm)*N_P), "allocate particles");

#ifdef TEST_PRDRAG_2D
    // isolate one production radiation substep so P-R damping can be compared with its exact constant-coefficient update
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(0.0, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(1.2, 0.0, 0.0);
        particle[idx].par_size = 0.05*std::pow(2.0, static_cast<real>(idx));
        particle[idx].par_numr = 1.0;
    }
    gpu_check(gpuMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, gpuMemcpyHostToDevice), "upload particles");
    real dt = 0.1;
    real *dev_optdepth = nullptr;
    gpu_check(gpuMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(real)*N_G), "allocate optical depth");
    gpu_check(gpuMemset(dev_optdepth, 0, sizeof(real)*N_G), "zero optical depth");
    ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt);
    ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, 1.0, dt);
    kernel_check("P-R drag response");
    gpu_check(gpuFree(dev_optdepth), "free optical depth");
#else
    // apply one reproducible production SDE displacement for pathwise analytical reconstruction
    for (int idx = 0; idx < N_P; idx++)
    {
#if defined(TEST_DIFFUSION_WEDGE_2D) || defined(TEST_DIFFUSION_WEDGE_3D)
        real seam_offset = 0.005;
        real x = (idx % 2 == 0) ? X_MIN + seam_offset : X_MAX - seam_offset;
#ifdef TEST_DIFFUSION_WEDGE_3D
        // stay one gas scale height off the midplane so Stokes-dependent diffusion can cross the seam
        real z = 0.5*M_PI - std::atan(ASPR_0);
        real lz = 0.12;
#else
        real z = 0.5*M_PI;
        real lz = 0.0;
#endif
        particle[idx].position = make_double3(x, 1.0, z);
        particle[idx].velocity = make_double3(0.7, 0.2, lz);
#else
        particle[idx].position = make_double3(0.0, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(0.7, 0.2, 0.0);
#endif
    }
#if defined(TEST_DIFFUSION_WEDGE_2D) || defined(TEST_DIFFUSION_WEDGE_3D)
    write_state(particle, "state_initial");
#endif // TEST_DIFFUSION_WEDGE_2D || TEST_DIFFUSION_WEDGE_3D
    gpu_check(gpuMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, gpuMemcpyHostToDevice), "upload particles");
    curs *dev_rngstate = nullptr;
    gpu_check(gpuMalloc(reinterpret_cast<void **>(&dev_rngstate), sizeof(curs)*N_P), "allocate random states");
    rngstate_init <<< NB_P, TPB >>> (dev_rngstate, 17);
#if defined(TEST_DIFFUSION_WEDGE_2D) || defined(TEST_DIFFUSION_WEDGE_3D)
    real dt = 0.002;
#else
    real dt = 0.02;
#endif
    diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    kernel_check("diffusion_pos");
    gpu_check(gpuFree(dev_rngstate), "free random states");
#endif

    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(dt, dt);
    gpu_check(gpuFree(dev_particle), "free particles");
#endif

    std::cout << "swarm verification case " << case_name() << " completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
