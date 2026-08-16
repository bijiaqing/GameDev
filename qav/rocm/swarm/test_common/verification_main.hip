#include <algorithm>      // std::fill
#include <cmath>          // std::log, std::pow, std::sqrt
#include <cstring>        // std::memcmp
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cerr, std::endl
#include <stdexcept>      // std::runtime_error
#include <string>         // std::string, std::to_string
#include <vector>         // std::vector

#include <hip/hip_runtime.h> // hipMalloc, hipMemcpy, hipFree, kernel launches

#ifdef COLLISION
#include <_collision.cuh>
#endif // COLLISION
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#if defined(TEST_INITIAL_3D) || defined(TEST_RESTART_2D)
#include <swarm_host.cuh>

std::mt19937 rand_generator;
#endif // TEST_INITIAL_3D || TEST_RESTART_2D

namespace
{

// shared native-HIP driver for isolated swarm formula, operator, and boundary tests
// each model selects one compile-time branch and writes raw arrays plus metadata for the independent Python validator

const std::string output_path = PATH_OUT;

// fail immediately with the HIP operation that produced the error
void hip_check (hipError_t status, const char *operation)
{
    if (status == hipSuccess) return;
    std::cerr << operation << ": " << hipGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

// expose asynchronous launch and execution failures at the tested operator boundary
void kernel_check (const char *kernel)
{
    hip_check(hipGetLastError(), kernel);
    hip_check(hipDeviceSynchronize(), kernel);
}

// keep every resolution artifact distinct inside one model output directory
std::string suffix ()
{
    return "_N" + std::to_string(VERIFY_RES) + ".dat";
}

// map the active compile-time test selector to its validator-facing label
const char *case_name ()
{
#if defined(TEST_GRID_1D)
    return "grid_1d";
#elif defined(TEST_GRID_2D)
    return "grid_2d";
#elif defined(TEST_GRID_3D)
    return "grid_3d";
#elif defined(TEST_ORBIT_1D)
    return "orbit_1d";
#elif defined(TEST_ORBIT_2D)
    return "orbit_2d";
#elif defined(TEST_DRAG_1D)
    return "drag_1d";
#elif defined(TEST_VISCFLOW_1D)
    return "viscflow_1d";
#elif defined(TEST_DRAG_2D)
    return "drag_2d";
#elif defined(TEST_DIFFUSION_1D)
    return "diffusion_1d";
#elif defined(TEST_DIFFUSION_2D)
    return "diffusion_2d";
#elif defined(TEST_DIFFUSION_3D)
    return "diffusion_3d";
#elif defined(TEST_INITIAL_3D)
    return "initial_3d";
#elif defined(TEST_RESTART_2D)
    return "restart_2d";
#elif defined(TEST_RADIATION_1D)
    return "radiation_1d";
#elif defined(TEST_RADIATION_2D)
    return "radiation_2d";
#elif defined(TEST_PRDRAG_1D)
    return "prdrag_1d";
#elif defined(TEST_PRDRAG_2D)
    return "prdrag_2d";
#elif defined(TEST_COLLISION_1D)
    return "collision_1d";
#elif defined(TEST_COLLISION_2D)
    return "collision_2d";
#elif defined(TEST_COLLISION_3D)
    return "collision_3d";
#elif defined(TEST_IMPORT_1D)
    return "import_1d";
#elif defined(TEST_BOUNDARY_1D)
    return "boundary_1d";
#elif defined(TEST_BOUNDARY_2D)
    return "boundary_2d";
#elif defined(TEST_BOUNDARY_3D)
    return "boundary_3d";
#elif defined(TEST_BOUNDARY_HALF)
    return "boundary_half";
#else
    return "unknown";
#endif
}

// write one contiguous float64 artifact consumed by the Python validator
void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::string path = output_path + name + suffix();
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open output file: " + path);
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    file.close();
    if (!file) throw std::runtime_error("cannot write output file: " + path);
}

// flatten particle phase space and optional species properties into stable test artifacts
void write_state_named (const std::string &name, const std::vector<swarm> &particle)
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

void write_state (const std::vector<swarm> &particle)
{
    write_state_named("state", particle);
}

// record the grid and integration parameters needed to reconstruct the analytical reference
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

// synchronize one completed device state back to the host representation
void copy_state_from_device (std::vector<swarm> &particle, const swarm *dev_particle)
{
    hip_check(hipMemcpy(particle.data(), dev_particle, sizeof(swarm)*N_P, hipMemcpyDeviceToHost), "copy state");
}

// locate the exact centroid of a logarithmic radial cell in the represented disk measure
real radial_centroid_offset ()
{
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real dimension = 2.0 + static_cast<real>(N_Z > 1);
    return log((dimension / (dimension + 1.0))*(pow(dy, dimension + 1.0) - 1.0)
        / (pow(dy, dimension) - 1.0)) / log(dy);
}

// place one equal-weight representative at every exact finite-volume centroid
void initialize_grid_particles (std::vector<swarm> &particle)
{
    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real dz = (N_Z > 1) ? (Z_MAX - Z_MIN) / static_cast<real>(N_Z) : 0.0;
    real radial_offset = radial_centroid_offset();

    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx = ix + iy*N_X + iz*N_X*N_Y;
                particle[idx].position.x = X_MIN + (static_cast<real>(ix) + 0.5)*dx;
                particle[idx].position.y = Y_MIN*pow(dy, static_cast<real>(iy) + radial_offset);
                particle[idx].position.z = (N_Z > 1)
                    ? Z_MIN + (static_cast<real>(iz) + 0.5)*dz : 0.5*M_PI;
                particle[idx].velocity = make_double3(0.0, 0.0, 0.0);
            }
        }
    }
}

#if defined(TEST_COLLISION_1D) || defined(TEST_COLLISION_2D) || defined(TEST_COLLISION_3D)
// evaluate collision volumes and kernels at fixed points for direct analytical comparison
__global__ void collision_math (real *result, const swarm *particle, const real *size, const real *number)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr real radius = 0.2;
    result[0] = _get_ball_measure(1.0, 0.5*M_PI, radius);
    result[1] = _get_ball_measure(Y_MIN + 0.25*radius, 0.5*M_PI, radius);
    result[2] = (N_X == 1 && N_Z == 1)
        ? _get_ball_measure(Y_MAX - 0.25*radius, 0.5*M_PI, radius)
        : (N_Z > 1) ? _get_ball_measure(1.0, Z_MIN + 0.25*radius, radius)
                    : _get_ball_measure(1.0, 0.5*M_PI, radius);
    result[3] = _get_col_rate_ij<CONSTANT_KERNEL>(particle, size, number, 0, 1, 0.3);
    result[4] = _get_col_rate_ij<LINEAR_KERNEL>(particle, size, number, 0, 1, 0.3);
    result[5] = _get_col_rate_ij<PRODUCT_KERNEL>(particle, size, number, 0, 1, 0.3);
    result[6] = (N_X == 1 && N_Z == 1)
        ? _get_ball_measure(1.0, 0.5*M_PI, 0.6) : 0.0;
}
#endif

#ifdef TEST_IMPORT_1D
// evaluate imported-gas Stokes and Reynolds closures at selected radial cells
__global__ void import_math (real *result, const real *gas_dens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    int iy = 2*idx;
    real dy = _get_dy();
    real mesh_dim = _get_mesh_dim();
    real radial_offset = log(
        (mesh_dim / (mesh_dim + 1.0))*(pow(dy, mesh_dim + 1.0) - 1.0)
        / (pow(dy, mesh_dim) - 1.0)
    ) / log(dy);
    real y = Y_MIN*pow(dy, static_cast<real>(iy) + radial_offset);
    real x = 0.5*(X_MIN + X_MAX);
    real z = 0.5*M_PI;
    real sigma_g = gas_dens[iy];
    real h_g = _get_hg(y);
    real stokes = _get_stokes(y, 0.0, h_g, S_0, x, y, z, gas_dens);
    real alpha = _get_alpha(y, h_g);

    result[idx] = stokes;
    result[N_P + idx] = _get_re_inv_sqrt(y, alpha, sigma_g);
}
#endif // TEST_IMPORT_1D

#if defined(TEST_BOUNDARY_1D) || defined(TEST_BOUNDARY_2D) \
    || defined(TEST_BOUNDARY_3D) || defined(TEST_BOUNDARY_HALF)
// pack one transformed boundary state into the validator's fixed six-field layout
__device__ __forceinline__
void write_boundary_state (real *result, int idx_case, real x, real y, real z, real lx, real vy, real lz)
{
    int idx_out = 6*idx_case;
    result[idx_out    ] = x;
    result[idx_out + 1] = y;
    result[idx_out + 2] = z;
    result[idx_out + 3] = lx;
    result[idx_out + 4] = vy;
    result[idx_out + 5] = lz;
}

// exercise diffusion reflection, transport outflow, periodic wrapping, and midplane reflection
__global__ void boundary_math (real *result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    real x_width = X_MAX - X_MIN;
    real y_width = Y_MAX - Y_MIN;
    real z_width = (N_Z > 1) ? Z_MAX - Z_MIN : 0.0;
    real z_mid = (N_Z > 1) ? 0.5*(Z_MIN + Z_MAX) : 0.5*M_PI;

    real x = X_MAX + 0.25*x_width;
    real y = Y_MIN - 0.125*y_width;
    real z = (N_Z > 1) ? Z_MIN - 0.125*z_width : 0.5*M_PI;
    _apply_diffusion_boundary(x, y, z);
    write_boundary_state(result, 0, x, y, z, 1.0, -2.0, 3.0);

    x = X_MIN - 0.25*x_width;
    y = Y_MAX + 0.125*y_width;
    z = (N_Z > 1) ? Z_MAX + 0.125*z_width : 0.5*M_PI;
    _apply_diffusion_boundary(x, y, z);
    write_boundary_state(result, 1, x, y, z, 1.0, -2.0, 3.0);

    x = 0.5*(X_MIN + X_MAX);
    y = Y_MIN - 2.25*y_width;
    z = z_mid;
    _apply_diffusion_boundary(x, y, z);
    write_boundary_state(result, 2, x, y, z, 1.0, -2.0, 3.0);

    x = X_MAX + 0.25*x_width;
    y = Y_MIN - 0.125*y_width;
    z = z_mid;
    real lx = 1.0;
    real vy = -2.0;
    real lz = 3.0;
    _apply_transport_boundary(x, y, z, lx, vy, lz);
    write_boundary_state(result, 3, x, y, z, lx, vy, lz);

    x = X_MIN - 0.25*x_width;
    y = Y_MAX + 0.125*y_width;
    z = z_mid;
    lx = 1.0;
    vy = 2.0;
    lz = 3.0;
    _apply_transport_boundary(x, y, z, lx, vy, lz);
    write_boundary_state(result, 4, x, y, z, lx, vy, lz);

    x = X_MAX + 0.25*x_width;
    y = 1.0;
    z = (N_Z > 1) ? Z_MIN - 0.125*z_width : 0.5*M_PI;
    lx = 1.0;
    vy = -2.0;
    lz = 3.0;
    _apply_transport_boundary(x, y, z, lx, vy, lz);
    write_boundary_state(result, 5, x, y, z, lx, vy, lz);

    x = X_MIN - 0.25*x_width;
    y = 1.0;
    z = (N_Z > 1) ? Z_MAX + 0.125*z_width : 0.5*M_PI;
    lx = 1.0;
    vy = 2.0;
    lz = 3.0;
    _apply_transport_boundary(x, y, z, lx, vy, lz);
    write_boundary_state(result, 6, x, y, z, lx, vy, lz);
}
#endif

}

int main ()
{
    std::vector<swarm> particle(N_P);
    swarm *dev_particle = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_particle), sizeof(swarm)*N_P), "allocate particles");

#if defined(TEST_GRID_1D) || defined(TEST_GRID_2D) || defined(TEST_GRID_3D)
    // verify particle deposition, exact cell measure, radial optical-depth accumulation, and azimuthal averaging
    initialize_grid_particles(particle);
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice), "upload particles");

    real *dev_dustdens = nullptr;
    real *dev_optdepth = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_dustdens), sizeof(real)*N_G), "allocate density");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(real)*N_G), "allocate optical depth");

    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);
    kernel_check("dustdens_init");
    optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
    kernel_check("optdepth_init");
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, 1.0);
    kernel_check("dustdens_depo");
    optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, 1.0);
    kernel_check("optdepth_depo");
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);
    kernel_check("dustdens_calc");
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
    kernel_check("optdepth_calc");
    optdepth_csum <<< (N_X*N_Z)/TPB + 1, TPB >>> (dev_optdepth);
    kernel_check("optdepth_csum");
    optdepth_mean <<< (N_Y*N_Z)/TPB + 1, TPB >>> (dev_optdepth);
    kernel_check("optdepth_mean");

    std::vector<real> dustdens(N_G);
    std::vector<real> optdepth(N_G);
    hip_check(hipMemcpy(dustdens.data(), dev_dustdens, sizeof(real)*N_G, hipMemcpyDeviceToHost), "copy density");
    hip_check(hipMemcpy(optdepth.data(), dev_optdepth, sizeof(real)*N_G, hipMemcpyDeviceToHost), "copy optical depth");
    write_binary("dustdens", dustdens);
    write_binary("optdepth", optdepth);
    write_state(particle);
    hip_check(hipFree(dev_dustdens), "free density");
    hip_check(hipFree(dev_optdepth), "free optical depth");
    write_meta(0.0, 0.0);

#elif defined(TEST_ORBIT_1D) || defined(TEST_ORBIT_2D)
    // integrate one Keplerian period to measure staggered semi-analytic orbit error
    for (int idx = 0; idx < N_P; idx++)
    {
        real x = (N_X > 1) ? X_MIN + (idx + 0.5)*(X_MAX - X_MIN)/N_P : 0.5*(X_MIN + X_MAX);
        particle[idx].position = make_double3(x, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(1.0, 0.0, 0.0);
    }
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice), "upload particles");
    real time_end = 2.0*M_PI;
    real dt = time_end / static_cast<real>(VERIFY_RES);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
    }
    kernel_check("ssa_transport");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(dt, time_end);

#elif defined(TEST_VISCFLOW_1D)
    // verify the steady particle velocity initialized against the viscous gas target
    std::vector<real> randposx(N_P, 0.0);
    std::vector<real> randposy(N_P);
    std::vector<real> randposz(N_P, 0.5*M_PI);
    std::vector<real> randsize(N_P, S_0);
    for (int idx = 0; idx < N_P; idx++)
    {
        randposy[idx] = 0.7 + 0.6*static_cast<real>(idx) / static_cast<real>(N_P - 1);
    }

    real *dev_randposx = nullptr;
    real *dev_randposy = nullptr;
    real *dev_randposz = nullptr;
    real *dev_randsize = nullptr;
    real *dev_mass_bank = nullptr;
    real domain_mass = 1.0;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_randposx), sizeof(real)*N_P), "allocate x positions");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_randposy), sizeof(real)*N_P), "allocate radial positions");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_randposz), sizeof(real)*N_P), "allocate z positions");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_randsize), sizeof(real)*N_P), "allocate grain sizes");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_mass_bank), sizeof(real)), "allocate domain mass");
    hip_check(hipMemcpy(dev_randposx, randposx.data(), sizeof(real)*N_P, hipMemcpyHostToDevice), "upload x positions");
    hip_check(hipMemcpy(dev_randposy, randposy.data(), sizeof(real)*N_P, hipMemcpyHostToDevice), "upload radial positions");
    hip_check(hipMemcpy(dev_randposz, randposz.data(), sizeof(real)*N_P, hipMemcpyHostToDevice), "upload z positions");
    hip_check(hipMemcpy(dev_randsize, randsize.data(), sizeof(real)*N_P, hipMemcpyHostToDevice), "upload grain sizes");
    hip_check(hipMemcpy(dev_mass_bank, &domain_mass, sizeof(real), hipMemcpyHostToDevice), "upload domain mass");

    particle_init <<< NB_P, TPB >>> (
        dev_particle, dev_randposx, dev_randposy, dev_randposz, dev_randsize, dev_mass_bank, 1, 1.0
    );
    kernel_check("viscous-flow initialization");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(0.0, 0.0);
    hip_check(hipFree(dev_randposx), "free x positions");
    hip_check(hipFree(dev_randposy), "free radial positions");
    hip_check(hipFree(dev_randposz), "free z positions");
    hip_check(hipFree(dev_randsize), "free grain sizes");
    hip_check(hipFree(dev_mass_bank), "free domain mass");

#elif defined(TEST_DRAG_1D) || defined(TEST_RADIATION_1D) || defined(TEST_PRDRAG_1D) \
    || defined(TEST_DRAG_2D) || defined(TEST_RADIATION_2D) || defined(TEST_PRDRAG_2D)
    // isolate drag, radiation pressure, and P-R damping over one fixed source step
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(0.0, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(1.2, 0.0, 0.0);
        particle[idx].par_size = 0.05*pow(2.0, static_cast<real>(idx));
        particle[idx].par_numr = 1.0;
    }
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice), "upload particles");
    real dt = 0.1;

#ifdef RADIATION
    real *dev_optdepth = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(real)*N_G), "allocate optical depth");
    hip_check(hipMemset(dev_optdepth, 0, sizeof(real)*N_G), "zero optical depth");
    ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt);
    ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, 1.0, dt);
    kernel_check("radiation drag response");
    hip_check(hipFree(dev_optdepth), "free optical depth");
#else
    ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
    kernel_check("drag response");
#endif
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(dt, dt);

#elif defined(TEST_DIFFUSION_1D) || defined(TEST_DIFFUSION_2D) || defined(TEST_DIFFUSION_3D)
    // apply one reproducible stochastic displacement for pathwise analytical reconstruction
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(0.0, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(0.7, 0.2, 0.0);
    }
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice), "upload particles");
    curs *dev_rngstate = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_rngstate), sizeof(curs)*N_P), "allocate random states");
    rngstate_init <<< NB_P, TPB >>> (dev_rngstate, 17);
    real dt = 0.02;
    diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    kernel_check("diffusion_pos");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(dt, dt);
    hip_check(hipFree(dev_rngstate), "free random states");

#elif defined(TEST_RESTART_2D)
    // compare uninterrupted evolution with save, deliberate clobber, reload, and continuation
    for (int idx = 0; idx < N_P; idx++)
    {
        real fraction = (static_cast<real>(idx) + 0.5) / static_cast<real>(N_P);
        particle[idx].position = make_double3(
            X_MIN + fraction*(X_MAX - X_MIN),
            0.75 + 0.5*fraction,
            0.5*M_PI
        );
        particle[idx].velocity = make_double3(
            particle[idx].position.y*(0.8 + 0.1*fraction),
            0.03 - 0.06*fraction,
            0.0
        );
    }

    swarm *dev_reference = nullptr;
    curs *dev_rngstate = nullptr;
    curs *dev_rngref = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_reference), sizeof(swarm)*N_P),
        "allocate reference particles");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_rngstate), sizeof(curs)*N_P),
        "allocate restart random states");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_rngref), sizeof(curs)*N_P),
        "allocate reference random states");
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice),
        "upload restart particles");
    hip_check(hipMemcpy(dev_reference, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice),
        "upload reference particles");
    rngstate_init <<< NB_P, TPB >>> (dev_rngstate, 29);
    rngstate_init <<< NB_P, TPB >>> (dev_rngref, 29);
    kernel_check("restart rngstate_init");

    real dt = 0.01;
    diffusion_pos <<< NB_P, TPB >>> (dev_reference, dev_rngref, dt);
    diffusion_pos <<< NB_P, TPB >>> (dev_reference, dev_rngref, dt);
    kernel_check("uninterrupted diffusion");

    diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    kernel_check("pre-checkpoint diffusion");
    if (!save_particle_data(output_path, 0, particle.data(), dev_particle, dev_rngstate))
        throw std::runtime_error("cannot save restart checkpoint");

    hip_check(hipMemset(dev_particle, 0, sizeof(swarm)*N_P), "clobber particles");
    hip_check(hipMemset(dev_rngstate, 0, sizeof(curs)*N_P), "clobber random states");
    std::fill(particle.begin(), particle.end(), swarm{});
    if (!load_particle_data(output_path, 0, particle.data(), dev_particle, dev_rngstate))
        throw std::runtime_error("cannot load restart checkpoint");

    std::vector<curs> rng_checkpoint(N_P);
    std::vector<curs> rng_loaded(N_P);
    std::string rng_path = output_path + "rngstate_" + frame_num(0) + ".dat";
    if (!load_host_binary(rng_path, rng_checkpoint.data(), N_P))
        throw std::runtime_error("cannot reread restart random-state checkpoint");
    hip_check(hipMemcpy(rng_loaded.data(), dev_rngstate, sizeof(curs)*N_P, hipMemcpyDeviceToHost),
        "copy reloaded random states");
    bool rng_equal = std::memcmp(
        rng_checkpoint.data(), rng_loaded.data(), sizeof(curs)*N_P
    ) == 0;

    diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    kernel_check("post-restart diffusion");

    std::vector<swarm> reference(N_P);
    std::vector<swarm> resumed(N_P);
    copy_state_from_device(reference, dev_reference);
    copy_state_from_device(resumed, dev_particle);
    write_state_named("reference", reference);
    write_state_named("resumed", resumed);
    write_binary("restart_status", {rng_equal ? 1.0 : 0.0});
    write_meta(dt, 2.0*dt);

    hip_check(hipFree(dev_reference), "free reference particles");
    hip_check(hipFree(dev_rngstate), "free restart random states");
    hip_check(hipFree(dev_rngref), "free reference random states");

#elif defined(TEST_INITIAL_3D)
    // validate continuous polydisperse containment, sampling CDFs, and represented-mass normalization
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

#elif defined(TEST_COLLISION_1D) || defined(TEST_COLLISION_2D) || defined(TEST_COLLISION_3D)
    // compare reduced-dimensional neighborhood measures and collision kernels with closed-form values
    particle[0].position = make_double3(0.0, 1.0, 0.5*M_PI);
    particle[1].position = make_double3((N_X > 1) ? 0.1 : 0.0, 1.0, 0.5*M_PI);
    particle[0].velocity = make_double3(1.0, 0.0, 0.0);
    particle[1].velocity = make_double3(1.0, 0.0, 0.0);
    particle[0].par_size = 1.0;
    particle[1].par_size = 2.0;
    particle[0].par_numr = 5.0;
    particle[1].par_numr = 7.0;
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice), "upload particles");

    real size[2] = {1.0, 2.0};
    real number[2] = {5.0, 7.0};
    real *dev_size = nullptr;
    real *dev_number = nullptr;
    real *dev_result = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_size), 2*sizeof(real)), "allocate sizes");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_number), 2*sizeof(real)), "allocate numbers");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_result), 7*sizeof(real)), "allocate results");
    hip_check(hipMemcpy(dev_size, size, 2*sizeof(real), hipMemcpyHostToDevice), "upload sizes");
    hip_check(hipMemcpy(dev_number, number, 2*sizeof(real), hipMemcpyHostToDevice), "upload numbers");
    collision_math <<< 1, 1 >>> (dev_result, dev_particle, dev_size, dev_number);
    kernel_check("collision_math");
    std::vector<real> result(7);
    hip_check(hipMemcpy(result.data(), dev_result, 7*sizeof(real), hipMemcpyDeviceToHost), "copy collision results");
    write_binary("collision", result);
    write_meta(0.0, 0.0);
    hip_check(hipFree(dev_size), "free sizes");
    hip_check(hipFree(dev_number), "free numbers");
    hip_check(hipFree(dev_result), "free results");

#elif defined(TEST_IMPORT_1D)
    // compare local closures against a prescribed external radial gas-density profile
    std::vector<real> gas_dens(N_G);
    for (int iy = 0; iy < N_Y; iy++) gas_dens[iy] = SIGMA_0*(1.0 + 0.1*static_cast<real>(iy));

    real *dev_gas_dens = nullptr;
    real *dev_result = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_gas_dens), sizeof(real)*N_G), "allocate gas density");
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_result), 2*sizeof(real)*N_P), "allocate import results");
    hip_check(hipMemcpy(dev_gas_dens, gas_dens.data(), sizeof(real)*N_G, hipMemcpyHostToDevice), "upload gas density");
    import_math <<< NB_P, TPB >>> (dev_result, dev_gas_dens);
    kernel_check("imported surface-density scaling");

    std::vector<real> result(2*N_P);
    hip_check(hipMemcpy(result.data(), dev_result, 2*sizeof(real)*N_P, hipMemcpyDeviceToHost), "copy import results");
    write_binary("import", result);
    write_meta(0.0, 0.0);
    hip_check(hipFree(dev_gas_dens), "free gas density");
    hip_check(hipFree(dev_result), "free import results");

#elif defined(TEST_BOUNDARY_1D) || defined(TEST_BOUNDARY_2D) \
    || defined(TEST_BOUNDARY_3D) || defined(TEST_BOUNDARY_HALF)
    // record deliberately crossed states after each configured boundary policy
    real *dev_result = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&dev_result), 42*sizeof(real)),
        "allocate boundary results");
    boundary_math <<< 1, 1 >>> (dev_result);
    kernel_check("boundary_math");

    std::vector<real> result(42);
    hip_check(hipMemcpy(result.data(), dev_result, 42*sizeof(real), hipMemcpyDeviceToHost),
        "copy boundary results");
    write_binary("boundary", result);
    write_meta(0.0, 0.0);
    hip_check(hipFree(dev_result), "free boundary results");
#endif

    hip_check(hipFree(dev_particle), "free particles");
    std::cout << "swarm verification case " << case_name() << " completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
