#include <cmath>          // std::log, std::pow, std::sqrt
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cerr, std::endl
#include <stdexcept>      // std::runtime_error
#include <string>         // std::string, std::to_string
#include <vector>         // std::vector

#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, cudaFree, kernel launches

#ifdef COLLISION
#include <_collision.cuh>
#endif // COLLISION
#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

#ifdef TEST_INITIAL_3D
#include <swarm_host.cuh>

std::mt19937 rand_generator;
#endif // TEST_INITIAL_3D

namespace
{

// =========================================================================================================================
// shared CUDA driver for isolated swarm verification models
//
// each model selects one compile-time branch, but the selected branch launches production kernels wherever an
// analytical reference permits; host construction replaces production initialization only when exact particle
// positions, random increments, imported gas, or crossed boundary states must be prescribed by the test
//
// raw arrays and metadata are written for an independent Python validator; this driver does not judge its own output
// =========================================================================================================================

const std::string output_path = PATH_OUT;

// fail immediately with the CUDA operation that produced the error
void cuda_check (cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return;
    std::cerr << operation << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

// expose asynchronous launch and execution failures at the tested operator boundary
void kernel_check (const char *kernel)
{
    cuda_check(cudaGetLastError(), kernel);
    cuda_check(cudaDeviceSynchronize(), kernel);
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
#elif defined(TEST_PARINIT_3D)
    return "parinit_3d";
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
void write_state (const std::vector<swarm> &particle)
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
    write_binary("state", state);

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
    cuda_check(cudaMemcpy(particle.data(), dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost), "copy state");
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
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_particle), sizeof(swarm)*N_P), "allocate particles");

#if defined(TEST_GRID_1D) || defined(TEST_GRID_2D) || defined(TEST_GRID_3D)
    // verify particle deposition, exact cell measure, radial optical-depth accumulation, and azimuthal averaging
    initialize_grid_particles(particle);
    cuda_check(cudaMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, cudaMemcpyHostToDevice), "upload particles");

    real *dev_dustdens = nullptr;
    real *dev_optdepth = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_dustdens), sizeof(real)*N_G), "allocate density");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(real)*N_G), "allocate optical depth");

    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);
    optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, 1.0);
    optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, 1.0);
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
    optdepth_csum <<< (N_X*N_Z)/TPB + 1, TPB >>> (dev_optdepth);
    optdepth_mean <<< (N_Y*N_Z)/TPB + 1, TPB >>> (dev_optdepth);
    kernel_check("grid diagnostics");

    std::vector<real> dustdens(N_G);
    std::vector<real> optdepth(N_G);
    cuda_check(cudaMemcpy(dustdens.data(), dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost), "copy density");
    cuda_check(cudaMemcpy(optdepth.data(), dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost), "copy optical depth");
    write_binary("dustdens", dustdens);
    write_binary("optdepth", optdepth);
    write_state(particle);
    cuda_check(cudaFree(dev_dustdens), "free density");
    cuda_check(cudaFree(dev_optdepth), "free optical depth");
    write_meta(0.0, 0.0);

#elif defined(TEST_ORBIT_1D) || defined(TEST_ORBIT_2D)
    // integrate one Keplerian period to measure staggered semi-analytic orbit error
    for (int idx = 0; idx < N_P; idx++)
    {
        real x = (N_X > 1) ? X_MIN + (idx + 0.5)*(X_MAX - X_MIN)/N_P : 0.5*(X_MIN + X_MAX);
        particle[idx].position = make_double3(x, 1.0, 0.5*M_PI);
        particle[idx].velocity = make_double3(1.0, 0.0, 0.0);
    }
    cuda_check(cudaMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, cudaMemcpyHostToDevice), "upload particles");
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
    for (int idx = 0; idx < N_P; idx++)
    {
        randposy[idx] = 0.7 + 0.6*static_cast<real>(idx) / static_cast<real>(N_P - 1);
    }

    real *dev_randposx = nullptr;
    real *dev_randposy = nullptr;
    real *dev_randposz = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposx), sizeof(real)*N_P), "allocate x positions");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposy), sizeof(real)*N_P), "allocate radial positions");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposz), sizeof(real)*N_P), "allocate z positions");
    cuda_check(cudaMemcpy(dev_randposx, randposx.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload x positions");
    cuda_check(cudaMemcpy(dev_randposy, randposy.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload radial positions");
    cuda_check(cudaMemcpy(dev_randposz, randposz.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload z positions");

    particle_init <<< NB_P, TPB >>> (dev_particle, dev_randposx, dev_randposy, dev_randposz);
    kernel_check("viscous-flow initialization");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(0.0, 0.0);
    cuda_check(cudaFree(dev_randposx), "free x positions");
    cuda_check(cudaFree(dev_randposy), "free radial positions");
    cuda_check(cudaFree(dev_randposz), "free z positions");

#elif defined(TEST_PARINIT_3D)
    // sample both sides of the midplane so one production launch exercises the complete resolved-vertical drift projection
    std::vector<real> randposx(N_P);
    std::vector<real> randposy(N_P);
    std::vector<real> randposz(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        real fraction = static_cast<real>(idx) / static_cast<real>(N_P - 1);
        real R = 0.7 + 0.6*fraction;
        real Z = 0.03*R*static_cast<real>(idx % 5 - 2);
        randposx[idx] = X_MIN + (static_cast<real>(idx) + 0.25)*(X_MAX - X_MIN)/static_cast<real>(N_P);
        randposy[idx] = sqrt(R*R + Z*Z);
        randposz[idx] = atan2(R, Z);
    }

    real *dev_randposx = nullptr;
    real *dev_randposy = nullptr;
    real *dev_randposz = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposx), sizeof(real)*N_P), "allocate x positions");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposy), sizeof(real)*N_P), "allocate radial positions");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_randposz), sizeof(real)*N_P), "allocate z positions");
    cuda_check(cudaMemcpy(dev_randposx, randposx.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload x positions");
    cuda_check(cudaMemcpy(dev_randposy, randposy.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload radial positions");
    cuda_check(cudaMemcpy(dev_randposz, randposz.data(), sizeof(real)*N_P, cudaMemcpyHostToDevice), "upload z positions");

    particle_init <<< NB_P, TPB >>> (dev_particle, dev_randposx, dev_randposy, dev_randposz);
    kernel_check("resolved-vertical particle initialization");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(0.0, 0.0);
    cuda_check(cudaFree(dev_randposx), "free x positions");
    cuda_check(cudaFree(dev_randposy), "free radial positions");
    cuda_check(cudaFree(dev_randposz), "free z positions");

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
    cuda_check(cudaMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, cudaMemcpyHostToDevice), "upload particles");
    real dt = 0.1;

#ifdef RADIATION
    real *dev_optdepth = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_optdepth), sizeof(real)*N_G), "allocate optical depth");
    cuda_check(cudaMemset(dev_optdepth, 0, sizeof(real)*N_G), "zero optical depth");
    ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt);
    ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, 1.0, dt);
    kernel_check("radiation drag response");
    cuda_check(cudaFree(dev_optdepth), "free optical depth");
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
    cuda_check(cudaMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, cudaMemcpyHostToDevice), "upload particles");
    curs *dev_rngstate = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_rngstate), sizeof(curs)*N_P), "allocate random states");
    rngstate_init <<< NB_P, TPB >>> (dev_rngstate, 17);
    real dt = 0.02;
    diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, dt);
    kernel_check("diffusion_pos");
    copy_state_from_device(particle, dev_particle);
    write_state(particle);
    write_meta(dt, dt);
    cuda_check(cudaFree(dev_rngstate), "free random states");

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
    cuda_check(cudaMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, cudaMemcpyHostToDevice), "upload particles");

    real size[2] = {1.0, 2.0};
    real number[2] = {5.0, 7.0};
    real *dev_size = nullptr;
    real *dev_number = nullptr;
    real *dev_result = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_size), 2*sizeof(real)), "allocate sizes");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_number), 2*sizeof(real)), "allocate numbers");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_result), 7*sizeof(real)), "allocate results");
    cuda_check(cudaMemcpy(dev_size, size, 2*sizeof(real), cudaMemcpyHostToDevice), "upload sizes");
    cuda_check(cudaMemcpy(dev_number, number, 2*sizeof(real), cudaMemcpyHostToDevice), "upload numbers");
    collision_math <<< 1, 1 >>> (dev_result, dev_particle, dev_size, dev_number);
    kernel_check("collision_math");
    std::vector<real> result(7);
    cuda_check(cudaMemcpy(result.data(), dev_result, 7*sizeof(real), cudaMemcpyDeviceToHost), "copy collision results");
    write_binary("collision", result);
    write_meta(0.0, 0.0);
    cuda_check(cudaFree(dev_size), "free sizes");
    cuda_check(cudaFree(dev_number), "free numbers");
    cuda_check(cudaFree(dev_result), "free results");

#elif defined(TEST_IMPORT_1D)
    // compare local closures against a prescribed external radial gas-density profile
    std::vector<real> gas_dens(N_G);
    for (int iy = 0; iy < N_Y; iy++) gas_dens[iy] = SIGMA_0*(1.0 + 0.1*static_cast<real>(iy));

    real *dev_gas_dens = nullptr;
    real *dev_result = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_gas_dens), sizeof(real)*N_G), "allocate gas density");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_result), 2*sizeof(real)*N_P), "allocate import results");
    cuda_check(cudaMemcpy(dev_gas_dens, gas_dens.data(), sizeof(real)*N_G, cudaMemcpyHostToDevice), "upload gas density");
    import_math <<< NB_P, TPB >>> (dev_result, dev_gas_dens);
    kernel_check("imported surface-density scaling");

    std::vector<real> result(2*N_P);
    cuda_check(cudaMemcpy(result.data(), dev_result, 2*sizeof(real)*N_P, cudaMemcpyDeviceToHost), "copy import results");
    write_binary("import", result);
    write_meta(0.0, 0.0);
    cuda_check(cudaFree(dev_gas_dens), "free gas density");
    cuda_check(cudaFree(dev_result), "free import results");

#elif defined(TEST_BOUNDARY_1D) || defined(TEST_BOUNDARY_2D) \
    || defined(TEST_BOUNDARY_3D) || defined(TEST_BOUNDARY_HALF)
    // record deliberately crossed states after each configured boundary policy
    real *dev_result = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev_result), 42*sizeof(real)),
        "allocate boundary results");
    boundary_math <<< 1, 1 >>> (dev_result);
    kernel_check("boundary_math");

    std::vector<real> result(42);
    cuda_check(cudaMemcpy(result.data(), dev_result, 42*sizeof(real), cudaMemcpyDeviceToHost),
        "copy boundary results");
    write_binary("boundary", result);
    write_meta(0.0, 0.0);
    cuda_check(cudaFree(dev_result), "free boundary results");
#endif

    cuda_check(cudaFree(dev_particle), "free particles");
    std::cout << "swarm verification case " << case_name() << " completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
