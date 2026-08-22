#include <cmath>          // cos, exp, fmin, pow, sin, sqrt, std::cyl_bessel_j, std::cyl_neumann
#include <cstddef>        // std::size_t
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <filesystem>     // std::filesystem::create_directories
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cerr, std::cout, std::endl
#include <string>         // std::string, std::to_string
#include <vector>         // std::vector

#include <hip/hip_runtime.h> // hipDeviceSynchronize, hipFree, hipMalloc, hipMemcpy

#include <fluid_kern.cuh>
#include <fluid_host.cuh>

// =========================================================================================================================
// shared HIP driver for the fluid verification models
//
// each model directory defines one VERIFY_* macro in flags.mk and includes this file as its main program; the preprocessor
// keeps only the selected initial condition, integration branch, and case name in the resulting executable; this lets every
// test call the production kernels while sharing identical allocation, output, and metadata logic
//
// the executable does not calculate error norms; it writes initial and final fields plus metadata for validate_case.py,
// which independently constructs analytical finite-volume reference solutions and performs the comparison
// =========================================================================================================================

namespace
{
const std::string PATH = PATH_OUT;

// integrate one scalar function with an eight-point Gauss-Legendre rule; the tables contain only the four positive nodes;
// evaluating each node at symmetric offsets supplies all eight quadrature points
template <typename Function>
real gauss8 (Function function, real lower, real upper)
{
    static constexpr real node[4] = {
        0.18343464249564980494, 0.52553240991632898582,
        0.79666647741362673959, 0.96028985649753623168
    };
    static constexpr real weight[4] = {
        0.36268378337836198297, 0.31370664587788728734,
        0.22238103445337447054, 0.10122853629037625915
    };

    real midpoint = 0.5*(lower + upper);
    real radius = 0.5*(upper - lower);
    real sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        real offset = radius*node[i];
        sum += weight[i]*(function(midpoint - offset) + function(midpoint + offset));
    }
    return radius*sum;
}

real compact_bump (real value, real lower, real upper)
{
    // this C-infinity profile is exactly zero outside its support and smooth at both endpoints, making it suitable for
    // transport convergence tests without discontinuities that would reduce the formal PPM order
    if (value <= lower || value >= upper) return 0.0;
    real center = 0.5*(lower + upper);
    real half_width = 0.5*(upper - lower);
    real u = (value - center) / half_width;
    return exp(1.0 - 1.0/(1.0 - u*u));
}

real spherical_j0 (real value)
{
    return sin(value) / value;
}

real spherical_y0 (real value)
{
    return -cos(value) / value;
}

real spherical_j0_deriv (real value)
{
    return (value*cos(value) - sin(value)) / (value*value);
}

real spherical_y0_deriv (real value)
{
    return (value*sin(value) + cos(value)) / (value*value);
}

real radial_mode (real y, int dimension)
{
    // construct a radial Laplacian eigenmode with zero derivative at Y_MIN; the selected k2 and k3 values also impose zero
    // derivative at Y_MAX, so the analytical mode is compatible with the diffusion kernel's zero-flux radial boundaries
    constexpr real k2 = 1.694299217770420;
    constexpr real k3 = 1.874562003084784;

    if (dimension == 2)
    {
        // cylindrical radial eigenfunctions are linear combinations of ordinary order-zero Bessel functions
        real value = std::cyl_bessel_j(0, k2*y)*std::cyl_neumann(1, k2*Y_MIN)
                   - std::cyl_neumann(0, k2*y)*std::cyl_bessel_j(1, k2*Y_MIN);
        real norm = std::cyl_bessel_j(0, k2*Y_MIN)*std::cyl_neumann(1, k2*Y_MIN)
                  - std::cyl_neumann(0, k2*Y_MIN)*std::cyl_bessel_j(1, k2*Y_MIN);
        return value / norm;
    }

    // spherical radial eigenfunctions use spherical Bessel functions instead
    real value = spherical_j0(k3*y)*spherical_y0_deriv(k3*Y_MIN)
               - spherical_y0(k3*y)*spherical_j0_deriv(k3*Y_MIN);
    real norm = spherical_j0(k3*Y_MIN)*spherical_y0_deriv(k3*Y_MIN)
              - spherical_y0(k3*Y_MIN)*spherical_j0_deriv(k3*Y_MIN);
    return value / norm;
}

real host_gas_dens (real R, real Z)
{
    // reproduce the gas profile used as the radial baseline of the ring density
    real sigma_g = SIGMA_0*pow(R/R_0, IDX_P);
    if (N_Z == 1) return sigma_g;

    real h_g = ASPR_0*pow(R/R_0, 0.5*(IDX_Q + 1.0));
    real rhog_mid = sigma_g / (sqrt(2.0*M_PI)*h_g*R);
    return rhog_mid*exp((R/sqrt(R*R + Z*Z) - 1.0)/(h_g*h_g));
}

const char *case_name ()
{
    // store a stable case identifier in metadata; the Python validator dispatches its analytical solution using this name
#if defined(VERIFY_X_TRANSPORT)
    return "x_transport";
#elif defined(VERIFY_Y_TRANSPORT_CYL)
    return "y_transport_cyl";
#elif defined(VERIFY_Y_TRANSPORT_SPH)
    return "y_transport_sph";
#elif defined(VERIFY_Y_OUTFLOW_2D)
    return "y_outflow_2d";
#elif defined(VERIFY_Y_OUTFLOW_3D)
    return "y_outflow_3d";
#elif defined(VERIFY_Z_TRANSPORT)
    return "z_transport";
#elif defined(VERIFY_X_DIFFUSION)
    return "x_diffusion";
#elif defined(VERIFY_Y_DIFFUSION_CYL)
    return "y_diffusion_cyl";
#elif defined(VERIFY_Y_DIFFUSION_SPH)
    return "y_diffusion_sph";
#elif defined(VERIFY_Z_DIFFUSION)
    return "z_diffusion";
#elif defined(VERIFY_SOURCE_DRAG)
    return "source_drag";
#elif defined(VERIFY_OPTDEPTH)
    return "optdepth";
#elif defined(VERIFY_RING_DIFFUSION) && defined(VERIFY_RING_RADIATION)
    return "ring_all_2d";
#elif defined(VERIFY_RING_DIFFUSION)
    return "ring_diffusion_2d";
#elif defined(VERIFY_RING_RADIATION)
    return "ring_radiation_2d";
#elif defined(VERIFY_RING)
    return "ring_transport_2d";
#else
    return "unknown";
#endif
}

std::string suffix ()
{
    // resolution belongs in every binary filename so results from several grid sizes can share one output directory
    return "_N" + std::to_string(VERIFY_RES) + ".dat";
}

// fill host-side cell averages of density and the three conserved momentum variables; initializing conserved quantities
// directly prevents a primitive-to-conserved conversion from becoming part of the test error
void initialize_state (std::vector<real> &rhod, std::vector<real> &mx,
    std::vector<real> &my, std::vector<real> &mz)
{
    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);

    // traverse cells in the same x-fastest ordering used by all production kernels and binary output files
    for (int iz = 0; iz < N_Z; iz++)
    {
        real z0 = Z_MIN + static_cast<real>(iz)*dz;
        real z1 = z0 + dz;
        real zc = z0 + 0.5*dz;

        for (int iy = 0; iy < N_Y; iy++)
        {
            real y0 = Y_MIN*pow(dy, static_cast<real>(iy));
            real y1 = y0*dy;
            real yc = sqrt(y0*y1);
            real Rc = yc*sin(zc);
            real Zc = yc*cos(zc);

            for (int ix = 0; ix < N_X; ix++)
            {
                real x0 = X_MIN + static_cast<real>(ix)*dx;
                real x1 = x0 + dx;
                int idx = ix + iy*N_X + iz*N_X*N_Y;

#if defined(VERIFY_X_TRANSPORT)
                // cell-average a periodic Fourier mode and assign specific angular momentum R^2, giving unit angular speed
                real q = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
                real velx = Rc*Rc;
                rhod[idx] = q;
                mx[idx] = q*velx;
                my[idx] = 0.0;
                mz[idx] = 0.0;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH)
                // integrate the compact bump and its radial momentum with the appropriate cylindrical or spherical volume
                // measure; dividing by volume produces the finite-volume cell averages consumed by the advection kernel
                int dimension = (N_Z > 1) ? 3 : 2;
                real volume = (pow(y1, dimension) - pow(y0, dimension))/static_cast<real>(dimension);
                real rho_int = gauss8([&](real y)
                {
                    return compact_bump(y, 1.0, 1.8)*pow(y, dimension - 1);
                }, y0, y1);
                real my_int = gauss8([&](real y)
                {
                    return compact_bump(y, 1.0, 1.8)*VERIFY_A*y*pow(y, dimension - 1);
                }, y0, y1);
                rhod[idx] = rho_int/volume;
                mx[idx] = 0.7*rhod[idx];
                my[idx] = my_int/volume;
                mz[idx] = 0.11*rhod[idx];
#elif defined(VERIFY_Y_OUTFLOW_2D) || defined(VERIFY_Y_OUTFLOW_3D)
                // initialize exact finite-volume averages of a smooth pulse touching the outer-boundary region
                // constant radial velocity then leaves the interior state and escaped mass analytically integrable
                int dimension = (N_Z > 1) ? 3 : 2;
                real volume = (pow(y1, dimension) - pow(y0, dimension))/static_cast<real>(dimension);
                real rho_int = gauss8([&](real y)
                {
                    return compact_bump(y, 1.7, 2.5)*pow(y, dimension - 1);
                }, y0, y1);
                rhod[idx] = rho_int/volume;
                mx[idx] = 0.0;
                my[idx] = VERIFY_A*rhod[idx];
                mz[idx] = 0.0;
#elif defined(VERIFY_Z_TRANSPORT)
                // average the compact bump with the polar finite-volume measure and choose the shell momentum so the exact
                // integrated polar face-area-to-volume factor gives the same known angular rate in every radial shell
                real volume = cos(z0) - cos(z1);
                real rho_int = gauss8([&](real z)
                {
                    return compact_bump(z, 0.80, 1.30);
                }, z0, z1);
                real area_z = 0.5*(y1*y1 - y0*y0);
                real vol_y = (y1*y1*y1 - y0*y0*y0) / 3.0;
                real lz = VERIFY_RATE_Z*yc*vol_y / area_z;
                rhod[idx] = rho_int/volume;
                mx[idx] = 0.7*rhod[idx];
                my[idx] = 0.05*rhod[idx];
                mz[idx] = lz*rhod[idx];
#elif defined(VERIFY_X_DIFFUSION)
                // a periodic Fourier mode is an exact azimuthal diffusion eigenfunction; constant primitive momentum ratios
                // also test whether diffusing mass transports all three conserved momentum components consistently
                real rhod_init = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
                rhod[idx] = rhod_init;
                mx[idx] = 0.7*rhod_init;
                my[idx] = -0.15*rhod_init;
                mz[idx] = 0.11*rhod_init;
#elif defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH)
                // average the zero-flux radial eigenmode with the correct dimension-dependent volume measure
                int dimension = (N_Z > 1) ? 3 : 2;
                real volume = (pow(y1, dimension) - pow(y0, dimension))/static_cast<real>(dimension);
                real mode_avg = gauss8([&](real y)
                {
                    return radial_mode(y, dimension)*pow(y, dimension - 1);
                }, y0, y1) / volume;
                real rhod_init = VERIFY_Q0 + VERIFY_EPS*mode_avg;
                rhod[idx] = rhod_init;
                mx[idx] = 0.7*rhod_init;
                my[idx] = -0.15*rhod_init;
                mz[idx] = 0.11*rhod_init;
#elif defined(VERIFY_Z_DIFFUSION)
                // p2(cos z) is an angular Laplacian eigenmode; multiplication by sin(z) supplies the spherical volume
                // measure before division by the exact polar cell volume
                real volume = cos(z0) - cos(z1);
                real mode_avg = gauss8([&](real z)
                {
                    real mu = cos(z);
                    return 0.5*(3.0*mu*mu - 1.0)*sin(z);
                }, z0, z1) / volume;
                real rhod_init = VERIFY_Q0 + VERIFY_EPS*mode_avg;
                rhod[idx] = rhod_init;
                mx[idx] = 0.7*rhod_init;
                my[idx] = -0.15*rhod_init;
                mz[idx] = 0.11*rhod_init;
#elif defined(VERIFY_SOURCE_DRAG)
                // eight x cells encode eight drag stiffnesses through the test parameter helpers; spatial variation is not
                // involved; one source update can then be checked from the non-stiff through the very stiff limit
                rhod[idx] = 1.0;
                mx[idx] = 1.3;
                my[idx] = -0.8;
                mz[idx] = 0.6;
#elif defined(VERIFY_OPTDEPTH)
                // choose surface density so the reconstructed midplane volume density is the requested radial power law
                real h_g = ASPR_0*pow(Rc/R_0, 0.5*(IDX_Q + 1.0));
                rhod[idx] = sqrt(2.0*M_PI)*h_g*Rc*pow(yc, static_cast<real>(VERIFY_POWER));
                mx[idx] = my[idx] = mz[idx] = 0.0;
#elif defined(VERIFY_RING)
                // initialize a relative Fourier density perturbation on circular equilibrium rings; radiation modifies
                // the equilibrium angular momentum, while optional diffusion supplies the known Fourier-mode damping
                real rhod_mode = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
#ifdef VERIFY_RING_RADIATION
                const real beta = BETA_0;
#else
                const real beta = 0.0;
#endif
                real ell = sqrt((1.0 - beta)*G*M_S*Rc);
                rhod[idx] = host_gas_dens(Rc, Zc)*rhod_mode;
                mx[idx] = rhod[idx]*ell;
                my[idx] = 0.0;
                mz[idx] = 0.0;
#endif
            }
        }
    }
}

void save_array (const std::string &name, const std::vector<real> &array)
{
    if (!save_host_binary(PATH + name + suffix(), array.data(), N_G))
    {
        std::cerr << "Failed to save " << name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void save_state (const std::string &stage, real *dev_dustdens, real *dev_dustmomx,
    real *dev_dustmomy, real *dev_dustmomz, real *dev_velx, real *dev_vely, real *dev_velz)
{
    // synchronize primitive variables with the current conserved fields before copying either representation to the host
    momentum_getv <<< NB_G, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_velx, dev_vely, dev_velz
    );
    HIP_KERNEL_CHECK("momentum_getv");
    HIP_CHECK(hipDeviceSynchronize());

    // ordinary host vectors are sufficient because test output is written only at the initial and final stages
    std::vector<real> rhod(N_G), mx(N_G), my(N_G), mz(N_G);
    std::vector<real> velx(N_G), vely(N_G), velz(N_G);
    HIP_CHECK(hipMemcpy(rhod.data(), dev_dustdens, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mx.data(), dev_dustmomx, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(my.data(), dev_dustmomy, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mz.data(), dev_dustmomz, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(velx.data(), dev_velx, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(vely.data(), dev_vely, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(velz.data(), dev_velz, sizeof(real)*N_G, hipMemcpyDeviceToHost));

    // production calculations store x and z primitives as specific angular momenta; convert copies to physical linear
    // velocities for output while retaining conserved angular momenta in the dustmomx and dustmomz files
    save_sam_as_velocity(velx.data(), velz.data());
    save_array("dustdens_" + stage, rhod);
    save_array("dustmomx_" + stage, mx);
    save_array("dustmomy_" + stage, my);
    save_array("dustmomz_" + stage, mz);
    save_array("dustvelx_" + stage, velx);
    save_array("dustvely_" + stage, vely);
    save_array("dustvelz_" + stage, velz);
}
}

int main ()
{
    // PATH_OUT is selected by the Makefile from the model name and optional parameter-variant tag
    std::filesystem::create_directories(PATH);

    // build exact finite-volume initial data on the host before allocating its device representation
    std::vector<real> rhod(N_G), mx(N_G), my(N_G), mz(N_G);
    initialize_state(rhod, mx, my, mz);

    // allocate the same conserved and primitive fields used by the production fluid integrator
    real *dev_dustdens, *dev_dustmomx, *dev_dustmomy, *dev_dustmomz;
    real *dev_velx, *dev_vely, *dev_velz;
    HIP_CHECK(hipMalloc((void**)&dev_dustdens, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_dustmomx, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_dustmomy, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_dustmomz, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_velx, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_vely, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_velz, sizeof(real)*N_G));

    // only conserved variables were initialized on the host; momentum_getv below derives synchronized primitives
    HIP_CHECK(hipMemcpy(dev_dustdens, rhod.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dustmomx, mx.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dustmomy, my.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dustmomz, mz.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));

    momentum_getv <<< NB_G, TPB >>> (
        dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_velx, dev_vely, dev_velz
    );
    HIP_KERNEL_CHECK("momentum_getv");

    // nonuniform radial and polar PPM kernels require precomputed face interpolation weights even when a particular test
    // evolves only one direction; keeping a common allocation path simplifies compile-time case selection
    real *dev_weight_y, *dev_weight_z;
    HIP_CHECK(hipMalloc((void**)&dev_weight_y, sizeof(real)*4*(N_Y + 1)));
    HIP_CHECK(hipMalloc((void**)&dev_weight_z, sizeof(real)*4*(N_Z + 1)));
    std::vector<real> weight_y(4*(N_Y + 1)), weight_z(4*(N_Z + 1));
    ppm_geometry_weights_calc(weight_y.data(), weight_z.data());
    HIP_CHECK(hipMemcpy(dev_weight_y, weight_y.data(), sizeof(real)*4*(N_Y + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_weight_z, weight_z.data(), sizeof(real)*4*(N_Z + 1), hipMemcpyHostToDevice));

    #ifdef FLUID_BLOCK_SWEEP
    // the block sweep shares one persistent 11-field advection workspace across all directions
    real *dev_adv_work;
    HIP_CHECK(hipMalloc(
        (void**)&dev_adv_work,
        sizeof(real)*static_cast<std::size_t>(BLOCK_ADV_FIELDS)*static_cast<std::size_t>(N_G)
    ));

    #ifdef DIFFUSION
    require_lds(reinterpret_cast<const void*>(diffusion_xbl), sizeof(real)*4*N_X, "diffusion_xbl");
    require_lds(reinterpret_cast<const void*>(diffusion_ybl), sizeof(real)*6*N_Y, "diffusion_ybl");
    require_lds(reinterpret_cast<const void*>(diffusion_zbl), sizeof(real)*6*N_Z, "diffusion_zbl");
    #endif // DIFFUSION
    #endif // FLUID_BLOCK_SWEEP

    // the transport tests that use velocity-dependent timesteps share the production CFL-rate kernel and host reduction
    real *dev_cfl_rate;
    HIP_CHECK(hipMalloc((void**)&dev_cfl_rate, sizeof(real)*N_G));

#ifdef RADIATION
    // radiation and standalone optical-depth builds require one cumulative optical-depth field
    real *dev_optdepth;
    HIP_CHECK(hipMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
#endif

    // saving the initial state enables both direct initialization checks and the later mass-conservation calculation
    save_state("initial", dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
        dev_velx, dev_vely, dev_velz);

    auto recover_velocity = [&]()
    {
        // conservative transport and diffusion update momentum, so primitives must be reconstructed before any subsequent
        // operator that reads velocity or specific angular momentum
        momentum_getv <<< NB_G, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_velx, dev_vely, dev_velz
        );
        HIP_KERNEL_CHECK("momentum_getv");
    };

    auto cfl_step = [&]()
    {
        // recompute rates from the current evolved state rather than retaining the initial timestep throughout the test
        recover_velocity();
        cfl_rate_calc <<< NB_X, TPB >>> (
            dev_cfl_rate, dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_velx, dev_vely, dev_velz
        );
        HIP_KERNEL_CHECK("cfl_rate_calc");
        HIP_CHECK(hipDeviceSynchronize());
        return get_dt_cfl(dev_cfl_rate, dev_velx, dev_vely, dev_velz, false);
    };

    // select one-thread-per-line or one-block-per-line transport without changing the test cases below
    auto apply_advection_x = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        advection_xbl <<< N_Y*N_Z, TPB_BLOCK >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_adv_work, dt
        );
        HIP_KERNEL_CHECK("advection_xbl");
        #else // !FLUID_BLOCK_SWEEP
        advection_xth <<< NB_X, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt);
        HIP_KERNEL_CHECK("advection_xth");
        #endif // FLUID_BLOCK_SWEEP
    };

    auto apply_advection_y = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        advection_ybl <<< N_X*N_Z, TPB_BLOCK >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dev_adv_work, dt
        );
        HIP_KERNEL_CHECK("advection_ybl");
        #else // !FLUID_BLOCK_SWEEP
        advection_yth <<< NB_Y, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dt);
        HIP_KERNEL_CHECK("advection_yth");
        #endif // FLUID_BLOCK_SWEEP
    };

    auto apply_advection_z = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        advection_zbl <<< N_X*N_Y, TPB_BLOCK >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dev_adv_work, dt
        );
        HIP_KERNEL_CHECK("advection_zbl");
        #else // !FLUID_BLOCK_SWEEP
        advection_zth <<< NB_Z, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dt);
        HIP_KERNEL_CHECK("advection_zth");
        #endif // FLUID_BLOCK_SWEEP
    };

    #ifdef DIFFUSION
    // select the matching line solver while preserving each test's timestep and operator order
    auto apply_diffusion_x = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        diffusion_xbl <<< N_Y*N_Z, TPB_BLOCK, sizeof(real)*4*N_X >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
        );
        HIP_KERNEL_CHECK("diffusion_xbl");
        #else // !FLUID_BLOCK_SWEEP
        diffusion_xth <<< NB_X, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt);
        HIP_KERNEL_CHECK("diffusion_xth");
        #endif // FLUID_BLOCK_SWEEP
    };

    auto apply_diffusion_y = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        diffusion_ybl <<< N_X*N_Z, TPB_BLOCK, sizeof(real)*6*N_Y >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
        );
        HIP_KERNEL_CHECK("diffusion_ybl");
        #else // !FLUID_BLOCK_SWEEP
        diffusion_yth <<< NB_Y, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt);
        HIP_KERNEL_CHECK("diffusion_yth");
        #endif // FLUID_BLOCK_SWEEP
    };

    auto apply_diffusion_z = [&](real dt)
    {
        #ifdef FLUID_BLOCK_SWEEP
        diffusion_zbl <<< N_X*N_Y, TPB_BLOCK, sizeof(real)*6*N_Z >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt
        );
        HIP_KERNEL_CHECK("diffusion_zbl");
        #else // !FLUID_BLOCK_SWEEP
        diffusion_zth <<< NB_Z, TPB >>> (dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt);
        HIP_KERNEL_CHECK("diffusion_zth");
        #endif // FLUID_BLOCK_SWEEP
    };
    #endif // DIFFUSION

    real clock = 0.0;
    int steps = 0;

#if defined(VERIFY_OPTDEPTH)
    // optical depth is a spatial quadrature test, not a time integration test; construct the cumulative radial field once,
    // save it, and leave clock and steps at zero for the metadata record
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth, dev_dustdens);
    HIP_KERNEL_CHECK("optdepth_calc");
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
    HIP_KERNEL_CHECK("optdepth_csum");
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<real> optdepth(N_G);
    HIP_CHECK(hipMemcpy(optdepth.data(), dev_optdepth, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    save_array("optdepth_final", optdepth);
#elif defined(VERIFY_SOURCE_DRAG)
    // apply exactly one full source step; the model-local source kernel prescribes gas velocities, stopping times, and
    // linearly varying forces whose closed-form solution is evaluated independently in validate_case.py
    source_update <<< NB_G, TPB >>> (
        dev_velx, dev_vely, dev_velz, dev_dustdens, VERIFY_TEND
    );
    HIP_KERNEL_CHECK("source_update");
    momentum_setv <<< NB_G, TPB >>> (
        dev_dustdens, dev_velx, dev_vely, dev_velz, dev_dustmomx, dev_dustmomy, dev_dustmomz
    );
    HIP_KERNEL_CHECK("momentum_setv");
    clock = VERIFY_TEND;
    steps = 1;
#elif defined(VERIFY_RING)
    // combined ring tests exercise the production operator composition; their target timestep is tied to one quarter of an
    // azimuthal cell at the fastest equilibrium orbit, then clipped so the final step lands exactly on VERIFY_TEND
    real dx = (X_MAX - X_MIN)/static_cast<real>(N_X);
    real omega_max = sqrt((1.0
        #ifdef VERIFY_RING_RADIATION
        - BETA_0
        #endif
        )*G*M_S/(Y_MIN*Y_MIN*Y_MIN));
    real dt_target = 0.25*dx/omega_max;

    while (clock < VERIFY_TEND)
    {
        real dt = fmin(dt_target, VERIFY_TEND - clock);

        #ifdef DIFFUSION
        // opening diffusion half-step in y then x; z is absent in these 2D ring models
        apply_diffusion_y(0.5*dt);
        apply_diffusion_x(0.5*dt);
        recover_velocity();
        #endif

        // opening transport half-step in x then y, recovering primitives after each conservative sweep
        apply_advection_x(0.5*dt);
        recover_velocity();
        apply_advection_y(0.5*dt);
        recover_velocity();

        #ifdef RADIATION
        // radiation builds recompute optical depth at the state presented to the centered source operator; the taper is one
        // in verification runs so the analytical acceleration is active for the entire step
        optdepth_calc <<< NB_G, TPB >>> (dev_optdepth, dev_dustdens);
        HIP_KERNEL_CHECK("optdepth_calc");
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        HIP_KERNEL_CHECK("optdepth_csum");
        source_update <<< NB_G, TPB >>> (dev_velx, dev_vely, dev_velz, dev_dustdens, dev_optdepth, 1.0, dt);
        #else
        source_update <<< NB_G, TPB >>> (dev_velx, dev_vely, dev_velz, dev_dustdens, dt);
        #endif
        HIP_KERNEL_CHECK("source_update");

        // source_update advances primitives, so rebuild conserved momenta before returning to conservative transport
        momentum_setv <<< NB_G, TPB >>> (
            dev_dustdens, dev_velx, dev_vely, dev_velz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        HIP_KERNEL_CHECK("momentum_setv");

        // closing transport half-step reverses the opening directional order for a symmetric composition
        apply_advection_y(0.5*dt);
        recover_velocity();
        apply_advection_x(0.5*dt);
        recover_velocity();

        #ifdef DIFFUSION
        // closing diffusion half-step also reverses its opening order
        apply_diffusion_x(0.5*dt);
        apply_diffusion_y(0.5*dt);
        #endif

        clock += dt;
        steps++;
    }
#else
    // isolated transport and diffusion tests advance only the kernel selected by the model's VERIFY_* macro
    while (clock < VERIFY_TEND)
    {
        real dt;
#if defined(VERIFY_X_TRANSPORT)
        // prescribe the azimuthal displacement in cell widths so integer and fractional FARGO shifts can be studied directly
        real dx = (X_MAX - X_MIN)/static_cast<real>(N_X);
        dt = fmin(static_cast<real>(VERIFY_SHIFT)*dx, VERIFY_TEND - clock);
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH) \
    || defined(VERIFY_Y_OUTFLOW_2D) || defined(VERIFY_Y_OUTFLOW_3D) || defined(VERIFY_Z_TRANSPORT)
        // radial and polar transport use the current global production CFL condition
        dt = fmin(cfl_step(), VERIFY_TEND - clock);
#else
        // diffusion is implicit, but a resolution-dependent timestep is retained to measure convergence of the complete
        // space-time discretization rather than allowing temporal error to remain fixed as the grid is refined
        real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
        real min_length = Y_MIN*(dy - 1.0);
        if (N_Z > 1)
        {
            min_length = fmin(min_length, Y_MIN*(Z_MAX - Z_MIN)/static_cast<real>(N_Z));
        }
        real dx_length = Y_MIN*(X_MAX - X_MIN)/static_cast<real>(N_X);
        min_length = fmin(min_length, dx_length);
        dt = fmin(0.25*min_length, VERIFY_TEND - clock);
#endif

        // compile exactly one of these calls into an isolated-kernel test executable
#if defined(VERIFY_X_TRANSPORT)
        apply_advection_x(dt);
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH) \
    || defined(VERIFY_Y_OUTFLOW_2D) || defined(VERIFY_Y_OUTFLOW_3D)
        apply_advection_y(dt);
#elif defined(VERIFY_Z_TRANSPORT)
        apply_advection_z(dt);
#elif defined(VERIFY_X_DIFFUSION)
        apply_diffusion_x(dt);
#elif defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH)
        apply_diffusion_y(dt);
#elif defined(VERIFY_Z_DIFFUSION)
        apply_diffusion_z(dt);
#endif
        clock += dt;
        steps++;
    }
#endif

#if !defined(VERIFY_OPTDEPTH)
    #ifdef RADIATION
    // save final optical depth for combined radiation cases after the last density update
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth, dev_dustdens);
    HIP_KERNEL_CHECK("optdepth_calc");
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
    HIP_KERNEL_CHECK("optdepth_csum");
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<real> optdepth_final(N_G);
    HIP_CHECK(hipMemcpy(optdepth_final.data(), dev_optdepth, sizeof(real)*N_G, hipMemcpyDeviceToHost));
    save_array("optdepth_final", optdepth_final);
    #endif

    // synchronize and save the final conserved fields and physical velocities for Python validation
    save_state("final", dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz,
        dev_velx, dev_vely, dev_velz);
#endif

    // record the compile-time configuration and realized integration statistics beside the raw binary arrays; Python uses
    // these values instead of inferring dimensions or final time from filenames and requested parameters
    std::ofstream meta(PATH + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"" << case_name() << "\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"time\": " << clock << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"cfl\": " << CFL_DYN << ",\n"
         << "  \"shift\": " << static_cast<real>(VERIFY_SHIFT) << ",\n"
         << "  \"power\": " << static_cast<real>(VERIFY_POWER) << "\n"
         << "}\n";

    // release all device allocations; host std::vectors are released automatically when main returns
    HIP_CHECK(hipFree(dev_dustdens));
    HIP_CHECK(hipFree(dev_dustmomx));
    HIP_CHECK(hipFree(dev_dustmomy));
    HIP_CHECK(hipFree(dev_dustmomz));
    HIP_CHECK(hipFree(dev_velx));
    HIP_CHECK(hipFree(dev_vely));
    HIP_CHECK(hipFree(dev_velz));
    HIP_CHECK(hipFree(dev_weight_y));
    HIP_CHECK(hipFree(dev_weight_z));
    #ifdef FLUID_BLOCK_SWEEP
    HIP_CHECK(hipFree(dev_adv_work));
    #endif // FLUID_BLOCK_SWEEP
    HIP_CHECK(hipFree(dev_cfl_rate));
#ifdef RADIATION
    HIP_CHECK(hipFree(dev_optdepth));
#endif

    std::cout << "Verification case " << case_name() << " completed at N="
              << VERIFY_RES << " in " << steps << " step(s)." << std::endl;
    return 0;
}
