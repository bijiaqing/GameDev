#include <cmath>          // cos, exp, fmin, pow, sin, sqrt, std::cyl_bessel_j, std::cyl_neumann
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <filesystem>     // std::filesystem::create_directories
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cerr, std::cout, std::endl
#include <string>         // std::string, std::to_string
#include <vector>         // std::vector

#include <cuda_runtime.h> // cudaDeviceSynchronize, cudaFree, cudaMalloc, cudaMemcpy

#include <fluid_kern.cuh>
#include <fluid_host.cuh>

namespace
{
const std::string PATH = PATH_OUT;

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
    constexpr real k2 = 1.694299217770420;
    constexpr real k3 = 1.874562003084784;

    if (dimension == 2)
    {
        real value = std::cyl_bessel_j(0, k2*y)*std::cyl_neumann(1, k2*Y_MIN)
                   - std::cyl_neumann(0, k2*y)*std::cyl_bessel_j(1, k2*Y_MIN);
        real norm = std::cyl_bessel_j(0, k2*Y_MIN)*std::cyl_neumann(1, k2*Y_MIN)
                  - std::cyl_neumann(0, k2*Y_MIN)*std::cyl_bessel_j(1, k2*Y_MIN);
        return value / norm;
    }

    real value = spherical_j0(k3*y)*spherical_y0_deriv(k3*Y_MIN)
               - spherical_y0(k3*y)*spherical_j0_deriv(k3*Y_MIN);
    real norm = spherical_j0(k3*Y_MIN)*spherical_y0_deriv(k3*Y_MIN)
              - spherical_y0(k3*Y_MIN)*spherical_j0_deriv(k3*Y_MIN);
    return value / norm;
}

real host_rhog (real R, real Z)
{
#ifdef VERIFY_UNIFORM_GAS
    (void)R;
    (void)Z;
    return 1.0;
#else
    real h_g = ASPR_0*pow(R/R_0, 0.5*(IDX_Q + 1.0));
    real sigma_g = SIGMA_0*pow(R/R_0, IDX_P);
    real rho_mid = sigma_g / (sqrt(2.0*M_PI)*h_g*R);
    return rho_mid*exp((R/sqrt(R*R + Z*Z) - 1.0)/(h_g*h_g));
#endif
}

const char *case_name ()
{
#if defined(VERIFY_X_TRANSPORT)
    return "x_transport";
#elif defined(VERIFY_Y_TRANSPORT_CYL)
    return "y_transport_cyl";
#elif defined(VERIFY_Y_TRANSPORT_SPH)
    return "y_transport_sph";
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
    return "_N" + std::to_string(VERIFY_RES) + ".dat";
}

void initialize_state (std::vector<real> &dens, std::vector<real> &momx,
    std::vector<real> &momy, std::vector<real> &momz)
{
    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX/Y_MIN, 1.0/static_cast<real>(N_Y));
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);

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
                real q = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
                real velx = Rc*Rc;
                dens[idx] = q;
                momx[idx] = q*velx;
                momy[idx] = 0.0;
                momz[idx] = 0.0;
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH)
                int dimension = (N_Z > 1) ? 3 : 2;
                real volume = (pow(y1, dimension) - pow(y0, dimension))/static_cast<real>(dimension);
                real rho_int = gauss8([&](real y)
                {
                    return compact_bump(y, 1.0, 1.8)*pow(y, dimension - 1);
                }, y0, y1);
                real momy_int = gauss8([&](real y)
                {
                    return compact_bump(y, 1.0, 1.8)*VERIFY_A*y*pow(y, dimension - 1);
                }, y0, y1);
                dens[idx] = rho_int/volume;
                momx[idx] = 0.7*dens[idx];
                momy[idx] = momy_int/volume;
                momz[idx] = 0.11*dens[idx];
#elif defined(VERIFY_Z_TRANSPORT)
                real volume = cos(z0) - cos(z1);
                real rho_int = gauss8([&](real z)
                {
                    return compact_bump(z, 0.80, 1.30);
                }, z0, z1);
                dens[idx] = rho_int/volume;
                momx[idx] = 0.7*dens[idx];
                momy[idx] = 0.05*dens[idx];
                momz[idx] = VERIFY_LZ*dens[idx];
#elif defined(VERIFY_X_DIFFUSION)
                real q = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
                dens[idx] = q;
                momx[idx] = 0.7*q;
                momy[idx] = -0.15*q;
                momz[idx] = 0.11*q;
#elif defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH)
                int dimension = (N_Z > 1) ? 3 : 2;
                real volume = (pow(y1, dimension) - pow(y0, dimension))/static_cast<real>(dimension);
                real mode_avg = gauss8([&](real y)
                {
                    return radial_mode(y, dimension)*pow(y, dimension - 1);
                }, y0, y1) / volume;
                real q = VERIFY_Q0 + VERIFY_EPS*mode_avg;
                dens[idx] = q;
                momx[idx] = 0.7*q;
                momy[idx] = -0.15*q;
                momz[idx] = 0.11*q;
#elif defined(VERIFY_Z_DIFFUSION)
                real volume = cos(z0) - cos(z1);
                real mode_avg = gauss8([&](real z)
                {
                    real mu = cos(z);
                    return 0.5*(3.0*mu*mu - 1.0)*sin(z);
                }, z0, z1) / volume;
                real q = VERIFY_Q0 + VERIFY_EPS*mode_avg;
                dens[idx] = q;
                momx[idx] = 0.7*q;
                momy[idx] = -0.15*q;
                momz[idx] = 0.11*q;
#elif defined(VERIFY_SOURCE_DRAG)
                dens[idx] = 1.0;
                momx[idx] = 1.3;
                momy[idx] = -0.8;
                momz[idx] = 0.6;
#elif defined(VERIFY_OPTDEPTH)
                dens[idx] = pow(yc, static_cast<real>(VERIFY_POWER));
                momx[idx] = momy[idx] = momz[idx] = 0.0;
#elif defined(VERIFY_RING)
                real q = VERIFY_Q0 + VERIFY_EPS*(sin(VERIFY_M*x1) - sin(VERIFY_M*x0))/(VERIFY_M*dx);
#ifdef VERIFY_RING_RADIATION
                const real beta = BETA_0;
#else
                const real beta = 0.0;
#endif
                real ell = sqrt((1.0 - beta)*G*M_S*Rc);
                dens[idx] = host_rhog(Rc, Zc)*q;
                momx[idx] = dens[idx]*ell;
                momy[idx] = 0.0;
                momz[idx] = 0.0;
#endif
            }
        }
    }
}

void save_array (const std::string &name, const std::vector<real> &array)
{
    if (!save_binary(PATH + name + suffix(), const_cast<real*>(array.data()), N_G))
    {
        std::cerr << "Failed to save " << name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void save_state (const std::string &stage, real *dev_dens, real *dev_momx,
    real *dev_momy, real *dev_momz, real *dev_velx, real *dev_vely, real *dev_velz)
{
    momentum_getv <<< NB_A, TPB >>> (
        dev_dens, dev_momx, dev_momy, dev_momz, dev_velx, dev_vely, dev_velz
    );
    CUDA_KERNEL_CHECK("momentum_getv");
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<real> dens(N_G), momx(N_G), momy(N_G), momz(N_G);
    std::vector<real> velx(N_G), vely(N_G), velz(N_G);
    CUDA_CHECK(cudaMemcpy(dens.data(), dev_dens, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(momx.data(), dev_momx, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(momy.data(), dev_momy, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(momz.data(), dev_momz, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velx.data(), dev_velx, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vely.data(), dev_vely, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velz.data(), dev_velz, sizeof(real)*N_G, cudaMemcpyDeviceToHost));

    save_sam_as_velocity(velx.data(), velz.data());
    save_array("dustdens_" + stage, dens);
    save_array("dustmomx_" + stage, momx);
    save_array("dustmomy_" + stage, momy);
    save_array("dustmomz_" + stage, momz);
    save_array("dustvelx_" + stage, velx);
    save_array("dustvely_" + stage, vely);
    save_array("dustvelz_" + stage, velz);
}
}

int main ()
{
    std::filesystem::create_directories(PATH);

    std::vector<real> dens(N_G), momx(N_G), momy(N_G), momz(N_G);
    initialize_state(dens, momx, momy, momz);

    real *dev_dens, *dev_momx, *dev_momy, *dev_momz;
    real *dev_velx, *dev_vely, *dev_velz;
    CUDA_CHECK(cudaMalloc((void**)&dev_dens, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_momx, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_momy, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_momz, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_velx, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_vely, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_velz, sizeof(real)*N_G));

    CUDA_CHECK(cudaMemcpy(dev_dens, dens.data(), sizeof(real)*N_G, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_momx, momx.data(), sizeof(real)*N_G, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_momy, momy.data(), sizeof(real)*N_G, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_momz, momz.data(), sizeof(real)*N_G, cudaMemcpyHostToDevice));

    momentum_getv <<< NB_A, TPB >>> (
        dev_dens, dev_momx, dev_momy, dev_momz, dev_velx, dev_vely, dev_velz
    );
    CUDA_KERNEL_CHECK("momentum_getv");

    real *dev_weight_y, *dev_weight_z;
    CUDA_CHECK(cudaMalloc((void**)&dev_weight_y, sizeof(real)*4*(N_Y + 1)));
    CUDA_CHECK(cudaMalloc((void**)&dev_weight_z, sizeof(real)*4*(N_Z + 1)));
    std::vector<real> weight_y(4*(N_Y + 1)), weight_z(4*(N_Z + 1));
    ppm_geometry_weights_calc(weight_y.data(), weight_z.data());
    CUDA_CHECK(cudaMemcpy(dev_weight_y, weight_y.data(), sizeof(real)*4*(N_Y + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_weight_z, weight_z.data(), sizeof(real)*4*(N_Z + 1), cudaMemcpyHostToDevice));

    real *dev_cfl_rate;
    CUDA_CHECK(cudaMalloc((void**)&dev_cfl_rate, sizeof(real)*N_G));

#ifdef RADIATION
    real *dev_optdepth;
    CUDA_CHECK(cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
#endif

    save_state("initial", dev_dens, dev_momx, dev_momy, dev_momz,
        dev_velx, dev_vely, dev_velz);

    auto recover_velocity = [&]()
    {
        momentum_getv <<< NB_A, TPB >>> (
            dev_dens, dev_momx, dev_momy, dev_momz, dev_velx, dev_vely, dev_velz
        );
        CUDA_KERNEL_CHECK("momentum_getv");
    };

    auto cfl_step = [&]()
    {
        recover_velocity();
        cfl_rate_calc <<< NB_X, TPB >>> (
            dev_cfl_rate, dev_dens, dev_momx, dev_momy, dev_momz,
            dev_velx, dev_vely, dev_velz
        );
        CUDA_KERNEL_CHECK("cfl_rate_calc");
        CUDA_CHECK(cudaDeviceSynchronize());
        return get_dt_cfl(dev_cfl_rate, dev_velx, dev_vely, dev_velz, false);
    };

    real clock = 0.0;
    int steps = 0;

#if defined(VERIFY_OPTDEPTH)
    optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dens);
    CUDA_KERNEL_CHECK("optdepth_calc");
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
    CUDA_KERNEL_CHECK("optdepth_csum");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<real> optdepth(N_G);
    CUDA_CHECK(cudaMemcpy(optdepth.data(), dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    save_array("optdepth_final", optdepth);
#elif defined(VERIFY_SOURCE_DRAG)
    source_update <<< NB_A, TPB >>> (
        dev_velx, dev_vely, dev_velz, dev_dens, VERIFY_TEND
    );
    CUDA_KERNEL_CHECK("source_update");
    momentum_setv <<< NB_A, TPB >>> (
        dev_dens, dev_velx, dev_vely, dev_velz, dev_momx, dev_momy, dev_momz
    );
    CUDA_KERNEL_CHECK("momentum_setv");
    clock = VERIFY_TEND;
    steps = 1;
#elif defined(VERIFY_RING)
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
        diffus_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("diffus_y_calc");
        diffus_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("diffus_x_calc");
        recover_velocity();
        #endif

        advect_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("advect_x_calc");
        recover_velocity();
        advect_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dev_weight_y, 0.5*dt);
        CUDA_KERNEL_CHECK("advect_y_calc");
        recover_velocity();

        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        source_update <<< NB_A, TPB >>> (dev_velx, dev_vely, dev_velz, dev_dens, dev_optdepth, 1.0, dt);
        #else
        source_update <<< NB_A, TPB >>> (dev_velx, dev_vely, dev_velz, dev_dens, dt);
        #endif
        CUDA_KERNEL_CHECK("source_update");
        momentum_setv <<< NB_A, TPB >>> (
            dev_dens, dev_velx, dev_vely, dev_velz, dev_momx, dev_momy, dev_momz
        );
        CUDA_KERNEL_CHECK("momentum_setv");

        advect_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dev_weight_y, 0.5*dt);
        CUDA_KERNEL_CHECK("advect_y_calc");
        recover_velocity();
        advect_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("advect_x_calc");
        recover_velocity();

        #ifdef DIFFUSION
        diffus_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("diffus_x_calc");
        diffus_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, 0.5*dt);
        CUDA_KERNEL_CHECK("diffus_y_calc");
        #endif

        clock += dt;
        steps++;
    }
#else
    while (clock < VERIFY_TEND)
    {
        real dt;
#if defined(VERIFY_X_TRANSPORT)
        real dx = (X_MAX - X_MIN)/static_cast<real>(N_X);
        dt = fmin(static_cast<real>(VERIFY_SHIFT)*dx, VERIFY_TEND - clock);
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH) || defined(VERIFY_Z_TRANSPORT)
        dt = fmin(cfl_step(), VERIFY_TEND - clock);
#else
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

#if defined(VERIFY_X_TRANSPORT)
        advect_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dt);
        CUDA_KERNEL_CHECK("advect_x_calc");
#elif defined(VERIFY_Y_TRANSPORT_CYL) || defined(VERIFY_Y_TRANSPORT_SPH)
        advect_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dev_weight_y, dt);
        CUDA_KERNEL_CHECK("advect_y_calc");
#elif defined(VERIFY_Z_TRANSPORT)
        advect_z_calc <<< NB_Z, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dev_weight_z, dt);
        CUDA_KERNEL_CHECK("advect_z_calc");
#elif defined(VERIFY_X_DIFFUSION)
        diffus_x_calc <<< NB_X, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dt);
        CUDA_KERNEL_CHECK("diffus_x_calc");
#elif defined(VERIFY_Y_DIFFUSION_CYL) || defined(VERIFY_Y_DIFFUSION_SPH)
        diffus_y_calc <<< NB_Y, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dt);
        CUDA_KERNEL_CHECK("diffus_y_calc");
#elif defined(VERIFY_Z_DIFFUSION)
        diffus_z_calc <<< NB_Z, TPB >>> (dev_dens, dev_momx, dev_momy, dev_momz, dt);
        CUDA_KERNEL_CHECK("diffus_z_calc");
#endif
        clock += dt;
        steps++;
    }
#endif

#if !defined(VERIFY_OPTDEPTH)
    #ifdef RADIATION
    optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dens);
    CUDA_KERNEL_CHECK("optdepth_calc");
    optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
    CUDA_KERNEL_CHECK("optdepth_csum");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<real> optdepth_final(N_G);
    CUDA_CHECK(cudaMemcpy(optdepth_final.data(), dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost));
    save_array("optdepth_final", optdepth_final);
    #endif

    save_state("final", dev_dens, dev_momx, dev_momy, dev_momz,
        dev_velx, dev_vely, dev_velz);
#endif

    std::ofstream meta(PATH + "meta_N" + std::to_string(VERIFY_RES) + ".txt");
    meta << std::setprecision(17)
         << "case=" << case_name() << '\n'
         << "resolution=" << VERIFY_RES << '\n'
         << "nx=" << N_X << '\n'
         << "ny=" << N_Y << '\n'
         << "nz=" << N_Z << '\n'
         << "time=" << clock << '\n'
         << "steps=" << steps << '\n'
         << "cfl=" << CFL_NUM << '\n'
         << "shift=" << static_cast<real>(VERIFY_SHIFT) << '\n'
         << "power=" << static_cast<real>(VERIFY_POWER) << '\n';

    CUDA_CHECK(cudaFree(dev_dens));
    CUDA_CHECK(cudaFree(dev_momx));
    CUDA_CHECK(cudaFree(dev_momy));
    CUDA_CHECK(cudaFree(dev_momz));
    CUDA_CHECK(cudaFree(dev_velx));
    CUDA_CHECK(cudaFree(dev_vely));
    CUDA_CHECK(cudaFree(dev_velz));
    CUDA_CHECK(cudaFree(dev_weight_y));
    CUDA_CHECK(cudaFree(dev_weight_z));
    CUDA_CHECK(cudaFree(dev_cfl_rate));
#ifdef RADIATION
    CUDA_CHECK(cudaFree(dev_optdepth));
#endif

    std::cout << "Verification case " << case_name() << " completed at N="
              << VERIFY_RES << " in " << steps << " step(s)." << std::endl;
    return 0;
}
