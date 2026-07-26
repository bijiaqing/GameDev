#include <cmath>        // for std::fmin, std::fmax
#include <filesystem>   // for creating output directory
#include <iomanip>      // for std::setfill, std::setw
#include <iostream>     // for std::cout, std::endl
#include <sstream>      // for std::stringstream

#include <graffiti_kern.cuh>
#include <graffiti_host.cuh>

const std::string PATH = PATH_OUT;

int main (int argc, char **argv)
{
    real *dustdens, *dev_dustdens;
    CUDA_CHECK(cudaMallocHost((void**)&dustdens, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_G));

    real *dustvelx, *dev_dustvelx, *dev_dustmomx;
    CUDA_CHECK(cudaMallocHost((void**)&dustvelx, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustvelx, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustmomx, sizeof(real)*N_G));

    real *dustvely, *dev_dustvely, *dev_dustmomy;
    CUDA_CHECK(cudaMallocHost((void**)&dustvely, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustvely, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustmomy, sizeof(real)*N_G));

    real *dustvelz, *dev_dustvelz, *dev_dustmomz;
    CUDA_CHECK(cudaMallocHost((void**)&dustvelz, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustvelz, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustmomz, sizeof(real)*N_G));

    real *dev_cfl_rate;
    CUDA_CHECK(cudaMalloc((void**)&dev_cfl_rate, sizeof(real)*N_G));

    int  *dev_badstate;
    CUDA_CHECK(cudaMalloc((void**)&dev_badstate, sizeof(int)));

    real *dev_weight_y, *dev_weight_z;
    CUDA_CHECK(cudaMalloc((void**)&dev_weight_y, sizeof(real)*4*(N_Y + 1)));
    CUDA_CHECK(cudaMalloc((void**)&dev_weight_z, sizeof(real)*4*(N_Z + 1)));
    
    std::vector<real> weight_y(4*(N_Y + 1));
    std::vector<real> weight_z(4*(N_Z + 1));
    ppm_geometry_weights_calc(weight_y.data(), weight_z.data());

    CUDA_CHECK(cudaMemcpy(dev_weight_y, weight_y.data(), sizeof(real)*4*(N_Y + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_weight_z, weight_z.data(), sizeof(real)*4*(N_Z + 1), cudaMemcpyHostToDevice));

    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    CUDA_CHECK(cudaMallocHost((void**)&optdepth, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
    #endif // RADIATION

    auto validate_finite_state = [&]()
    {
        CUDA_CHECK(cudaMemset(dev_badstate, 0, sizeof(int)));
        finite_verify <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz,
            #ifdef RADIATION
            dev_optdepth,
            #endif
            dev_badstate
        );
        CUDA_KERNEL_CHECK("finite_verify");

        int badstate = 0;
        CUDA_CHECK(cudaMemcpy(&badstate, dev_badstate, sizeof(int), cudaMemcpyDeviceToHost));
        if (badstate != 0)
        {
            int idx_bad = badstate - 1;
            int ix_bad = idx_bad % N_X;
            int iy_bad = (idx_bad / N_X) % N_Y;
            int iz_bad = idx_bad / (N_X * N_Y);
            
            std::cerr 
            << "Error: non-finite simulation state at cell ("
            << ix_bad << "," << iy_bad << "," << iz_bad << ")"
            << std::endl;
            
            std::exit(EXIT_FAILURE);
        }
    };

    int idx_from;
    real clock_sim = 0.0;
    real clock_out = 0.0;

    if (argc <= 1)
    {
        idx_from = 0;

        // Precompute convolved METAL_Z*Sigma_g dust surface density on host.
        real *initdens, *dev_initdens;
        CUDA_CHECK(cudaMallocHost((void**)&initdens, sizeof(real)*(N_Y + 1)));
        CUDA_CHECK(cudaMalloc((void**)&dev_initdens, sizeof(real)*(N_Y + 1)));

        convpow_calc(initdens);
        CUDA_CHECK(cudaMemcpy(dev_initdens, initdens, sizeof(real)*(N_Y + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(initdens));

        f_rho_initial <<< NB_A, TPB >>> (dev_dustdens, dev_initdens);
        CUDA_KERNEL_CHECK("f_rho_initial");

        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        
        CUDA_CHECK(cudaDeviceSynchronize());
        #endif // RADIATION

        f_vel_initial <<< NB_A, TPB >>> (
            dev_dustvelx, dev_dustvely, dev_dustvelz
            #ifdef DIFFUSION
            , dev_dustdens
            #endif
        );
        CUDA_KERNEL_CHECK("f_vel_initial");
        
        f_moment_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_setv");
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dev_initdens));

        validate_finite_state();

        std::filesystem::create_directories(PATH);
        if (!save_variable(PATH + "variables.txt"))
        {
            std::cerr << "Error: failed to save simulation parameters" << std::endl;
            return 1;
        }

        SAVE_DUSTDATA_TO_FILE(idx_from);
        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_from);
        #endif // RADIATION

        msg_output(0);
    }
    else
    {
        std::stringstream ss{argv[1]};
        if (!(ss >> idx_from))
        {
            std::cerr << "Error: invalid resume frame number: " << argv[1] << "\n";
            return 1;
        }

        if (!validate_restart_config(PATH + "variables.txt")) return 1;

        LOAD_DUSTDATA_TO_VRAM(idx_from);

        f_moment_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_setv");
        
        CUDA_CHECK(cudaDeviceSynchronize());

        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        
        CUDA_CHECK(cudaDeviceSynchronize());
        #endif // RADIATION

        validate_finite_state();

        msg_output(idx_from);
    }

    clock_sim = static_cast<real>(idx_from)*DT_OUT;

    msg_step_title();

    auto recover_dust_velocity = [&]()
    {
        f_moment_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_getv");
    };

    auto recalc_dt_cfl = [&](bool verbose)
    {
        cfl_rate_calc <<< NB_X, TPB >>> (
            dev_cfl_rate, dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("cfl_rate_calc");
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return get_dt_cfl(dev_cfl_rate, dev_dustvelx, dev_dustvely, dev_dustvelz, verbose);
    };

    // advance one complete directional interval while refreshing the CFL bound after every substep
    // each direction exhausts its own time budget before the next Strang operator begins
    auto advance_x = [&](real time_interval)
    {
        real time_remain = time_interval;
        while (time_remain > 0.0)
        {
            real dt_sub = std::fmin(time_remain, recalc_dt_cfl(false));

            f_advection_x <<< NB_X, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt_sub
            );
            CUDA_KERNEL_CHECK("f_advection_x");

            recover_dust_velocity();
            time_remain -= dt_sub;
        }
    };

    auto advance_y = [&](real time_interval)
    {
        real time_remain = time_interval;
        while (time_remain > 0.0)
        {
            real dt_sub = std::fmin(time_remain, recalc_dt_cfl(false));

            f_advection_y <<< NB_Y, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dt_sub
            );
            CUDA_KERNEL_CHECK("f_advection_y");

            recover_dust_velocity();
            time_remain -= dt_sub;
        }
    };

    auto advance_z = [&](real time_interval)
    {
        real time_remain = time_interval;
        while (time_remain > 0.0)
        {
            real dt_sub = std::fmin(time_remain, recalc_dt_cfl(false));

            f_advection_z <<< NB_Z, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dt_sub
            );
            CUDA_KERNEL_CHECK("f_advection_z");

            recover_dust_velocity();
            time_remain -= dt_sub;
        }
    };

    while (idx_from < SAVE_MAX) // main simulation loop
    {
        real dt_cfl_begin = recalc_dt_cfl(true);
        real dt = dt_cfl_begin;
        real dt_to_out = DT_OUT - clock_out;
        bool output_due = (dt >= dt_to_out);
        if (output_due) dt = dt_to_out;

        #ifdef DIFFUSION
        f_diffusion_y <<< NB_Y, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_y");
        
        f_diffusion_x <<< NB_X, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_x");
        
        f_diffusion_z <<< NB_Z, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_z");
        
        f_moment_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_getv");
        #endif // DIFFUSION

        // =====================================================
        // Symmetric diffusion/advection/source composition:
        // D_y D_x D_z | A_x A_y A_z | source | A_z A_y A_x | D_z D_x D_y
        // Every D and A entry is a half-step; source is a full step.
        // =====================================================

        // First half advection (X Y Z): each operator covers exactly dt/2
        real adv_interval = 0.5*dt;

        advance_x(adv_interval);
        advance_y(adv_interval);

        if (N_Z > 1)
        {
            advance_z(adv_interval);
        }

        // Recompute τ from midpoint density ρ_d^{n+1/2}
        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        
        validate_finite_state();
        #endif // RADIATION

        // Full source step (exponential drag/force update, uses tau^{n+1/2})
        #ifdef RADIATION
        real taper = (clock_sim + 0.5*dt) / T_BETA;
        taper = std::fmin(std::fmax(taper, 0.0), 1.0);
        real beta_taper = taper*taper*(3.0 - 2.0*taper);
        #endif // RADIATION

        f_source_step <<< NB_A, TPB >>> (
            dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens,
            #ifdef RADIATION
            dev_optdepth, beta_taper,
            #endif // RADIATION
            dt
        );
        CUDA_KERNEL_CHECK("f_source_step");

        f_moment_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_setv");

        // Second half advection (Z Y X): independently bounded directional operators
        // The source and each preceding sweep may change the next sweep's transport velocity
        if (N_Z > 1)
        {
            advance_z(adv_interval);
        }

        advance_y(adv_interval);
        advance_x(adv_interval);

        #ifdef DIFFUSION
        f_diffusion_z <<< NB_Z, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_z");
        
        f_diffusion_x <<< NB_X, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_x");
        
        f_diffusion_y <<< NB_Y, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("f_diffusion_y");

        f_moment_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_getv");
        #endif // DIFFUSION

        validate_finite_state();

        CUDA_CHECK(cudaDeviceSynchronize());

        clock_sim += dt;
        clock_out += dt;

        msg_step(idx_from, dt, clock_out, clock_sim);

        if (output_due)
        {
            idx_from++;
            clock_out = 0.0;

            #ifdef RADIATION
            optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
            CUDA_KERNEL_CHECK("optdepth_calc");
            
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_csum");
            
            validate_finite_state();
            #endif // RADIATION

            SAVE_DUSTDATA_TO_FILE(idx_from);
            #ifdef RADIATION
            SAVE_OPTDEPTH_TO_FILE(idx_from);
            #endif // RADIATION

            msg_output(idx_from);
        }
    }

    CUDA_CHECK(cudaFreeHost(dustdens));
    CUDA_CHECK(cudaFree(dev_dustdens));
    CUDA_CHECK(cudaFree(dev_cfl_rate));
    CUDA_CHECK(cudaFree(dev_badstate));
    CUDA_CHECK(cudaFree(dev_weight_y));
    CUDA_CHECK(cudaFree(dev_weight_z));
    CUDA_CHECK(cudaFreeHost(dustvelx));
    CUDA_CHECK(cudaFree(dev_dustvelx));
    CUDA_CHECK(cudaFree(dev_dustmomx));
    CUDA_CHECK(cudaFreeHost(dustvely));
    CUDA_CHECK(cudaFree(dev_dustvely));
    CUDA_CHECK(cudaFree(dev_dustmomy));
    CUDA_CHECK(cudaFreeHost(dustvelz));
    CUDA_CHECK(cudaFree(dev_dustvelz));
    CUDA_CHECK(cudaFree(dev_dustmomz));

    #ifdef RADIATION
    CUDA_CHECK(cudaFreeHost(optdepth));
    CUDA_CHECK(cudaFree(dev_optdepth));
    #endif // RADIATION

    return 0;
}
