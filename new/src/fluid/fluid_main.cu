#include <cmath>          // std::fmax, std::fmin
#include <cstdlib>        // std::exit, EXIT_FAILURE
#include <filesystem>     // std::filesystem::create_directories
#include <iostream>       // std::cerr, std::endl
#include <sstream>        // std::stringstream
#include <string>         // std::string
#include <vector>         // std::vector

#include <cuda_runtime.h> // cudaDeviceSynchronize, cudaFree, cudaFreeHost, ...

#include <fluid_kern.cuh>
#include <fluid_host.cuh>

const std::string PATH = PATH_OUT;

int main (int argc, char **argv)
{
    // allocate host output buffers and persistent device fields
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

    // precompute nonuniform PPM face weights and upload them once
    ppm_geometry_weights_calc(weight_y.data(), weight_z.data());

    CUDA_CHECK(cudaMemcpy(dev_weight_y, weight_y.data(), sizeof(real)*4*(N_Y + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_weight_z, weight_z.data(), sizeof(real)*4*(N_Z + 1), cudaMemcpyHostToDevice));

    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    CUDA_CHECK(cudaMallocHost((void**)&optdepth, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
    #endif

    // abort at the first cell containing a nonfinite evolved value
    auto validate_finite_state = [&]()
    {
        CUDA_CHECK(cudaMemset(dev_badstate, 0, sizeof(int)));
        inf_cell_flag <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz,
            #ifdef RADIATION
            dev_optdepth,
            #endif
            dev_badstate
        );
        CUDA_KERNEL_CHECK("inf_cell_flag");

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

        // initialize density from the convolved surface profile
        real *initdens, *dev_initdens;
        CUDA_CHECK(cudaMallocHost((void**)&initdens, sizeof(real)*(N_Y + 1)));
        CUDA_CHECK(cudaMalloc((void**)&dev_initdens, sizeof(real)*(N_Y + 1)));

        convpow_calc(initdens);
        CUDA_CHECK(cudaMemcpy(dev_initdens, initdens, sizeof(real)*(N_Y + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(initdens));

        init_rho_calc <<< NB_A, TPB >>> (dev_dustdens, dev_initdens);
        CUDA_KERNEL_CHECK("init_rho_calc");

        #ifdef RADIATION
        // construct the initial cumulative radial optical depth
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");

        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");

        CUDA_CHECK(cudaDeviceSynchronize());
        #endif

        // initialize primitive velocities and build their conserved fields
        init_vel_calc <<< NB_A, TPB >>> (
            dev_dustvelx, dev_dustvely, dev_dustvelz
            #ifdef DIFFUSION
            , dev_dustdens
            #endif
        );
        CUDA_KERNEL_CHECK("init_vel_calc");

        momentum_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("momentum_setv");

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dev_initdens));

        validate_finite_state();

        // create the output directory and save the initial state
        std::filesystem::create_directories(PATH);
        if (!save_variable(PATH + "variables.txt"))
        {
            std::cerr << "Error: failed to save simulation parameters" << std::endl;
            return 1;
        }

        SAVE_DUSTDENS_TO_FILE(idx_from);
        SAVE_DUST_VEL_TO_FILE(idx_from);
        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_from);
        #endif

        msg_output(0);
    }
    else
    {
        // restore density and linear velocity output from the selected frame
        std::stringstream ss{argv[1]};
        if (!(ss >> idx_from))
        {
            std::cerr << "Error: invalid resume frame number: " << argv[1] << "\n";
            return 1;
        }

        LOAD_DUSTDATA_TO_VRAM(idx_from);

        momentum_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("momentum_setv");

        CUDA_CHECK(cudaDeviceSynchronize());

        #ifdef RADIATION
        // reconstruct optical depth from the restored density
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");

        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");

        CUDA_CHECK(cudaDeviceSynchronize());
        #endif

        validate_finite_state();

        msg_output(idx_from);
    }

    clock_sim = static_cast<real>(idx_from)*DT_OUT;

    msg_step_title();

    // recover synchronized primitives after every conservative operator
    auto recover_dust_velocity = [&]()
    {
        momentum_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("momentum_getv");
    };

    // recompute the global transport timestep from the current state
    auto recalc_dt_cfl = [&](bool verbose)
    {
        cfl_rate_calc <<< NB_X, TPB >>> (
            dev_cfl_rate, dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("cfl_rate_calc");

        CUDA_CHECK(cudaDeviceSynchronize());

        return get_dt_cfl(dev_cfl_rate, dev_dustvelx, dev_dustvely, dev_dustvelz, verbose);
    };

    // advance each directional transport operator with fresh CFL-limited substeps
    auto advance_x = [&](real time_interval)
    {
        real time_remain = time_interval;
        while (time_remain > 0.0)
        {
            real dt_sub = std::fmin(time_remain, recalc_dt_cfl(false));

            advect_x_calc <<< NB_X, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt_sub
            );
            CUDA_KERNEL_CHECK("advect_x_calc");

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

            advect_y_calc <<< NB_Y, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dt_sub
            );
            CUDA_KERNEL_CHECK("advect_y_calc");

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

            advect_z_calc <<< NB_Z, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dt_sub
            );
            CUDA_KERNEL_CHECK("advect_z_calc");

            recover_dust_velocity();
            time_remain -= dt_sub;
        }
    };

    while (idx_from < SAVE_MAX)
    {
        // clip the global step at the next output time
        real dt_cfl_begin = recalc_dt_cfl(true);
        real dt = dt_cfl_begin;
        real dt_to_out = DT_OUT - clock_out;
        bool output_due = (dt >= dt_to_out);
        if (output_due) dt = dt_to_out;

        // apply the opening half of the symmetric diffusion composition
        #ifdef DIFFUSION
        diffus_y_calc <<< NB_Y, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_y_calc");

        diffus_x_calc <<< NB_X, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_x_calc");

        diffus_z_calc <<< NB_Z, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_z_calc");

        momentum_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("momentum_getv");
        #endif

        // apply the opening half of the symmetric directional transport composition
        real adv_interval = 0.5*dt;

        advance_x(adv_interval);
        advance_y(adv_interval);

        if (N_Z > 1)
        {
            advance_z(adv_interval);
        }

        // evaluate optical depth at the source-step midpoint state
        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");

        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");

        validate_finite_state();
        #endif

        // ramp radiation pressure smoothly during the configured startup interval
        #ifdef RADIATION
        real taper = (T_BETA > 0.0) ? (clock_sim + 0.5*dt) / T_BETA : 1.0;
        taper = std::fmin(std::fmax(taper, 0.0), 1.0);
        real beta_taper = taper*taper*(3.0 - 2.0*taper);
        #endif

        // advance the centred source operator and synchronize conserved momentum
        source_update <<< NB_A, TPB >>> (
            dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens,
            #ifdef RADIATION
            dev_optdepth, beta_taper,
            #endif
            dt
        );
        CUDA_KERNEL_CHECK("source_update");

        momentum_setv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("momentum_setv");

        // close the symmetric directional transport composition in reverse order
        if (N_Z > 1)
        {
            advance_z(adv_interval);
        }

        advance_y(adv_interval);
        advance_x(adv_interval);

        // close the symmetric diffusion composition in reverse order
        #ifdef DIFFUSION
        diffus_z_calc <<< NB_Z, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_z_calc");

        diffus_x_calc <<< NB_X, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_x_calc");

        diffus_y_calc <<< NB_Y, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, 0.5*dt
        );
        CUDA_KERNEL_CHECK("diffus_y_calc");

        momentum_getv <<< NB_A, TPB >>> (
            dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("momentum_getv");
        #endif

        validate_finite_state();

        CUDA_CHECK(cudaDeviceSynchronize());

        // advance simulation and output clocks after the accepted step
        clock_sim += dt;
        clock_out += dt;

        msg_step(idx_from, dt, clock_out, clock_sim);

        if (output_due)
        {
            // refresh derived output fields and save the completed frame
            idx_from++;
            clock_out = 0.0;

            #ifdef RADIATION
            optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
            CUDA_KERNEL_CHECK("optdepth_calc");

            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_csum");

            validate_finite_state();
            #endif

            SAVE_DUSTDENS_TO_FILE(idx_from);
            SAVE_DUST_VEL_TO_FILE(idx_from);
            #ifdef RADIATION
            SAVE_OPTDEPTH_TO_FILE(idx_from);
            #endif

            msg_output(idx_from);
        }
    }

    // release all persistent host and device allocations
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
    #endif

    return 0;
}
