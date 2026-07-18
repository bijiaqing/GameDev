#include <cmath>        // for std::ceil, std::fmin, std::fmax
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

    int *dev_bad_state;
    CUDA_CHECK(cudaMalloc((void**)&dev_bad_state, sizeof(int)));

    // Geometry-aware finite-volume PPM weights are fixed by the mesh and reused by every
    // radial and polar sweep.  Four coefficients are stored for each cell face.
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

    auto validate_finite_state = [&](bool check_optdepth)
    {
        CUDA_CHECK(cudaMemset(dev_bad_state, 0, sizeof(int)));
        state_finite_check <<< NB_A, TPB >>> (
            dev_bad_state,
            dev_dustdens,
            dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_dustvelx, dev_dustvely, dev_dustvelz
            #ifdef RADIATION
            , dev_optdepth, check_optdepth
            #endif
        );
        CUDA_KERNEL_CHECK("state_finite_check");

        int bad_state = 0;
        CUDA_CHECK(cudaMemcpy(&bad_state, dev_bad_state, sizeof(int), cudaMemcpyDeviceToHost));
        if (bad_state != 0)
        {
            int idx_bad = bad_state - 1;
            int ix_bad = idx_bad % N_X;
            int iy_bad = (idx_bad / N_X) % N_Y;
            int iz_bad = idx_bad / (N_X*N_Y);
            std::cerr << "Error: non-finite simulation state at cell ("
                      << ix_bad << "," << iy_bad << "," << iz_bad << ").\n";
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
        real *conv_pow, *dev_conv_pow;
        CUDA_CHECK(cudaMallocHost((void**)&conv_pow, sizeof(real)*(N_Y + 1)));
        CUDA_CHECK(cudaMalloc((void**)&dev_conv_pow, sizeof(real)*(N_Y + 1)));
        
        convpow_calc(conv_pow);
        CUDA_CHECK(cudaMemcpy(dev_conv_pow, conv_pow, sizeof(real)*(N_Y + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(conv_pow));

        f_rho_initial <<< NB_A, TPB >>> (dev_dustdens, dev_conv_pow);
        CUDA_KERNEL_CHECK("f_rho_initial");

        #ifdef RADIATION
        // Build the initial optical depth from rho_d before velocities and momenta are finalized.
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        CUDA_CHECK(cudaDeviceSynchronize());
        #endif // RADIATION

        f_vel_initial <<< NB_A, TPB >>> (dev_dustvelx, dev_dustvely, dev_dustvelz);
        CUDA_KERNEL_CHECK("f_vel_initial");
        f_moment_sync <<< NB_A, TPB >>> (
            dev_dustdens, 
            dev_dustvelx, dev_dustvely, dev_dustvelz,
            dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_sync");
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dev_conv_pow));

        validate_finite_state(
            #ifdef RADIATION
            true
            #else
            false
            #endif
        );

        std::filesystem::create_directories(PATH);
        if (!save_variable(PATH + "variables.txt"))
        {
            std::cerr << "Error: failed to save simulation parameters.\n";
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

        // Reconstruct momentum densities from loaded rho and v_d (not saved separately)
        f_moment_sync <<< NB_A, TPB >>> (
            dev_dustdens, 
            dev_dustvelx, dev_dustvely, dev_dustvelz,
            dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_sync");
        CUDA_CHECK(cudaDeviceSynchronize());

        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        CUDA_CHECK(cudaDeviceSynchronize());
        #endif // RADIATION

        validate_finite_state(
            #ifdef RADIATION
            true
            #else
            false
            #endif
        );

        msg_output(idx_from);
    }

    // Every saved frame is separated by the fixed interval DT_OUT.
    clock_sim = static_cast<real>(idx_from)*DT_OUT;

    msg_step_title();

    // Keep primitive velocity arrays synchronized with the authoritative conserved state.
    auto recover_dust_velocity = [&]()
    {
        f_moment_recv <<< NB_A, TPB >>> (
            dev_dustdens,
            dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_recv");
    };

    // Kernel completion is required before the host reduction chooses a substep size.
    auto recalc_dt_cfl = [&](bool verbose)
    {
        cfl_rate_calc <<< NB_X, TPB >>> (
            dev_cfl_rate, dev_dustdens,
            dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("cfl_rate_calc");
        CUDA_CHECK(cudaDeviceSynchronize());
        return get_dt_cfl(dev_cfl_rate, dev_dustvelx, dev_dustvely, dev_dustvelz, verbose);
    };

    while (idx_from < SAVE_MAX) // main simulation loop
    {
        // ---- Adaptive CFL timestep ----
        real dt_cfl_begin = recalc_dt_cfl(true);
        real dt = dt_cfl_begin;

        // Cap to remaining time before next output — no overshoot, no artificial floor
        real dt_to_out = DT_OUT - clock_out;
        if (dt > dt_to_out) dt = dt_to_out;

        // Palindromic diffusion half-step.  The matching reverse half-step follows dynamics.
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
        f_moment_recv <<< NB_A, TPB >>> (
            dev_dustdens,
            dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_recv");
        #endif // DIFFUSION

        // =====================================================
        // Symmetric diffusion/advection/source composition:
        // D_y D_x D_z | A_x A_y A_z | source | A_z A_y A_x | D_z D_x D_y
        // Every D and A entry is a half-step; source is a full step.
        // =====================================================

        // ---- First half advection (X Y Z): each operator covers exactly dt/2 ----
        // Refresh the bound only between directional operators.  The substep counts may
        // differ, but n_dir*dt_dir = dt/2 for every active direction.
        real adv_interval = 0.5*dt;

        if (N_X > 1)
        {
            #ifdef DIFFUSION
            real dt_cfl_x_fwd = recalc_dt_cfl(false);
            #else
            // No operator has changed the beginning-of-step state, so reuse its bound.
            real dt_cfl_x_fwd = dt_cfl_begin;
            #endif // DIFFUSION
            int n_x_fwd = static_cast<int>(std::ceil(adv_interval / dt_cfl_x_fwd));
            if (n_x_fwd < 1) n_x_fwd = 1;
            real dt_x_fwd = adv_interval / static_cast<real>(n_x_fwd);
            for (int iad = 0; iad < n_x_fwd; iad++)
            {
                f_advection_x <<< NB_X, TPB >>> (
                    dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt_x_fwd
                );
                CUDA_KERNEL_CHECK("f_advection_x");
            }

            recover_dust_velocity();
        }

        real dt_cfl_y_fwd = recalc_dt_cfl(false);
        int n_y_fwd = static_cast<int>(std::ceil(adv_interval / dt_cfl_y_fwd));
        if (n_y_fwd < 1) n_y_fwd = 1;
        real dt_y_fwd = adv_interval / static_cast<real>(n_y_fwd);
        for (int iad = 0; iad < n_y_fwd; iad++)
        {
            f_advection_y <<< NB_Y, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dt_y_fwd
            );
            CUDA_KERNEL_CHECK("f_advection_y");
        }

        recover_dust_velocity();
        if (N_Z > 1)
        {
            real dt_cfl_z_fwd = recalc_dt_cfl(false);
            int n_z_fwd = static_cast<int>(std::ceil(adv_interval / dt_cfl_z_fwd));
            if (n_z_fwd < 1) n_z_fwd = 1;
            real dt_z_fwd = adv_interval / static_cast<real>(n_z_fwd);
            for (int iad = 0; iad < n_z_fwd; iad++)
            {
                f_advection_z <<< NB_Z, TPB >>> (
                    dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dt_z_fwd
                );
                CUDA_KERNEL_CHECK("f_advection_z");
            }

            // Recover v_d from midpoint momentum: v_d^{n+1/2} = mom^{n+1/2} / rho^{n+1/2}
            recover_dust_velocity();
        }

        // ---- Recompute τ from midpoint density ρ_d^{n+1/2} ----
        #ifdef RADIATION
        optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
        CUDA_KERNEL_CHECK("optdepth_calc");
        optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
        CUDA_KERNEL_CHECK("optdepth_csum");
        validate_finite_state(true);
        #endif // RADIATION

        // ---- Full source step (exponential drag/force update, uses tau^{n+1/2}) ----
        #ifdef RADIATION
        // C1-smooth turn-on from zero to full radiation over T_BETA
        // Evaluate the prescribed time dependence at the source step's temporal midpoint
        real taper = (clock_sim + 0.5*dt) / T_BETA;
        taper = std::fmin(std::fmax(taper, 0.0), 1.0);
        real beta_taper = taper*taper*(3.0 - 2.0*taper);
        #endif // RADIATION

        f_source_term <<< NB_A, TPB >>> (dev_dustvelx, dev_dustvely, dev_dustvelz, dev_dustdens,
            #ifdef RADIATION
            dev_optdepth, beta_taper,
            #endif // RADIATION
            dt
        );
        CUDA_KERNEL_CHECK("f_source_term");
        // Sync moms = rho * v_d_new before second half-advection
        f_moment_sync <<< NB_A, TPB >>> (
            dev_dustdens, 
            dev_dustvelx, dev_dustvely, dev_dustvelz,
            dev_dustmomx, dev_dustmomy, dev_dustmomz
        );
        CUDA_KERNEL_CHECK("f_moment_sync");

        // ---- Second half advection (Z Y X): independently bounded directional operators ----
        // The source and each preceding sweep may change the next sweep's transport velocity.
        if (N_Z > 1)
        {
            real dt_cfl_z_rev = recalc_dt_cfl(false);
            int n_z_rev = static_cast<int>(std::ceil(adv_interval / dt_cfl_z_rev));
            if (n_z_rev < 1) n_z_rev = 1;
            real dt_z_rev = adv_interval / static_cast<real>(n_z_rev);
            for (int iad = 0; iad < n_z_rev; iad++)
            {
                f_advection_z <<< NB_Z, TPB >>> (
                    dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_z, dt_z_rev
                );
                CUDA_KERNEL_CHECK("f_advection_z");
            }

            recover_dust_velocity();
        }
        real dt_cfl_y_rev = recalc_dt_cfl(false);
        int n_y_rev = static_cast<int>(std::ceil(adv_interval / dt_cfl_y_rev));
        if (n_y_rev < 1) n_y_rev = 1;
        real dt_y_rev = adv_interval / static_cast<real>(n_y_rev);
        for (int iad = 0; iad < n_y_rev; iad++)
        {
            f_advection_y <<< NB_Y, TPB >>> (
                dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dev_weight_y, dt_y_rev
            );
            CUDA_KERNEL_CHECK("f_advection_y");
        }

        recover_dust_velocity();
        if (N_X > 1)
        {
            real dt_cfl_x_rev = recalc_dt_cfl(false);
            int n_x_rev = static_cast<int>(std::ceil(adv_interval / dt_cfl_x_rev));
            if (n_x_rev < 1) n_x_rev = 1;
            real dt_x_rev = adv_interval / static_cast<real>(n_x_rev);
            for (int iad = 0; iad < n_x_rev; iad++)
            {
                f_advection_x <<< NB_X, TPB >>> (
                    dev_dustdens, dev_dustmomx, dev_dustmomy, dev_dustmomz, dt_x_rev
                );
                CUDA_KERNEL_CHECK("f_advection_x");
            }

            // Recover final v_d^{n+1} = mom^{n+1} / rho^{n+1}
            recover_dust_velocity();
        }

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

        // Recover primitive velocity from the density and momentum transported by diffusion.
        f_moment_recv <<< NB_A, TPB >>> (
            dev_dustdens,
            dev_dustmomx, dev_dustmomy, dev_dustmomz,
            dev_dustvelx, dev_dustvely, dev_dustvelz
        );
        CUDA_KERNEL_CHECK("f_moment_recv");
        #endif // DIFFUSION

        // Validate the final dust state before advancing clocks or writing an output frame.
        validate_finite_state(false);

        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- Advance clocks ----
        clock_sim += dt;
        clock_out += dt;

        msg_step(idx_from, dt, clock_out, clock_sim);

        // ---- Output ----
        if (clock_out >= DT_OUT - OUTPUT_TIME_TOL)
        {
            idx_from++;
            clock_out = 0.0;

            // Recompute τ for output
            #ifdef RADIATION
            optdepth_calc <<< NB_A, TPB >>> (dev_optdepth, dev_dustdens);
            CUDA_KERNEL_CHECK("optdepth_calc");
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_csum");
            validate_finite_state(true);
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
    CUDA_CHECK(cudaFree(dev_bad_state));
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
