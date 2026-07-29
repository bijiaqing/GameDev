#include <chrono>           // std::chrono::system_clock
#include <filesystem>       // std::filesystem::create_directories
#include <iomanip>          // std::setw, std::setfill
#include <iostream>         // std::cout, std::endl
#include <sstream>          // std::stringstream
#include <stdexcept>        // std::runtime_error

#if defined(TRANSPORT) || defined(COLLISION)
#include <thrust/device_ptr.h>  // thrust::device_ptr
#include <thrust/extrema.h>     // thrust::max_element
#endif // TRANSPORT || COLLISION

#include <swarm_host.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <adaptive_morton.cuh>
#endif // COLLISION_MORTON

std::mt19937 rand_generator;

const std::string PATH = PATH_OUT; // convert the Makefile string literal to the output-path string used below

// =========================================================================================================================
// main program
// initialize or resume a swarm and advance enabled operators between successive output frames
//
// combined dynamics sequence:
//   1 half collision step
//   2 half spatial-diffusion step
//   3 full staggered semi-analytic transport step with optional midpoint radiation reconstruction
//   4 half spatial-diffusion step
//   5 half collision step
// =========================================================================================================================

int main (int argc, char **argv)
{
    const real total_dust_mass = get_total_dust_mass();

    int idx_from;
    real clock_sim;   // total simulated time
    real clock_out;   // elapsed time in the current output interval
    real dt_out;      // duration of the current output interval

    #ifdef TRANSPORT
    int count_dyn;    // dynamics steps completed in the current output interval
    real dt_dyn;      // current dynamics timestep
    #endif // TRANSPORT
    
    #ifdef COLLISION
    int count_col;    // collision batches completed in the current dynamics interval
    real clock_dyn;   // elapsed collision time in the current dynamics interval
    real dt_col;      // current collision-batch timestep
    #endif // COLLISION
    
    // allocate the particle state and feature-dependent work arrays
    swarm *particle, *dev_particle;
    CUDA_CHECK(cudaMallocHost((void**)&particle, sizeof(swarm)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_P));

    #ifdef TRANSPORT
    real *dev_dt_rate;
    CUDA_CHECK(cudaMalloc((void**)&dev_dt_rate, sizeof(real)*N_P));
    #endif // TRANSPORT
    
    #ifdef SAVE_DENS
    real *dustdens, *dev_dustdens;
    CUDA_CHECK(cudaMallocHost((void**)&dustdens, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_G));
    #endif // SAVE_DENS
    
    #ifdef IMPORTGAS
    real *gas_dens, *dev_gas_dens;
    CUDA_CHECK(cudaMallocHost((void**)&gas_dens,  sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_dens,  sizeof(real)*N_G));

    real *gas_velx, *dev_gas_velx;
    CUDA_CHECK(cudaMallocHost((void**)&gas_velx,  sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_velx,  sizeof(real)*N_G));

    real *gas_vely, *dev_gas_vely;
    CUDA_CHECK(cudaMallocHost((void**)&gas_vely,  sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_vely,  sizeof(real)*N_G));

    real *gas_velz, *dev_gas_velz;
    CUDA_CHECK(cudaMallocHost((void**)&gas_velz,  sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_velz,  sizeof(real)*N_G));

    real *dev_gas_dens_next, *dev_gas_velx_next, *dev_gas_vely_next, *dev_gas_velz_next;
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_dens_next, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_velx_next, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_vely_next, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_gas_velz_next, sizeof(real)*N_G));
    #endif // IMPORTGAS
    
    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    CUDA_CHECK(cudaMallocHost((void**)&optdepth, sizeof(real)*N_G));
    CUDA_CHECK(cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
    #endif // RADIATION

    #ifdef COLLISION
    #ifdef COLLISION_KDTREE
    bbox *dev_boundbox;
    CUDA_CHECK(cudaMalloc((void**)&dev_boundbox, sizeof(bbox)));

    tree *dev_col_tree;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_tree, sizeof(tree)*N_T));
    #else  // COLLISION_MORTON
    float3 *dev_col_point;
    unsigned int *dev_col_overflow;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_point, sizeof(float3)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_overflow, sizeof(unsigned int)*N_P));
    adaptive_morton_index col_morton;
    #endif // COLLISION_KDTREE

    real *dev_size_old, *dev_numr_old, *dev_col_rate, *dev_col_dist;
    unsigned long long *dev_col_hash;
    unsigned int *dev_col_count;
    CUDA_CHECK(cudaMalloc((void**)&dev_size_old, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_numr_old, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_dist, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_hash, sizeof(unsigned long long)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_count, sizeof(unsigned int)*N_P));
    #endif // COLLISION

    #if defined(COLLISION) || defined(DIFFUSION)
    curs *dev_rngstate;
    CUDA_CHECK(cudaMalloc((void**)&dev_rngstate, sizeof(curs)*N_P));
    #endif // COLLISION || DIFFUSION

    if (argc <= 1)
	{
        // construct a fresh realization from the configured analytic or imported distribution
        
        idx_from = 0;

        real *randposx, *dev_randposx;
        CUDA_CHECK(cudaMallocHost((void**)&randposx, sizeof(real)*N_P));
        CUDA_CHECK(cudaMalloc((void**)&dev_randposx, sizeof(real)*N_P));

        real *randposy, *dev_randposy;
        CUDA_CHECK(cudaMallocHost((void**)&randposy, sizeof(real)*N_P));
        CUDA_CHECK(cudaMalloc((void**)&dev_randposy, sizeof(real)*N_P));

        real *randposz, *dev_randposz;
        CUDA_CHECK(cudaMallocHost((void**)&randposz, sizeof(real)*N_P));
        CUDA_CHECK(cudaMalloc((void**)&dev_randposz, sizeof(real)*N_P));

        #ifdef MULTISIZE
        real *randsize, *dev_randsize;
        CUDA_CHECK(cudaMallocHost((void**)&randsize, sizeof(real)*N_P));
        CUDA_CHECK(cudaMalloc((void**)&dev_randsize, sizeof(real)*N_P));
        #endif // MULTISIZE

        rand_generator.seed(0); // keep initialization reproducible across runs

        #ifdef MULTISIZE
        // sample grain properties before positions so settled spatial distributions can depend on size
        real power_idx = -0.5;  // equal represented mass per swarm
        #ifdef RADIATION
        power_idx = -1.5;       // equal represented surface area per swarm, see _get_grain_number
        #endif // RADIATION
        #ifdef COLLISION_LINEAR_TEST
        rand_gamma_k2(randsize, N_P);
        #else  // STANDARD_INITIALIZATION
        rand_powerlaw(randsize, N_P, INIT_SMIN, INIT_SMAX, power_idx);
        #endif // COLLISION_LINEAR_TEST

        // correct finite-sample size fluctuations so represented masses sum exactly to total_dust_mass
        real mass_norm = get_mass_norm(randsize, total_dust_mass);
        #endif // MULTISIZE

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);

        real *epsilon;
        CUDA_CHECK(cudaMallocHost((void**)&epsilon,  sizeof(real)*N_G));
        
        if (!load_epsilon(PATH, idx_from, epsilon))
        {
            std::cerr << "Error: Failed to load gas data files for frame " << idx_from << std::endl;
            return 1;
        }

        // use one imported total-dust spatial distribution for all previously sampled grain species
        rand_from_file(randposx, randposy, randposz, N_P, gas_dens, epsilon);
        
        CUDA_CHECK(cudaFreeHost(epsilon));
        #else  // NO IMPORTGAS
        #if defined(MULTISIZE) && defined(DIFFUSION)
        rand_disk_poly(randposx, randposy, randposz, randsize, N_P);
        #else  // !(MULTISIZE && DIFFUSION)
        rand_disk_mono(randposx, randposy, randposz, S_0, N_P);
        #endif // MULTISIZE && DIFFUSION
        #endif // IMPORTGAS

        CUDA_CHECK(cudaMemcpy(dev_randposx, randposx, sizeof(real)*N_P, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_randposy, randposy, sizeof(real)*N_P, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_randposz, randposz, sizeof(real)*N_P, cudaMemcpyHostToDevice));

        #ifdef MULTISIZE
        CUDA_CHECK(cudaMemcpy(dev_randsize, randsize, sizeof(real)*N_P, cudaMemcpyHostToDevice));
        #endif // MULTISIZE

        // convert sampled coordinates and sizes into the device particle state
        particle_init <<< NB_P, TPB >>> (dev_particle, dev_randposx, dev_randposy, dev_randposz
            #ifdef MULTISIZE
            , dev_randsize, mass_norm
            #endif // MULTISIZE
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        CUDA_KERNEL_CHECK("particle_init");

        CUDA_CHECK(cudaFreeHost(randposx));
        CUDA_CHECK(cudaFree(dev_randposx));
        CUDA_CHECK(cudaFreeHost(randposy));
        CUDA_CHECK(cudaFree(dev_randposy));
        CUDA_CHECK(cudaFreeHost(randposz));
        CUDA_CHECK(cudaFree(dev_randposz));

        #ifdef MULTISIZE
        CUDA_CHECK(cudaFreeHost(randsize));
        CUDA_CHECK(cudaFree(dev_randsize));
        #endif // MULTISIZE
        
        #if defined(COLLISION) || defined(DIFFUSION)
        rngstate_init <<< NB_P, TPB >>> (dev_rngstate);
        CUDA_KERNEL_CHECK("rngstate_init");
        #endif // COLLISION || DIFFUSION
        
        // write the initial state and active configuration before evolution
        std::filesystem::create_directories(PATH);
        save_variable(PATH + "variables.txt", total_dust_mass);

        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_from, false);
        #endif // RADIATION

        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_from);
        #endif // SAVE_DENS

        SAVE_PARTICLE_TO_FILE(idx_from);

        msg_output(0);
    }
    else
    {
        // resume particle and imported-gas states from the requested output frame
        std::stringstream frame_stream{argv[1]};

        if (!(frame_stream >> idx_from))
        {
            std::cerr << "Error: Invalid resume file number: " << argv[1] << std::endl;
            return 1;
        }

        LOAD_PARTICLE_TO_VRAM(idx_from);

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);
        #endif // IMPORTGAS
        
        msg_output(idx_from);
    }

    #ifdef LOGTIMING
    clock_sim = (idx_from == 0) ? 0.0 : int_pow(LOG_BASE, idx_from)*DT_OUT;
    #else  // LOGOUTPUT or LINEAR
    clock_sim =                         static_cast<real>(idx_from)*DT_OUT;
    #endif // LOGTIMING

    #ifdef COLLISION
    bool save_knn_probe = true;

    // evolve collisions over a fixed-position interval with controlled frozen-rate Bernoulli batches
    auto evolve_collisions = [&] (real duration, real time_start)
    {
        // collisions change grain properties but not positions, so one search structure serves the full interval
        auto knn_start = std::chrono::steady_clock::now();
        #ifdef COLLISION_KDTREE
        col_tree_init <<< NB_P, TPB >>> (dev_col_tree, dev_particle);
        CUDA_KERNEL_CHECK("col_tree_init");
        cukd::buildTree <tree, tree_traits> (dev_col_tree, N_T, dev_boundbox);
        CUDA_KERNEL_CHECK("cukd::buildTree");
        #else  // COLLISION_MORTON
        col_tree_init <<< NB_P, TPB >>> (dev_col_point, dev_particle);
        CUDA_KERNEL_CHECK("col_tree_init");

        float root_width = 2.0002f*static_cast<float>(Y_MAX);
        float3 root_origin = make_float3(
            -0.5f*root_width, -0.5f*root_width, -0.5f*root_width
        );
        col_morton.build(
            dev_col_point, N_P, root_origin, root_width,
            (N_Z > 1) ? 3 : 2, MORTON_LEAF, MORTON_LEVEL
        );
        #endif // COLLISION_KDTREE
        CUDA_CHECK(cudaDeviceSynchronize());
        double knn_build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - knn_start
        ).count();

        #ifdef COLLISION_KDTREE
        std::size_t knn_bytes = sizeof(tree)*static_cast<std::size_t>(N_T) + sizeof(bbox);
        std::cout << "[KNN] backend=kdtree";
        #else  // COLLISION_MORTON
        std::size_t knn_bytes = col_morton.persistent_bytes()
            + (sizeof(float3) + sizeof(unsigned int))*static_cast<std::size_t>(N_P);
        std::cout << "[KNN] backend=morton";
        #endif // COLLISION_KDTREE
        std::cout
            << " build_ms=" << knn_build_ms
            << " persistent_bytes=" << knn_bytes
            << std::endl;

        real elapsed = 0.0;
        while (elapsed < duration)
        {
            // freeze only the species fields changed by collisions while positions and velocities remain fixed
            col_snap_save <<< NB_P, TPB >>> (dev_size_old, dev_numr_old, dev_particle);
            CUDA_KERNEL_CHECK("col_snap_save");
            CUDA_CHECK(cudaDeviceSynchronize());
            knn_start = std::chrono::steady_clock::now();
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< NB_T, TPB >>> (
                dev_col_rate, dev_col_dist, dev_col_hash, dev_col_count, dev_particle,
                dev_size_old, dev_numr_old, dev_col_tree, dev_boundbox,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / (N_K - 1.0) / total_dust_mass, save_knn_probe
            );
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
                dev_col_rate, dev_col_dist, dev_col_hash, dev_col_count, dev_col_overflow, dev_particle,
                dev_size_old, dev_numr_old, dev_col_point, col_morton.view(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / (N_K - 1.0) / total_dust_mass, save_knn_probe
            );
            #endif // COLLISION_KDTREE
            CUDA_KERNEL_CHECK("col_rate_calc");

            #ifdef COLLISION_MORTON
            thrust::device_ptr <const unsigned int> col_overflow_ptr(dev_col_overflow);
            unsigned int max_col_overflow = *thrust::max_element(
                col_overflow_ptr, col_overflow_ptr + N_P
            );
            if (max_col_overflow != 0)
                throw std::runtime_error("adaptive Morton traversal stack overflow in col_rate_calc");
            #endif // COLLISION_MORTON

            // use the largest total propensity to control every representative's event probability
            thrust::device_ptr <const real> col_rate_ptr(dev_col_rate);
            real max_col_rate = *thrust::max_element(col_rate_ptr, col_rate_ptr + N_P);
            double knn_rate_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - knn_start
            ).count();

            if (save_knn_probe)
            {
                if (!save_device_binary(PATH + "col_rate_probe.dat", dev_col_rate, N_P)
                    || !save_device_binary(PATH + "col_dist_probe.dat", dev_col_dist, N_P)
                    || !save_device_binary(PATH + "col_hash_probe.dat", dev_col_hash, N_P)
                    || !save_device_binary(PATH + "col_count_probe.dat", dev_col_count, N_P))
                    throw std::runtime_error("failed to save the first collision-rate probe");
                save_knn_probe = false;
            }

            real remaining = duration - elapsed;

            if (!(max_col_rate > 0.0))
            {
                // consume the remaining interval when no collision channel is active
                dt_col = remaining;
                elapsed = duration;
                clock_dyn = elapsed;
                std::cout
                    << "[PROGRESS] phase=collision"
                    << " time=" << std::scientific << std::setprecision(6) << time_start + elapsed
                    << " dt=" << dt_col
                    << " batch=" << count_col
                    << " operator_elapsed=" << elapsed
                    << " operator_duration=" << duration
                    << " status=no_active_rate"
                    << std::endl;
                break;
            }

            // keep the fastest frozen propensity below CFL_COL before sampling one event at most
            dt_col = fmin(CFL_COL / max_col_rate, remaining);
            knn_start = std::chrono::steady_clock::now();
            #ifdef COLLISION_KDTREE
            col_event_run <<< NB_T, TPB >>> (dev_particle, dev_rngstate, dev_col_rate, dev_col_dist,
                dev_size_old, dev_numr_old, dev_col_tree, dev_boundbox,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / (N_K - 1.0) / total_dust_mass,
                dt_col
            );
            #else  // COLLISION_MORTON
            col_event_run <<< N_P, MORTON_TPB >>> (
                dev_particle, dev_rngstate, dev_col_rate, dev_col_dist, dev_col_overflow,
                dev_size_old, dev_numr_old, dev_col_point, col_morton.view(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / (N_K - 1.0) / total_dust_mass,
                dt_col
            );
            #endif // COLLISION_KDTREE
            CUDA_KERNEL_CHECK("col_event_run");
            CUDA_CHECK(cudaDeviceSynchronize());

            #ifdef COLLISION_MORTON
            max_col_overflow = *thrust::max_element(col_overflow_ptr, col_overflow_ptr + N_P);
            if (max_col_overflow != 0)
                throw std::runtime_error("adaptive Morton traversal stack overflow in col_event_run");
            #endif // COLLISION_MORTON
            double knn_event_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - knn_start
            ).count();
            std::cout
                << "[KNN] rate_ms=" << knn_rate_ms
                << " event_ms=" << knn_event_ms
                << " max_rate=" << max_col_rate
                << std::endl;

            elapsed += dt_col;
            clock_dyn = elapsed;
            count_col++;
            std::cout
                << "[PROGRESS] phase=collision"
                << " time=" << std::scientific << std::setprecision(6) << time_start + elapsed
                << " dt=" << dt_col
                << " batch=" << count_col
                << " operator_elapsed=" << elapsed
                << " operator_duration=" << duration
                << std::endl;
        }
    };
    #endif // COLLISION

    for (int idx_file = idx_from + 1; idx_file <= SAVE_MAX; idx_file++)
    {
        // preload the next external frame and advance exactly one output interval
        dt_out = _get_dt_out(idx_file);

        #ifdef IMPORTGAS
        LOAD_GAS_NEXT_TO_VRAM(idx_file);
        real gas_frac = 0.0;
        #endif // IMPORTGAS
        
        clock_out = 0.0;
        
        #ifdef TRANSPORT
        count_dyn = 0;
        #endif // TRANSPORT

        PRINT_TITLE_TO_SCREEN();
        
        do
        {
            #ifdef TRANSPORT
            // reduce all local inverse rates to a globally valid dynamics timestep
            dyn_rate_calc <<< NB_P, TPB >>> (dev_dt_rate, dev_particle
                #ifdef IMPORTGAS
                , dev_gas_velx, dev_gas_vely, dev_gas_velz
                , dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next
                #endif // IMPORTGAS
            );
            CUDA_KERNEL_CHECK("dyn_rate_calc");
            thrust::device_ptr <const real> dt_rate_ptr(dev_dt_rate);
            real max_dt_rate = *thrust::max_element(dt_rate_ptr, dt_rate_ptr + N_P);
            dt_dyn = fmin(DT_MAX, fmin(1.0 / max_dt_rate, dt_out - clock_out));
            std::cout
                << "[PROGRESS] phase=dynamics"
                << " time=" << std::scientific << std::setprecision(6) << clock_sim
                << " dt=" << dt_dyn
                << " target=" << clock_sim + dt_dyn
                << " step=" << count_dyn + 1
                << std::endl;

            #ifdef IMPORTGAS
            // interpolate the working gas fields to the midpoint time of this dynamics step
            real gas_target = (clock_out + 0.5*dt_dyn) / dt_out;
            real gas_blend = (gas_target - gas_frac) / (1.0 - gas_frac);
            gas_lerp_calc <<< NB_G, TPB >>> (
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, gas_blend
            );
            CUDA_KERNEL_CHECK("gas_lerp_calc");
            gas_frac = gas_target;
            #endif // IMPORTGAS

            #ifdef COLLISION
            // begin the symmetric composition with half a collision interval
            count_col = 0;
            clock_dyn = 0.0;
            evolve_collisions(0.5*dt_dyn, clock_sim);
            #endif // COLLISION

            #ifdef DIFFUSION
            // apply the first half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn);
            CUDA_KERNEL_CHECK("diffusion_pos");
            #endif // DIFFUSION

            #ifdef RADIATION
            // drift to midpoint positions and reconstruct the optical depth used by the force solve
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            CUDA_KERNEL_CHECK("ssa_substep_1");
            optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_init");
            optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, total_dust_mass);
            CUDA_KERNEL_CHECK("optdepth_depo");
            optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_calc");
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            CUDA_KERNEL_CHECK("optdepth_csum");

            real taper = (T_BETA > 0.0) ? (clock_sim + 0.5*dt_dyn) / T_BETA : 1.0;
            taper = fmin(fmax(taper, 0.0), 1.0);
            real beta_taper = taper*taper*(3.0 - 2.0*taper);

            ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                beta_taper,
                dt_dyn
            );
            CUDA_KERNEL_CHECK("ssa_substep_2");
            #else  // NO RADIATION
            // complete transport in one launch when no midpoint radiation field is required
            ssa_transport <<< NB_P, TPB >>> (dev_particle,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                dt_dyn
            );
            CUDA_KERNEL_CHECK("ssa_transport");
            #endif // RADIATION

            #ifdef DIFFUSION
            // apply the second half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn);
            CUDA_KERNEL_CHECK("diffusion_pos");
            #endif // DIFFUSION

            #ifdef COLLISION
            // close the symmetric composition with half a collision interval
            evolve_collisions(0.5*dt_dyn, clock_sim + 0.5*dt_dyn);
            #endif // COLLISION

            CUDA_CHECK(cudaDeviceSynchronize());
            clock_sim += dt_dyn;
            clock_out += dt_dyn;
            count_dyn++;
            PRINT_VALUE_TO_SCREEN();
            #endif // TRANSPORT

            #if defined(COLLISION) && !defined(TRANSPORT)
            // collision-only runs evolve directly across the complete output interval
            count_col = 0;
            clock_dyn = 0.0;
            real duration = dt_out - clock_out;
            
            #ifdef IMPORTGAS
            real gas_target = (clock_out + 0.5*duration) / dt_out;
            real gas_blend = (gas_target - gas_frac) / (1.0 - gas_frac);
            gas_lerp_calc <<< NB_G, TPB >>> (
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, gas_blend
            );
            CUDA_KERNEL_CHECK("gas_lerp_calc");
            gas_frac = gas_target;
            #endif // IMPORTGAS
            
            evolve_collisions(duration, clock_sim);
            clock_out += duration;
            clock_sim += duration;
            PRINT_VALUE_TO_SCREEN();
            #endif // COLLISION && !TRANSPORT
        } while (clock_out < dt_out);

        #ifdef IMPORTGAS
        // replace the incrementally blended working fields by the exact endpoint snapshot
        CUDA_CHECK(cudaMemcpy(dev_gas_dens, dev_gas_dens_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(dev_gas_velx, dev_gas_velx_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(dev_gas_vely, dev_gas_vely_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(dev_gas_velz, dev_gas_velz_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice));
        #endif // IMPORTGAS

        // reconstruct requested mesh fields and save particle frames under the configured output cadence
        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_file, false);
        #endif // RADIATION
    
        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_file);
        #endif // SAVE_DENS

        #ifdef LOGTIMING
        SAVE_PARTICLE_TO_FILE(idx_file);
        #elif defined(LOGOUTPUT)
        if (is_log_power(idx_file)) SAVE_PARTICLE_TO_FILE(idx_file);
        #else  // LINEAR_OUTPUT
        if (idx_file % LIN_BASE == 0) SAVE_PARTICLE_TO_FILE(idx_file);
        #endif // LOGTIMING

        msg_output(idx_file);
    }
 
    return 0;
}

// =========================================================================================================================
