#define COL_DIAGNOSTICS // Validation retains collision audit and event records.
// VAL runtime override: retain deterministic seeds, geometry accounting, timing, and reduced cache diagnostics outside src/
#include <chrono>           // std::chrono clocks and durations
#include <cmath>            // std::fabs, std::fmin, std::sin
#include <cstdlib>          // EXIT_FAILURE, std::exit
#include <filesystem>       // std::filesystem::create_directories
#include <fstream>          // std::ofstream
#include <iomanip>          // std::setw, std::setfill, std::setprecision
#include <iostream>         // std::cout, std::endl
#include <limits>           // std::numeric_limits
#include <sstream>          // std::stringstream
#include <stdexcept>        // std::runtime_error
#include <string>           // std::string, std::to_string
#include <vector>           // std::vector

#if defined(TRANSPORT) || defined(COLLISION)
#include <thrust/device_ptr.h>  // thrust::device_ptr
#include <thrust/extrema.h>     // thrust::max_element
#endif // TRANSPORT || COLLISION

#include <swarm_host.cuh>
#include <swarm_kern.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)
#include <_col_chain.cuh>
#include <local_gpu.cuh>
#endif // COLLISION && !BERNOULLI

#ifdef KNN_CACHE
#include <_col_cache.cuh>
#endif // KNN_CACHE

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
#endif // COLLISION_MORTON

#if defined(COL_PERF_VAL) && !defined(COL_GEOM_VAL)
#error "COL_PERF_VAL requires COL_GEOM_VAL"
#endif

#if defined(COL_CACHE_VAL) && (!defined(COLLISION) || (defined(BERNOULLI) && !defined(KNN_CACHE)))
#error "COL_CACHE_VAL requires a collision integrator with a persistent neighbor cache"
#endif

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
    #ifdef HALF_DISK
    if (N_Z > 1 && std::fabs(Z_MAX - 0.5*M_PI) > 16.0*std::numeric_limits<real>::epsilon())
        throw std::runtime_error("HALF_DISK requires Z_MAX = pi/2");
    #endif // HALF_DISK

    std::vector <real> mass_bank;
    initmass_calc(mass_bank);
    const real total_dust_mass = get_total_dust_mass(mass_bank);

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
    real *dev_dyn_rate;
    CUDA_CHECK(cudaMalloc((void**)&dev_dyn_rate, sizeof(real)*N_P));
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
    unsigned char *dev_col_active;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_active, sizeof(unsigned char)*N_P));

    int *dev_bad_part;
    CUDA_CHECK(cudaMalloc((void**)&dev_bad_part, sizeof(int)));

    #ifdef COLLISION_KDTREE
    kdtree_boxf *dev_kdtree_box;
    CUDA_CHECK(cudaMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)));

    kdtree_node *dev_kdtree_node;
    CUDA_CHECK(cudaMalloc((void**)&dev_kdtree_node, sizeof(kdtree_node)*N_T));
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx, *dev_search_dist;
    unsigned int *dev_morton_overflow;
    CUDA_CHECK(cudaMalloc((void**)&dev_morton_point, sizeof(float3)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_morton_posx, sizeof(float)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_search_dist, sizeof(float)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_morton_overflow, sizeof(unsigned int)*N_P));
    morton_ghost_index morton_owner;
    #endif // COLLISION_KDTREE

    real *dev_size_old, *dev_numr_old, *dev_col_rate;
    CUDA_CHECK(cudaMalloc((void**)&dev_size_old, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_numr_old, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_P));

    #if !defined(BERNOULLI) || defined(KNN_CACHE)
    const std::size_t col_neighbor_count = static_cast<std::size_t>(N_P)*N_K;
    int *dev_col_neighbor;
    real *dev_col_measure;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_neighbor, sizeof(int)*col_neighbor_count));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_measure, sizeof(real)*N_P));

    #ifdef COL_CACHE_VAL
    col_cache_val_sample *dev_cacheval;
    CUDA_CHECK(cudaMalloc((void**)&dev_cacheval, sizeof(col_cache_val_sample)*N_P));
    #endif // COL_CACHE_VAL
    #endif // FROZEN_BATH || KNN_CACHE

    #ifndef BERNOULLI
    const int col_raw_count = _get_col_raw_count();
    int *dev_col_events, *dev_col_spatial;
    int *dev_col_count, *dev_col_binmap, *dev_col_error, *dev_col_unfinished;
    real *dev_col_time, *dev_col_hazard;
    real *dev_col_jump1_int, *dev_col_jump2_int, *dev_col_jumpmax_int;
    unsigned char *dev_col_complete;
    col_rate_bin *dev_col_ratebin;
    col_audit_accum *dev_col_audit;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_events, sizeof(int)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_spatial, sizeof(int)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_count, sizeof(int)*col_raw_count));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_binmap, sizeof(int)*col_raw_count));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_error, sizeof(int)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_unfinished, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_time, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_hazard, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_jump1_int, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_jump2_int, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_jumpmax_int, sizeof(real)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_complete, sizeof(unsigned char)*N_P));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_ratebin, sizeof(col_rate_bin)*col_raw_count));
    CUDA_CHECK(cudaMalloc((void**)&dev_col_audit, sizeof(col_audit_accum)*col_raw_count));
    CUDA_CHECK(cudaMemset(dev_col_error, 0, sizeof(int)*N_P));
    #elif !defined(KNN_CACHE)  // DIRECT_BERNOULLI
    real *dev_col_dist;
    CUDA_CHECK(cudaMalloc((void**)&dev_col_dist, sizeof(real)*N_P));
    #endif // FROZEN_BATH / KNN_CACHE / DIRECT_BERNOULLI
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

        real *dev_mass_bank;
        CUDA_CHECK(cudaMalloc((void**)&dev_mass_bank, sizeof(real)*mass_bank.size()));
        CUDA_CHECK(cudaMemcpy(dev_mass_bank, mass_bank.data(), sizeof(real)*mass_bank.size(), cudaMemcpyHostToDevice));
        #endif // MULTISIZE

        unsigned int init_seed = 0;
        #ifdef VAL_RUNTIME_SEED
        // vary the sampled particle population independently of the device collision stream
        if (const char *seed_text = std::getenv("GAMEDEV_VAL_INIT_SEED"))
            init_seed = static_cast<unsigned int>(std::stoul(seed_text));
        #endif // VAL_RUNTIME_SEED
        rand_generator.seed(init_seed);

        #ifdef MULTISIZE
        // sample grain properties before positions so settled spatial distributions can depend on size
        real power_idx = -0.5;  // full-column equal-mass sampling proposal
        #ifdef RADIATION
        power_idx = -1.5;       // full-column equal-area sampling proposal, see _get_grain_number
        #endif // RADIATION
        #ifdef COLLISION_LINEAR_TEST
        rand_gamma_k2(randsize, N_P);
        #else  // STANDARD_INITIALIZATION
        rand_powerlaw(randsize, N_P, INIT_SMIN, INIT_SMAX, power_idx);
        #endif // COLLISION_LINEAR_TEST

        // correct size sampling and finite-domain containment so represented masses sum exactly to total_dust_mass
        real mass_norm = get_mass_norm(randsize, mass_bank, total_dust_mass);
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
            , dev_randsize, dev_mass_bank, static_cast<int>(mass_bank.size()), mass_norm
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
        CUDA_CHECK(cudaFree(dev_mass_bank));
        #endif // MULTISIZE
        
        #if defined(COLLISION) || defined(DIFFUSION)
        #ifdef VAL_RUNTIME_SEED
        // vary the stochastic device stream independently of the sampled initial particles
        int rng_seed = 1;
        if (const char *seed_text = std::getenv("GAMEDEV_VAL_RNG_SEED"))
            rng_seed = std::stoi(seed_text);
        rngstate_init <<< NB_P, TPB >>> (dev_rngstate, rng_seed);
        #else  // PRODUCTION_SEED
        rngstate_init <<< NB_P, TPB >>> (dev_rngstate);
        #endif // VAL_RUNTIME_SEED
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

        #ifndef COL_CACHE_VAL
        SAVE_PARTICLE_TO_FILE(idx_from);
        #endif // COL_CACHE_VAL

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
    bool col_geom_valid = false;
    #ifdef COL_CACHE_VAL
    bool cache_val_saved = false;
    #endif // COL_CACHE_VAL

    #ifdef COL_GEOM_VAL
    int col_geom_calls = 0;
    int col_geom_builds = 0;
    int col_geom_reuses = 0;
    int col_geom_invalidations = 0;
    #endif // COL_GEOM_VAL

    #ifdef COL_PERF_VAL
    double col_perf_total_ms = 0.0;
    double col_perf_geometry_ms = 0.0;
    double col_perf_rate_ms = 0.0;
    double col_perf_event_ms = 0.0;
    double col_perf_audit_ms = 0.0;
    long long col_perf_batches = 0;
    long long col_perf_launches = 0;

    // synchronize benchmark boundaries so every phase owns all of its GPU work
    auto col_perf_start = [] ()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::steady_clock::now();
    };
    auto col_perf_stop = [] (const std::chrono::steady_clock::time_point &start)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();
    };
    #endif // COL_PERF_VAL

    float image_dist_min = -1.0f;
    if (X_WEDGE)
    {
        image_dist_min = _get_image_dist_min(
            static_cast<float>(X_MIN), static_cast<float>(X_MAX), static_cast<float>(Y_MIN),
            static_cast<float>(Z_MIN), static_cast<float>(Z_MAX)
        );
    }
    #ifndef BERNOULLI
    local_workspace local(PATH);
    bool local_geometry_valid=false;
    col_controller_summary col_summary;
    #endif // FROZEN_BATH

    // invalidate the search package only when a position update ends its current geometry epoch
    auto invalidate_col_geometry = [&] ()
    {
        #ifdef COL_GEOM_VAL
        if (col_geom_valid) col_geom_invalidations++;
        #endif // COL_GEOM_VAL
        col_geom_valid = false;
        #ifndef BERNOULLI
        local_geometry_valid=false;
        #endif
    };

    // evolve collisions over a fixed-position interval with the configured collision integrator
    auto evolve_collisions = [&] (real duration)
    {
        #ifdef COL_PERF_VAL
        auto col_operator_start = col_perf_start();
        #endif // COL_PERF_VAL

        #ifdef KNN_FRESH
        // VAL baseline: deliberately rebuild an otherwise reusable fixed-position search package
        col_geom_valid = false;
        #ifndef BERNOULLI
        local_geometry_valid=false;
        #endif
        #endif // KNN_FRESH

        #ifdef COL_GEOM_VAL
        col_geom_calls++;
        #endif // COL_GEOM_VAL

        // geometry reuse must not suppress the per-operator nonfinite-state failure path
        if (col_geom_valid)
        {
            CUDA_CHECK(cudaMemset(dev_bad_part, 0, sizeof(int)));
            colstate_flag <<< NB_P, TPB >>> (dev_particle, dev_bad_part);
            CUDA_KERNEL_CHECK("colstate_flag");
            int bad_part = 0;
            CUDA_CHECK(cudaMemcpy(&bad_part, dev_bad_part, sizeof(int), cudaMemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // rebuild the complete geometric search package only after particle positions change
        if (!col_geom_valid)
        {
            #ifdef COL_PERF_VAL
            auto col_geometry_start = col_perf_start();
            #endif // COL_PERF_VAL

            CUDA_CHECK(cudaMemset(dev_bad_part, 0, sizeof(int)));
            #ifdef COLLISION_KDTREE
            col_site_init <<< NB_P, TPB >>> (
                dev_kdtree_node, dev_col_active, dev_particle, dev_bad_part
            );
            CUDA_KERNEL_CHECK("col_site_init");
            int bad_part = 0;
            CUDA_CHECK(cudaMemcpy(&bad_part, dev_bad_part, sizeof(int), cudaMemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }
            kdtree::buildTree <kdtree_node, kdtree_traits> (
                dev_kdtree_node, N_T, dev_kdtree_box
            );
            CUDA_KERNEL_CHECK("kdtree::buildTree");
            #else  // COLLISION_MORTON
            col_site_init <<< NB_P, TPB >>> (
                dev_morton_point, dev_morton_posx, dev_search_dist,
                dev_col_active, dev_particle, dev_bad_part
            );
            CUDA_KERNEL_CHECK("col_site_init");
            int bad_part = 0;
            CUDA_CHECK(cudaMemcpy(&bad_part, dev_bad_part, sizeof(int), cudaMemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }

            thrust::device_ptr <const float> search_dist_ptr(dev_search_dist);
            float max_search_dist = *thrust::max_element(search_dist_ptr, search_dist_ptr + N_P);
            bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*max_search_dist;
            morton_owner.build(
                dev_morton_point, dev_morton_posx, N_P, max_search_dist,
                static_cast<float>(X_MIN), static_cast<float>(X_MAX),
                static_cast<float>(Y_MAX), X_WEDGE, unique_ids,
                (N_Z > 1) ? 3 : 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
            );
            #endif // COLLISION_KDTREE

            #if !defined(BERNOULLI) || defined(KNN_CACHE)
            // retain fixed physical neighbors while collision properties continue to evolve
            #ifdef COLLISION_KDTREE
            col_cache_get <<< NB_T, TPB >>> (
                dev_col_neighbor, dev_col_measure, dev_kdtree_node, dev_kdtree_box,
                dev_col_active, dev_particle, image_dist_min
            );
            CUDA_KERNEL_CHECK("col_cache_get");
            #else  // COLLISION_MORTON
            col_cache_get <<< N_P, MORTON_TPB >>> (
                dev_col_neighbor, dev_col_measure, dev_morton_overflow, dev_morton_point,
                dev_col_active, dev_particle, morton_owner.view(), morton_owner.unique_ids()
            );
            CUDA_KERNEL_CHECK("col_cache_get");
            thrust::device_ptr <const unsigned int> morton_overflow_ptr(dev_morton_overflow);
            unsigned int max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_cache_get");
            #endif // COLLISION_KDTREE
            #endif // FROZEN_BATH || KNN_CACHE

            #ifdef COL_CACHE_VAL
            if (!cache_val_saved)
            {
                cache_val_get <<< NB_P, TPB >>> (
                    dev_cacheval, dev_col_neighbor, dev_col_measure,
                    dev_col_active, dev_particle
                );
                CUDA_KERNEL_CHECK("cache_val_get");
                std::vector<col_cache_val_sample> cache_val(N_P);
                CUDA_CHECK(cudaMemcpy(
                    cache_val.data(), dev_cacheval,
                    sizeof(col_cache_val_sample)*N_P, cudaMemcpyDeviceToHost
                ));
                std::string cache_val_file = PATH + "collision_cache.json";
                if (!save_col_cache_val(cache_val_file, cache_val, "cuda"))
                    throw std::runtime_error("failed to save collision-cache diagnostics");
                cache_val_saved = true;
            }
            #endif // COL_CACHE_VAL

            #ifndef BERNOULLI
            col_space_bin <<< NB_P, TPB >>> (dev_col_spatial, dev_particle);
            CUDA_KERNEL_CHECK("col_space_bin");
            #endif // FROZEN_BATH

            // publish validity only after every required hierarchy, cache, and guard has completed
            col_geom_valid = true;
            #ifdef COL_GEOM_VAL
            col_geom_builds++;
            #endif // COL_GEOM_VAL
            #ifdef COL_PERF_VAL
            col_perf_geometry_ms += col_perf_stop(col_geometry_start);
            #endif // COL_PERF_VAL
        }
        #ifdef COL_GEOM_VAL
        else
        {
            col_geom_reuses++;
        }
        #endif // COL_GEOM_VAL

        #ifndef BERNOULLI
        #include <local_evolve.inc>
        #else  // BERNOULLI
        real elapsed = 0.0;
        while (elapsed < duration)
        {
            #ifdef COL_PERF_VAL
            auto col_rate_start = col_perf_start();
            #endif // COL_PERF_VAL

            // freeze only the species fields changed by collisions while positions and velocities remain fixed
            col_snap_save <<< NB_P, TPB >>> (dev_size_old, dev_numr_old, dev_particle);
            CUDA_KERNEL_CHECK("col_snap_save");
            #ifdef KNN_CACHE
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< NB_P, TPB >>> (
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
            #endif // COLLISION_KDTREE
                dev_col_rate, dev_particle, dev_col_neighbor, dev_col_measure,
                dev_col_active, dev_size_old, dev_numr_old,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #else  // DIRECT_BERNOULLI
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< NB_T, TPB >>> (dev_col_rate, dev_col_dist, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
                dev_col_rate, dev_col_dist, dev_morton_overflow, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_morton_point,
                morton_owner.view(), morton_owner.unique_ids(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #endif // COLLISION_KDTREE
            #endif // KNN_CACHE
            CUDA_KERNEL_CHECK("col_rate_calc");

            #if defined(COLLISION_MORTON) && !defined(KNN_CACHE)
            thrust::device_ptr <const unsigned int> morton_overflow_ptr(dev_morton_overflow);
            unsigned int max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_rate_calc");
            #endif // COLLISION_MORTON && !KNN_CACHE

            // use the largest total propensity to control every representative's event probability
            thrust::device_ptr <const real> col_rate_ptr(dev_col_rate);
            real max_col_rate = *thrust::max_element(col_rate_ptr, col_rate_ptr + N_P);
            real remaining = duration - elapsed;
            #ifdef COL_PERF_VAL
            col_perf_rate_ms += col_perf_stop(col_rate_start);
            #endif // COL_PERF_VAL

            if (!(max_col_rate > 0.0))
            {
                // consume the remaining interval when no collision channel is active
                dt_col = remaining;
                elapsed = duration;
                break;
            }

            // keep the fastest frozen propensity below CFL_COL before sampling one event at most
            dt_col = fmin(CFL_COL / max_col_rate, remaining);
            #ifdef COL_PERF_VAL
            auto col_event_start = col_perf_start();
            #endif // COL_PERF_VAL
            #ifdef KNN_CACHE
            #ifdef COLLISION_KDTREE
            col_event_run <<< NB_P, TPB >>> (
            #else  // COLLISION_MORTON
            col_event_run <<< N_P, MORTON_TPB >>> (
            #endif // COLLISION_KDTREE
                dev_particle, dev_rngstate, dev_col_rate, dev_col_neighbor, dev_col_measure,
                dev_col_active, dev_size_old, dev_numr_old,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #else  // DIRECT_BERNOULLI
            #ifdef COLLISION_KDTREE
            col_event_run <<< NB_T, TPB >>> (dev_particle, dev_rngstate, dev_col_rate, dev_col_dist,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #else  // COLLISION_MORTON
            col_event_run <<< N_P, MORTON_TPB >>> (
                dev_particle, dev_rngstate, dev_col_rate, dev_col_dist, dev_morton_overflow,
                dev_col_active, dev_size_old, dev_numr_old, dev_morton_point,
                morton_owner.view(), morton_owner.unique_ids(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #endif // COLLISION_KDTREE
            #endif // KNN_CACHE
            CUDA_KERNEL_CHECK("col_event_run");
            CUDA_CHECK(cudaDeviceSynchronize());

            #if defined(COLLISION_MORTON) && !defined(KNN_CACHE)
            max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_event_run");
            #endif // COLLISION_MORTON && !KNN_CACHE

            #ifdef COL_PERF_VAL
            col_perf_event_ms += col_perf_stop(col_event_start);
            col_perf_batches++;
            col_perf_launches++;
            #endif // COL_PERF_VAL

            real elapsed_old = elapsed;
            elapsed += dt_col;
            if (!(elapsed > elapsed_old))
                throw std::runtime_error("collision timestep cannot advance the operator clock");
            clock_dyn = elapsed;
            count_col++;
        }
        #endif // FROZEN_BATH

        #ifdef COL_PERF_VAL
        col_perf_total_ms += col_perf_stop(col_operator_start);
        #endif // COL_PERF_VAL
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

        #ifndef BERNOULLI
        // restart controller memory at checkpoint boundaries while retaining it across split operators
        std::fill(local.state.begin(),local.state.end(),col_bath_state{});
        col_summary = col_controller_summary{};
        #endif // FROZEN_BATH

        PRINT_TITLE_TO_SCREEN();
        
        do
        {
            #ifdef TRANSPORT
            // reduce all local inverse rates to a globally valid dynamics timestep
            dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle
                #ifdef IMPORTGAS
                , dev_gas_velx, dev_gas_vely, dev_gas_velz
                , dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next
                #endif // IMPORTGAS
            );
            CUDA_KERNEL_CHECK("dyn_rate_calc");
            thrust::device_ptr <const real> dt_rate_ptr(dev_dyn_rate);
            real max_dt_rate = *thrust::max_element(dt_rate_ptr, dt_rate_ptr + N_P);
            dt_dyn = fmin(DT_MAX, fmin(1.0 / max_dt_rate, dt_out - clock_out));

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
            evolve_collisions(0.5*dt_dyn);
            #endif // COLLISION

            #ifdef DIFFUSION
            // apply the first half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn);
            CUDA_KERNEL_CHECK("diffusion_pos");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // DIFFUSION

            #ifdef RADIATION
            // drift to midpoint positions and reconstruct the optical depth used by the force solve
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            CUDA_KERNEL_CHECK("ssa_substep_1");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
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
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #else  // NO RADIATION
            // complete transport in one launch when no midpoint radiation field is required
            ssa_transport <<< NB_P, TPB >>> (dev_particle,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                dt_dyn
            );
            CUDA_KERNEL_CHECK("ssa_transport");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // RADIATION

            #ifdef DIFFUSION
            // apply the second half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn);
            CUDA_KERNEL_CHECK("diffusion_pos");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // DIFFUSION

            #ifdef COLLISION
            // close the symmetric composition with half a collision interval
            evolve_collisions(0.5*dt_dyn);
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
            
            evolve_collisions(duration);
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

        #ifdef COL_CACHE_VAL
        // cache probes archive only reduced diagnostics, not large particle and RNG checkpoints
        #elif defined(LOGTIMING)
        SAVE_PARTICLE_TO_FILE(idx_file);
        #elif defined(LOGOUTPUT)
        if (is_log_power(idx_file)) SAVE_PARTICLE_TO_FILE(idx_file);
        #else  // LINEAR_OUTPUT
        if (idx_file % LIN_BASE == 0) SAVE_PARTICLE_TO_FILE(idx_file);
        #endif // COL_CACHE_VAL / LOGTIMING / LOGOUTPUT / LINEAR_OUTPUT

        #ifndef BERNOULLI
        std::string controller_file = PATH + "collision_chain_" + frame_num(idx_file) + ".json";
        if (!save_col_controller(controller_file, col_summary))
        {
            std::cerr << "Error: Failed to save file: " << controller_file << std::endl;
            return 1;
        }
        #endif // FROZEN_BATH

        #ifdef COL_GEOM_VAL
        std::string geometry_file = PATH + "collision_geometry_" + frame_num(idx_file) + ".json";
        std::ofstream geometry_stream(geometry_file);
        if (!geometry_stream)
        {
            std::cerr << "Error: Failed to save file: " << geometry_file << std::endl;
            return 1;
        }
        geometry_stream
            << "{\n"
            << "  \"schema\": 1,\n"
            << "  \"backend\": \"cuda\",\n"
            << "  \"frame\": " << idx_file << ",\n"
            << "  \"search_calls\": " << col_geom_calls << ",\n"
            << "  \"geometry_builds\": " << col_geom_builds << ",\n"
            << "  \"geometry_reuses\": " << col_geom_reuses << ",\n"
            << "  \"geometry_invalidations\": " << col_geom_invalidations << "\n"
            << "}\n";
        if (!geometry_stream)
        {
            std::cerr << "Error: Failed to write file: " << geometry_file << std::endl;
            return 1;
        }
        #endif // COL_GEOM_VAL

        #ifdef COL_PERF_VAL
        std::string performance_file = PATH + "collision_performance_" + frame_num(idx_file) + ".json";
        std::ofstream performance_stream(performance_file);
        if (!performance_stream)
        {
            std::cerr << "Error: Failed to save file: " << performance_file << std::endl;
            return 1;
        }
        double col_perf_accounted_ms = col_perf_geometry_ms + col_perf_rate_ms
            + col_perf_event_ms + col_perf_audit_ms;
        performance_stream
            << std::setprecision(17)
            << "{\n"
            << "  \"schema\": 1,\n"
            << "  \"backend\": \"cuda\",\n"
            << "  \"frame\": " << idx_file << ",\n"
            << "  \"particles\": " << N_P << ",\n"
            << "  \"neighbors\": " << N_K << ",\n"
            << "  \"integrator\": \""
            #ifndef BERNOULLI
            << "frozen_bath"
            #else  // BERNOULLI
            #ifdef KNN_CACHE
            << "bernoulli_cache"
            #else  // DIRECT_BERNOULLI
            << "bernoulli_direct"
            #endif // KNN_CACHE
            #endif // FROZEN_BATH
            << "\",\n"
            << "  \"search\": \""
            #ifdef COLLISION_KDTREE
            << "kdtree"
            #else  // COLLISION_MORTON
            << "morton"
            #endif // COLLISION_KDTREE
            << "\",\n"
            << "  \"forced_fresh\": "
            #ifdef KNN_FRESH
            << "true"
            #else  // REUSED_GEOMETRY
            << "false"
            #endif // KNN_FRESH
            << ",\n"
            << "  \"neighbor_cache\": "
            #if !defined(BERNOULLI) || defined(KNN_CACHE)
            << "true"
            #else  // DIRECT_BERNOULLI
            << "false"
            #endif // FROZEN_BATH || KNN_CACHE
            << ",\n"
            << "  \"neighbor_cache_bytes\": "
            #if !defined(BERNOULLI) || defined(KNN_CACHE)
            << sizeof(int)*col_neighbor_count + sizeof(real)*static_cast<std::size_t>(N_P)
            #else  // DIRECT_BERNOULLI
            << 0
            #endif // FROZEN_BATH || KNN_CACHE
            << ",\n"
            << "  \"collision_calls\": " << col_geom_calls << ",\n"
            << "  \"geometry_builds\": " << col_geom_builds << ",\n"
            << "  \"geometry_reuses\": " << col_geom_reuses << ",\n"
            << "  \"geometry_invalidations\": " << col_geom_invalidations << ",\n"
            << "  \"collision_batches\": " << col_perf_batches << ",\n"
            << "  \"event_launches\": " << col_perf_launches << ",\n"
            << "  \"timing_ms\": {\n"
            << "    \"collision_total\": " << col_perf_total_ms << ",\n"
            << "    \"geometry\": " << col_perf_geometry_ms << ",\n"
            << "    \"rate_and_control\": " << col_perf_rate_ms << ",\n"
            << "    \"events\": " << col_perf_event_ms << ",\n"
            << "    \"audit\": " << col_perf_audit_ms << ",\n"
            << "    \"other\": " << col_perf_total_ms - col_perf_accounted_ms << "\n"
            << "  }\n"
            << "}\n";
        if (!performance_stream)
        {
            std::cerr << "Error: Failed to write file: " << performance_file << std::endl;
            return 1;
        }
        #endif // COL_PERF_VAL

        msg_output(idx_file);
    }
 
    return 0;
}

// =========================================================================================================================
