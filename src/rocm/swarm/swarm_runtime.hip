#include <chrono>           // std::chrono::system_clock
#include <cmath>            // std::fabs, std::fmin, std::sin
#include <cstdlib>          // EXIT_FAILURE, std::exit
#include <filesystem>       // std::filesystem::create_directories
#include <iomanip>          // std::setw, std::setfill
#include <iostream>         // std::cout, std::endl
#include <limits>           // std::numeric_limits
#include <sstream>          // std::stringstream
#include <stdexcept>        // std::runtime_error
#include <vector>           // std::vector

#if defined(TRANSPORT) || defined(COLLISION)
#include <thrust/device_ptr.h>  // thrust::device_ptr
#include <thrust/extrema.h>     // thrust::max_element
#endif // TRANSPORT || COLLISION

#include <swarm_host.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
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
    HIP_CHECK(hipHostMalloc((void**)&particle, sizeof(swarm)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_particle, sizeof(swarm)*N_P));

    #ifdef TRANSPORT
    real *dev_dyn_rate;
    HIP_CHECK(hipMalloc((void**)&dev_dyn_rate, sizeof(real)*N_P));
    #endif // TRANSPORT
    
    #ifdef SAVE_DENS
    real *dustdens, *dev_dustdens;
    HIP_CHECK(hipHostMalloc((void**)&dustdens, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_dustdens, sizeof(real)*N_G));
    #endif // SAVE_DENS
    
    #ifdef IMPORTGAS
    real *gas_dens, *dev_gas_dens;
    HIP_CHECK(hipHostMalloc((void**)&gas_dens,  sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_dens,  sizeof(real)*N_G));

    real *gas_velx, *dev_gas_velx;
    HIP_CHECK(hipHostMalloc((void**)&gas_velx,  sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_velx,  sizeof(real)*N_G));

    real *gas_vely, *dev_gas_vely;
    HIP_CHECK(hipHostMalloc((void**)&gas_vely,  sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_vely,  sizeof(real)*N_G));

    real *gas_velz, *dev_gas_velz;
    HIP_CHECK(hipHostMalloc((void**)&gas_velz,  sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_velz,  sizeof(real)*N_G));

    real *dev_gas_dens_next, *dev_gas_velx_next, *dev_gas_vely_next, *dev_gas_velz_next;
    HIP_CHECK(hipMalloc((void**)&dev_gas_dens_next, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_velx_next, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_vely_next, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_gas_velz_next, sizeof(real)*N_G));
    #endif // IMPORTGAS
    
    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    HIP_CHECK(hipHostMalloc((void**)&optdepth, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
    #endif // RADIATION

    #ifdef COLLISION
    unsigned char *dev_col_active;
    HIP_CHECK(hipMalloc((void**)&dev_col_active, sizeof(unsigned char)*N_P));

    int *dev_bad_part;
    HIP_CHECK(hipMalloc((void**)&dev_bad_part, sizeof(int)));

    #ifdef COLLISION_KDTREE
    kdtree_boxf *dev_kdtree_box;
    HIP_CHECK(hipMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)));

    kdtree_node *dev_kdtree_node;
    HIP_CHECK(hipMalloc((void**)&dev_kdtree_node, sizeof(kdtree_node)*N_T));
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx, *dev_search_dist;
    unsigned int *dev_morton_overflow;
    HIP_CHECK(hipMalloc((void**)&dev_morton_point, sizeof(float3)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_morton_posx, sizeof(float)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_search_dist, sizeof(float)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_morton_overflow, sizeof(unsigned int)*N_P));
    morton_ghost_index morton_owner;
    #endif // COLLISION_KDTREE

    real *dev_size_old, *dev_numr_old, *dev_col_rate, *dev_col_dist;
    HIP_CHECK(hipMalloc((void**)&dev_size_old, sizeof(real)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_numr_old, sizeof(real)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_col_rate, sizeof(real)*N_P));
    HIP_CHECK(hipMalloc((void**)&dev_col_dist, sizeof(real)*N_P));
    #endif // COLLISION

    #if defined(COLLISION) || defined(DIFFUSION)
    curs *dev_rngstate;
    HIP_CHECK(hipMalloc((void**)&dev_rngstate, sizeof(curs)*N_P));
    #endif // COLLISION || DIFFUSION

    if (argc <= 1)
	{
        // construct a fresh realization from the configured analytic or imported distribution
        
        idx_from = 0;

        real *randposx, *dev_randposx;
        HIP_CHECK(hipHostMalloc((void**)&randposx, sizeof(real)*N_P));
        HIP_CHECK(hipMalloc((void**)&dev_randposx, sizeof(real)*N_P));

        real *randposy, *dev_randposy;
        HIP_CHECK(hipHostMalloc((void**)&randposy, sizeof(real)*N_P));
        HIP_CHECK(hipMalloc((void**)&dev_randposy, sizeof(real)*N_P));

        real *randposz, *dev_randposz;
        HIP_CHECK(hipHostMalloc((void**)&randposz, sizeof(real)*N_P));
        HIP_CHECK(hipMalloc((void**)&dev_randposz, sizeof(real)*N_P));

        #ifdef MULTISIZE
        real *randsize, *dev_randsize;
        HIP_CHECK(hipHostMalloc((void**)&randsize, sizeof(real)*N_P));
        HIP_CHECK(hipMalloc((void**)&dev_randsize, sizeof(real)*N_P));

        real *dev_mass_bank;
        HIP_CHECK(hipMalloc((void**)&dev_mass_bank, sizeof(real)*mass_bank.size()));
        HIP_CHECK(hipMemcpy(dev_mass_bank, mass_bank.data(), sizeof(real)*mass_bank.size(), hipMemcpyHostToDevice));
        #endif // MULTISIZE

        rand_generator.seed(0); // keep initialization reproducible across runs

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
        HIP_CHECK(hipHostMalloc((void**)&epsilon,  sizeof(real)*N_G));
        
        if (!load_epsilon(PATH, idx_from, epsilon))
        {
            std::cerr << "Error: Failed to load gas data files for frame " << idx_from << std::endl;
            return 1;
        }

        // use one imported total-dust spatial distribution for all previously sampled grain species
        rand_from_file(randposx, randposy, randposz, N_P, gas_dens, epsilon);
        
        HIP_CHECK(hipHostFree(epsilon));
        #else  // NO IMPORTGAS
        #if defined(MULTISIZE) && defined(DIFFUSION)
        rand_disk_poly(randposx, randposy, randposz, randsize, N_P);
        #else  // !(MULTISIZE && DIFFUSION)
        rand_disk_mono(randposx, randposy, randposz, S_0, N_P);
        #endif // MULTISIZE && DIFFUSION
        #endif // IMPORTGAS

        HIP_CHECK(hipMemcpy(dev_randposx, randposx, sizeof(real)*N_P, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dev_randposy, randposy, sizeof(real)*N_P, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dev_randposz, randposz, sizeof(real)*N_P, hipMemcpyHostToDevice));

        #ifdef MULTISIZE
        HIP_CHECK(hipMemcpy(dev_randsize, randsize, sizeof(real)*N_P, hipMemcpyHostToDevice));
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
        HIP_KERNEL_CHECK("particle_init");

        HIP_CHECK(hipHostFree(randposx));
        HIP_CHECK(hipFree(dev_randposx));
        HIP_CHECK(hipHostFree(randposy));
        HIP_CHECK(hipFree(dev_randposy));
        HIP_CHECK(hipHostFree(randposz));
        HIP_CHECK(hipFree(dev_randposz));

        #ifdef MULTISIZE
        HIP_CHECK(hipHostFree(randsize));
        HIP_CHECK(hipFree(dev_randsize));
        HIP_CHECK(hipFree(dev_mass_bank));
        #endif // MULTISIZE
        
        #if defined(COLLISION) || defined(DIFFUSION)
        rngstate_init <<< NB_P, TPB >>> (dev_rngstate);
        HIP_KERNEL_CHECK("rngstate_init");
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
    // evolve collisions over a fixed-position interval with controlled frozen-rate Bernoulli batches
    auto evolve_collisions = [&] (real duration)
    {
        // collisions change grain properties but not positions, so one search index serves the full interval
        HIP_CHECK(hipMemset(dev_bad_part, 0, sizeof(int)));
        #ifdef COLLISION_KDTREE
        col_site_init <<< NB_P, TPB >>> (dev_kdtree_node, dev_col_active, dev_particle, dev_bad_part);
        HIP_KERNEL_CHECK("col_site_init");
        int bad_part = 0;
        HIP_CHECK(hipMemcpy(&bad_part, dev_bad_part, sizeof(int), hipMemcpyDeviceToHost));
        if (bad_part != 0)
        {
            std::cerr << "Error: non-finite particle state before collision search at particle "
                << bad_part - 1 << std::endl;
            std::exit(EXIT_FAILURE);
        }
        kdtree::buildTree <kdtree_node, kdtree_traits> (
            dev_kdtree_node, N_T, dev_kdtree_box
        );
        HIP_KERNEL_CHECK("kdtree::buildTree");
        float image_dist_min = -1.0f;
        if (N_T > N_P)
        {
            image_dist_min = static_cast<float>(
                2.0*Y_MIN*std::fmin(std::sin(Z_MIN), std::sin(Z_MAX))
                *std::fabs(std::sin(0.5*(X_MAX - X_MIN)))
            );
        }
        #else  // COLLISION_MORTON
        col_site_init <<< NB_P, TPB >>> (
            dev_morton_point, dev_morton_posx, dev_search_dist, dev_col_active, dev_particle, dev_bad_part
        );
        HIP_KERNEL_CHECK("col_site_init");
        int bad_part = 0;
        HIP_CHECK(hipMemcpy(&bad_part, dev_bad_part, sizeof(int), hipMemcpyDeviceToHost));
        if (bad_part != 0)
        {
            std::cerr << "Error: non-finite particle state before collision search at particle "
                << bad_part - 1 << std::endl;
            std::exit(EXIT_FAILURE);
        }

        thrust::device_ptr <const float> search_dist_ptr(dev_search_dist);
        float max_search_dist = *thrust::max_element(search_dist_ptr, search_dist_ptr + N_P);
        morton_owner.build(
            dev_morton_point, dev_morton_posx, N_P, max_search_dist,
            static_cast<float>(X_MIN), static_cast<float>(X_MAX),
            static_cast<float>(Y_MIN), static_cast<float>(Y_MAX),
            static_cast<float>(Z_MIN), static_cast<float>(Z_MAX),
            N_X > 1, (N_Z > 1) ? 3 : 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
        );
        #endif // COLLISION_KDTREE

        real elapsed = 0.0;
        while (elapsed < duration)
        {
            // freeze only the species fields changed by collisions while positions and velocities remain fixed
            col_snap_save <<< NB_P, TPB >>> (dev_size_old, dev_numr_old, dev_particle);
            HIP_KERNEL_CHECK("col_snap_save");
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< NB_T, TPB >>> (dev_col_rate, dev_col_dist, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / (N_K - 1.0) / total_dust_mass
            );
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
                dev_col_rate, dev_col_dist, dev_morton_overflow, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_morton_point,
                morton_owner.view(), morton_owner.unique_ids(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / (N_K - 1.0) / total_dust_mass
            );
            #endif // COLLISION_KDTREE
            HIP_KERNEL_CHECK("col_rate_calc");

            HIP_CHECK(hipMemset(dev_bad_part, 0, sizeof(int)));
            inf_rate_flag <<< NB_P, TPB >>> (dev_col_rate, dev_col_dist, dev_bad_part);
            HIP_KERNEL_CHECK("inf_rate_flag");
            int bad_result = 0;
            HIP_CHECK(hipMemcpy(&bad_result, dev_bad_part, sizeof(int), hipMemcpyDeviceToHost));
            if (bad_result != 0)
            {
                std::cerr << "Error: non-finite collision result at particle "
                    << bad_result - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }

            #ifdef COLLISION_MORTON
            thrust::device_ptr <const unsigned int> morton_overflow_ptr(dev_morton_overflow);
            unsigned int max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_rate_calc");
            #endif // COLLISION_MORTON

            // use the largest total propensity to control every representative's event probability
            thrust::device_ptr <const real> col_rate_ptr(dev_col_rate);
            real max_col_rate = *thrust::max_element(col_rate_ptr, col_rate_ptr + N_P);
            real remaining = duration - elapsed;

            if (!(max_col_rate > 0.0))
            {
                // consume the remaining interval when no collision channel is active
                dt_col = remaining;
                elapsed = duration;
                break;
            }

            // keep the fastest frozen propensity below CFL_COL before sampling one event at most
            dt_col = fmin(CFL_COL / max_col_rate, remaining);
            #ifdef COLLISION_KDTREE
            col_event_run <<< NB_T, TPB >>> (dev_particle, dev_rngstate, dev_col_rate, dev_col_dist,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / (N_K - 1.0) / total_dust_mass,
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
                N_P / (N_K - 1.0) / total_dust_mass,
                dt_col
            );
            #endif // COLLISION_KDTREE
            HIP_KERNEL_CHECK("col_event_run");
            HIP_CHECK(hipDeviceSynchronize());

            #ifdef COLLISION_MORTON
            max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_event_run");
            #endif // COLLISION_MORTON

            elapsed += dt_col;
            clock_dyn = elapsed;
            count_col++;
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
            dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle
                #ifdef IMPORTGAS
                , dev_gas_velx, dev_gas_vely, dev_gas_velz
                , dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next
                #endif // IMPORTGAS
            );
            HIP_KERNEL_CHECK("dyn_rate_calc");
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
            HIP_KERNEL_CHECK("gas_lerp_calc");
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
            HIP_KERNEL_CHECK("diffusion_pos");
            #endif // DIFFUSION

            #ifdef RADIATION
            // drift to midpoint positions and reconstruct the optical depth used by the force solve
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            HIP_KERNEL_CHECK("ssa_substep_1");
            optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
            HIP_KERNEL_CHECK("optdepth_init");
            optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, total_dust_mass);
            HIP_KERNEL_CHECK("optdepth_depo");
            optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
            HIP_KERNEL_CHECK("optdepth_calc");
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            HIP_KERNEL_CHECK("optdepth_csum");

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
            HIP_KERNEL_CHECK("ssa_substep_2");
            #else  // NO RADIATION
            // complete transport in one launch when no midpoint radiation field is required
            ssa_transport <<< NB_P, TPB >>> (dev_particle,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                dt_dyn
            );
            HIP_KERNEL_CHECK("ssa_transport");
            #endif // RADIATION

            #ifdef DIFFUSION
            // apply the second half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn);
            HIP_KERNEL_CHECK("diffusion_pos");
            #endif // DIFFUSION

            #ifdef COLLISION
            // close the symmetric composition with half a collision interval
            evolve_collisions(0.5*dt_dyn);
            #endif // COLLISION

            HIP_CHECK(hipDeviceSynchronize());
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
            HIP_KERNEL_CHECK("gas_lerp_calc");
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
        HIP_CHECK(hipMemcpy(dev_gas_dens, dev_gas_dens_next, sizeof(real)*N_G, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(dev_gas_velx, dev_gas_velx_next, sizeof(real)*N_G, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(dev_gas_vely, dev_gas_vely_next, sizeof(real)*N_G, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(dev_gas_velz, dev_gas_velz_next, sizeof(real)*N_G, hipMemcpyDeviceToDevice));
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
