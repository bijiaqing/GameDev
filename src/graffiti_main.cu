#include <chrono>           // for std::chrono::system_clock
#include <filesystem>       // for std::filesystem::create_directories
#include <iomanip>          // for std::setw, std::setfill
#include <iostream>         // for std::cout, std::endl
#include <sstream>          // for std::stringstream

#if defined(TRANSPORT) || defined(COLLISION)
#include <thrust/device_ptr.h>  // for thrust::device_ptr
#include <thrust/extrema.h>     // for thrust::max_element
#endif

#include <graffiti_kern.cuh>
#include <graffiti_host.cuh>

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
    std::string fname;
    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_P);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_P);

    #ifdef TRANSPORT
    real *dev_dt_rate;
    cudaMalloc((void**)&dev_dt_rate, sizeof(real)*N_P);
    #endif
    
    #ifdef SAVE_DENS
    real *dustdens, *dev_dustdens;
    cudaMallocHost((void**)&dustdens, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_G);
    #endif // SAVE_DENS
    
    #ifdef IMPORTGAS
    real *gasdens, *dev_gasdens;
    real *gasvelx, *dev_gasvelx;
    real *gasvely, *dev_gasvely;
    real *gasvelz, *dev_gasvelz;
    cudaMallocHost((void**)&gasdens,  sizeof(real)*N_G);
    cudaMallocHost((void**)&gasvelx,  sizeof(real)*N_G);
    cudaMallocHost((void**)&gasvely,  sizeof(real)*N_G);
    cudaMallocHost((void**)&gasvelz,  sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasdens,  sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvelx,  sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvely,  sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvelz,  sizeof(real)*N_G);
    real *dev_gasdens_next, *dev_gasvelx_next, *dev_gasvely_next, *dev_gasvelz_next;
    cudaMalloc((void**)&dev_gasdens_next, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvelx_next, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvely_next, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_gasvelz_next, sizeof(real)*N_G);
    #endif // IMPORTGAS
    
    #if defined(TRANSPORT) && defined(RADIATION)
    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G);
    #endif // TRANSPORT && RADIATION

    #ifdef COLLISION
    bbox *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(bbox));

    // azimuthal wedges require two periodic image nodes per representative
    int col_tree_size = (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12) ? 3*N_P : N_P;
    tree *dev_col_tree;
    cudaMalloc((void**)&dev_col_tree, sizeof(tree)*col_tree_size);

    swarm *dev_particle_old;
    cudaMalloc((void**)&dev_particle_old, sizeof(swarm)*N_P);

    real *dev_col_rate;
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_P);
    #endif // COLLISION

    #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
    curs *dev_rs_swarm;
    cudaMalloc((void**)&dev_rs_swarm, sizeof(curs)*N_P);
    #endif // COLLISION || (TRANSPORT && DIFFUSION)

    if (argc <= 1)
	{
        // construct a fresh realization from the configured analytic or imported distribution
        
        idx_from = 0;

        real *random_x, *dev_random_x;
        real *random_y, *dev_random_y;
        real *random_z, *dev_random_z;

        cudaMallocHost((void**)&random_x, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_x, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_y, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_y, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_z, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_z, sizeof(real)*N_P);

        #ifdef MULTISIZE
        real *random_s, *dev_random_s;
        cudaMallocHost((void**)&random_s, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_s, sizeof(real)*N_P);
        #endif // MULTISIZE

        rand_generator.seed(0); // keep initialization reproducible across runs

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);

        real *epsilon;
        cudaMallocHost((void**)&epsilon,  sizeof(real)*N_G);
        
        if (!load_epsilon(PATH, idx_from, epsilon))
        {
            std::cerr << "Error: Failed to load gas data files for frame " << idx_from << std::endl;
            return 1;
        }

        // use one imported total-dust spatial distribution for all grain species sampled below
        rand_from_file(random_x, random_y, random_z, N_P, gasdens, epsilon);
        
        cudaFreeHost(epsilon);
        #else // NOT IMPORTGAS
        rand_uniform(random_x, N_P, INIT_XMIN, INIT_XMAX);
        rand_disk(random_y, random_z, N_P, S_0);
        #endif // IMPORTGAS
        
        #ifdef MULTISIZE
        real idx_swarm = -0.5;  // equal represented mass per swarm
        #if defined(TRANSPORT) && defined(RADIATION)
        idx_swarm = -1.5;       // equal represented surface area per swarm, see _get_grain_number
        #endif // TRANSPORT && RADIATION
        #ifdef COLLISION_LINEAR_TEST
        rand_4_linear(random_s, N_P);
        #else
        rand_pow_law(random_s, N_P, INIT_SMIN, INIT_SMAX, idx_swarm);
        #endif
        #endif // MULTISIZE

        cudaMemcpy(dev_random_x, random_x, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_y, random_y, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_z, random_z, sizeof(real)*N_P, cudaMemcpyHostToDevice);

        #ifdef MULTISIZE
        cudaMemcpy(dev_random_s, random_s, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        #endif // MULTISIZE

        // convert sampled coordinates and sizes into the device particle state
        particle_init <<< NB_P, TPB >>> (dev_particle, dev_random_x, dev_random_y, dev_random_z
            #ifdef MULTISIZE
            , dev_random_s
            #endif // MULTISIZE
        );

        cudaFreeHost(random_x); cudaFree(dev_random_x);
        cudaFreeHost(random_y); cudaFree(dev_random_y);
        cudaFreeHost(random_z); cudaFree(dev_random_z);

        #ifdef MULTISIZE
        cudaFreeHost(random_s); cudaFree(dev_random_s);
        #endif // MULTISIZE
        
        #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
        rs_swarm_init <<< NB_P, TPB >>> (dev_rs_swarm);
        #endif // COLLISION || (TRANSPORT && DIFFUSION)
        
        // write the initial state and active configuration before evolution
        std::filesystem::create_directories(PATH);
        save_variable(PATH + "variables.txt");

        #if defined(TRANSPORT) && defined(RADIATION)
        SAVE_OPTDEPTH_TO_FILE(idx_from, false);
        #endif // TRANSPORT && RADIATION

        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_from);
        #endif // SAVE_DENS

        SAVE_PARTICLE_TO_FILE(idx_from);

        msg_output(0);
    }
    else
    {
        // resume particle and imported-gas states from the requested output frame

        std::stringstream convert{argv[1]};
        if (!(convert >> idx_from))
        {
            std::cerr << "Error: Invalid resume file number: " << argv[1] << std::endl;
            return 1;
        }

        LOAD_PARTICLE_TO_VRAM(idx_from);

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);
        #endif // IMPORTGAS

        #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
        rs_swarm_init <<< NB_P, TPB >>> (dev_rs_swarm);
        #endif // COLLISION || (TRANSPORT && DIFFUSION)
        
        msg_output(idx_from);
    }

    #if !defined(TRANSPORT) && !defined(COLLISION)
    {
        std::cerr << "Error: No evolution module is enabled." << std::endl;
        return 1;
    }
    #endif // NO TRANSPORT and NO COLLISION

    #ifdef LOGTIMING
    #ifdef LOGOUTPUT
    {
        std::cerr << "Error: LOGTIMING and LOGOUTPUT cannot be enabled simultaneously." << std::endl;
        return 1;
    }
    #endif // LOGOUTPUT
    #ifdef TRANSPORT
    {
        std::cerr << "Error: LOGTIMING is not compatible with TRANSPORT module." << std::endl;
        return 1;
    }
    #endif // TRANSPORT
    #ifdef SAVE_DENS
    {
        std::cerr << "Error: LOGTIMING is not compatible with SAVE_DENS module." << std::endl;
        return 1;
    }
    #endif // SAVE_DENS
    #endif // LOGTIMING

    #ifdef LOGTIMING
    clock_sim = (idx_from == 0) ? 0.0 : int_pow(LOG_BASE, idx_from)*DT_OUT;
    #else  // LOGOUTPUT or LINEAR
    clock_sim =                         static_cast<real>(idx_from)*DT_OUT;
    #endif // LOGTIMING

    #ifdef COLLISION
    // evolve collisions over a fixed-position interval with controlled frozen-rate Bernoulli batches
    auto evolve_collisions = [&] (real duration)
    {
        // collisions change grain properties but not positions, so one KD tree serves the full interval
        col_tree_init <<< NB_P, TPB >>> (dev_col_tree, dev_particle);
        cukd::buildTree <tree, tree_traits> (dev_col_tree, col_tree_size, dev_boundbox);

        real elapsed = 0.0;
        while (elapsed < duration)
        {
            // calculate rates from a read-only snapshot and copy its diagnostics back to the live array
            cudaMemcpy(dev_particle_old, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToDevice);
            int tree_blocks = col_tree_size/TPB + 1;
            col_rate_calc <<< tree_blocks, TPB >>> (dev_col_rate, dev_particle_old, dev_col_tree, dev_boundbox
                #ifdef IMPORTGAS
                , dev_gasdens
                #endif
            );
            cudaMemcpy(dev_particle, dev_particle_old, sizeof(swarm)*N_P, cudaMemcpyDeviceToDevice);

            // use the largest total propensity to control every representative's event probability
            thrust::device_ptr <const real> rate_ptr(dev_col_rate);
            real max_rate = *thrust::max_element(rate_ptr, rate_ptr + N_P);
            real remaining = duration - elapsed;

            if (!(max_rate > 0.0))
            {
                // consume the remaining interval when no collision channel is active
                dt_col = remaining;
                elapsed = duration;
                break;
            }

            // keep the fastest frozen propensity below CFL_COL before sampling one event at most
            dt_col = fmin(CFL_COL/max_rate, remaining);
            col_proc_exec <<< tree_blocks, TPB >>> (dev_particle, dev_particle_old, dev_rs_swarm, dt_col,
                dev_col_tree, dev_boundbox
                #ifdef IMPORTGAS
                , dev_gasdens
                #endif
            );
            cudaDeviceSynchronize();

            elapsed += dt_col;
            clock_dyn = elapsed;
            count_col++;
        }
    };
    #endif

    for (int idx_file = idx_from + 1; idx_file <= SAVE_MAX; idx_file++)
    {
        // preload the next external frame and advance exactly one output interval
        
        dt_out = _get_dt_out(idx_file);

        #ifdef IMPORTGAS
        LOAD_GAS_NEXT_TO_VRAM(idx_file);
        real gas_frac = 0.0;
        #endif
        
        clock_out = 0.0;
        
        #ifdef TRANSPORT
        count_dyn = 0;
        #endif // TRANSPORT

        PRINT_TITLE_TO_SCREEN();
        
        do
        {
            #ifdef TRANSPORT
            // reduce all local inverse rates to a globally valid dynamics timestep
            dt_rate_calc <<< NB_P, TPB >>> (dev_dt_rate, dev_particle
                #ifdef IMPORTGAS
                , dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz,
                  dev_gasdens_next, dev_gasvelx_next, dev_gasvely_next, dev_gasvelz_next
                #endif
            );
            thrust::device_ptr <const real> dt_rate_ptr(dev_dt_rate);
            real max_dt_rate = *thrust::max_element(dt_rate_ptr, dt_rate_ptr + N_P);
            dt_dyn = fmin(DT_DYN, fmin(1.0/max_dt_rate, dt_out - clock_out));
            if (dt_dyn < DT_MIN) break;

            #ifdef IMPORTGAS
            // interpolate the working gas fields to the midpoint time of this dynamics step
            real gas_target = (clock_out + 0.5*dt_dyn)/dt_out;
            real gas_blend = (gas_target - gas_frac)/(1.0 - gas_frac);
            gas_interp_calc <<< NB_A, TPB >>> (dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz,
                dev_gasdens_next, dev_gasvelx_next, dev_gasvely_next, dev_gasvelz_next, gas_blend);
            gas_frac = gas_target;
            #endif

            #ifdef COLLISION
            // begin the symmetric composition with half a collision interval
            count_col = 0;
            clock_dyn = 0.0;
            evolve_collisions(0.5*dt_dyn);
            #endif

            #ifdef DIFFUSION
            // apply the first half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rs_swarm, 0.5*dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens
                #endif
            );
            #endif

            #ifdef RADIATION
            // drift to midpoint positions and reconstruct the optical depth used by the force solve
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            optdepth_init <<< NB_A, TPB >>> (dev_optdepth);
            optdepth_scat <<< NB_P, TPB >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, TPB >>> (dev_optdepth);
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz
                #endif
            );
            #else
            // complete transport in one launch when no midpoint radiation field is required
            ssa_transport <<< NB_P, TPB >>> (dev_particle, dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz
                #endif
            );
            #endif

            #ifdef DIFFUSION
            // apply the second half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rs_swarm, 0.5*dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens
                #endif
            );
            #endif

            #ifdef COLLISION
            // close the symmetric composition with half a collision interval
            evolve_collisions(0.5*dt_dyn);
            #endif

            cudaDeviceSynchronize();
            clock_sim += dt_dyn;
            clock_out += dt_dyn;
            count_dyn++;
            PRINT_VALUE_TO_SCREEN();
            #endif

            #if defined(COLLISION) && !defined(TRANSPORT)
            // collision-only runs evolve directly across the complete output interval
            count_col = 0;
            clock_dyn = 0.0;
            real duration = dt_out - clock_out;
            #ifdef IMPORTGAS
            real gas_target = (clock_out + 0.5*duration)/dt_out;
            real gas_blend = (gas_target - gas_frac)/(1.0 - gas_frac);
            gas_interp_calc <<< NB_A, TPB >>> (dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz,
                dev_gasdens_next, dev_gasvelx_next, dev_gasvely_next, dev_gasvelz_next, gas_blend);
            gas_frac = gas_target;
            #endif
            evolve_collisions(duration);
            clock_out += duration;
            clock_sim += duration;
            PRINT_VALUE_TO_SCREEN();
            #endif
        } while (clock_out < dt_out);

        #ifdef IMPORTGAS
        // replace the incrementally blended working fields by the exact endpoint snapshot
        cudaMemcpy(dev_gasdens, dev_gasdens_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_gasvelx, dev_gasvelx_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_gasvely, dev_gasvely_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_gasvelz, dev_gasvelz_next, sizeof(real)*N_G, cudaMemcpyDeviceToDevice);
        #endif // IMPORTGAS

        // reconstruct requested mesh fields and save particle frames under the configured output cadence
        #if defined(TRANSPORT) && defined(RADIATION)
        SAVE_OPTDEPTH_TO_FILE(idx_file, false);
        #endif // TRANSPORT && RADIATION
    
        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_file);
        #endif // SAVE_DENS

        #ifdef LOGTIMING
        {
            SAVE_PARTICLE_TO_FILE(idx_file);
        }
        #elif defined(LOGOUTPUT)
        if (is_log_power(idx_file))
        {
            SAVE_PARTICLE_TO_FILE(idx_file);
        }
        #else
        if (idx_file % LIN_BASE == 0)
        {
            SAVE_PARTICLE_TO_FILE(idx_file);
        }
        #endif // LOGTIMING

        msg_output(idx_file);
    }
 
    return 0;
}

// =========================================================================================================================
