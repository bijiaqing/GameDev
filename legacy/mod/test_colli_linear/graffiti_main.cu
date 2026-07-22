#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill
#include <iostream>         // for std::cout, std::endl
#include <sstream>          // for std::stringstream

#ifdef COLLISION
#include <thrust/device_ptr.h>  // for thrust::device_ptr
#include <thrust/extrema.h>     // for thrust::max_element
#endif // COLLISION

#include <graffiti_kern.cuh>
#include <graffiti_host.cuh>

std::mt19937 rand_generator;

const std::string PATH = PATH_OUT; // PATH_OUT is defined by Makefile as a string literal, convert to std::string

// =========================================================================================================================

int main (int argc, char **argv)
{   // pre-loop memory allocation
    
    int idx_from;
    real clock_sim;   // total simulation time
    real clock_out;   // time accumulated toward the next output
    real dt_out;      // output timestep

    #ifdef TRANSPORT
    int count_dyn;    // how many dynamics timesteps in one output timestep
    real dt_dyn;      // dynamical timestep
    #endif // TRANSPORT
    
    #ifdef COLLISION
    int count_col;    // how many collision calculations in one dynamics timestep
    real clock_dyn;   // time accumulated toward the next dynamics timestep
    real dt_col;      // collisional timestep
    #endif // COLLISION
    
    std::string fname;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_P);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_P);
    
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
    #endif // IMPORTGAS
    
    #if defined(TRANSPORT) && defined(RADIATION)
    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G);
    #endif // RADIATION

    #ifdef COLLISION
    bbox *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(bbox));

    tree *dev_col_tree;
    cudaMalloc((void**)&dev_col_tree, sizeof(tree)*N_P);

    curs *dev_rs_grids;
    cudaMalloc((void**)&dev_rs_grids, sizeof(curs)*N_G);
    
    real *dev_col_rand, *dev_col_expt, *dev_col_rate;
    cudaMalloc((void**)&dev_col_rand, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_col_expt, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_G);

    int  *dev_col_flag;
    cudaMalloc((void**)&dev_col_flag, sizeof(int) *N_G);
    #endif // COLLISION

    #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
    curs *dev_rs_swarm;
    cudaMalloc((void**)&dev_rs_swarm, sizeof(curs)*N_P);
    #endif // COLLISION or DIFFUSION

    if (argc <= 1)
	{   // no argument input, start a new simulation with certain initial conditions
        
        idx_from = 0;

        real *random_x, *dev_random_x;
        real *random_y, *dev_random_y;
        real *random_z, *dev_random_z;
        real *random_s, *dev_random_s;

        cudaMallocHost((void**)&random_x, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_x, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_y, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_y, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_z, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_z, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_s, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_s, sizeof(real)*N_P);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from); // for gasdens, gasvelx, gasvely, gasvelz, and dev_*

        real *epsilon;
        cudaMallocHost((void**)&epsilon,  sizeof(real)*N_G);
        
        if (!load_epsilon(PATH, idx_from, epsilon))
        {
            std::cerr << "Error: Failed to load gas data files for frame " << idx_from << std::endl;
            return 1;
        }

        // note: the current implementation only works for equal-size, equal-mass dust particles
        rand_from_file(random_x, random_y, random_z, N_P, gasdens, epsilon);
        
        cudaFreeHost(epsilon);
        #else // NOT IMPORTGAS
        real idx_rho_g = IDX_P - 0.5*IDX_Q - 1.5; // volumetric gas density power-law index

        rand_uniform(random_x, N_P, INIT_XMIN, INIT_XMAX);
        rand_convpow(random_y, N_P, INIT_YMIN, INIT_YMAX, idx_rho_g, 0.05*R_0, N_Y);
        rand_uniform(random_z, N_P, INIT_ZMIN, INIT_ZMAX);
        #endif // IMPORTGAS
        
        real idx_swarm = -0.5;  // all swarms have an equal mass
        #if defined(TRANSPORT) && defined(RADIATION)
        idx_swarm = -1.5;       // all swarms have an equal surface area, see _get_grain_number
        #endif // TRANSPORT && RADIATION
        rand_4_linear(random_s, N_P);

        cudaMemcpy(dev_random_x, random_x, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_y, random_y, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_z, random_z, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_s, random_s, sizeof(real)*N_P, cudaMemcpyHostToDevice);

        particle_init <<< NB_P, TPB >>> (dev_particle, dev_random_x, dev_random_y, dev_random_z, dev_random_s);

        cudaFreeHost(random_x); cudaFree(dev_random_x);
        cudaFreeHost(random_y); cudaFree(dev_random_y);
        cudaFreeHost(random_z); cudaFree(dev_random_z);
        cudaFreeHost(random_s); cudaFree(dev_random_s);
        
        #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
        rs_swarm_init <<< NB_P, TPB >>> (dev_rs_swarm);
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, TPB >>> (dev_rs_grids);
        #endif // COLLISION

        save_variable(PATH + "variables.txt");

        #if defined(TRANSPORT) && defined(RADIATION)
        SAVE_OPTDEPTH_TO_FILE(idx_from, true);
        #endif // RADIATION

        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_from);
        #endif // SAVE_DENS

        SAVE_PARTICLE_TO_FILE(idx_from);

        msg_output(0);
    }
    else
    {   // with argument input, resume from a previous simulation by loading data files

        std::stringstream convert{argv[1]}; // read resume file number from input argument
        if (!(convert >> idx_from)) // set idx_from by input argument, exit if failed
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
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, TPB >>> (dev_rs_grids);
        #endif // COLLISION

        msg_output(idx_from);
    }

    #if defined(COLLISION) && !defined(TRANSPORT) // only need to build KD-tree once and can use it forever
    col_tree_init <<< NB_P, TPB >>> (dev_col_tree, dev_particle);
    cukd::buildTree <tree, tree_traits> (dev_col_tree, N_P, dev_boundbox);
    #endif // COLLISION but NO TRANSPORT

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

    for (int idx_file = idx_from + 1; idx_file <= SAVE_MAX; idx_file++)
    {   // parental loop over output files
        
        dt_out = _get_dt_out(idx_file);
        
        clock_out = 0.0; // reset output clock
        
        #ifdef TRANSPORT
        count_dyn = 0; // reset dynamical step counter
        #endif // TRANSPORT

        PRINT_TITLE_TO_SCREEN();
        
        do  // begin of while (clock_out < dt_out)
        {   // loop over dynamics timesteps within one output timestep
            
            #ifdef TRANSPORT
            dt_dyn = fmin(DT_DYN, dt_out - clock_out);
            
            if (dt_dyn < DT_MIN)
            {
                break; // jump to file save if dt_dyn is too small
            }

            #ifdef RADIATION
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            optdepth_init <<< NB_A, TPB >>> (dev_optdepth);
            optdepth_scat <<< NB_P, TPB >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, TPB >>> (dev_optdepth);
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth, dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz
                #endif // IMPORTGAS
            );
            #else  // NO RADIATION
            ssa_transport <<< NB_P, TPB >>> (dev_particle, dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens, dev_gasvelx, dev_gasvely, dev_gasvelz
                #endif // IMPORTGAS
            );
            #endif // RADIATION
            
            #ifdef DIFFUSION
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rs_swarm, dt_dyn
                #ifdef IMPORTGAS
                , dev_gasdens
                #endif // IMPORTGAS
            );
            #endif // DIFFUSION

            cudaDeviceSynchronize();

            #ifndef COLLISION // only update simulation clock here if NO COLLISION
            clock_sim += dt_dyn;
            #endif // NO COLLISION

            clock_out += dt_dyn;
            count_dyn ++;

            PRINT_VALUE_TO_SCREEN();
            #endif // TRANSPORT
            
            #ifdef COLLISION
            count_col = 0;

            #ifdef TRANSPORT
            clock_dyn = 0.0;
            
            // need to rebuild KD-tree for every dynamical timestep
            col_tree_init <<< NB_P, TPB >>> (dev_col_tree, dev_particle);
            cukd::buildTree <tree, tree_traits> (dev_col_tree, N_P, dev_boundbox);
            #endif // TRANSPORT
        
            do  // begin of while (clock_dyn < dt_dyn) or while (clock_out < dt_out)
            {   // loop over collision calculations within one dynamics timestep or output timestep
                
                col_rate_init <<< NB_A, TPB >>> (dev_col_rate, dev_col_expt, dev_col_rand);
                col_rate_calc <<< NB_P, TPB >>> (dev_col_rate, dev_particle, dev_col_tree, dev_boundbox
                    #ifdef IMPORTGAS
                    , dev_gasdens
                    #endif
                );
                
                // use thrust to find maximum collision rate on device
                thrust::device_ptr <const real> dev_ptr(dev_col_rate);
                thrust::device_ptr <const real> max_ptr = thrust::max_element(dev_ptr, dev_ptr + N_G);
                real max_rate = *max_ptr; // dereference triggers single-value cudaMemcpy from device to host
                
                // Compute collisional timestep
                dt_col = 1.0 / max_rate;

                if (dt_col < DT_MIN)
                {
                    std::cerr << "Error: Minimum collisional timestep reached due to high collision rate." << std::endl;
                    return 1;
                }
                
                #ifdef TRANSPORT
                dt_col = fmin(dt_col, dt_dyn - clock_dyn);
                #else  // NO TRANSPORT
                dt_col = fmin(dt_col, dt_out - clock_out);
                #endif // TRANSPORT

                if (dt_col < DT_MIN)
                {
                    break; // jump to transport, or file save, if dt_col is too small
                }

                col_flag_calc <<< NB_A, TPB >>> (dev_col_flag, dev_rs_grids, dev_col_rand, dev_col_rate, dt_col);
                col_proc_exec <<< NB_P, TPB >>> (dev_particle, dev_rs_swarm, dev_col_expt, dev_col_rand, dev_col_flag, dev_col_tree, dev_boundbox
                    #ifdef IMPORTGAS
                    , dev_gasdens
                    #endif
                );

                cudaDeviceSynchronize();

                #ifdef TRANSPORT
                clock_dyn += dt_col;
                #else  // NO TRANSPORT
                clock_out += dt_col;
                #endif // TRANSPORT
                
                clock_sim += dt_col;
                count_col ++;

                PRINT_VALUE_TO_SCREEN();
            }
            #ifdef TRANSPORT
            while (clock_dyn < dt_dyn);
            #else  // NO TRANSPORT
            while (clock_out < dt_out);
            #endif // TRANSPORT
            #endif // COLLISION

            // prevent infinite loop when COLLISION but NO TRANSPORT
            #if defined(COLLISION) && !defined(TRANSPORT)
            if (dt_out - clock_out < DT_MIN) 
            {
                break; // jump to file save if remaining time is too small
            }
            #endif // COLLISION but NO TRANSPORT

        } while (clock_out < dt_out);

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_file);
        #endif // IMPORTGAS

        #if defined(TRANSPORT) && defined(RADIATION)
        SAVE_OPTDEPTH_TO_FILE(idx_file, false);
        #endif // RADIATION
    
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
