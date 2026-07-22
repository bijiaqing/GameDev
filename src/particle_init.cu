

#include <graffiti_kern.cuh>
#include <helpers_paramphys.cuh>  // for _get_grain_mass

// =========================================================================================================================
// represented grain-number normalization
// =========================================================================================================================

// calculate the physical grain count represented by one equal-mass or equal-area swarm
__device__ __forceinline__
real _get_grain_number (real size)
{
    // the question is, if we want 
    // (1) the grain size distribution follows a -3.5 power-law,
    // (2) all swarms have an equal total surface area, and
    // (3) a certain total particle number N_P and represented domain dust mass M_D,
    // what is the size distribution of swarms (to be fixed in main.cu), and
    // and what is the grain number inside each swarm (to be solved here)

    // n_p(s) is the number of swarms for dust grains of individual grain size s
    // n_d(s) is the number of dust grains of individual grain size s in a swarm
    // m(s) = C_m*s^3 is one compact-grain mass, with C_m = pi*RHO_0/6
    // essentially, we want to solve n_d(s)
    
    // then the total mass of grains of size s is
    // (1) dm(s) = n_p(s) * n_d(s) * m(s) * ds
    // and  the total number of grains of size s is
    // (2) dn(s) = n_p(s) * n_d(s) * ds = n_0 * s^-3.5 * ds (n_0 = const)
    
    // to achieve all swarms having the same total surface area, there is
    // (3) n_d(s) = n_1 * s^-2 (n_1 = const)
    // combining (2) and (3) there is 
    // (4) n_p(s) = n_2 * s^-1.5 (n_2 = const), which explains why pow_idx = -1.5 in main.cu
    
    // (note) if all swarms have the same total mass, there is
    // (3') n_d(s) = n_1 * s^-3 (n_1 = const)
    // combining (2) and (3') there is 
    // (4') n_p(s) = n_2 * s^-0.5 (n_2 = const), which explains why pow_idx = -0.5 in main.cu

    // with (4), the total number of all swarms is 
    // (5) N_P = integrate( n_p(s) ds ) = integrate( n_2 * s^-1.5 ds )
    // with (5), there is
    // (6) n_2 = 0.5*N_P / (s_min^-0.5 - s_max^-0.5)
    
    // with (1), the total mass of all swarms is 
    // (7) M_D = integrate( dm(s) ) = integrate( n_p(s) * n_d(s) * C_m * s^3 ds )
    // with (3), (4), and (7) there is
    // (8) M_D = 2*n_1*n_2*C_m*(s_max^0.5 - s_min^0.5)
    
    // with (6) and (8) there is
    // (9) n_1 = M_D / N_P / C_m / (s_max^0.5 - s_min^0.5) * (s_min^-0.5 - s_max^-0.5)

    // finally, combining (3) and (9), we know n_d(s)

    real numr = M_D / N_P / _get_grain_mass(size);

    #if defined(TRANSPORT) && defined(RADIATION) // keep total surface area for individual swarms the same
    {
        if (INIT_SMIN != INIT_SMAX) // if min = max, converge back to the ordinary case
        {
            numr *= size;
            numr *= (pow(INIT_SMIN, -0.5) - pow(INIT_SMAX, -0.5));
            numr /= (pow(INIT_SMAX,  0.5) - pow(INIT_SMIN,  0.5));
        }
    }
    #endif // TRANSPORT && RADIATION

    return numr;
}

// =========================================================================================================================
// kernel: particle_init
// initialize representative positions, cylindrical Keplerian rotation, and optional grain properties
//
// parallelization: one thread per representative particle
// =========================================================================================================================

__global__
void particle_init (swarm *dev_particle, 
    const real *dev_random_x, const real *dev_random_y, const real *dev_random_z
    #ifdef MULTISIZE
    , const real *dev_random_s
    #endif // MULTISIZE
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        // copy host-sampled spherical positions into the particle state
        dev_particle[idx].position.x = dev_random_x[idx];
        dev_particle[idx].position.y = dev_random_y[idx];
        dev_particle[idx].position.z = dev_random_z[idx];
        
        real R = dev_random_y[idx]*sin(dev_random_z[idx]);

        // initialize cylindrical Keplerian specific angular momentum with no meridional motion
        dev_particle[idx].velocity.x = sqrt(G*M_S*R);
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;

        #ifdef MULTISIZE
        // attach the sampled grain species and its represented physical grain count
        dev_particle[idx].par_size   = dev_random_s[idx];
        dev_particle[idx].par_numr   = _get_grain_number(dev_random_s[idx]);
        #endif // MULTISIZE

        #ifdef COLLISION
        // clear collision diagnostics before the first KNN rate calculation
        dev_particle[idx].col_rate   = 0.0;
        dev_particle[idx].max_dist   = 0.0;
        #endif // COLLISION
    }
}

// =========================================================================================================================
