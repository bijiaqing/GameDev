#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// represented grain-number normalization
// =========================================================================================================================

// calculate the physical grain count represented by one normalized equal-mass or equal-area swarm
__device__ __forceinline__
real _get_grain_number (real size, real mass_norm)
{
    // the question is, if we want
    // (1) the grain size distribution follows a -3.5 power-law,
    // (2) all swarms have an equal total surface area, and
    // (3) a certain total particle number N_P and represented domain dust mass dust_mass,
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
    // (7) dust_mass = integrate( dm(s) ) = integrate( n_p(s) * n_d(s) * C_m * s^3 ds )
    // with (3), (4), and (7) there is
    // (8) dust_mass = 2*n_1*n_2*C_m*(s_max^0.5 - s_min^0.5)

    // with (6) and (8) there is
    // (9) n_1 = dust_mass / N_P / C_m / (s_max^0.5 - s_min^0.5) * (s_min^-0.5 - s_max^-0.5)

    // finally, combining (3) and (9), we know n_d(s)
    // replace the continuous normalization by mass_norm so the finite ensemble sums to dust_mass within roundoff

    return mass_norm*_get_mass_weight(size) / static_cast<real>(N_P) / _get_grain_mass(size);
}

// =========================================================================================================================
// kernel: particle_init
// initialize representative positions, steady drag-coupled drift, and optional grain properties
//
// parallelization: one thread per representative particle
// =========================================================================================================================

__global__
void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z
    #ifdef MULTISIZE
    , const real *dev_random_s, real mass_norm
    #endif // MULTISIZE
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    // copy host-sampled spherical positions into the particle state
    dev_particle[idx].position.x = dev_random_x[idx];
    dev_particle[idx].position.y = dev_random_y[idx];
    dev_particle[idx].position.z = (N_Z > 1) ? dev_random_z[idx] : 0.5*M_PI;

    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    real R = y*sin(z);
    real Z = y*cos(z);

    #ifdef MULTISIZE
    real size = dev_random_s[idx];
    #else  // MONOSIZE
    real size = S_0;
    #endif // MULTISIZE

    // reproduce the fluid steady single-species drift in cylindrical coordinates
    real h_g = _get_hg(R);
    real omega = _get_omegaK(R);
    real v_K = R*omega;
    real eta = _get_eta(R, Z, h_g);
    real vgas_x = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(R, Z, size, h_g
        #ifdef IMPORTGAS
        , dev_particle[idx].position.x, y, z, nullptr
        #endif // IMPORTGAS
    );

    real vgas_R = 0.0;
    #ifdef VISC_ACCRETION
    vgas_R = _get_visc_vel(R, Z, h_g);
    #endif // VISC_ACCRETION

    real vel_R = (vgas_R + 2.0*stokes*(vgas_x - v_K)) / (1.0 + stokes*stokes);
    real vel_x = vgas_x - 0.5*stokes*vel_R;
    real vel_Z = (N_Z > 1) ? -stokes*omega*Z : 0.0;

    real vel_y = vel_R*sin(z) + vel_Z*cos(z);
    real vel_z = vel_R*cos(z) - vel_Z*sin(z);

    dev_particle[idx].velocity.x = R*vel_x;
    dev_particle[idx].velocity.y = vel_y;
    dev_particle[idx].velocity.z = y*vel_z;

    #ifdef MULTISIZE
    // attach the sampled grain species and its represented physical grain count
    dev_particle[idx].par_size   = size;
    dev_particle[idx].par_numr   = _get_grain_number(size, mass_norm);
    #endif // MULTISIZE
}

// =========================================================================================================================
