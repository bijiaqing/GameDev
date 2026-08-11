#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// represented grain-number normalization
// =========================================================================================================================

#ifdef MULTISIZE
// calculate the physical grain count represented by one containment-weighted sampled swarm
__device__ __forceinline__
real _get_grain_number (real size, real domain_mass, real mass_norm)
{
    // target an MRN number spectrum dN/ds proportional to s^-3.5 with compact-grain mass m_g(s) proportional to s^3
    // the corresponding physical mass spectrum is therefore dM/ds proportional to s^-0.5
    // sample q(s) proportional to s^-0.5 for equal full-column swarm mass when radiation is disabled
    // sample q(s) proportional to s^-1.5 for equal full-column swarm area when radiation is enabled
    // convert either proposal to the target mass spectrum with the importance factor _get_mass_weight(s)
    // multiply by domain_mass=I(s) because the finite radial-polar domain contains a size-dependent fraction of each column
    // assign represented swarm mass M_i=mass_norm*w(s_i)*I(s_i)/N_P
    // choose mass_norm on the host so sum_i M_i equals total_dust_mass within roundoff
    // convert represented mass to physical grain count with N_i=M_i/m_g(s_i)

    return mass_norm*_get_mass_weight(size)*domain_mass
         / static_cast<real>(N_P) / _get_grain_mass(size);
}
#endif // MULTISIZE

// =========================================================================================================================
// kernel: particle_init
// initialize representative positions, steady drag-coupled drift, and optional grain properties
//
// parallelization: one thread per representative particle
// =========================================================================================================================

__global__
void particle_init (swarm *dev_particle, const real *dev_randposx, const real *dev_randposy, const real *dev_randposz
    #ifdef MULTISIZE
    , const real *dev_randsize, const real *dev_mass_bank, int mass_bin_count, real mass_norm
    #endif // MULTISIZE
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    // copy host-sampled spherical positions into the particle state
    dev_particle[idx].position.x = (N_X > 1) ? dev_randposx[idx] : 0.5*(X_MIN + X_MAX);
    dev_particle[idx].position.y = dev_randposy[idx];
    dev_particle[idx].position.z = (N_Z > 1) ? dev_randposz[idx] : 0.5*M_PI;

    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);

    #ifdef MULTISIZE
    real size = dev_randsize[idx];
    #else  // MONOSIZE
    real size = S_0;
    #endif // MULTISIZE

    // reproduce the fluid steady single-species drift in cylindrical coordinates
    real h_g = _get_hg(R);
    real omega = _get_omegaK(R);
    real v_K = R*omega;
    real eta = _get_eta(R, Z, h_g);
    real vx_g = v_K*sqrt(fmax(1.0 - 2.0*eta, 0.0));
    real stokes = _get_stokes(R, Z, h_g, size
        #ifdef IMPORTGAS
        , dev_particle[idx].position.x, y, z, dev_gas_dens
        #endif // IMPORTGAS
    );

    real vR_g = 0.0;
    #ifdef VISC_FLOW
    vR_g = _get_visc_vel(R, Z, h_g);
    #endif // VISC_FLOW

    real vR = (vR_g + 2.0*stokes*(vx_g - v_K)) / (1.0 + stokes*stokes);
    real vx = vx_g - 0.5*stokes*vR;
    real vZ = (N_Z > 1) ? -stokes*omega*Z : 0.0;

    dev_particle[idx].velocity.x = R*vx;
    if (N_Z == 1)
    {
        dev_particle[idx].velocity.y = vR;
        dev_particle[idx].velocity.z = 0.0;
    }
    else
    {
        real vy = vR*sin(z) + vZ*cos(z);
        real vz = vR*cos(z) - vZ*sin(z);

        dev_particle[idx].velocity.y = vy;
        dev_particle[idx].velocity.z = y*vz;
    }

    #ifdef MULTISIZE
    // attach the sampled grain species and its represented physical grain count
    real domain_mass = _get_domain_mass(size, dev_mass_bank, mass_bin_count);
    dev_particle[idx].par_size   = size;
    dev_particle[idx].par_numr   = _get_grain_number(size, domain_mass, mass_norm);
    #endif // MULTISIZE
}

// =========================================================================================================================
