#ifndef COAG_SWARM_HOST_CUH
#define COAG_SWARM_HOST_CUH
// Reuse production I/O, MRN size sampling, mass normalization and runtime helpers.
// Replace only the settled/smoothed spatial initializer and its domain-mass integral.
#define initmass_calc production_initmass_calc
#define rand_disk_poly production_rand_disk_poly
#ifdef GAMEDEV_ROCM
#include "../../../../inc/rocm/swarm/swarm_host.cuh"
#else
#include "../../../../inc/cuda/swarm/swarm_host.cuh"
#endif
#undef initmass_calc
#undef rand_disk_poly
#include "initial_profile.hpp"

inline void initmass_calc(std::vector<real> &mass_bank)
{
    // Every initial size has the same spatial distribution and containment.
    mass_bank.assign(1,initial_dust_mass());
}

inline void rand_disk_poly(real *x, real *r, real *theta, const real *, int count)
{
    for (int i=0; i<count; ++i) {
        x[i] = 0.5*(X_MIN+X_MAX);
        sample_initial_position(rand_generator,r[i],theta[i]);
    }
}
#endif
