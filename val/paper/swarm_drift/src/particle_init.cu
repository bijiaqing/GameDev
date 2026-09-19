#include <swarm_kern.cuh>

// Fung & Muley (2019), Section 3.3, Eqs. 42-43, as printed.
// These are approximate equilibrium initial conditions, not an exact trajectory.
__global__ void particle_init(swarm *particle, const real *, const real *, const real *)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i >= N_P) return;
    const real h2 = ASPR_0*ASPR_0;
    const real deficit = h2/(1.0 + sqrt(1.0 - h2));
    const real st2 = STOKES_0*STOKES_0;
    const real denom = 1.0 + st2;
    const real L = deficit/denom*(1.0 + 1.5*st2*deficit/(denom*denom));
    particle[i].position = {0.0, R_0, 0.5*M_PI};
    particle[i].velocity = {1.0 - L, -2.0*L*(1.0 + 0.5*L)*STOKES_0, 0.0};
}
