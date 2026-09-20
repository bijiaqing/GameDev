#ifndef COAG_SWARM_HOST_CUH
#define COAG_SWARM_HOST_CUH
// Reuse production I/O, size sampling, mass normalization and runtime helpers.
// Replace only the settled/smoothed spatial initializer and its domain-mass integral.
#define initmass_calc production_initmass_calc
#define rand_disk_poly production_rand_disk_poly
#include "../../../../inc/swarm/swarm_host.cuh"
#undef initmass_calc
#undef rand_disk_poly
#include <cmath>
#include <random>
#include <const_defs.cuh>

// Well-mixed dust: rho_d = METAL_Z*rho_g, using the production spherical
// hydrostatic gas profile. r is in au; the returned weight includes r^2 sin(theta).
inline double initial_weight(double r, double theta)
{
    double R = r*std::sin(theta);
    double h2 = ASPR_0*ASPR_0*std::sqrt(R);
    return std::pow(r, -0.25)*std::pow(std::sin(theta), -1.25)
         * std::exp((std::sin(theta)-1.0)/h2);
}

// Simpson integration over the actual spherical domain, including both hemispheres.
inline double initial_integral(int n = 512, int radial_power = 0)
{
    double dr = (Y_MAX-Y_MIN)/AU/n, dtheta = (Z_MAX-Z_MIN)/n;
    double sum = 0.0;
    for (int i=0; i<=n; ++i) {
        double r = Y_MIN/AU + i*dr;
        int wi = (i==0 || i==n) ? 1 : (i%2 ? 4 : 2);
        for (int j=0; j<=n; ++j) {
            int wj = (j==0 || j==n) ? 1 : (j%2 ? 4 : 2);
            sum += wi*wj*initial_weight(r,Z_MIN+j*dtheta)*std::pow(r,radial_power);
        }
    }
    return sum*dr*dtheta/9.0;
}

inline double initial_dust_mass()
{
    double rho0 = METAL_Z*SIGMA_0/(std::sqrt(2.0*M_PI)*ASPR_0*AU);
    return 2.0*M_PI*rho0*AU*AU*AU*initial_integral();
}

inline void sample_initial_position(std::mt19937 &rng, double &r, double &theta)
{
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    // For the fixed p=-1, q=-1/2 setup, stratification <=1 and this bounds
    // r^-1/4 sin(theta)^-5/4 throughout the domain.
    const double bound = std::pow(Y_MIN/AU,-0.25)*std::pow(std::sin(Z_MIN),-1.25);
    do {
        r = Y_MIN/AU + (Y_MAX-Y_MIN)/AU*uniform(rng);
        theta = Z_MIN + (Z_MAX-Z_MIN)*uniform(rng);
    } while (uniform(rng)*bound > initial_weight(r,theta));
    r *= AU;
}
static_assert(IDX_P == -1.0 && IDX_Q == -0.5 && R_0 == AU);

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
