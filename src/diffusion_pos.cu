#ifdef DIFFUSION

#include <_diffusion.cuh>
#include <_transport.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: diffusion_pos
// apply one cylindrical diffusion SDE step while preserving physical Cartesian velocity
//
// parallelization: one thread and one independent cuRAND stream per representative particle
//
// per call:
//   1 reconstruct the physical velocity before spatial redistribution
//   2 sample azimuthal, cylindrical-radial, and vertical drift-diffusion increments
//   3 map back to spherical position and reconstruct the stored velocity variables
// =========================================================================================================================

__global__
void diffusion_pos (swarm *dev_particle, curs *dev_rngstate, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;

    if (!_is_particle_active(y, z)) return;

    real R = y*sin(z);

    real lx = dev_particle[idx].velocity.x;
    real vy = dev_particle[idx].velocity.y;
    real lz = dev_particle[idx].velocity.z;

    // reconstruct the pre-displacement velocity in a fixed Cartesian basis
    real vphi  = lx / R;
    real vtheta = lz / y;
    real vel_R = vy*sin(z) + vtheta*cos(z);
    real vel_Z = vy*cos(z) - vtheta*sin(z);
    real vel_Cx = vel_R*cos(x) - vphi*sin(x);
    real vel_Cy = vel_R*sin(x) + vphi*cos(x);

    real h_g = _get_hg(R);
    real nu = _get_nu(R, h_g);

    curs rngstate = dev_rngstate[idx]; // keep the random state local until all directional draws are complete

    real delta_x = 0.0;
    real delta_R = 0.0;
    real delta_Z = 0.0;

    if (N_X > 1)
    {
        // convert physical azimuthal diffusion length to angular noise
        real coeff_x = nu / SCHMIDT_X;

        // retain the density-diffusion drift placeholder for a future azimuthally varying diffusivity
        real grad_x = 0.0; // replace with the partial derivative of coeff_x with respect to x
        real avg_x = dt*grad_x / (R*R);
        real std_x = sqrt(2.0*dt*coeff_x) / R;

        delta_x = avg_x + std_x*curand_normal_double(&rngstate);
    }

    if (N_Y > 1)
    {
        // include the cylindrical Ito drift and variable-diffusivity drift
        real coeff_R = nu / SCHMIDT_R;
        
        real avg_R = dt*_get_diff_drift_R(R, coeff_R);
        real std_R = sqrt(2.0*dt*coeff_R);

        delta_R = avg_R + std_R*curand_normal_double(&rngstate);
    }

    if (N_Z > 1)
    {
        // apply physical vertical diffusion at fixed cylindrical radius
        real coeff_Z = nu / SCHMIDT_Z;

        // retain the density-diffusion drift placeholder for a future vertically varying diffusivity
        real grad_Z = 0.0; // replace with the partial derivative of coeff_Z with respect to Z
        real avg_Z = dt*grad_Z;
        real std_Z = sqrt(2.0*dt*coeff_Z);

        delta_Z = avg_Z + std_Z*curand_normal_double(&rngstate);
    }

    real x_new = x + delta_x;
    real R_new = R + delta_R;
    real Z_new = Z + delta_Z;

    // map a negative cylindrical radius to the equivalent positive-radius coordinate
    if (R_new < 0.0)
    {
        R_new = -R_new;
        x_new += M_PI;
    }

    real y_new = sqrt(R_new*R_new + Z_new*Z_new);
    real z_new = atan2(R_new, Z_new);

    _apply_diffusion_boundary(x_new, y_new, z_new);
    R_new = y_new*sin(z_new);
    Z_new = y_new*cos(z_new);

    // project the unchanged Cartesian velocity into the new local spherical basis
    real sin_znew = R_new / y_new;
    real cos_znew = Z_new / y_new;
    real vel_Rnew = vel_Cx*cos(x_new) + vel_Cy*sin(x_new);
    real vphi_new = vel_Cy*cos(x_new) - vel_Cx*sin(x_new);

    lx = vphi_new*R_new;
    vy = vel_Rnew*sin_znew + vel_Z*cos_znew;
    lz = (vel_Rnew*cos_znew - vel_Z*sin_znew)*y_new;

    dev_particle[idx].position.x = x_new;
    dev_particle[idx].position.y = y_new;
    dev_particle[idx].position.z = z_new;
    dev_particle[idx].velocity.x = lx;
    dev_particle[idx].velocity.y = vy;
    dev_particle[idx].velocity.z = lz;

    dev_rngstate[idx] = rngstate; // persist the advanced random stream
}

// =========================================================================================================================

#endif // DIFFUSION
