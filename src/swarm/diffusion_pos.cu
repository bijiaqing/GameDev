#include <gpu.cuh>
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
void diffusion_pos (swarm *dev_particle, curs *dev_rngstate, real dt
#ifdef IMPORTGAS
    , const real *dev_gas_dens
#endif
)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;

    if (!_is_particle_active(y, z)) return;

    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);

    real lx = dev_particle[idx].velocity.x;
    real vy = dev_particle[idx].velocity.y;
    real lz = dev_particle[idx].velocity.z;

    if constexpr (N_Z == 1)
    {
        // evolve the vertically integrated disk directly in cylindrical coordinates
        real vx = lx / R;
        real vR = vy;
        real vx_cart = vR*cos(x) - vx*sin(x);
        real vy_cart = vR*sin(x) + vx*cos(x);

        auto diffusion = _get_dust_diffusion(x,y,z,
#ifdef MULTISIZE
            dev_particle[idx].par_size
#else
            S_0
#endif
#ifdef IMPORTGAS
            ,dev_gas_dens
#endif
        );
        real nu = diffusion.nu;
        curs rngstate = dev_rngstate[idx];

        real delta_x = 0.0;
        if (N_X > 1)
        {
            real diff_x = nu / SCHMIDT_X;
            real avg_x = dt*diff_x*diffusion.drift_phi_per_D / (R*R);
            real std_x = sqrt(2.0*dt*diff_x) / R;

            delta_x = avg_x + std_x*gpuRandNormalDouble(&rngstate);
        }

        real diff_R = nu / SCHMIDT_R;
        real avg_R = dt*diff_R*diffusion.drift_R_per_D;
        real std_R = sqrt(2.0*dt*diff_R);

        real x_new = x + delta_x;
        real R_new = R + avg_R + std_R*gpuRandNormalDouble(&rngstate);
        real z_new = 0.5*M_PI;
        real x_basis = x_new;
        _apply_diffusion_boundary(x_new, R_new, z_new);

        real vR_new = vx_cart*cos(x_basis) + vy_cart*sin(x_basis);
        real vx_new = vy_cart*cos(x_basis) - vx_cart*sin(x_basis);

        dev_particle[idx].position.x = x_new;
        dev_particle[idx].position.y = R_new;
        dev_particle[idx].position.z = 0.5*M_PI;
        dev_particle[idx].velocity.x = R_new*vx_new;
        dev_particle[idx].velocity.y = vR_new;
        dev_particle[idx].velocity.z = 0.0;
        dev_rngstate[idx] = rngstate;
        return;
    }

    // reconstruct the pre-displacement velocity in a fixed Cartesian basis
    real vx = lx / R;
    real vz = lz / y;
    real vR = vy*sin(z) + vz*cos(z);
    real vZ = vy*cos(z) - vz*sin(z);
    real vx_cart = vR*cos(x) - vx*sin(x);
    real vy_cart = vR*sin(x) + vx*cos(x);

    auto diffusion = _get_dust_diffusion(x,y,z,
#ifdef MULTISIZE
        dev_particle[idx].par_size
#else
        S_0
#endif
#ifdef IMPORTGAS
        ,dev_gas_dens
#endif
    );
    real nu = diffusion.nu;

    curs rngstate = dev_rngstate[idx]; // keep the random state local until all directional draws are complete

    real delta_x = 0.0;
    real delta_R = 0.0;
    real delta_Z = 0.0;

    if (N_X > 1)
    {
        // convert physical azimuthal diffusion length to angular noise
        real diff_x = nu / SCHMIDT_X;

        // include azimuthal diffusivity and selected gas-concentration drift
        real avg_x = dt*diff_x*diffusion.drift_phi_per_D / (R*R);
        real std_x = sqrt(2.0*dt*diff_x) / R;

        delta_x = avg_x + std_x*gpuRandNormalDouble(&rngstate);
    }

    if (N_Y > 1)
    {
        // include the cylindrical Ito drift and variable-diffusivity drift
        real diff_R = nu / SCHMIDT_R;

        real avg_R = dt*diff_R*diffusion.drift_R_per_D;
        real std_R = sqrt(2.0*dt*diff_R);

        delta_R = avg_R + std_R*gpuRandNormalDouble(&rngstate);
    }

    if (N_Z > 1)
    {
        // apply physical vertical diffusion at fixed cylindrical radius
        real diff_Z = nu / SCHMIDT_Z;

        // include vertical diffusivity and selected gas-concentration drift
        real avg_Z = dt*diff_Z*diffusion.drift_Z_per_D;
        real std_Z = sqrt(2.0*dt*diff_Z);

        delta_Z = avg_Z + std_Z*gpuRandNormalDouble(&rngstate);
    }

    real x_new = x + delta_x;
    real R_new = R + delta_R;
    real Z_new = Z + delta_Z;

    // continue across the cylindrical axis before reconstructing the spherical coordinates
    if (R_new < 0.0)
    {
        R_new = -R_new;
        x_new += M_PI;
    }

    real y_new = sqrt(R_new*R_new + Z_new*Z_new);
    real z_new = atan2(R_new, Z_new);

    // periodic storage wrapping must not rotate the physical Cartesian velocity basis
    real x_basis = x_new;
    _apply_diffusion_boundary(x_new, y_new, z_new);
    R_new = _get_cyl_R(y_new, z_new);
    Z_new = _get_cyl_Z(y_new, z_new);

    // project the unchanged Cartesian velocity into the new local spherical basis
    real sinz_new = R_new / y_new;
    real cosz_new = Z_new / y_new;
    real vR_new = vx_cart*cos(x_basis) + vy_cart*sin(x_basis);
    real vx_new = vy_cart*cos(x_basis) - vx_cart*sin(x_basis);

    lx = vx_new*R_new;
    vy = vR_new*sinz_new + vZ*cosz_new;
    lz = (vR_new*cosz_new - vZ*sinz_new)*y_new;

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
