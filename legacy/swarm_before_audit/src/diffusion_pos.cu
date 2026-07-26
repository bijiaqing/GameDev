#if defined(TRANSPORT) && defined(DIFFUSION)

#include <graffiti_kern.cuh>
#include <helpers_paramphys.cuh>  // for _get_hg, _get_nu
#include <helpers_diffusion.cuh>  // for _get_term_grad_cyl
#include <helpers_transport.cuh>  // for _if_out_of_box

// =========================================================================================================================
// Kernel: diffusion_pos
// Purpose: Apply turbulent diffusion effects to particle positions (radial and vertical)
// Dependencies: graffiti.cuh (for types and constants),
//               helpers_paramphys.cuh (for _get_hg, _get_nu),
//               helpers_diffusion.cuh (for _get_term_grad_cyl),
//               curand (for normal distribution random numbers)
// =========================================================================================================================

__global__
void diffusion_pos (swarm *dev_particle, curs *dev_rs_swarm, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif // IMPORTGAS
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        real x = dev_particle[idx].position.x;
        real y = dev_particle[idx].position.y;
        real z = dev_particle[idx].position.z;

        real R = y*sin(z);
        real Z = y*cos(z);
        
        real h_g = _get_hg(R);
        real nu = _get_nu(R, h_g);
        
        curs rs_swarm = dev_rs_swarm[idx]; // use a local state for less global memory traffic 

        real delta_x = 0.0;
        real delta_R = 0.0;
        real delta_Z = 0.0;

        real term_x, term_R, term_Z;
        
        _get_term_grad_cyl(x, y, z, term_x, term_R, term_Z
            #ifdef IMPORTGAS
            , dev_gasdens
            #endif // IMPORTGAS
        );
        
        if (N_X > 1)
        {
            real coeff_x = nu / SC_X;
            
            real avg_x1 = dt*coeff_x*term_x;
            real var_x1 = dt*coeff_x*2.0;
            
            real avg_x2 = 0.0; // assuming both nu and alpha are constant in x-direction
            real var_x2 = avg_x2*avg_x2;
            
            delta_x = (avg_x1 + avg_x2) + sqrt(var_x1 + var_x2)*curand_normal_double(&rs_swarm);
        }
        
        if (N_Y > 1)
        {
            real coeff_R = nu / SC_R;
            
            real avg_R1 = dt*coeff_R*term_R;
            real var_R1 = dt*coeff_R*2.0;

            #ifndef CONST_NU
            // given nu ~ alpha*H_gas*c_s ~ pow(R, IDX_Q + 1.5), then d(coeff_R)/dR = (IDX_Q + 1.5)*coeff_R / R
            real avg_R2 = dt*coeff_R*(IDX_Q + 1.5) / R;
            #else  // CONST_NU
            real avg_R2 = 0.0;
            #endif // NOT CONST_NU
            real var_R2 = avg_R2*avg_R2;

            delta_R = (avg_R1 + avg_R2) + sqrt(var_R1 + var_R2)*curand_normal_double(&rs_swarm);
        }

        if (N_Z > 1)
        {
            real coeff_Z = nu / SC_Z;

            real avg_Z1 = dt*coeff_Z*term_Z;
            real var_Z1 = dt*coeff_Z*2.0;

            real avg_Z2 = 0.0; // assuming both nu and alpha are constant in Z-direction
            real var_Z2 = avg_Z2*avg_Z2;

            delta_Z = (avg_Z1 + avg_Z2) + sqrt(var_Z1 + var_Z2)*curand_normal_double(&rs_swarm);
        }
        
        real x_new = x + delta_x;
        real R_new = R + delta_R;
        real Z_new = Z + delta_Z;

        real y_new = sqrt(R_new*R_new + Z_new*Z_new);
        real z_new = atan2(R_new, Z_new);

        real lx = dev_particle[idx].velocity.x;
        real vy = dev_particle[idx].velocity.y;
        real lz = dev_particle[idx].velocity.z;

        _if_out_of_box(x_new, y_new, z_new, lx, vy, lz);

        dev_particle[idx].position.x = x_new;
        dev_particle[idx].position.y = y_new;
        dev_particle[idx].position.z = z_new;
        dev_particle[idx].velocity.x = lx;
        dev_particle[idx].velocity.y = vy;
        dev_particle[idx].velocity.z = lz;

        dev_rs_swarm[idx] = rs_swarm; // update the global state, otherwise, always the same number
    }
}

// =========================================================================================================================

#endif // DIFFUSION
