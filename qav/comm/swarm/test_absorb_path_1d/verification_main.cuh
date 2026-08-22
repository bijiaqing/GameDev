#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <_transport.cuh>
#include <device_api.cuh>
#include <param_grid.cuh>
#include <swarm_kern.cuh>

// test-only driver for deterministic endpoint absorption
// constant radial paths expose exact inner and outer crossing times, while the final inactive state is passed through the
// production dynamics-rate and particle-to-grid routines to verify that absorbed representatives no longer contribute

namespace
{
const std::string output_path = PATH_OUT;
constexpr real time_end = 1.0;
constexpr real outer_hit = 0.73;
constexpr real inner_hit = 0.41;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open absorption output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write absorption output");
}
}

int main ()
{
    constexpr real x_mid = 0.5*(X_MIN + X_MAX);
    constexpr real z_mid = 0.5*M_PI;
    const real y_initial[N_P] = {1.2, 0.8, 1.0};
    const real vy_initial[N_P] = {
        (Y_MAX - y_initial[0])/outer_hit,
        (Y_MIN - y_initial[1])/inner_hit,
        0.1,
    };

    std::vector<swarm> particle(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(x_mid, y_initial[idx], z_mid);
        particle[idx].velocity = make_double3(0.0, vy_initial[idx], 0.0);
    }

    swarm *dev_particle;
    qav_malloc(&dev_particle, N_P, "allocate absorption particles");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload absorption particles");

    const real dt = time_end/static_cast<real>(VERIFY_RES);
    std::vector<real> absorbed_time(N_P, -1.0);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
        qav_kernel_check("constant radial absorption path");
        qav_copy_d2h(particle.data(), dev_particle, N_P, "inspect absorption state");

        for (int idx = 0; idx < N_P; idx++)
        {
            if (absorbed_time[idx] < 0.0 && particle[idx].position.y == 0.0)
            {
                // archive the accepted-step endpoint at which the production sentinel first appears
                absorbed_time[idx] = static_cast<real>(step + 1)*dt;
            }
        }
    }

    real *dev_dyn_rate, *dev_dustdens;
    qav_malloc(&dev_dyn_rate, N_P, "allocate absorption dynamics rates");
    qav_malloc(&dev_dustdens, N_G, "allocate absorption density grid");

    dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle);
    qav_kernel_check("inactive dynamics-rate exclusion");
    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, static_cast<real>(N_P));
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);
    qav_kernel_check("inactive density-deposition exclusion");

    std::vector<real> dyn_rate(N_P), dustdens(N_G);
    qav_copy_d2h(dyn_rate.data(), dev_dyn_rate, N_P, "copy absorption dynamics rates");
    qav_copy_d2h(dustdens.data(), dev_dustdens, N_G, "copy absorption density grid");

    std::vector<real> state(6*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx] = particle[idx].position.x;
        state[N_P + idx] = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
    }
    write_binary("state", state);
    write_binary("absorption_time", absorbed_time);
    write_binary("dyn_rate", dyn_rate);
    write_binary("dustdens", dustdens);

    real deposited_mass = 0.0;
    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            real cell_measure = _get_vol_x()*_get_vol_y(iy)*_get_vol_z(iz);
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                deposited_mass += dustdens[idx_cell]*cell_measure;
            }
        }
    }

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"absorb_path_1d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"ng\": " << N_G << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"outer_hit\": " << outer_hit << ",\n"
         << "  \"inner_hit\": " << inner_hit << ",\n"
         << "  \"survivor_y_initial\": " << y_initial[2] << ",\n"
         << "  \"survivor_vy\": " << vy_initial[2] << ",\n"
         << "  \"deposited_mass\": " << deposited_mass << ",\n"
         << "  \"constant_radial_specialization\": " << (QAV_CONSTANT_RADIAL_PATH ? "true" : "false") << ",\n"
         << "  \"production_transport_boundary\": true,\n"
         << "  \"production_downstream_exclusion\": true\n"
         << "}\n";

    qav_free(dev_dustdens, "free absorption density grid");
    qav_free(dev_dyn_rate, "free absorption dynamics rates");
    qav_free(dev_particle, "free absorption particles");
    std::cout << "swarm deterministic absorption completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
