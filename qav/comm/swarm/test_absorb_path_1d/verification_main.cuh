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

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
#endif // COLLISION_MORTON

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
    const real y_initial[N_P] = {1.2, 0.8, 1.0, 1.0};
    const real vy_initial[N_P] = {
        (Y_MAX - y_initial[0])/outer_hit,
        (Y_MIN - y_initial[1])/inner_hit,
        0.1, -0.1,
    };

    std::vector<swarm> particle(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(x_mid, y_initial[idx], z_mid);
        particle[idx].velocity = make_double3(0.0, vy_initial[idx], 0.0);
        particle[idx].par_size = S_0;
        // assign unit represented mass without calling the device-only grain-mass helper from host code
        particle[idx].par_numr = 6.0 / (M_PI*RHO_0*S_0*S_0*S_0);
    }

    swarm *dev_particle;
    real *dev_zero_optdepth;
    qav_malloc(&dev_particle, N_P, "allocate absorption particles");
    qav_malloc(&dev_zero_optdepth, N_G, "allocate absorption transport optical depth");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload absorption particles");
    std::vector<real> zero_optdepth(N_G, 0.0);
    qav_copy_h2d(dev_zero_optdepth, zero_optdepth.data(), N_G, "upload absorption transport optical depth");

    const real dt = time_end/static_cast<real>(VERIFY_RES);
    std::vector<real> absorbed_time(N_P, -1.0);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        // call both production radiation-split transport kernels while the test specialization retains constant radial motion
        ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt);
        ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_zero_optdepth, 0.0, dt);
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

    real *dev_dyn_rate, *dev_dustdens, *dev_optdepth;
    qav_malloc(&dev_dyn_rate, N_P, "allocate absorption dynamics rates");
    qav_malloc(&dev_dustdens, N_G, "allocate absorption density grid");
    qav_malloc(&dev_optdepth, N_G, "allocate absorption optical-depth grid");

    dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle);
    qav_kernel_check("inactive dynamics-rate exclusion");
    dustdens_init <<< NB_G, TPB >>> (dev_dustdens);
    dustdens_depo <<< NB_P, TPB >>> (dev_dustdens, dev_particle, static_cast<real>(N_P));
    dustdens_calc <<< NB_G, TPB >>> (dev_dustdens);
    qav_kernel_check("inactive density-deposition exclusion");

    // exercise the complete production extinction chain after absorption; inactive representatives must deposit nothing
    optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
    optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, static_cast<real>(N_P));
    optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
    optdepth_csum <<< N_X*N_Z, 1 >>> (dev_optdepth);
    qav_kernel_check("inactive optical-depth exclusion");

    std::vector<real> dyn_rate(N_P), dustdens(N_G), optdepth(N_G);
    qav_copy_d2h(dyn_rate.data(), dev_dyn_rate, N_P, "copy absorption dynamics rates");
    qav_copy_d2h(dustdens.data(), dev_dustdens, N_G, "copy absorption density grid");
    qav_copy_d2h(optdepth.data(), dev_optdepth, N_G, "copy absorption optical-depth grid");

    // connect the inactive sentinel to the production collision-search mask; standalone KNN tests verify that both
    // backends exclude every record whose corresponding mask byte is zero
    unsigned char *dev_col_active;
    real *dev_col_rate, *dev_col_dist, *dev_size_old, *dev_numr_old;
    qav_malloc(&dev_col_active, N_P, "allocate absorption collision mask");
    qav_malloc(&dev_col_rate, N_P, "allocate absorption collision rates");
    qav_malloc(&dev_col_dist, N_P, "allocate absorption collision radii");
    qav_malloc(&dev_size_old, N_P, "allocate absorption frozen sizes");
    qav_malloc(&dev_numr_old, N_P, "allocate absorption frozen numbers");
    int *dev_bad_part;
    qav_malloc(&dev_bad_part, 1, "allocate absorption bad-particle flag");
    int bad_part = 0;
    qav_copy_h2d(dev_bad_part, &bad_part, 1, "clear absorption bad-particle flag");
    #ifdef COLLISION_KDTREE
    kdtree_node *dev_kdtree_node;
    kdtree_boxf *dev_kdtree_box;
    qav_malloc(&dev_kdtree_node, N_T, "allocate absorption KD-tree sites");
    qav_malloc(&dev_kdtree_box, 1, "allocate absorption KD-tree bounds");
    col_site_init <<< NB_P, TPB >>> (dev_kdtree_node, dev_col_active, dev_particle, dev_bad_part);
    qav_kernel_check("inactive collision-site exclusion");
    kdtree::buildTree <kdtree_node, kdtree_traits> (dev_kdtree_node, N_T, dev_kdtree_box);
    qav_kernel_check("absorption KD-tree build");
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx, *dev_search_dist;
    qav_malloc(&dev_morton_point, N_P, "allocate absorption Morton sites");
    qav_malloc(&dev_morton_posx, N_P, "allocate absorption Morton azimuths");
    qav_malloc(&dev_search_dist, N_P, "allocate absorption Morton search radii");
    col_site_init <<< NB_P, TPB >>> (
        dev_morton_point, dev_morton_posx, dev_search_dist, dev_col_active, dev_particle, dev_bad_part
    );
    qav_kernel_check("inactive collision-site exclusion");
    std::vector<float> search_dist(N_P);
    qav_copy_d2h(search_dist.data(), dev_search_dist, N_P, "copy absorption Morton search radii");
    float max_search_dist = 0.0f;
    for (float value : search_dist)
    {
        if (value > max_search_dist) max_search_dist = value;
    }
    morton_ghost_index morton_owner;
    morton_owner.build(
        dev_morton_point, dev_morton_posx, N_P, max_search_dist,
        static_cast<float>(X_MIN), static_cast<float>(X_MAX),
        static_cast<float>(Y_MIN), static_cast<float>(Y_MAX),
        static_cast<float>(Z_MIN), static_cast<float>(Z_MAX),
        N_X > 1, 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
    );
    #endif // COLLISION_KDTREE

    unsigned int morton_overflow_max = 0;
    col_snap_save <<< NB_P, TPB >>> (dev_size_old, dev_numr_old, dev_particle);
    #ifdef COLLISION_KDTREE
    col_rate_calc <<< NB_T, TPB >>> (
        dev_col_rate, dev_col_dist, dev_particle, dev_col_active, dev_size_old, dev_numr_old,
        dev_kdtree_node, dev_kdtree_box, -1.0f, 1.0
    );
    #else  // COLLISION_MORTON
    unsigned int *dev_morton_overflow;
    qav_malloc(&dev_morton_overflow, N_P, "allocate absorption Morton overflow flags");
    std::vector<unsigned int> zero_overflow(N_P, 0);
    qav_copy_h2d(dev_morton_overflow, zero_overflow.data(), N_P, "clear absorption Morton overflow flags");
    col_rate_calc <<< N_P, MORTON_TPB >>> (
        dev_col_rate, dev_col_dist, dev_morton_overflow, dev_particle, dev_col_active,
        dev_size_old, dev_numr_old, dev_morton_point, morton_owner.view(), morton_owner.unique_ids(), 1.0
    );
    std::vector<unsigned int> morton_overflow(N_P);
    qav_copy_d2h(morton_overflow.data(), dev_morton_overflow, N_P, "copy absorption Morton overflow flags");
    for (unsigned int value : morton_overflow)
    {
        if (value > morton_overflow_max) morton_overflow_max = value;
    }
    #endif // COLLISION_KDTREE
    qav_kernel_check("inactive collision-rate exclusion");
    std::vector<unsigned char> col_active(N_P);
    qav_copy_d2h(col_active.data(), dev_col_active, N_P, "copy absorption collision mask");
    std::vector<real> col_active_real(N_P);
    std::vector<real> col_rate(N_P), col_dist(N_P);
    qav_copy_d2h(col_rate.data(), dev_col_rate, N_P, "copy absorption collision rates");
    qav_copy_d2h(col_dist.data(), dev_col_dist, N_P, "copy absorption collision radii");
    for (int idx = 0; idx < N_P; idx++)
    {
        col_active_real[idx] = static_cast<real>(col_active[idx]);
    }

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
    write_binary("optdepth", optdepth);
    write_binary("col_active", col_active_real);
    write_binary("col_rate", col_rate);
    write_binary("col_dist", col_dist);

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
         << "  \"survivor2_y_initial\": " << y_initial[3] << ",\n"
         << "  \"survivor2_vy\": " << vy_initial[3] << ",\n"
         << "  \"deposited_mass\": " << deposited_mass << ",\n"
         << "  \"constant_radial_specialization\": " << (QAV_CONSTANT_RADIAL_PATH ? "true" : "false") << ",\n"
         << "  \"production_transport_boundary\": true,\n"
         << "  \"production_downstream_exclusion\": true,\n"
         << "  \"production_optdepth_exclusion\": true,\n"
         << "  \"production_collision_mask\": true,\n"
         << "  \"production_collision_rate\": true,\n"
         << "  \"morton_overflow_max\": " << morton_overflow_max << "\n"
         << "}\n";

    #ifdef COLLISION_KDTREE
    qav_free(dev_kdtree_box, "free absorption KD-tree bounds");
    qav_free(dev_kdtree_node, "free absorption KD-tree sites");
    #else  // COLLISION_MORTON
    qav_free(dev_morton_overflow, "free absorption Morton overflow flags");
    qav_free(dev_search_dist, "free absorption Morton search radii");
    qav_free(dev_morton_posx, "free absorption Morton azimuths");
    qav_free(dev_morton_point, "free absorption Morton sites");
    #endif // COLLISION_KDTREE
    qav_free(dev_numr_old, "free absorption frozen numbers");
    qav_free(dev_size_old, "free absorption frozen sizes");
    qav_free(dev_col_dist, "free absorption collision radii");
    qav_free(dev_col_rate, "free absorption collision rates");
    qav_free(dev_bad_part, "free absorption bad-particle flag");
    qav_free(dev_col_active, "free absorption collision mask");
    qav_free(dev_optdepth, "free absorption optical-depth grid");
    qav_free(dev_dustdens, "free absorption density grid");
    qav_free(dev_dyn_rate, "free absorption dynamics rates");
    qav_free(dev_zero_optdepth, "free absorption transport optical depth");
    qav_free(dev_particle, "free absorption particles");
    std::cout << "swarm deterministic absorption completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
