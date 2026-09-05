#include <algorithm>  // std::max_element
#include <array>      // std::array
#include <cmath>      // sqrt
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

#include <_col_cache.cuh>
#include <_collision.cuh>
#include <device_api.cuh>
#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
#endif // COLLISION_MORTON

// replace stochastic evolution by fixed pair probes while retaining production collision formulas
namespace
{
constexpr int turbulence_count = 16;
constexpr int device_result_count = 32;
constexpr int result_count = device_result_count + N_P*N_K + N_P;
constexpr real seam_offset = 1.0e-3;
const std::string output_path = PATH_OUT;

// evaluate every physical-rate branch at fixed inputs without running a stochastic collision history
__global__ void collision_physics (real *result, const swarm *particle, const real *size, const real *number)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr real R = 1.0;
    constexpr real size_ratio = 0.3;
    real h_g = _get_hg(R);
    real sigma_g = _get_sigma_g(R);
    real alpha = _get_alpha(R, h_g);
    real re_inv_sqrt = _get_re_inv_sqrt(R, alpha, sigma_g);
    result[0] = re_inv_sqrt;

    // the first six points lie inside the six Ormel-Cuzzi regimes; the remaining pairs straddle every branch boundary
    real stokes_large[turbulence_count];
    stokes_large[0] = 0.02*re_inv_sqrt;
    stokes_large[1] = 0.40*re_inv_sqrt;
    stokes_large[2] = 2.00*re_inv_sqrt;
    stokes_large[3] = sqrt(5.0*re_inv_sqrt*0.2);
    stokes_large[4] = 0.5;
    stokes_large[5] = 2.0;

    real boundary[5] = {
        0.2*re_inv_sqrt,
        re_inv_sqrt / 1.6,
        5.0*re_inv_sqrt,
        0.2,
        1.0
    };
    for (int idx = 0; idx < 5; idx++)
    {
        stokes_large[6 + 2*idx] = boundary[idx]*(1.0 - 1.0e-6);
        stokes_large[7 + 2*idx] = boundary[idx]*(1.0 + 1.0e-6);
    }
    for (int idx = 0; idx < turbulence_count; idx++)
        result[1 + idx] = _get_vrel_t(R, stokes_large[idx], size_ratio*stokes_large[idx], h_g, sigma_g);

    #ifdef CODE_UNIT
    result[17] = 0.0;
    result[18] = 0.0;
    #else  // PHYSICAL_UNIT
    result[17] = _get_vrel_b(R, size[0], size[1], h_g);
    result[18] = _get_vrel_b(R, 1.0e-12, 1.0e-12, h_g);
    #endif // CODE_UNIT

    result[19] = _get_vrel_pair(particle, size[0], size[1], 0, 1, 0);
    result[20] = _get_col_rate_ij<CUSTOM_KERNEL>(particle, size, number, 0, 1, 0, 0.3);

    real3 velocity_i = _get_cart_vel(particle[0]);
    real3 velocity_j = _get_cart_vel(particle[1]);
    real dvx = velocity_i.x - velocity_j.x;
    real dvy = velocity_i.y - velocity_j.y;
    real dvz = velocity_i.z - velocity_j.z;
    result[21] = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

    // compare one local pair represented inside the wedge and across its rotational seam
    swarm image_particle[4]{};
    constexpr real image_z = (N_Z == 1) ? 0.5*M_PI : 0.9;
    constexpr real image_lz = (N_Z == 1) ? 0.0 : 0.3;
    image_particle[0].position = make_double3(X_MIN + seam_offset, 1.0, image_z);
    image_particle[1].position = make_double3(X_MAX - seam_offset, 1.0, image_z);
    image_particle[2].position = make_double3(+seam_offset, 1.0, image_z);
    image_particle[3].position = make_double3(-seam_offset, 1.0, image_z);
    for (int idx = 0; idx < 4; idx++)
        image_particle[idx].velocity = make_double3(1.5, 0.2, image_lz);
    real image_size[4] = {size[0], size[1], size[0], size[1]};
    real image_number[4] = {number[0], number[1], number[0], number[1]};
    int neighbor = _encode_col_neighbor(1, 1);
    result[22] = _get_vrel_pair(image_particle, image_size[0], image_size[1], 0, 1, 1);
    result[23] = _get_vrel_pair(image_particle, image_size[2], image_size[3], 2, 3, 0);
    result[24] = _get_vrel_pair(image_particle, image_size[0], image_size[1], 0, 1, 0);
    result[25] = _get_col_rate_ij<CUSTOM_KERNEL>(
        image_particle, image_size, image_number, 0, 1, 1, 0.3
    );
    result[26] = _get_col_rate_ij<CUSTOM_KERNEL>(
        image_particle, image_size, image_number, 2, 3, 0, 0.3
    );
    result[27] = static_cast<real>(neighbor);
    result[28] = static_cast<real>(_get_col_idx_old(neighbor));
    result[29] = static_cast<real>(_get_col_image(neighbor));
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open collision-physics output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write collision-physics output");
}
}

int main ()
{
    std::array<swarm, N_P> particle{};
    std::array<real, N_P> size = {0.5, 1.75};
    std::array<real, N_P> number = {3.0, 7.0};

    particle[0].position = make_double3(0.3, 1.0, 0.5*M_PI);
    particle[1].position = make_double3(-0.4, 1.0, 0.5*M_PI);
    particle[0].velocity = make_double3(0.2, -0.03, 0.0);
    particle[1].velocity = make_double3(-0.1, 0.04, 0.0);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].par_size = size[idx];
        particle[idx].par_numr = number[idx];
    }

    // isolate the cache-path check from the fixed interior-pair probes above
    std::array<swarm, N_P> cache_particle{};
    constexpr real cache_z = (N_Z == 1) ? 0.5*M_PI : 0.9;
    constexpr real cache_lz = (N_Z == 1) ? 0.0 : 0.3;
    cache_particle[0].position = make_double3(X_MIN + seam_offset, 1.0, cache_z);
    cache_particle[1].position = make_double3(X_MAX - seam_offset, 1.0, cache_z);
    for (int idx = 0; idx < N_P; idx++)
    {
        cache_particle[idx].velocity = make_double3(1.5, 0.2, cache_lz);
        cache_particle[idx].par_size = size[idx];
        cache_particle[idx].par_numr = number[idx];
    }

    swarm *dev_particle;
    swarm *dev_cache_particle;
    real *dev_size;
    real *dev_number;
    real *dev_result;
    int *dev_col_neighbor;
    int *dev_bad_part;
    real *dev_col_measure;
    unsigned char *dev_col_active;
    val_malloc(&dev_particle, N_P, "allocate collision-physics particles");
    val_malloc(&dev_cache_particle, N_P, "allocate collision-physics cache particles");
    val_malloc(&dev_size, N_P, "allocate collision-physics sizes");
    val_malloc(&dev_number, N_P, "allocate collision-physics numbers");
    val_malloc(&dev_result, device_result_count, "allocate collision-physics results");
    val_malloc(&dev_col_neighbor, N_P*N_K, "allocate collision-physics neighbors");
    val_malloc(&dev_col_measure, N_P, "allocate collision-physics measures");
    val_malloc(&dev_col_active, N_P, "allocate collision-physics active flags");
    val_copy_h2d(dev_particle, particle.data(), N_P, "upload collision-physics particles");
    val_copy_h2d(dev_cache_particle, cache_particle.data(), N_P,
        "upload collision-physics cache particles");
    val_copy_h2d(dev_size, size.data(), N_P, "upload collision-physics sizes");
    val_copy_h2d(dev_number, number.data(), N_P, "upload collision-physics numbers");

    collision_physics <<< 1, 1 >>> (dev_result, dev_particle, dev_size, dev_number);
    val_kernel_check("collision_physics");

    // exercise the production search-to-cache handoff before either rate consumer reads it
    int bad_part = 0;
    val_malloc(&dev_bad_part, 1, "allocate collision-physics bad-particle flag");
    val_copy_h2d(dev_bad_part, &bad_part, 1, "clear collision-physics bad-particle flag");
    float image_dist_min = _get_image_dist_min(
        static_cast<float>(X_MIN), static_cast<float>(X_MAX), static_cast<float>(Y_MIN),
        static_cast<float>(Z_MIN), static_cast<float>(Z_MAX)
    );

    #ifdef COLLISION_KDTREE
    kdtree_node *dev_kdtree_node;
    kdtree_boxf *dev_kdtree_box;
    val_malloc(&dev_kdtree_node, N_T, "allocate collision-physics KD nodes");
    val_malloc(&dev_kdtree_box, 1, "allocate collision-physics KD bounds");
    col_site_init <<< NB_P, TPB >>> (
        dev_kdtree_node, dev_col_active, dev_cache_particle, dev_bad_part
    );
    val_kernel_check("collision-physics KD site initialization");
    kdtree::buildTree <kdtree_node, kdtree_traits> (
        dev_kdtree_node, N_T, dev_kdtree_box
    );
    val_kernel_check("collision-physics KD build");
    col_cache_get <<< NB_T, TPB >>> (
        dev_col_neighbor, dev_col_measure, dev_kdtree_node, dev_kdtree_box,
        dev_col_active, dev_cache_particle, image_dist_min
    );
    val_kernel_check("collision-physics KD cache");
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx;
    float *dev_search_dist;
    unsigned int *dev_morton_overflow;
    val_malloc(&dev_morton_point, N_P, "allocate collision-physics Morton points");
    val_malloc(&dev_morton_posx, N_P, "allocate collision-physics Morton azimuths");
    val_malloc(&dev_search_dist, N_P, "allocate collision-physics search distances");
    val_malloc(&dev_morton_overflow, N_P, "allocate collision-physics Morton overflow flags");
    col_site_init <<< NB_P, TPB >>> (
        dev_morton_point, dev_morton_posx, dev_search_dist,
        dev_col_active, dev_cache_particle, dev_bad_part
    );
    val_kernel_check("collision-physics Morton site initialization");
    std::array<float, N_P> search_dist;
    val_copy_d2h(search_dist.data(), dev_search_dist, N_P,
        "copy collision-physics search distances");
    float max_search_dist = *std::max_element(search_dist.begin(), search_dist.end());
    bool unique_ids = image_dist_min > 2.0f*max_search_dist;
    morton_ghost_index morton_owner;
    morton_owner.build(
        dev_morton_point, dev_morton_posx, N_P, max_search_dist,
        static_cast<float>(X_MIN), static_cast<float>(X_MAX),
        static_cast<float>(Y_MAX), true, unique_ids,
        (N_Z > 1) ? 3 : 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
    );
    col_cache_get <<< N_P, MORTON_TPB >>> (
        dev_col_neighbor, dev_col_measure, dev_morton_overflow, dev_morton_point,
        dev_col_active, dev_cache_particle, morton_owner.view(), morton_owner.unique_ids()
    );
    val_kernel_check("collision-physics Morton cache");
    std::array<unsigned int, N_P> morton_overflow;
    val_copy_d2h(morton_overflow.data(), dev_morton_overflow, N_P,
        "copy collision-physics Morton overflow flags");
    if (*std::max_element(morton_overflow.begin(), morton_overflow.end()) != 0)
        throw std::runtime_error("collision-physics Morton cache overflow");
    #endif // COLLISION_KDTREE

    val_copy_d2h(&bad_part, dev_bad_part, 1, "copy collision-physics bad-particle flag");
    if (bad_part != 0) throw std::runtime_error("collision-physics search rejected a particle");

    #ifdef COLLISION_KDTREE
    col_rate_calc <<< NB_P, TPB >>> (
    #else  // COLLISION_MORTON
    col_rate_calc <<< N_P, MORTON_TPB >>> (
    #endif // COLLISION_KDTREE
        dev_result + 30, dev_cache_particle, dev_col_neighbor, dev_col_measure,
        dev_col_active, dev_size, dev_number, 0.3
    );
    val_kernel_check("cached collision-rate probe");

    std::vector<real> result(result_count);
    val_copy_d2h(result.data(), dev_result, device_result_count,
        "copy collision-physics results");
    std::array<int, N_P*N_K> col_neighbor;
    std::array<real, N_P> col_measure;
    val_copy_d2h(col_neighbor.data(), dev_col_neighbor, N_P*N_K,
        "copy collision-physics neighbors");
    val_copy_d2h(col_measure.data(), dev_col_measure, N_P,
        "copy collision-physics measures");
    for (int idx = 0; idx < N_P*N_K; idx++)
        result[device_result_count + idx] = static_cast<real>(col_neighbor[idx]);
    for (int idx = 0; idx < N_P; idx++)
        result[device_result_count + N_P*N_K + idx] = col_measure[idx];
    write_binary("collision_physics", result);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    if (!meta) throw std::runtime_error("cannot open collision-physics metadata");
    meta << std::setprecision(17)
         << "{\n"
         #ifdef CODE_UNIT
         #ifdef TEST_COLPHYS_3D
         << "  \"case\": \"colphys_3d\",\n"
         #else
         << "  \"case\": \"colphys_code\",\n"
         #endif // TEST_COLPHYS_3D
         << "  \"code_unit\": true,\n"
         #else  // PHYSICAL_UNIT
         << "  \"case\": \"colphys_cgs\",\n"
         << "  \"code_unit\": false,\n"
         #endif // CODE_UNIT
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"dimension\": " << ((N_Z == 1) ? 2 : 3) << ",\n"
         << "  \"result_count\": " << result_count << ",\n"
         << "  \"turbulence_count\": " << turbulence_count << ",\n"
         << "  \"position_x\": [0.3, -0.4],\n"
         << "  \"velocity_lx\": [0.2, -0.1],\n"
         << "  \"velocity_y\": [-0.03, 0.04],\n"
         << "  \"size\": [0.5, 1.75],\n"
         << "  \"number\": [3.0, 7.0],\n"
         << "  \"x_min\": " << X_MIN << ",\n"
         << "  \"x_max\": " << X_MAX << ",\n"
         << "  \"seam_offset\": " << seam_offset << ",\n"
         << "  \"seam_z\": " << ((N_Z == 1) ? 0.5*M_PI : 0.9) << ",\n"
         << "  \"seam_velocity\": [1.5, 0.2, " << ((N_Z == 1) ? 0.0 : 0.3) << "]\n"
         << "}\n";
    if (!meta) throw std::runtime_error("cannot write collision-physics metadata");

    val_free(dev_col_active, "free collision-physics active flags");
    val_free(dev_col_measure, "free collision-physics measures");
    val_free(dev_col_neighbor, "free collision-physics neighbors");
    val_free(dev_result, "free collision-physics results");
    val_free(dev_number, "free collision-physics numbers");
    val_free(dev_size, "free collision-physics sizes");
    val_free(dev_cache_particle, "free collision-physics cache particles");
    val_free(dev_particle, "free collision-physics particles");
    val_free(dev_bad_part, "free collision-physics bad-particle flag");
    #ifdef COLLISION_KDTREE
    val_free(dev_kdtree_box, "free collision-physics KD bounds");
    val_free(dev_kdtree_node, "free collision-physics KD nodes");
    #else  // COLLISION_MORTON
    val_free(dev_morton_overflow, "free collision-physics Morton overflow flags");
    val_free(dev_search_dist, "free collision-physics search distances");
    val_free(dev_morton_posx, "free collision-physics Morton azimuths");
    val_free(dev_morton_point, "free collision-physics Morton points");
    #endif // COLLISION_KDTREE
    std::cout << "swarm physical collision kernel completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
