#include <algorithm>  // std::max_element
#include <cmath>      // M_PI, std::sqrt
#include <iostream>   // std::cerr, std::cout
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <vector>     // std::vector

#include <swarm_kern.cuh>

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
#endif // COLLISION_MORTON

namespace
{

constexpr int IDX_BAD = 7;

void hip_check (hipError_t status, const char *operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

real injected_value (const std::string &value)
{
    if (value == "finite") return 1.0;
    if (value == "nan") return std::numeric_limits<real>::quiet_NaN();
    if (value == "inf") return std::numeric_limits<real>::infinity();
    throw std::runtime_error("unknown injected value: " + value);
}

void inject_particle (swarm &particle, const std::string &field, real value)
{
    if      (field == "position_x") particle.position.x = value;
    else if (field == "position_y") particle.position.y = value;
    else if (field == "position_z") particle.position.z = value;
    else if (field == "velocity_x") particle.velocity.x = value;
    else if (field == "velocity_y") particle.velocity.y = value;
    else if (field == "velocity_z") particle.velocity.z = value;
    else if (field == "par_size")   particle.par_size   = value;
    else if (field == "par_numr")   particle.par_numr   = value;
    else if (field != "none") throw std::runtime_error("unknown particle field: " + field);
}

} // namespace

int main (int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: gamedev site|result field finite|nan|inf" << std::endl;
        return 2;
    }

    const std::string mode = argv[1];
    const std::string field = argv[2];
    const std::string value_name = argv[3];
    const real value = injected_value(value_name);

    int *dev_bad_part;
    hip_check(hipMalloc((void**)&dev_bad_part, sizeof(int)), "allocate failure flag");
    hip_check(hipMemset(dev_bad_part, 0, sizeof(int)), "clear failure flag");

    if (mode == "result")
    {
        std::vector<real> col_rate(N_P, 1.0);
        std::vector<real> col_dist(N_P, 1.0);
        if (field == "col_rate") col_rate[IDX_BAD] = value;
        else if (field == "col_dist") col_dist[IDX_BAD] = value;
        else if (field != "none") throw std::runtime_error("unknown collision-result field: " + field);

        real *dev_col_rate, *dev_col_dist;
        hip_check(hipMalloc((void**)&dev_col_rate, sizeof(real)*N_P), "allocate collision rates");
        hip_check(hipMalloc((void**)&dev_col_dist, sizeof(real)*N_P), "allocate collision radii");
        hip_check(hipMemcpy(dev_col_rate, col_rate.data(), sizeof(real)*N_P, hipMemcpyHostToDevice),
            "copy collision rates");
        hip_check(hipMemcpy(dev_col_dist, col_dist.data(), sizeof(real)*N_P, hipMemcpyHostToDevice),
            "copy collision radii");

        inf_rate_flag <<< NB_P, TPB >>> (dev_col_rate, dev_col_dist, dev_bad_part);
        hip_check(hipGetLastError(), "launch inf_rate_flag");

        int bad_part = 0;
        hip_check(hipMemcpy(&bad_part, dev_bad_part, sizeof(int), hipMemcpyDeviceToHost),
            "read collision-result failure flag");
        if (bad_part != 0)
        {
            std::cerr << "Error: non-finite collision result at particle " << bad_part - 1 << std::endl;
            return 1;
        }

        std::cout << "clean collision result accepted" << std::endl;
        return 0;
    }

    if (mode != "site") throw std::runtime_error("unknown failure-test mode: " + mode);

    std::vector<swarm> particle(N_P);
    const real y_mid = std::sqrt(Y_MIN*Y_MAX);
    for (int idx = 0; idx < N_P; idx++)
    {
        particle[idx].position = make_double3(
            X_MIN + (static_cast<real>(idx) + 0.5)*(X_MAX - X_MIN) / static_cast<real>(N_P),
            y_mid, 0.5*M_PI
        );
        particle[idx].velocity = make_double3(1.0, 0.0, 0.0);
        particle[idx].par_size = 1.0;
        particle[idx].par_numr = 1.0;
    }
    inject_particle(particle[IDX_BAD], field, value);

    swarm *dev_particle;
    unsigned char *dev_col_active;
    hip_check(hipMalloc((void**)&dev_particle, sizeof(swarm)*N_P), "allocate particles");
    hip_check(hipMalloc((void**)&dev_col_active, sizeof(unsigned char)*N_P), "allocate collision activity");
    hip_check(hipMemcpy(dev_particle, particle.data(), sizeof(swarm)*N_P, hipMemcpyHostToDevice),
        "copy particles");

    #ifdef COLLISION_KDTREE
    kdtree_node *dev_kdtree_node;
    kdtree_boxf *dev_kdtree_box;
    hip_check(hipMalloc((void**)&dev_kdtree_node, sizeof(kdtree_node)*N_T), "allocate KD-tree nodes");
    hip_check(hipMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)), "allocate KD-tree bounds");
    col_site_init <<< NB_P, TPB >>> (dev_kdtree_node, dev_col_active, dev_particle, dev_bad_part);
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx, *dev_search_dist;
    hip_check(hipMalloc((void**)&dev_morton_point, sizeof(float3)*N_P), "allocate Morton points");
    hip_check(hipMalloc((void**)&dev_morton_posx, sizeof(float)*N_P), "allocate Morton azimuths");
    hip_check(hipMalloc((void**)&dev_search_dist, sizeof(float)*N_P), "allocate Morton search radii");
    col_site_init <<< NB_P, TPB >>> (
        dev_morton_point, dev_morton_posx, dev_search_dist, dev_col_active, dev_particle, dev_bad_part
    );
    #endif // COLLISION_KDTREE
    hip_check(hipGetLastError(), "launch col_site_init");

    int bad_part = 0;
    hip_check(hipMemcpy(&bad_part, dev_bad_part, sizeof(int), hipMemcpyDeviceToHost),
        "read particle failure flag");
    if (bad_part != 0)
    {
        std::cerr << "Error: non-finite particle state before collision search at particle "
            << bad_part - 1 << std::endl;
        return 1;
    }

    // require the clean control to continue through the selected production index builder
    #ifdef COLLISION_KDTREE
    kdtree::buildTree <kdtree_node, kdtree_traits> (dev_kdtree_node, N_T, dev_kdtree_box);
    hip_check(hipGetLastError(), "build clean KD-tree");
    hip_check(hipDeviceSynchronize(), "finish clean KD-tree build");
    #else  // COLLISION_MORTON
    std::vector<float> search_dist(N_P);
    hip_check(hipMemcpy(search_dist.data(), dev_search_dist, sizeof(float)*N_P, hipMemcpyDeviceToHost),
        "copy Morton search radii");
    float max_search_dist = *std::max_element(search_dist.begin(), search_dist.end());
    morton_ghost_index morton_owner;
    morton_owner.build(
        dev_morton_point, dev_morton_posx, N_P, max_search_dist,
        static_cast<float>(X_MIN), static_cast<float>(X_MAX),
        static_cast<float>(Y_MIN), static_cast<float>(Y_MAX),
        static_cast<float>(Z_MIN), static_cast<float>(Z_MAX),
        N_X > 1, (N_Z > 1) ? 3 : 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
    );
    #endif // COLLISION_KDTREE

    std::cout << "clean collision search input accepted" << std::endl;
    return 0;
}
