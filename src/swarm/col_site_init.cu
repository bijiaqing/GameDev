#include <gpu.cuh>
#ifdef COLLISION

#include <_transport.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: colstate_flag
// retain the collision-entry finite-state guard when a valid geometry package skips site reconstruction
//
// parallelization: one thread per representative particle
// =====================================================================================================================

__global__
void colstate_flag (const swarm *dev_particle, int *dev_bad_part)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    bool finite = isfinite(dev_particle[idx].position.x)
        && isfinite(dev_particle[idx].position.y)
        && isfinite(dev_particle[idx].position.z)
        && isfinite(dev_particle[idx].velocity.x)
        && isfinite(dev_particle[idx].velocity.y)
        && isfinite(dev_particle[idx].velocity.z);
    #ifdef MULTISIZE
    finite = finite && isfinite(dev_particle[idx].par_size) && isfinite(dev_particle[idx].par_numr);
    #endif // MULTISIZE
    if (!finite) atomicCAS(dev_bad_part, 0, idx + 1);
}

// =====================================================================================================================
// kernel: col_site_init
// construct Cartesian collision-search records from spherical particle positions
//
// parallelization: one thread per representative particle
// =====================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_site_init (kdtree_node *dev_kdtree_node, unsigned char *dev_col_active,
    const swarm *dev_particle, int *dev_bad_part)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    bool finite = isfinite(dev_particle[idx].position.x)
        && isfinite(dev_particle[idx].position.y)
        && isfinite(dev_particle[idx].position.z)
        && isfinite(dev_particle[idx].velocity.x)
        && isfinite(dev_particle[idx].velocity.y)
        && isfinite(dev_particle[idx].velocity.z);
    #ifdef MULTISIZE
    finite = finite && isfinite(dev_particle[idx].par_size) && isfinite(dev_particle[idx].par_numr);
    #endif // MULTISIZE
    if (!finite)
    {
        dev_col_active[idx] = 0;
        atomicCAS(dev_bad_part, 0, idx + 1);
        return;
    }

    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    bool active = _is_particle_active(y, z);
    dev_col_active[idx] = static_cast<unsigned char>(active);
    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);

    float3 cartesian = (N_X == 1 && N_Z == 1)
        ? make_float3(static_cast<float>(R), 0.0f, 0.0f)
        : make_float3(
            static_cast<float>(R*cos(x)), static_cast<float>(R*sin(x)), static_cast<float>(Z)
        );
    dev_kdtree_node[idx].cartesian = cartesian;
    dev_kdtree_node[idx].idx_old = idx;
    dev_kdtree_node[idx].image = 0;

    // append both rotated wedge images when azimuth covers less than a complete period
    if (X_WEDGE)
    {
        float width = static_cast<float>(X_MAX - X_MIN);
        float sin_width;
        float cos_width;
        sincosf(width, &sin_width, &cos_width);
        for (int image = 0; image < 2; image++)
        {
            int idx_image = idx + (image + 1)*N_P;
            float sin_angle = (image == 0) ? -sin_width : sin_width;
            dev_kdtree_node[idx_image].cartesian = make_float3(
                cos_width*cartesian.x - sin_angle*cartesian.y,
                sin_angle*cartesian.x + cos_width*cartesian.y,
                cartesian.z
            );
            dev_kdtree_node[idx_image].idx_old = idx;
            dev_kdtree_node[idx_image].image = image + 1;
        }
    }
}
#else  // COLLISION_MORTON
__global__
void col_site_init (float3 *dev_morton_point, float *dev_morton_posx, float *dev_search_dist,
    unsigned char *dev_col_active, const swarm *dev_particle, int *dev_bad_part)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    bool finite = isfinite(dev_particle[idx].position.x)
        && isfinite(dev_particle[idx].position.y)
        && isfinite(dev_particle[idx].position.z)
        && isfinite(dev_particle[idx].velocity.x)
        && isfinite(dev_particle[idx].velocity.y)
        && isfinite(dev_particle[idx].velocity.z);
    #ifdef MULTISIZE
    finite = finite && isfinite(dev_particle[idx].par_size) && isfinite(dev_particle[idx].par_numr);
    #endif // MULTISIZE
    if (!finite)
    {
        dev_col_active[idx] = 0;
        atomicCAS(dev_bad_part, 0, idx + 1);
        return;
    }

    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    bool active = _is_particle_active(y, z);
    dev_col_active[idx] = static_cast<unsigned char>(active);
    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);

    // retain only physical records because the Morton owner builds its compact ghost list later
    dev_morton_point[idx] = (N_X == 1 && N_Z == 1)
        ? make_float3(static_cast<float>(R), 0.0f, 0.0f)
        : make_float3(
            static_cast<float>(R*cos(x)), static_cast<float>(R*sin(x)), static_cast<float>(Z)
        );
    dev_morton_posx[idx] = static_cast<float>(x);
    dev_search_dist[idx] = active ? static_cast<float>(H_SEARCH*_get_hg(R)*R) : 0.0f;
}
#endif // COLLISION_KDTREE

// =====================================================================================================================

#endif // COLLISION
