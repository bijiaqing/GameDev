#ifdef COLLISION

#include <param_phys.cuh>
#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: col_tree_init
// construct Cartesian collision-search records from spherical particle positions
//
// parallelization: one thread per representative particle
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_tree_init (kdtree_node *dev_col_tree, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    float x = static_cast<float>(dev_particle[idx].position.x);
    float y = static_cast<float>(dev_particle[idx].position.y);
    float z = static_cast<float>(dev_particle[idx].position.z);

    float3 cartesian = make_float3(y*sin(z)*cos(x), y*sin(z)*sin(x), y*cos(z));
    dev_col_tree[idx].cartesian = cartesian;
    dev_col_tree[idx].index_old = idx;
    dev_col_tree[idx].image = 0;

    if (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12)
    {
        float width = static_cast<float>(X_MAX - X_MIN);
        float sin_width;
        float cos_width;
        sincosf(width, &sin_width, &cos_width);
        for (int image = 0; image < 2; image++)
        {
            int idx_image = idx + (image + 1)*N_P;
            float sin_angle = (image == 0) ? -sin_width : sin_width;
            dev_col_tree[idx_image].cartesian = make_float3(
                cos_width*cartesian.x - sin_angle*cartesian.y,
                sin_angle*cartesian.x + cos_width*cartesian.y,
                cartesian.z
            );
            dev_col_tree[idx_image].index_old = idx;
            dev_col_tree[idx_image].image = image + 1;
        }
    }
}
#else  // COLLISION_MORTON
__global__
void col_tree_init (float3 *dev_col_point, float *dev_col_x, float *dev_col_cutoff,
    const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    real R = y*sin(z);

    dev_col_point[idx] = make_float3(
        static_cast<float>(R*cos(x)),
        static_cast<float>(R*sin(x)),
        static_cast<float>(y*cos(z))
    );
    dev_col_x[idx] = static_cast<float>(x);
    dev_col_cutoff[idx] = static_cast<float>(H_SEARCH*_get_hg(R)*R);
}
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // COLLISION
