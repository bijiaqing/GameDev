#ifdef COLLISION

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: col_tree_init
// construct the backend Cartesian search records from spherical particle positions
//
// parallelization: one thread per representative particle
// =========================================================================================================================

#ifdef COLLISION_KDTREE
__global__
void col_tree_init (tree *dev_col_tree, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    float x = static_cast<float>(dev_particle[idx].position.x);
    float y = static_cast<float>(dev_particle[idx].position.y);
    float z = static_cast<float>(dev_particle[idx].position.z);

    // store the primary Cartesian node and its stable particle-array index
    float3 cartesian = make_float3(y*sin(z)*cos(x), y*sin(z)*sin(x), y*cos(z));
    dev_col_tree[idx].cartesian   = cartesian;
    dev_col_tree[idx].index_old   = idx;
    dev_col_tree[idx].image       = 0;

    if (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12)
    {
        // rotate the rounded physical point so ghost and query images use the same Cartesian isometry
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
            dev_col_tree[idx_image].image     = image + 1;
        }
    }
}
#else  // COLLISION_MORTON
__global__
void col_tree_init (float3 *dev_col_point, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    float x = static_cast<float>(dev_particle[idx].position.x);
    float y = static_cast<float>(dev_particle[idx].position.y);
    float z = static_cast<float>(dev_particle[idx].position.z);

    dev_col_point[idx] = make_float3(y*sin(z)*cos(x), y*sin(z)*sin(x), y*cos(z));
}
#endif // COLLISION_KDTREE

// =========================================================================================================================

#endif // COLLISION
