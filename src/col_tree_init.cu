#ifdef COLLISION

#include <swarm_kern.cuh>

// =========================================================================================================================
// kernel: col_tree_init
// construct Cartesian KD-tree nodes and periodic azimuthal images from spherical particle positions
//
// parallelization: one thread per representative particle with up to two additional image nodes
// =========================================================================================================================

__global__
void col_tree_init (tree *dev_col_tree, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= N_P) return;

    float x = static_cast<float>(dev_particle[idx].position.x);
    float y = static_cast<float>(dev_particle[idx].position.y);
    float z = static_cast<float>(dev_particle[idx].position.z);

    // store the primary Cartesian node and its stable particle-array index
    dev_col_tree[idx].cartesian.x = y*sin(z)*cos(x);
    dev_col_tree[idx].cartesian.y = y*sin(z)*sin(x);
    dev_col_tree[idx].cartesian.z = y*cos(z);
    dev_col_tree[idx].index_old   = idx;
    dev_col_tree[idx].image       = 0;

    if (N_X > 1 && X_MAX - X_MIN < 2.0*M_PI - 1.0e-12)
        {
            // rotate copies by one wedge width so KNN queries cross periodic seams
            real width = X_MAX - X_MIN;
            for (int image = 0; image < 2; image++)
            {
                int idx_image = idx + (image + 1)*N_P;
                float x_image = static_cast<float>(x + ((image == 0) ? -width : width));
                dev_col_tree[idx_image].cartesian.x = y*sin(z)*cos(x_image);
                dev_col_tree[idx_image].cartesian.y = y*sin(z)*sin(x_image);
                dev_col_tree[idx_image].cartesian.z = y*cos(z);
                dev_col_tree[idx_image].index_old   = idx;
                dev_col_tree[idx_image].image       = image + 1;
            }
    }
}

// =========================================================================================================================

#endif // COLLISION
