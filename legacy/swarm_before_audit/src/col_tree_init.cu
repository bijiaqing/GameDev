#ifdef COLLISION

#include <graffiti_kern.cuh>

// =========================================================================================================================
// Kernel: col_tree_init
// Purpose: Initialize KD-tree nodes from particle positions (spherical to Cartesian conversion)
// Dependencies: tree structure, swarm structure
// =========================================================================================================================

__global__
void col_tree_init (tree *dev_col_tree, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        float x = static_cast<float>(dev_particle[idx].position.x);
        float y = static_cast<float>(dev_particle[idx].position.y);
        float z = static_cast<float>(dev_particle[idx].position.z);

        dev_col_tree[idx].cartesian.x = y*sin(z)*cos(x);
        dev_col_tree[idx].cartesian.y = y*sin(z)*sin(x);
        dev_col_tree[idx].cartesian.z = y*cos(z);
        dev_col_tree[idx].index_old   = idx;
    }
}

// =========================================================================================================================

#endif // COLLISION
