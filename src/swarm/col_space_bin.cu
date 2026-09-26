#include <gpu.cuh>

#ifdef COLLISION
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_space_bin
// assign fixed geometry bins used by every bath controller audit
// =====================================================================================================================

__global__
void col_space_bin (int *dev_col_spatial, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;
    real x = dev_particle[idx].position.x;
    real y = dev_particle[idx].position.y;
    real z = dev_particle[idx].position.z;
    int bin_x = 0;
    int bin_y = static_cast<int>(floor((y - Y_MIN) / (Y_MAX - Y_MIN)*COL_BIN_Y));
    int bin_z = 0;
    if constexpr (N_X > 1)
        bin_x = static_cast<int>(floor((x - X_MIN) / (X_MAX - X_MIN)*COL_BIN_X));
    if constexpr (N_Z > 1)
    {
        real extent_z = Z_MAX - Z_MIN;
        bin_z = static_cast<int>(floor((z - Z_MIN) / extent_z*COL_BIN_Z));
    }
    bin_x = (bin_x < 0) ? 0 : ((bin_x >= COL_BIN_X) ? COL_BIN_X - 1 : bin_x);
    bin_y = (bin_y < 0) ? 0 : ((bin_y >= COL_BIN_Y) ? COL_BIN_Y - 1 : bin_y);
    bin_z = (bin_z < 0) ? 0 : ((bin_z >= COL_BIN_Z) ? COL_BIN_Z - 1 : bin_z);
    int count_x = (N_X > 1) ? COL_BIN_X : 1;
    int count_y = COL_BIN_Y;
    dev_col_spatial[idx] = bin_x + count_x*(bin_y + count_y*bin_z);
}

#endif // COLLISION
