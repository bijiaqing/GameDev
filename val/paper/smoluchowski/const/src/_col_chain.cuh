#pragma once
#include <swarm_kern.cuh>

// Scratch extrema are reset globally; retained bounds change only for refreshed groups.
constexpr int moving_groups = ((N_X>1)?COL_BIN_X:1)*COL_BIN_Y*((N_Z>1)?COL_BIN_Z:1);
static __device__ unsigned long long moving_min[moving_groups], moving_max[moving_groups];
static __device__ real moving_lower[moving_groups], moving_upper[moving_groups];

static __global__ void reset_size_extrema() {
    int g=threadIdx.x+blockIdx.x*blockDim.x;
    if (g<moving_groups) { moving_min[g]=__double_as_longlong(INFINITY); moving_max[g]=0; }
}
static __global__ void reduce_size_extrema(const int *ids,int count,const swarm *particles,
    const int *spatial,const unsigned char *active) {
    int slot=threadIdx.x+blockIdx.x*blockDim.x;
    if (slot>=count) return;
    int i=ids[slot];
    real size=particles[i].par_size;
    if (!active[i] || !(size>0) || !isfinite(size)) return;
    // Positive doubles have the same ordering as their unsigned bit patterns.
    auto bits=static_cast<unsigned long long>(__double_as_longlong(size));
    atomicMin(&moving_min[spatial[i]],bits);
    atomicMax(&moving_max[spatial[i]],bits);
}
static __global__ void publish_size_bounds() {
    int g=threadIdx.x+blockIdx.x*blockDim.x;
    if (g<moving_groups && moving_max[g]!=0) {
        moving_lower[g]=0.5*__longlong_as_double(moving_min[g]);
        moving_upper[g]=8.0*__longlong_as_double(moving_max[g]);
    }
}
__device__ __forceinline__ int moving_sizebin(real size,int group) {
    if (!isfinite(size) || !(size>0)) return 0;
    real fraction=log(size/moving_lower[group])/log(moving_upper[group]/moving_lower[group]);
    return max(0,min(COL_BIN_S-1,static_cast<int>(floor(fraction*COL_BIN_S))));
}
#define COL_SIZEBIN(size, spatial) moving_sizebin(size, spatial)
#define COL_REFRESH_SIZE_BINS(ids, count, particles, spatial, active) do { \
    reset_size_extrema<<<(moving_groups+TPB-1)/TPB,TPB>>>(); \
    GPU_KERNEL_CHECK("reset_size_extrema"); \
    reduce_size_extrema<<<((count)+TPB-1)/TPB,TPB>>>(ids,count,particles,spatial,active); \
    GPU_KERNEL_CHECK("reduce_size_extrema"); \
    publish_size_bounds<<<(moving_groups+TPB-1)/TPB,TPB>>>(); \
    GPU_KERNEL_CHECK("publish_size_bounds"); \
} while (0)
#include "../../../../../inc/swarm/_col_chain.cuh"
#undef COL_REFRESH_SIZE_BINS
#undef COL_SIZEBIN
