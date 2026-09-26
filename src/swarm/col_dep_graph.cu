#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_image.cuh>
#include <_col_types.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_dep_graph
// build the group dependency graph once per geometry epoch: edge c->d when an owner in c has a neighbor in d
// duplicate edges are reduced within each owner before issuing atomics, so no per-neighbor data reaches the host
// =====================================================================================================================

__global__
void col_dep_graph (unsigned int *edges, const int *spatial,
    const int *neighbors, const unsigned char *active)
{
    #ifdef GAMEDEV_ROCM
    // assign 32 lanes to each owner and OR-reduce their neighbor masks by shuffles
    if constexpr(TPB % 32 == 0)
    {
    const int lane = threadIdx.x % 32;
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) / 32;
    if (i >= N_P || !active[i]) return;
    unsigned int bits[LOCAL_WORDS] = {};
    for (int k = lane; k < N_K; k += 32)
    {
        int entry = neighbors[_get_col_offset(i, k)];
        if (entry < 0)continue;
        int j = _get_col_idx_old(entry);
        int c = spatial[j];
        bits[c / 32] |= 1u << (c % 32);
    }
    for (int w = 0; w < LOCAL_WORDS; ++w)
    {
        unsigned int value = bits[w];
        for (int delta = 16; delta; delta /= 2)value |= __shfl_down(value, delta, 32);
        if (lane == 0 && value)atomicOr(edges + spatial[i]*LOCAL_WORDS + w, value);
    }
    }
    else
    #endif // GAMEDEV_ROCM
    {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_P || !active[i]) return;
    unsigned int bits[LOCAL_WORDS] = {};
    for (int k = 0; k < N_K; ++k)
    {
        int entry = neighbors[_get_col_offset(i, k)];
        if (entry < 0) continue;
        int j = _get_col_idx_old(entry);
        int c = spatial[j];
        bits[c / 32] |= 1u << (c % 32);
    }
    for (int w = 0; w < LOCAL_WORDS; ++w)
    {
        if (bits[w]) atomicOr(edges + spatial[i]*LOCAL_WORDS + w, bits[w]);
    }
    }
}

#endif // COLLISION && !BERNOULLI
