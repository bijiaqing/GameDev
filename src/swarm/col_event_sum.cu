#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_types.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_event_sum
// reduce per-owner diagnostics once per collision half-step with one block per event category
// =====================================================================================================================

__global__
void col_event_sum (const event_work *work, event_work *sum)
{
    const int k = blockIdx.x;
    const int t = threadIdx.x;
    __shared__ unsigned long long counts[TPB];
    __shared__ real growth[TPB];
    unsigned long long n = 0;
    real g = 0.0;
    for (int i = t; i < N_P; i += TPB)
    {
        n += work[i].count[k];
        g += work[i].log_mass[k];
    }
    counts[t] = n;
    growth[t] = g;
    __syncthreads();
    for (int stride = TPB / 2; stride > 0; stride /= 2)
    {
        if (t < stride)
        {
            counts[t] += counts[t + stride];
            growth[t] += growth[t + stride];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        sum->count[k] = counts[0];
        sum->log_mass[k] = growth[0];
    }
}
static_assert(TPB > 0 && (TPB & (TPB - 1)) == 0, "event reduction needs power-of-two TPB");

#endif // COLLISION && !BERNOULLI
