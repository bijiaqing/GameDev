#include <gpu.cuh>

#ifdef COLLISION
#include <_col_rates.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_skip_scan
// evaluate the cached no-event path with one thread per owner
// complete owners whose first waiting time spans the interval and queue only the rest for col_chain_run
// =====================================================================================================================

__global__
void col_skip_scan (const int *ids, int count, curs *rng,
    const unsigned char *active, const real *measure, const int *spatial, const real *steps,
    const cached_rate_moments *cached, real *time, int *events, unsigned char *complete,
    real *hazard, real *jump1, real *jump2, real *jumpmax, int *queue, int *queued)
{
    int slot = blockIdx.x*blockDim.x + threadIdx.x;
    if (slot >= count) return;
    int i = ids[slot];
    real end = steps[spatial[i]];
    if (!active[i] || !(measure[i] > 0.0))
    {
        time[i] = end;
        complete[i] = 1;
        return;
    }
    if (complete[i]) return;
    if (time[i] == 0.0 && events[i] == 0)
    {
        const auto c = cached[i];
        if (isfinite(c.rate) && c.rate >= 0.0)
        {
            if (c.rate == 0.0)
            {
                time[i] = end;
                complete[i] = 1;
                return;
            }
            curs state = rng[i];
            real wait = -log(_get_col_uniform(&state)) / c.rate;
            real remaining = end - time[i];
            if (isfinite(wait) && wait > 0.0 && wait >= remaining)
            {
                hazard[i] += c.rate*remaining;
                jump1[i] += c.first*remaining;
                jump2[i] += c.second*remaining;
                jumpmax[i] = fmax(jumpmax[i], c.maximum);
                time[i] = end;
                complete[i] = 1;
                rng[i] = state;
                return;
            }
            // leave the trial RNG draw uncommitted because the full chain repeats the same draw
        }
    }
    queue[atomicAdd(queued, 1)] = i;
}

#endif // COLLISION
