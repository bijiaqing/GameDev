#include <gpu.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)

#include <_col_rates.cuh>
#include <_col_sizes.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_audit_bin
// aggregate predicted moments and realized changes for post-bath validation
// =====================================================================================================================

__global__
void col_audit_bin (const int *owner_ids, int owner_count, col_audit_accum *dev_col_bin, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const real *dev_col_rate,
    const real *dev_col_hazard, const real *dev_col_jump1_int,
    const real *dev_col_jump2_int, const real *dev_col_jumpmax_int,
    const int *dev_col_events, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active, const real *group_step)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_spatial = dev_col_spatial[idx];
    real duration = group_step[idx_spatial];
    int idx_start_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_size_old[idx], idx_spatial);
    int idx_end_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size, dev_col_spatial[idx]);
    int idx_start = dev_col_binmap[idx_start_raw];
    int idx_end = dev_col_binmap[idx_end_raw];
    real weight = dev_numr_old[idx]*_get_grain_mass(dev_size_old[idx]);
    real probability = -expm1(-dev_col_rate[idx]*duration);
    real expected_events = dev_col_hazard[idx];
    real events = static_cast<real>(dev_col_events[idx]);
    col_audit_accum *bin = dev_col_bin + idx_start;
    atomicAdd(&bin->owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0) || !isfinite(probability)
        || !isfinite(expected_events) || expected_events < 0.0 || !isfinite(events)
        || !isfinite(dev_col_jump1_int[idx]) || dev_col_jump1_int[idx] < 0.0
        || !isfinite(dev_col_jump2_int[idx]) || dev_col_jump2_int[idx] < 0.0
        || !isfinite(dev_col_jumpmax_int[idx]) || dev_col_jumpmax_int[idx] < 0.0
        || !isfinite(dev_particle[idx].par_size) || !(dev_particle[idx].par_size > 0.0))
    {
        atomicAdd(&bin->invalid_count, 1);
        return;
    }
    atomicAdd(&bin->mass, weight);
    atomicAdd(&bin->predicted_f, weight*probability);
    atomicAdd(&bin->predicted_e, weight*expected_events);
    atomicAdd(&bin->predicted_var_f, weight*weight*probability*(1.0 - probability));
    atomicAdd(&bin->predicted_var_e, weight*weight*expected_events);
    atomicAdd(&bin->predicted_g, weight*dev_col_jump1_int[idx]);
    atomicAdd(&bin->predicted_var_g, weight*weight*dev_col_jump2_int[idx]);
    _col_atomic_max(&bin->maximum_weight, weight);
    _col_atomic_max(&bin->maximum_g_jump, weight*dev_col_jumpmax_int[idx]);
    if (dev_col_events[idx] > 0) atomicAdd(&bin->touched, weight);
    atomicAdd(&bin->event_weight, weight*events);
    atomicAdd(&bin->growth, weight*fabs(log(dev_particle[idx].par_size / dev_size_old[idx])));
    atomicAdd(&dev_col_bin[idx_end].end_mass, weight);
}

#endif // COLLISION && !BERNOULLI
