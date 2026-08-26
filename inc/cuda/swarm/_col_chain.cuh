#ifndef SWARM_COL_CHAIN_CUH
#define SWARM_COL_CHAIN_CUH

#ifdef COL_CHAIN

#include <algorithm>  // std::max, std::min
#include <climits>    // INT_MAX
#include <cmath>      // std::abs, std::isfinite, std::log, std::sqrt
#include <cstddef>    // std::size_t
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::runtime_error
#include <vector>     // std::vector

#include <_collision.cuh>
#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

struct col_rate_bin
{
    real mass;
    real weighted_rate;
    int owner_count;
    int invalid_count;
};

struct col_audit_accum
{
    real mass;
    real predicted_f;
    real predicted_e;
    real predicted_var_f;
    real predicted_var_e;
    real predicted_g;
    real predicted_var_g;
    real maximum_weight;
    real maximum_g_jump;
    real touched;
    real event_weight;
    real growth;
    real end_mass;
    int owner_count;
    int invalid_count;
};

struct col_bath_state
{
    real limit_scale = 1.0;
    int activity_streak = 0;
    int quiet_streak = 0;
};

struct col_bath_result
{
    real max_f = 0.0;
    real max_e = 0.0;
    real max_touched = 0.0;
    real max_events = 0.0;
    real max_g = 0.0;
    real max_g_upper = 0.0;
    real d_bath = 0.0;
    bool activity_overshoot = false;
    bool distribution_overshoot = false;
};

__host__ __device__ __forceinline__
std::size_t _get_col_offset (int idx_owner, int idx_neighbor)
{
    return static_cast<std::size_t>(idx_owner)*static_cast<std::size_t>(N_K)
        + static_cast<std::size_t>(idx_neighbor);
}

__host__ __device__ __forceinline__
int _get_col_sizebin (real size)
{
    if (!isfinite(size) || !(size > 0.0)) return 0;
    real fraction = log(size / COL_SIZE_MIN) / log(COL_SIZE_MAX / COL_SIZE_MIN);
    int idx_bin = static_cast<int>(floor(fraction*COL_BIN_S));
    if (idx_bin < 0) return 0;
    if (idx_bin >= COL_BIN_S) return COL_BIN_S - 1;
    return idx_bin;
}

__device__ __forceinline__
real _get_col_uniform (curs *rngstate)
{
    return fmin(curand_uniform_double(rngstate), nextafter(1.0, 0.0));
}

__device__ __forceinline__
void _col_atomic_max (real *address, real value)
{
    auto integer = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *integer;
    while (value > __longlong_as_double(static_cast<long long>(old)))
    {
        unsigned long long assumed = old;
        old = atomicCAS(integer, assumed,
            static_cast<unsigned long long>(__double_as_longlong(value)));
        if (old == assumed) break;
    }
}

// evaluate one pair against the mutable owner size and frozen partner reservoir
template <KernelType kernel> __device__ __forceinline__
real _get_col_chain_rate (const swarm *dev_particle, real size_i,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    int idx_old_i, int idx_old_j, real lambda_0)
{
    real size_j = dev_size_old[idx_old_j];
    real numr_j = dev_numr_old[idx_old_j];
    if constexpr (kernel == CONSTANT_KERNEL)
    {
        return lambda_0*numr_j;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        return lambda_0*numr_j*(_get_grain_mass(size_i) + _get_grain_mass(size_j));
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        return lambda_0*numr_j*_get_grain_mass(size_i)*_get_grain_mass(size_j);
    }
    else if constexpr (kernel == CUSTOM_KERNEL)
    {
        real vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        real rate = numr_j*vrel*M_PI*(size_i + size_j)*(size_i + size_j) / 4.0;
        if constexpr (N_Z == 1)
        {
            real R_i = _get_cyl_R(
                dev_particle[idx_old_i].position.y, dev_particle[idx_old_i].position.z
            );
            real R_j = _get_cyl_R(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            );
            real H_gi = R_i*_get_hg(R_i);
            real H_gj = R_j*_get_hg(R_j);
            rate /= sqrt(2.0*M_PI*(H_gi*H_gi + H_gj*H_gj));
        }
        return rate;
    }
    else
    {
        assert(false);
        return 0.0;
    }
}

__device__ __forceinline__
void _get_col_jump (real size_i, real size_j, bool fragmentation,
    real &mean, real &second, real &maximum)
{
    real merged_size = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);
    real log_growth = log(merged_size / size_i);
    if (!fragmentation)
    {
        mean = log_growth;
        second = log_growth*log_growth;
        maximum = log_growth;
        return;
    }

    real log_floor = log(INIT_SMIN / size_i);
    real u_floor = fmin(fmax(sqrt(INIT_SMIN / merged_size), 0.0), 1.0);
    real u_zero = fmin(fmax(sqrt(size_i / merged_size), u_floor), 1.0);
    mean = fmax(log_growth - 2.0 + 4.0*u_zero - 2.0*u_floor, 0.0);
    real second_at_one = log_growth*log_growth - 4.0*log_growth + 8.0;
    real second_at_floor = u_floor*(log_floor*log_floor - 4.0*log_floor + 8.0);
    second = fmax(u_floor*log_floor*log_floor + second_at_one - second_at_floor, 0.0);
    maximum = fmax(log_growth, -log_floor);
}

// cache the fixed physical top-K neighborhood once per collision operator
#ifdef COLLISION_KDTREE
__global__
void col_cache_get (int *dev_col_neighbor, real *dev_col_measure,
    const kdtree_node *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    const unsigned char *dev_col_active, const swarm *dev_particle,
    float image_dist_min)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= N_T || dev_kdtree_node[idx_tree].image != 0) return;

    int idx_old_i = dev_kdtree_node[idx_tree].idx_old;
    if (dev_col_active[idx_old_i] == 0)
    {
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        dev_col_measure[idx_old_i] = 0.0;
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // deduplicate overlapping wedge images using the same exact physical-id heap as the legacy search
    bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*search_dist;
    kdtree_heap near_result(search_dist, dev_kdtree_node, !unique_ids, dev_col_active);
    kdtree::cct::knn <kdtree_heap, kdtree_node, kdtree_traits> (
        near_result, dev_kdtree_node[idx_tree].cartesian,
        *dev_kdtree_box, dev_kdtree_node, N_T
    );

    float max_dist_sq = 0.0f;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = near_result.returnIndex(idx_neighbor);
        dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = idx_old_j;
        if (idx_old_j >= 0)
            max_dist_sq = fmaxf(max_dist_sq, near_result.returnDist2(idx_neighbor));
    }

    real radius = sqrt(static_cast<real>(max_dist_sq));
    real measure = _get_ball_measure(y, z, radius);
    #ifdef COLLISION_UNIT_VOLUME
    measure = 1.0;
    #endif // COLLISION_UNIT_VOLUME
    dev_col_measure[idx_old_i] = measure;
}
#else  // COLLISION_MORTON
__global__
void col_cache_get (int *dev_col_neighbor, real *dev_col_measure,
    unsigned int *dev_morton_overflow, const float3 *dev_morton_point,
    const unsigned char *dev_col_active, const swarm *dev_particle,
    morton_view morton_data, bool unique_ids)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    if (dev_col_active[idx_old_i] == 0)
    {
        for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        if (threadIdx.x == 0)
        {
            dev_col_measure[idx_old_i] = 0.0;
            dev_morton_overflow[idx_old_i] = 0;
        }
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    __shared__ float work_dist_sq[MORTON_WORK_SIZE];
    __shared__ int work_idx_old[MORTON_WORK_SIZE];
    __shared__ int idx_node_stack[256];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;

    _morton_ghost_topk<N_K, MORTON_TPB, MORTON_WORK_SIZE, 256>(
        morton_data, dev_morton_point[idx_old_i], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow, dev_col_active
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        int idx_old_j = work_idx_old[idx_neighbor];
        dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)]
            = (idx_old_j == INT_MAX) ? -1 : idx_old_j;
    }
    if (threadIdx.x == 0)
    {
        float max_dist_sq = 0.0f;
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
            if (work_idx_old[idx_neighbor] != INT_MAX)
                max_dist_sq = fmaxf(max_dist_sq, work_dist_sq[idx_neighbor]);
        real radius = sqrt(static_cast<real>(max_dist_sq));
        real measure = _get_ball_measure(y, z, radius);
        #ifdef COLLISION_UNIT_VOLUME
        measure = 1.0;
        #endif // COLLISION_UNIT_VOLUME
        dev_col_measure[idx_old_i] = measure;
        dev_morton_overflow[idx_old_i] = stack_overflow;
    }
}
#endif // COLLISION_KDTREE

// freeze the partner reservoir and reset continuation state for one bath
__global__
void col_bath_init (real *dev_size_old, real *dev_numr_old, real *dev_col_time,
    int *dev_col_events, unsigned char *dev_col_complete, const swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P) return;
    dev_size_old[idx] = dev_particle[idx].par_size;
    dev_numr_old[idx] = dev_particle[idx].par_numr;
    dev_col_time[idx] = 0.0;
    dev_col_events[idx] = 0;
    dev_col_complete[idx] = 0;
}

// assign fixed geometry bins used by every bath controller audit
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

// calculate bath-start rates and logarithmic jump moments from the cached neighborhood
__global__
void col_momnt_get (real *dev_col_rate, real *dev_col_jump1, real *dev_col_jump2,
    real *dev_col_jumpmax, const swarm *dev_particle, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    __shared__ real rate_work[N_K];
    __shared__ real mean_work[N_K];
    __shared__ real second_work[N_K];
    __shared__ real maximum_work[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        rate_work[idx_neighbor] = 0.0;
        mean_work[idx_neighbor] = 0.0;
        second_work[idx_neighbor] = 0.0;
        maximum_work[idx_neighbor] = 0.0;
        int idx_old_j = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
        if (dev_col_active[idx_old_i] == 0 || idx_old_j < 0 || idx_old_j == idx_old_i
            || !(dev_col_measure[idx_old_i] > 0.0)) continue;

        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];
        real pair_rate = _get_col_chain_rate <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, size_i, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, lambda_0
        ) / dev_col_measure[idx_old_i];
        bool fragmentation = false;
        if constexpr (COAG_KERNEL == CUSTOM_KERNEL)
        {
            real vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j
                #ifdef IMPORTGAS
                , dev_gas_dens
                #endif // IMPORTGAS
            );
            fragmentation = vrel > V_FRAG;
        }
        real mean = 0.0, second = 0.0, maximum = 0.0;
        _get_col_jump(size_i, size_j, fragmentation, mean, second, maximum);
        rate_work[idx_neighbor] = pair_rate;
        mean_work[idx_neighbor] = pair_rate*mean;
        second_work[idx_neighbor] = pair_rate*second;
        maximum_work[idx_neighbor] = maximum;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        real rate = 0.0, mean = 0.0, second = 0.0, maximum = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
        {
            rate += rate_work[idx_neighbor];
            mean += mean_work[idx_neighbor];
            second += second_work[idx_neighbor];
            maximum = fmax(maximum, maximum_work[idx_neighbor]);
        }
        dev_col_rate[idx_old_i] = rate;
        dev_col_jump1[idx_old_i] = mean;
        dev_col_jump2[idx_old_i] = second;
        dev_col_jumpmax[idx_old_i] = maximum;
    }
}

__global__
void col_count_bin (int *dev_col_count, const swarm *dev_particle,
    const int *dev_col_spatial, const unsigned char *dev_col_active)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P || dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
    atomicAdd(dev_col_count + idx_raw, 1);
}

__global__
void col_rate_bins (col_rate_bin *dev_col_bin, const swarm *dev_particle,
    const real *dev_col_rate, const real *dev_col_jump1, const real *dev_col_jump2,
    const real *dev_col_jumpmax, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P || dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
    int idx_bin = dev_col_binmap[idx_raw];
    real weight = dev_particle[idx].par_numr*_get_grain_mass(dev_particle[idx].par_size);
    atomicAdd(&dev_col_bin[idx_bin].owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0)
        || !isfinite(dev_col_rate[idx]) || dev_col_rate[idx] < 0.0
        || !isfinite(dev_col_jump1[idx]) || dev_col_jump1[idx] < 0.0
        || !isfinite(dev_col_jump2[idx]) || dev_col_jump2[idx] < 0.0
        || !isfinite(dev_col_jumpmax[idx]) || dev_col_jumpmax[idx] < 0.0)
    {
        atomicAdd(&dev_col_bin[idx_bin].invalid_count, 1);
        return;
    }
    atomicAdd(&dev_col_bin[idx_bin].mass, weight);
    atomicAdd(&dev_col_bin[idx_bin].weighted_rate, weight*dev_col_rate[idx]);
}

// evolve every owner against one frozen reservoir with bounded continuation
__global__
void col_chain_run (swarm *dev_particle, curs *dev_rngstate, int *dev_col_error,
    int *dev_col_unfinished, real *dev_col_time, int *dev_col_events,
    unsigned char *dev_col_complete, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, real bath_end)
{
    int idx_old_i = blockIdx.x;
    if (idx_old_i >= N_P) return;

    __shared__ real pair_rate[N_K];
    __shared__ real size_i;
    __shared__ real numr_i;
    __shared__ real time_i;
    __shared__ curs rngstate;
    __shared__ int event_count;
    __shared__ int accepted;
    __shared__ bool keep_running;

    if (threadIdx.x == 0)
    {
        size_i = dev_particle[idx_old_i].par_size;
        numr_i = dev_particle[idx_old_i].par_numr;
        time_i = dev_col_time[idx_old_i];
        rngstate = dev_rngstate[idx_old_i];
        event_count = dev_col_events[idx_old_i];
        accepted = 0;
        keep_running = dev_col_complete[idx_old_i] == 0;
        if (dev_col_active[idx_old_i] == 0 || !(dev_col_measure[idx_old_i] > 0.0))
        {
            time_i = bath_end;
            dev_col_complete[idx_old_i] = 1;
            keep_running = false;
        }
    }
    __syncthreads();

    while (true)
    {
        if (threadIdx.x == 0 && keep_running && accepted >= COL_EVENT_CAP)
            keep_running = false;
        __syncthreads();
        if (!keep_running) break;

        for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
        {
            int idx_old_j = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
            pair_rate[idx_neighbor] = (idx_old_j < 0 || idx_old_j == idx_old_i)
                ? 0.0
                : _get_col_chain_rate <static_cast<KernelType>(COAG_KERNEL)> (
                    dev_particle, size_i, dev_size_old, dev_numr_old,
                    #ifdef IMPORTGAS
                    dev_gas_dens,
                    #endif // IMPORTGAS
                    idx_old_i, idx_old_j, lambda_0
                ) / dev_col_measure[idx_old_i];
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            real total_rate = 0.0;
            int last_positive = -1;
            for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
            {
                if (!isfinite(pair_rate[idx_neighbor]) || pair_rate[idx_neighbor] < 0.0)
                {
                    dev_col_error[idx_old_i] = 1;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                    break;
                }
                if (pair_rate[idx_neighbor] > 0.0) last_positive = idx_neighbor;
                total_rate += pair_rate[idx_neighbor];
            }
            if (keep_running && (!isfinite(total_rate) || total_rate < 0.0))
            {
                dev_col_error[idx_old_i] = 2;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            if (keep_running && total_rate == 0.0)
            {
                time_i = bath_end;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            if (keep_running)
            {
                real wait = -log(_get_col_uniform(&rngstate)) / total_rate;
                real remaining = bath_end - time_i;
                if (!isfinite(wait) || !(wait > 0.0))
                {
                    dev_col_error[idx_old_i] = 3;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else if (wait >= remaining)
                {
                    time_i = bath_end;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else
                {
                    real event_time = time_i + wait;
                    if (!(event_time > time_i))
                    {
                        dev_col_error[idx_old_i] = 4;
                        dev_col_complete[idx_old_i] = 1;
                        keep_running = false;
                    }
                    else
                    {
                        real target = _get_col_uniform(&rngstate)*total_rate;
                        real cumulative = 0.0;
                        int idx_slot = -1;
                        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
                        {
                            if (pair_rate[idx_neighbor] <= 0.0) continue;
                            cumulative += pair_rate[idx_neighbor];
                            if (cumulative > target)
                            {
                                idx_slot = idx_neighbor;
                                break;
                            }
                        }
                        if (idx_slot < 0) idx_slot = last_positive;
                        if (idx_slot < 0)
                        {
                            dev_col_error[idx_old_i] = 5;
                            dev_col_complete[idx_old_i] = 1;
                            keep_running = false;
                        }
                        else
                        {
                            int idx_old_j = dev_col_neighbor[_get_col_offset(idx_old_i, idx_slot)];
                            real size_j = dev_size_old[idx_old_j];
                            real vrel = 0.0;
                            if constexpr (COAG_KERNEL == CUSTOM_KERNEL)
                            {
                                vrel = _get_vrel_pair(
                                    dev_particle, size_i, size_j, idx_old_i, idx_old_j
                                    #ifdef IMPORTGAS
                                    , dev_gas_dens
                                    #endif // IMPORTGAS
                                );
                            }
                            real mass_before = numr_i*size_i*size_i*size_i;
                            real size_new = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);
                            if (vrel > V_FRAG)
                            {
                                real frag_sample = _get_col_uniform(&rngstate);
                                size_new = fmax(INIT_SMIN, size_new*frag_sample*frag_sample);
                            }
                            numr_i = mass_before / (size_new*size_new*size_new);
                            size_i = size_new;
                            time_i = event_time;
                            event_count++;
                            accepted++;
                            real mass_after = numr_i*size_i*size_i*size_i;
                            real mass_scale = fmax(fabs(mass_before), fabs(mass_after));
                            if (!isfinite(size_i) || !isfinite(numr_i)
                                || !(size_i > 0.0) || !(numr_i > 0.0)
                                || fabs(mass_after - mass_before) > 2.0e-12*mass_scale)
                            {
                                dev_col_error[idx_old_i] = 6;
                                dev_col_complete[idx_old_i] = 1;
                                keep_running = false;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        dev_particle[idx_old_i].par_size = size_i;
        dev_particle[idx_old_i].par_numr = numr_i;
        dev_rngstate[idx_old_i] = rngstate;
        dev_col_time[idx_old_i] = time_i;
        dev_col_events[idx_old_i] = event_count;
        if (dev_col_complete[idx_old_i] == 0) atomicAdd(dev_col_unfinished, 1);
    }
}

__global__
void col_audit_bin (col_audit_accum *dev_col_bin, const swarm *dev_particle,
    const real *dev_size_old, const real *dev_numr_old, const real *dev_col_rate,
    const real *dev_col_jump1, const real *dev_col_jump2, const real *dev_col_jumpmax,
    const int *dev_col_events, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active, real duration)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_P || dev_col_active[idx] == 0) return;
    int idx_spatial = dev_col_spatial[idx];
    int idx_start_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_size_old[idx]);
    int idx_end_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
    int idx_start = dev_col_binmap[idx_start_raw];
    int idx_end = dev_col_binmap[idx_end_raw];
    real weight = dev_numr_old[idx]*_get_grain_mass(dev_size_old[idx]);
    real probability = -expm1(-dev_col_rate[idx]*duration);
    real expected_events = dev_col_rate[idx]*duration;
    real events = static_cast<real>(dev_col_events[idx]);
    col_audit_accum *bin = dev_col_bin + idx_start;
    atomicAdd(&bin->owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0) || !isfinite(probability)
        || !isfinite(expected_events) || !isfinite(events)
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
    atomicAdd(&bin->predicted_g, weight*dev_col_jump1[idx]*duration);
    atomicAdd(&bin->predicted_var_g, weight*weight*dev_col_jump2[idx]*duration);
    _col_atomic_max(&bin->maximum_weight, weight);
    _col_atomic_max(&bin->maximum_g_jump, weight*dev_col_jumpmax[idx]);
    if (dev_col_events[idx] > 0) atomicAdd(&bin->touched, weight);
    atomicAdd(&bin->event_weight, weight*events);
    atomicAdd(&bin->growth, weight*fabs(log(dev_particle[idx].par_size / dev_size_old[idx])));
    atomicAdd(&dev_col_bin[idx_end].end_mass, weight);
}

inline
int _get_col_raw_count ()
{
    int count_x = (N_X > 1) ? COL_BIN_X : 1;
    int count_z = (N_Z > 1) ? COL_BIN_Z : 1;
    return count_x*COL_BIN_Y*count_z*COL_BIN_S;
}

inline
int _build_col_binmap (const std::vector<int> &raw_count, std::vector<int> &raw_to_merged)
{
    int spatial_count = _get_col_raw_count() / COL_BIN_S;
    raw_to_merged.assign(raw_count.size(), -1);
    int merged_count = 0;
    for (int idx_spatial = 0; idx_spatial < spatial_count; idx_spatial++)
    {
        int last_occupied = -1;
        for (int idx_size = 0; idx_size < COL_BIN_S; idx_size++)
            if (raw_count[idx_spatial*COL_BIN_S + idx_size] > 0) last_occupied = idx_size;
        if (last_occupied < 0)
        {
            int idx_merged = merged_count++;
            for (int idx_size = 0; idx_size < COL_BIN_S; idx_size++)
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            continue;
        }

        int idx_begin = 0;
        int idx_previous = -1;
        while (idx_begin <= last_occupied)
        {
            int idx_end = idx_begin;
            int count = 0;
            while (idx_end <= last_occupied && count < COL_BIN_MIN)
                count += raw_count[idx_spatial*COL_BIN_S + idx_end++];
            bool merge_tail = idx_end > last_occupied && count < COL_BIN_MIN
                && idx_previous >= 0;
            int idx_merged = merge_tail ? idx_previous : merged_count++;
            for (int idx_size = idx_begin; idx_size < idx_end; idx_size++)
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            idx_previous = idx_merged;
            idx_begin = idx_end;
        }
        if (last_occupied + 1 < COL_BIN_S)
        {
            int idx_merged = merged_count++;
            for (int idx_size = last_occupied + 1; idx_size < COL_BIN_S; idx_size++)
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
        }
    }
    return merged_count;
}

inline
real _choose_col_bath (const std::vector<col_rate_bin> &bin, int bin_count,
    real remaining, real limit_scale)
{
    real duration = std::min(remaining, COL_BATH_MAX);
    real tolerance = COL_BATH_EPS*limit_scale;
    for (int idx_bin = 0; idx_bin < bin_count; idx_bin++)
    {
        const col_rate_bin &value = bin[idx_bin];
        if (value.invalid_count != 0)
            throw std::runtime_error("collision bath controller returned invalid rate moments");
        if (!(value.mass > 0.0) || !(value.weighted_rate > 0.0)) continue;
        duration = std::min(duration, tolerance*value.mass / value.weighted_rate);
    }
    if (!(duration > 0.0) || !std::isfinite(duration))
        throw std::runtime_error("collision bath controller selected an invalid duration");
    return duration;
}

inline
col_bath_result _finish_col_bath (const std::vector<col_audit_accum> &bin,
    int bin_count, col_bath_state &state)
{
    col_bath_result result;
    int active_count = 0;
    real total_mass = 0.0;
    for (int idx_bin = 0; idx_bin < bin_count; idx_bin++)
    {
        const col_audit_accum &value = bin[idx_bin];
        if (value.invalid_count != 0)
            throw std::runtime_error("collision bath controller returned an invalid audit state");
        if (value.mass > 0.0) active_count++;
        total_mass += value.mass;
    }
    if (!(total_mass > 0.0))
        throw std::runtime_error("collision bath controller found no active represented mass");
    real confidence_log = log(2.0*static_cast<real>(active_count) / COL_BATH_ALPHA);
    real mass_difference = 0.0;
    for (int idx_bin = 0; idx_bin < bin_count; idx_bin++)
    {
        const col_audit_accum &value = bin[idx_bin];
        mass_difference += std::abs(value.end_mass - value.mass);
        if (!(value.mass > 0.0)) continue;
        real mass_sq = value.mass*value.mass;
        real predicted_f = value.predicted_f / value.mass;
        real predicted_e = value.predicted_e / value.mass;
        real predicted_g = value.predicted_g / value.mass;
        real touched = value.touched / value.mass;
        real events = value.event_weight / value.mass;
        real growth = value.growth / value.mass;
        real weight_max = value.maximum_weight / value.mass;
        real jump_max = value.maximum_g_jump / value.mass;
        real upper_f = std::min(1.0, predicted_f
            + std::sqrt(2.0*value.predicted_var_f / mass_sq*confidence_log)
            + weight_max*confidence_log / 3.0);
        real upper_e = predicted_e
            + std::sqrt(2.0*value.predicted_var_e / mass_sq*confidence_log)
            + weight_max*confidence_log / 3.0;
        real upper_g = predicted_g
            + std::sqrt(2.0*value.predicted_var_g / mass_sq*confidence_log)
            + jump_max*confidence_log / 3.0;
        result.max_f = std::max(result.max_f, predicted_f);
        result.max_e = std::max(result.max_e, predicted_e);
        result.max_touched = std::max(result.max_touched, touched);
        result.max_events = std::max(result.max_events, events);
        result.max_g = std::max(result.max_g, growth);
        result.max_g_upper = std::max(result.max_g_upper, upper_g);
        result.activity_overshoot = result.activity_overshoot
            || touched > upper_f || events > upper_e;
        result.distribution_overshoot = result.distribution_overshoot
            || growth > std::max(COL_BATH_EPS, upper_g);
    }
    result.d_bath = mass_difference / (2.0*total_mass);
    result.distribution_overshoot = result.distribution_overshoot
        || result.d_bath > COL_BATH_EPS;

    bool issue = result.activity_overshoot || result.distribution_overshoot;
    if (issue)
    {
        state.activity_streak++;
        state.quiet_streak = 0;
    }
    else
    {
        state.activity_streak = 0;
        state.quiet_streak++;
    }
    if (result.distribution_overshoot || state.activity_streak >= 2)
    {
        state.limit_scale = std::max(0.25, 0.5*state.limit_scale);
        state.activity_streak = 0;
    }
    else if (state.quiet_streak >= 3)
    {
        state.limit_scale = std::min(1.0, 1.25*state.limit_scale);
        state.quiet_streak = 0;
    }
    return result;
}

#endif // COL_CHAIN

#endif // SWARM_COL_CHAIN_CUH
