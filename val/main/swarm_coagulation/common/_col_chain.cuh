// Lab copy: active-owner dispatch, per-group durations, compact continuations.
#ifndef SWARM_COL_CHAIN_CUH
#define SWARM_COL_CHAIN_CUH

#if defined(COLLISION) && !defined(BERNOULLI)

#include <algorithm>  // std::max, std::min
#include <climits>    // INT_MAX
#include <cmath>      // std::abs, std::isfinite, std::log, std::sqrt
#include <cstddef>    // std::size_t
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <vector>     // std::vector

#include <_col_cache.cuh>
#include <_collision.cuh>
#include "query_environment.cuh"
#include "change_limit.hpp"
#include "event_work.cuh"
#include "erosion_outcome.cuh"
#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

// retain mass-weighted bath-start rate moments for one merged controller bin
struct col_rate_bin
{
    real mass;
    real weighted_rate;
    real weighted_change;
    real weighted_second;
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
    int floor_streak = 0;
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
    bool persistent_overshoot = false;
};

struct col_bath_record
{
    int group_index = -1;
    int operator_index = 0;
    int bath_index = 0;
    int merged_bins = 0;
    int continuation_launches = 0;
    real duration = 0.0;
    real limit_before = 1.0;
    real limit_after = 1.0;
    col_bath_result result;
};

struct col_controller_summary
{
    int operator_count = 0;
    int bath_count = 0;
    int continuation_launches = 0;
    int activity_overshoots = 0;
    int distribution_overshoots = 0;
    int persistent_overshoots = 0;
    real minimum_duration = std::numeric_limits<real>::infinity();
    real maximum_duration = 0.0;
    real minimum_limit_scale = 1.0;
    real maximum_f = 0.0;
    real maximum_e = 0.0;
    real maximum_touched = 0.0;
    real maximum_events = 0.0;
    real maximum_g = 0.0;
    real maximum_g_upper = 0.0;
    real maximum_d_bath = 0.0;
    std::vector<col_bath_record> baths;
};

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
    #ifdef GAMEDEV_CUDA
    return fmin(curand_uniform_double(rngstate), nextafter(1.0, 0.0));
    #else  // GAMEDEV_ROCM
    return fmin(hiprand_uniform_double(rngstate), nextafter(1.0, 0.0));
    #endif // GAMEDEV_CUDA
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

// evaluate one pair with the same synthetic or physical normalization as _get_col_rate_ij
template <KernelType kernel> __device__ __forceinline__
real _get_col_chain_rate (const swarm *dev_particle, real size_i,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    int idx_old_i, int idx_old_j, int image_j, real lambda_0, real &vrel,
    const query_environment &environment)
{
    vrel = 0.0;
    real size_j = dev_size_old[idx_old_j];
    // include the owner's own swarm when i == j, using the large-number approximation N_i - 1 ~= N_i
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
        vrel = cached_pair_velocity(environment,size_i,size_j);
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

// freeze the partner reservoir and reset continuation state for one bath
__global__
void col_bath_init (const int *owner_ids, int owner_count, real *dev_size_old, real *dev_numr_old, real *dev_col_time,
    int *dev_col_events, unsigned char *dev_col_complete, const swarm *dev_particle)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
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

// calculate bath-start rates used by the pre-bath duration controller
__global__
void col_bath_rate (const int *owner_ids, int owner_count, real *dev_col_rate, real *change_rate, real *second_rate, const swarm *dev_particle, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const query_environment *environment, cached_rate_moments *cached)
{
    int slot = blockIdx.x;
    if (slot >= owner_count) return;
    int idx_old_i = owner_ids[slot];

    __shared__ real rate_work[N_K];
    __shared__ real change_work[N_K], second_work[N_K], maximum_work[N_K];
    for (int idx_neighbor = threadIdx.x; idx_neighbor < N_K; idx_neighbor += blockDim.x)
    {
        rate_work[idx_neighbor] = 0.0;
        change_work[idx_neighbor] = second_work[idx_neighbor] = maximum_work[idx_neighbor] = 0.0;
        int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
        if (dev_col_active[idx_old_i] == 0 || neighbor < 0
            || !(dev_col_measure[idx_old_i] > 0.0)) continue;
        int idx_old_j = _get_col_idx_old(neighbor);
        int image_j = _get_col_image(neighbor);

        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];
        real vrel = 0.0;
        real pair_rate = _get_col_chain_rate <static_cast<KernelType>(COAG_KERNEL)> (
            dev_particle, size_i, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment[idx_old_i]
        ) / dev_col_measure[idx_old_i];
        real mean=0, second=0, maximum=0;
        pair_rate *= erosion_outcome_moments(size_i,size_j,vrel>=V_FRAG,mean,second,maximum);
        maximum_work[idx_neighbor]=pair_rate>0.0 ? maximum : 0.0;
        change_work[idx_neighbor]=pair_rate*mean;
        second_work[idx_neighbor]=pair_rate*second;
        rate_work[idx_neighbor] = pair_rate;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        real rate = 0.0, change=0.0, second=0.0, maximum=0.0;
        bool valid=true;
        for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
        {
            valid=valid && isfinite(rate_work[idx_neighbor]) && rate_work[idx_neighbor]>=0.0;
            maximum=fmax(maximum,maximum_work[idx_neighbor]);
            rate += rate_work[idx_neighbor];
            change += change_work[idx_neighbor];
            second += second_work[idx_neighbor];
        }
        cached[idx_old_i]={valid ? rate : -1.0,change,second,maximum};
        dev_col_rate[idx_old_i] = rate;
        change_rate[idx_old_i]=change; second_rate[idx_old_i]=second;
    }
}

// count occupied size bins before merging statistically undersampled tails
__global__
void col_count_bin (const int *owner_ids, int owner_count, int *dev_col_count, const swarm *dev_particle,
    const int *dev_col_spatial, const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
    atomicAdd(dev_col_count + idx_raw, 1);
}

// accumulate mass-weighted rates for the pre-bath duration bound
__global__
void col_rate_bins (const int *owner_ids, int owner_count, col_rate_bin *dev_col_bin, const swarm *dev_particle,
    const real *dev_col_rate, const real *change_rate, const real *second_rate, const int *dev_col_spatial, const int *dev_col_binmap,
    const unsigned char *dev_col_active)
{
    int slot = threadIdx.x + blockDim.x*blockIdx.x;
    if (slot >= owner_count) return;
    int idx = owner_ids[slot];
    if (dev_col_active[idx] == 0) return;
    int idx_raw = dev_col_spatial[idx]*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
    int idx_bin = dev_col_binmap[idx_raw];
    real weight = dev_particle[idx].par_numr*_get_grain_mass(dev_particle[idx].par_size);
    atomicAdd(&dev_col_bin[idx_bin].owner_count, 1);
    if (!isfinite(weight) || !(weight > 0.0)
        || !isfinite(dev_col_rate[idx]) || dev_col_rate[idx] < 0.0
        || !isfinite(change_rate[idx]) || change_rate[idx]<0
        || !isfinite(second_rate[idx]) || second_rate[idx]<0)
    {
        atomicAdd(&dev_col_bin[idx_bin].invalid_count, 1);
        return;
    }
    atomicAdd(&dev_col_bin[idx_bin].mass, weight);
    atomicAdd(&dev_col_bin[idx_bin].weighted_rate, weight*dev_col_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_change,weight*change_rate[idx]);
    atomicAdd(&dev_col_bin[idx_bin].weighted_second,weight*second_rate[idx]);
}

// evolve every owner against one frozen reservoir with bounded continuation
__global__
void col_chain_run (const int *owner_ids, int owner_count, swarm *dev_particle, curs *dev_rngstate, int *dev_col_error,
    int *dev_col_unfinished, real *dev_col_time, int *dev_col_events,
    unsigned char *dev_col_complete, real *dev_col_hazard,
    real *dev_col_jump1_int, real *dev_col_jump2_int, real *dev_col_jumpmax_int,
    const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    real lambda_0, const real *group_step, const int *spatial,
    int *unfinished_ids, int *error_flag, event_work *work,
    const query_environment *environment, const cached_rate_moments *cached)
{
    int slot = blockIdx.x;
    if (slot >= owner_count) return;
    int idx_old_i = owner_ids[slot];
    real bath_end = group_step[spatial[idx_old_i]];

    __shared__ real pair_rate[N_K];
    __shared__ real pair_jump1[N_K];
    __shared__ real pair_jump2[N_K];
    __shared__ real pair_jumpmax[N_K];
    __shared__ real size_i;
    __shared__ real numr_i;
    __shared__ real time_i;
    __shared__ real hazard_i;
    __shared__ real jump1_i;
    __shared__ real jump2_i;
    __shared__ real jumpmax_i;
    #ifdef GAMEDEV_CUDA
    __shared__ curs rngstate;
    #else  // GAMEDEV_ROCM
    curs rngstate; // keep the HIP RNG object local because shared objects cannot be initialized
    #endif // GAMEDEV_CUDA
    __shared__ int event_count;
    __shared__ int accepted;
    __shared__ event_work event_stats;
    __shared__ bool keep_running;

    if (threadIdx.x == 0)
    {
        size_i = dev_particle[idx_old_i].par_size;
        numr_i = dev_particle[idx_old_i].par_numr;
        time_i = dev_col_time[idx_old_i];
        hazard_i = dev_col_hazard[idx_old_i];
        jump1_i = dev_col_jump1_int[idx_old_i];
        jump2_i = dev_col_jump2_int[idx_old_i];
        jumpmax_i = dev_col_jumpmax_int[idx_old_i];
        rngstate = dev_rngstate[idx_old_i];
        event_count = dev_col_events[idx_old_i];
        accepted = 0;
        event_stats = work[idx_old_i];
        keep_running = dev_col_complete[idx_old_i] == 0;
        if (dev_col_active[idx_old_i] == 0 || !(dev_col_measure[idx_old_i] > 0.0))
        {
            time_i = bath_end;
            dev_col_complete[idx_old_i] = 1;
            keep_running = false;
        }
    }
    __syncthreads();

    // Rates were computed after this wave's publication, with no intervening writes
    // to reservoir arrays. Never use the start cache after an event/continuation.
    if (threadIdx.x==0 && keep_running && time_i==0.0 && event_count==0) {
        const auto c=cached[idx_old_i];
        if (isfinite(c.rate) && c.rate>=0.0) {
            if (c.rate==0.0) {
                time_i=bath_end; dev_col_complete[idx_old_i]=1; keep_running=false;
            } else {
                curs before=rngstate;
                real wait=-log(_get_col_uniform(&rngstate))/c.rate;
                real remaining=bath_end-time_i;
                if (isfinite(wait) && wait>0.0 && wait>=remaining) {
                    hazard_i+=c.rate*remaining;
                    jump1_i+=c.first*remaining; jump2_i+=c.second*remaining;
                    jumpmax_i=fmax(jumpmax_i,c.maximum);
                    time_i=bath_end; dev_col_complete[idx_old_i]=1; keep_running=false;
                } else {
                    // Normal path handles both events and invalid waits with its original draw.
                    rngstate=before;
                }
            }
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
            int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
            pair_rate[idx_neighbor] = 0.0;
            pair_jump1[idx_neighbor] = 0.0;
            pair_jump2[idx_neighbor] = 0.0;
            pair_jumpmax[idx_neighbor] = 0.0;
            if (neighbor >= 0)
            {
                int idx_old_j = _get_col_idx_old(neighbor);
                int image_j = _get_col_image(neighbor);
                real size_j = dev_size_old[idx_old_j];
                real vrel = 0.0;
                real pair_value = _get_col_chain_rate <static_cast<KernelType>(COAG_KERNEL)> (
                    dev_particle, size_i, dev_size_old, dev_numr_old,
                    #ifdef IMPORTGAS
                    dev_gas_dens,
                    #endif // IMPORTGAS
                    idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment[idx_old_i]
                ) / dev_col_measure[idx_old_i];
                real mean = 0.0, second = 0.0, maximum = 0.0;
                pair_value *= erosion_outcome_moments(size_i,size_j,vrel>=V_FRAG,mean,second,maximum);
                pair_rate[idx_neighbor] = pair_value;
                pair_jump1[idx_neighbor] = pair_value*mean;
                pair_jump2[idx_neighbor] = pair_value*second;
                pair_jumpmax[idx_neighbor] = (pair_value > 0.0) ? maximum : 0.0;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            real total_rate = 0.0;
            real total_jump1 = 0.0;
            real total_jump2 = 0.0;
            real total_jumpmax = 0.0;
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
                total_jump1 += pair_jump1[idx_neighbor];
                total_jump2 += pair_jump2[idx_neighbor];
                total_jumpmax = fmax(total_jumpmax, pair_jumpmax[idx_neighbor]);
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
                    hazard_i += total_rate*remaining;
                    jump1_i += total_jump1*remaining;
                    jump2_i += total_jump2*remaining;
                    jumpmax_i = fmax(jumpmax_i, total_jumpmax);
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
                        hazard_i += total_rate*wait;
                        jump1_i += total_jump1*wait;
                        jump2_i += total_jump2*wait;
                        jumpmax_i = fmax(jumpmax_i, total_jumpmax);
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
                            int neighbor = dev_col_neighbor[_get_col_offset(idx_old_i, idx_slot)];
                            int idx_old_j = _get_col_idx_old(neighbor);
                            int image_j = _get_col_image(neighbor);
                            real size_j = dev_size_old[idx_old_j];
                            real vrel = 0.0;
                            if constexpr (COAG_KERNEL == CUSTOM_KERNEL)
                            {
                                vrel = cached_pair_velocity(environment[idx_old_i],size_i,size_j);
                            }
                            real mass_before = numr_i*size_i*size_i*size_i;
                            bool high_speed=vrel>=V_FRAG;
                            real sample=high_speed ? _get_col_uniform(&rngstate) : 0.0;
                            int category; real log_mass;
                            real size_new=sample_erosion_outcome(size_i,size_j,
                                high_speed,sample,category,log_mass);
                            record_event_work(event_stats,category,log_mass);
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
        work[idx_old_i] = event_stats;
        dev_particle[idx_old_i].par_size = size_i;
        dev_particle[idx_old_i].par_numr = numr_i;
        dev_rngstate[idx_old_i] = rngstate;
        dev_col_time[idx_old_i] = time_i;
        dev_col_events[idx_old_i] = event_count;
        dev_col_hazard[idx_old_i] = hazard_i;
        dev_col_jump1_int[idx_old_i] = jump1_i;
        dev_col_jump2_int[idx_old_i] = jump2_i;
        dev_col_jumpmax_int[idx_old_i] = jumpmax_i;
        if (dev_col_error[idx_old_i]) atomicMax(error_flag, dev_col_error[idx_old_i]);
        if (dev_col_complete[idx_old_i] == 0)
            unfinished_ids[atomicAdd(dev_col_unfinished, 1)] = idx_old_i;
    }
}

// aggregate predicted moments and realized changes for post-bath validation
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
    int idx_start_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_size_old[idx]);
    int idx_end_raw = idx_spatial*COL_BIN_S + _get_col_sizebin(dev_particle[idx].par_size);
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

// count raw spatial and logarithmic-size controller bins
inline
int _get_col_raw_count ()
{
    int count_x = (N_X > 1) ? COL_BIN_X : 1;
    int count_z = (N_Z > 1) ? COL_BIN_Z : 1;
    return count_x*COL_BIN_Y*count_z*COL_BIN_S;
}

// merge adjacent sparse size bins independently inside every spatial bin
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

// Mean absolute log-size change and compound-Poisson variance constraints.
inline
real _choose_col_bath (const std::vector<col_rate_bin> &bin, int bin_count,
    real remaining, real limit_scale, int *binding=nullptr)
{
    real duration=std::min(remaining,COL_BATH_MAX);
    real tolerance=COL_BATH_EPS*limit_scale;
    int reason=0;
    for (int b=0;b<bin_count;++b) {
        const auto &v=bin[b];
        if (v.invalid_count) throw std::runtime_error("invalid change-based rate moments");
        if (!(v.mass>0)) continue;
        auto bound=change_limit(v.weighted_change/v.mass,v.weighted_second/v.mass,
                                tolerance,duration);
        if (bound.duration<duration) { duration=bound.duration; reason=bound.reason; }
    }
    if (binding) *binding=reason;
    return duration;
}

// compare realized bath changes with concentration bounds and adapt the next limit
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
        // Independent weight sums can put a fully touched fraction a few ulps
        // above one. Keep the raw diagnostic, but enforce its physical ceiling.
        result.activity_overshoot = result.activity_overshoot
            || std::min(touched, real(1.0)) > upper_f || events > upper_e;
        result.distribution_overshoot = result.distribution_overshoot
            || growth > std::max(COL_BATH_EPS, upper_g);
    }
    result.d_bath = mass_difference / (2.0*total_mass);
    result.distribution_overshoot = result.distribution_overshoot
        || result.d_bath > COL_BATH_EPS;

    bool issue = result.activity_overshoot || result.distribution_overshoot;
    bool at_floor = state.limit_scale
        <= 0.25*(1.0 + 8.0*std::numeric_limits<real>::epsilon());
    if (issue)
    {
        state.activity_streak++;
        state.quiet_streak = 0;
        state.floor_streak = at_floor ? state.floor_streak + 1 : 0;
    }
    else
    {
        state.activity_streak = 0;
        state.quiet_streak++;
        state.floor_streak = 0;
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
    result.persistent_overshoot = state.floor_streak >= 2;
    return result;
}

// retain the adaptive schedule and its strongest controller diagnostics
inline
void _record_col_bath (col_controller_summary &summary, const col_bath_record &record)
{
    const col_bath_result &result = record.result;
    summary.bath_count++;
    summary.continuation_launches += record.continuation_launches;
    summary.activity_overshoots += result.activity_overshoot ? 1 : 0;
    summary.distribution_overshoots += result.distribution_overshoot ? 1 : 0;
    summary.persistent_overshoots += result.persistent_overshoot ? 1 : 0;
    summary.minimum_duration = std::min(summary.minimum_duration, record.duration);
    summary.maximum_duration = std::max(summary.maximum_duration, record.duration);
    summary.minimum_limit_scale = std::min(summary.minimum_limit_scale, record.limit_after);
    summary.maximum_f = std::max(summary.maximum_f, result.max_f);
    summary.maximum_e = std::max(summary.maximum_e, result.max_e);
    summary.maximum_touched = std::max(summary.maximum_touched, result.max_touched);
    summary.maximum_events = std::max(summary.maximum_events, result.max_events);
    summary.maximum_g = std::max(summary.maximum_g, result.max_g);
    summary.maximum_g_upper = std::max(summary.maximum_g_upper, result.max_g_upper);
    summary.maximum_d_bath = std::max(summary.maximum_d_bath, result.d_bath);
    summary.baths.push_back(record);
}

// archive one output interval without mixing controller state into particle checkpoints
inline
bool save_col_controller (const std::string &file_name, const col_controller_summary &summary)
{
    std::ofstream file(file_name);
    if (!file) return false;
    real minimum_duration = std::isfinite(summary.minimum_duration)
        ? summary.minimum_duration : 0.0;
    file << std::setprecision(17);
    file << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"operator_count\": " << summary.operator_count << ",\n"
         << "  \"bath_count\": " << summary.bath_count << ",\n"
         << "  \"continuation_launches\": " << summary.continuation_launches << ",\n"
         << "  \"activity_overshoots\": " << summary.activity_overshoots << ",\n"
         << "  \"distribution_overshoots\": " << summary.distribution_overshoots << ",\n"
         << "  \"persistent_overshoots\": " << summary.persistent_overshoots << ",\n"
         << "  \"minimum_duration\": " << minimum_duration << ",\n"
         << "  \"maximum_duration\": " << summary.maximum_duration << ",\n"
         << "  \"minimum_limit_scale\": " << summary.minimum_limit_scale << ",\n"
         << "  \"maximum_f\": " << summary.maximum_f << ",\n"
         << "  \"maximum_e\": " << summary.maximum_e << ",\n"
         << "  \"maximum_touched\": " << summary.maximum_touched << ",\n"
         << "  \"maximum_events\": " << summary.maximum_events << ",\n"
         << "  \"maximum_g\": " << summary.maximum_g << ",\n"
         << "  \"maximum_g_upper\": " << summary.maximum_g_upper << ",\n"
         << "  \"maximum_d_bath\": " << summary.maximum_d_bath << ",\n"
         << "  \"baths\": [\n";
    for (std::size_t idx = 0; idx < summary.baths.size(); idx++)
    {
        const col_bath_record &record = summary.baths[idx];
        const col_bath_result &result = record.result;
        file << "    {\"operator\": " << record.operator_index
             << ", \"group\": " << record.group_index
             << ", \"bath\": " << record.bath_index
             << ", \"merged_bins\": " << record.merged_bins
             << ", \"continuation_launches\": " << record.continuation_launches
             << ", \"duration\": " << record.duration
             << ", \"limit_before\": " << record.limit_before
             << ", \"limit_after\": " << record.limit_after
             << ", \"max_f\": " << result.max_f
             << ", \"max_e\": " << result.max_e
             << ", \"max_touched\": " << result.max_touched
             << ", \"max_events\": " << result.max_events
             << ", \"max_g\": " << result.max_g
             << ", \"max_g_upper\": " << result.max_g_upper
             << ", \"d_bath\": " << result.d_bath
             << ", \"activity_overshoot\": "
             << (result.activity_overshoot ? "true" : "false")
             << ", \"distribution_overshoot\": "
             << (result.distribution_overshoot ? "true" : "false")
             << ", \"persistent_overshoot\": "
             << (result.persistent_overshoot ? "true" : "false")
             << "}" << ((idx + 1 < summary.baths.size()) ? "," : "") << "\n";
    }
    file << "  ]\n}\n";
    return static_cast<bool>(file);
}

#endif // COLLISION && !BERNOULLI

#endif // SWARM_COL_CHAIN_CUH
