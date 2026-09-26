#include <gpu.cuh>

#ifdef COLLISION
#include <_col_event.cuh>
#include <_col_rates.cuh>
#include <swarm_kern.cuh>

// =====================================================================================================================
// kernel: col_chain_run
// evolve every owner against one frozen reservoir with bounded continuation
//
// parallelization: one block per queued owner; threads evaluate neighbor pair rates and thread 0 advances the chain
//
// per loop iteration:
//   1 recompute all owner-dependent pair rates and jump moments from the current owner size
//   2 reduce the total rate, jump moments, and last positive slot
//   3 draw one waiting time; stop at the bath end or apply one sampled event and update owner size and number
// the loop stops after COL_EVENT_CAP accepted events; unfinished owners are appended to the next continuation queue
// =====================================================================================================================

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
    #ifdef COL_DIAGNOSTICS
    __shared__ event_work event_stats;
    #endif // COL_DIAGNOSTICS
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
        #ifdef COL_DIAGNOSTICS
        event_stats = work[idx_old_i];
        #endif // COL_DIAGNOSTICS
        keep_running = dev_col_complete[idx_old_i] == 0;
        if (dev_col_active[idx_old_i] == 0 || !(dev_col_measure[idx_old_i] > 0.0))
        {
            time_i = bath_end;
            dev_col_complete[idx_old_i] = 1;
            keep_running = false;
        }
    }
    __syncthreads();

    // use the bath-start cache only before the first event of this bath, because it matches the published reservoir
    // after an event or in a continuation the owner has changed and every rate must be recomputed
    if (threadIdx.x == 0 && keep_running && time_i == 0.0 && event_count == 0)
    {
        const auto c = cached[idx_old_i];
        if (isfinite(c.rate) && c.rate >= 0.0)
        {
            if (c.rate == 0.0)
            {
                time_i = bath_end;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
            }
            else
            {
                curs before = rngstate;
                real wait = -log(_get_col_uniform(&rngstate)) / c.rate;
                real remaining = bath_end - time_i;
                if (isfinite(wait) && wait > 0.0 && wait >= remaining)
                {
                    hazard_i += c.rate*remaining;
                    jump1_i += c.first*remaining;
                    jump2_i += c.second*remaining;
                    jumpmax_i = fmax(jumpmax_i, c.maximum);
                    time_i = bath_end;
                    dev_col_complete[idx_old_i] = 1;
                    keep_running = false;
                }
                else
                {
                    // restore the draw so the full path handles events and invalid waits with the same deviate
                    rngstate = before;
                }
            }
        }
    }
    __syncthreads();

    while (true)
    {
        if (threadIdx.x == 0 && keep_running && accepted >= COL_EVENT_CAP)
        {
            keep_running = false;
        }
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
                real pair_value = _get_col_chain_rate <static_cast<kernel_type>(COAG_KERNEL)> (
                    dev_particle, size_i, dev_size_old, dev_numr_old,
                    #ifdef IMPORTGAS
                    dev_gas_dens,
                    #endif // IMPORTGAS
                    idx_old_i, idx_old_j, image_j, lambda_0, vrel, environment
                ) / dev_col_measure[idx_old_i];
                real mean = 0.0;
                real second = 0.0;
                real maximum = 0.0;
                pair_value *= _erosion_outcome_moments(size_i, size_j, vrel >= V_FRAG, mean, second, maximum);
                pair_rate[idx_neighbor] = pair_value;
                pair_jump1[idx_neighbor] = pair_value*mean;
                pair_jump2[idx_neighbor] = pair_value*second;
                pair_jumpmax[idx_neighbor] = (pair_value > 0.0) ? maximum : 0.0;
            }
        }
        constexpr bool wave_reduction_shape = N_K == 256 && COL_BATH_TPB == 128;
        // keep the generic allocation for other shapes; the shape choice is compile-time
        #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
        constexpr int total_slots = wave_reduction_shape ? 128 : N_K;
        #else  // !(GAMEDEV_ROCM && __gfx942__)
        constexpr int total_slots = N_K;
        #endif // GAMEDEV_ROCM && __gfx942__
        __shared__ real total_work[total_slots];
        __shared__ int last_work[total_slots];
        #if defined(GAMEDEV_ROCM) && defined(__gfx942__)
        // MI300A fast path for N_K=256 and 128 threads, as in col_bath_rate
        if constexpr(wave_reduction_shape)
        {
          const int t = threadIdx.x;
          __syncthreads();
          real left = pair_rate[t];
          real right = pair_rate[t + 128];
          total_work[t] = left + right;
          last_work[t] = (!isfinite(left) || left < 0.0 || !isfinite(right) || right < 0.0)
              ? -2 : (right > 0.0 ? t + 128 : (left > 0.0 ? t : -1));
          pair_jump1[t] += pair_jump1[t + 128];
          pair_jump2[t] += pair_jump2[t + 128];
          pair_jumpmax[t] = fmax(pair_jumpmax[t], pair_jumpmax[t + 128]);
          __syncthreads();
          if (t < 64)
          {
            real r = total_work[t] + total_work[t + 64];
            real a = pair_jump1[t] + pair_jump1[t + 64];
            real b = pair_jump2[t] + pair_jump2[t + 64];
            real m = fmax(pair_jumpmax[t], pair_jumpmax[t + 64]);
            int l = last_work[t];
            int other = last_work[t + 64];
            l = (l == -2 || other == -2) ? -2 : max(l, other);
            for (int delta = 32; delta; delta /= 2)
            {
              r += __shfl_down(r, delta, 64);
              a += __shfl_down(a, delta, 64);
              b += __shfl_down(b, delta, 64);
              m = fmax(m, __shfl_down(m, delta, 64));
              other = __shfl_down(l, delta, 64);
              l = (l == -2 || other == -2) ? -2 : max(l, other);
            }
            if (t == 0)
            {
                total_work[0] = r;
                pair_jump1[0] = a;
                pair_jump2[0] = b;
                pair_jumpmax[0] = m;
                last_work[0] = l;
            }
          }
        }
        else
        #endif // GAMEDEV_ROCM && __gfx942__
        {
        // reduce into separate arrays so pair_rate stays intact for serial partner sampling
        for (int j = threadIdx.x; j < N_K; j += blockDim.x)
        {
            total_work[j] = pair_rate[j];
            // -2 propagates invalid input; -1 denotes no positive contribution
            last_work[j] = (!isfinite(pair_rate[j]) || pair_rate[j] < 0.0)
                ? -2 : (pair_rate[j] > 0.0 ? j : -1);
        }
        __syncthreads();
        for (int count = N_K; count > 1; count = (count + 1) / 2)
        {
            int half = (count + 1) / 2;
            for (int j = threadIdx.x; j < count / 2; j += blockDim.x)
            {
                total_work[j] += total_work[j + half];
                pair_jump1[j] += pair_jump1[j + half];
                pair_jump2[j] += pair_jump2[j + half];
                pair_jumpmax[j] = fmax(pair_jumpmax[j], pair_jumpmax[j + half]);
                int left = last_work[j];
                int right = last_work[j + half];
                last_work[j] = (left == -2 || right == -2) ? -2
                    : (left > right ? left : right);
            }
            __syncthreads();
        }

        }

        if (threadIdx.x == 0)
        {
            real total_rate = total_work[0];
            real total_jump1 = pair_jump1[0];
            real total_jump2 = pair_jump2[0];
            real total_jumpmax = pair_jumpmax[0];
            int last_positive = last_work[0];
            if (last_positive == -2)
            {
                dev_col_error[idx_old_i] = 1;
                dev_col_complete[idx_old_i] = 1;
                keep_running = false;
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
                        // select the partner by walking cumulative pair rates, falling back to the last positive slot
                        // on roundoff
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
                                #ifdef COL_QUERY_ENV_CACHE
                                vrel = _cached_pair_velocity(environment[idx_old_i], size_i, size_j);
                                #else  // !COL_QUERY_ENV_CACHE
                                vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j, image_j
                                    #ifdef IMPORTGAS
                                    , dev_gas_dens
                                    #endif // IMPORTGAS
                                );
                                #endif // COL_QUERY_ENV_CACHE
                            }
                            // apply the outcome while conserving represented mass through the grain count
                            real mass_before = numr_i*size_i*size_i*size_i;
                            bool high_speed = vrel >= V_FRAG;
                            real sample = high_speed ? _get_col_uniform(&rngstate) : 0.0;
                            int category;
                            real log_mass;
                            real size_new = _sample_erosion_outcome(size_i, size_j,
                                high_speed, sample, category, log_mass);
                            #ifdef COL_DIAGNOSTICS
                            _record_event_work(event_stats, category, log_mass);
                            #endif // COL_DIAGNOSTICS
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
        #ifdef COL_DIAGNOSTICS
        work[idx_old_i] = event_stats;
        #endif // COL_DIAGNOSTICS
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
        {
            unfinished_ids[atomicAdd(dev_col_unfinished, 1)] = idx_old_i;
        }
    }
}

#endif // COLLISION
