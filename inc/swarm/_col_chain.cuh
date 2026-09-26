// frozen-bath collision chain: active-owner dispatch, local refresh durations, compact continuations, and cached rates
#ifndef GAMEDEV_SWARM_COL_CHAIN_CUH
#define GAMEDEV_SWARM_COL_CHAIN_CUH

#if defined(COLLISION) && !defined(BERNOULLI)

#include <algorithm> // std::max, std::min, std::min_element
#include <chrono>    // std::chrono clocks and durations
#include <cmath>     // std::abs, std::isfinite, std::log, std::sqrt
#include <cstddef>   // std::size_t
#include <cstdint>   // std::uint64_t
#include <fstream>   // std::ofstream
#include <iomanip>   // std::setprecision
#include <numeric>   // std::iota
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

#include <_col_bound.cuh>
#include <_col_cache.cuh>
#include <_col_sched.cuh>
#include <_col_types.cuh>
#include <_collision.cuh>
#include <swarm_kern.cuh>

// per-group scheduler statistics written to the COL_DIAGNOSTICS JSONL record
struct local_group_stats
{
    std::uint64_t updates = 0;
    int overshoots = 0, persistent = 0;
    real max_age = 0, max_growth = 0, max_activity = 0, max_requested_ratio = 0;
};

// persistent device scratch and host scheduler state for the local collision workflow
// scratch allocations survive all operator calls; particle IDs retain their RNG identity even when
// continuation queues are reordered by atomic append
struct local_workspace
{
    query_environment *environment = nullptr;
    cached_rate_moments *cached = nullptr;
    event_work *work = nullptr, *work_sum = nullptr;
    int *ids = nullptr, *queue_a = nullptr, *queue_b = nullptr, *error = nullptr;
    real *dt = nullptr, *change_rate = nullptr, *second_rate = nullptr;
    unsigned int *graph = nullptr;
    std::vector<std::vector<int>> owners;
    std::vector<unsigned int> edges;
    std::vector<col_bath_state> state;
    #ifdef COL_DIAGNOSTICS
    std::ofstream log;
    std::uint64_t operator_id = 0;
    #endif // COL_DIAGNOSTICS
    local_workspace (const std::string &path): owners(LOCAL_GROUPS), edges(LOCAL_GROUPS*LOCAL_WORDS),
        state(LOCAL_GROUPS)
    {
        #ifdef COL_QUERY_ENV_CACHE
        GPU_CHECK(gpuMalloc((void**)&environment, sizeof(query_environment)*N_P));
        #endif // COL_QUERY_ENV_CACHE
        GPU_CHECK(gpuMalloc((void**)&cached, sizeof(cached_rate_moments)*N_P));
        #ifdef COL_DIAGNOSTICS
        GPU_CHECK(gpuMalloc((void**)&work, sizeof(event_work)*N_P));
        GPU_CHECK(gpuMalloc((void**)&work_sum, sizeof(event_work)));
        #endif // COL_DIAGNOSTICS
        GPU_CHECK(gpuMalloc((void**)&ids, sizeof(int)*N_P));
        GPU_CHECK(gpuMalloc((void**)&queue_a, sizeof(int)*N_P));
        GPU_CHECK(gpuMalloc((void**)&queue_b, sizeof(int)*N_P));
        GPU_CHECK(gpuMalloc((void**)&error, sizeof(int)));
        GPU_CHECK(gpuMalloc((void**)&dt, sizeof(real)*LOCAL_GROUPS));
        GPU_CHECK(gpuMalloc((void**)&change_rate, sizeof(real)*N_P));
        GPU_CHECK(gpuMalloc((void**)&second_rate, sizeof(real)*N_P));
        GPU_CHECK(gpuMalloc((void**)&graph, sizeof(unsigned int)*edges.size()));
        #ifdef COL_DIAGNOSTICS
        auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        log.open(path+"collision_local_"+std::to_string(stamp)+".jsonl");
        if (!log) throw std::runtime_error("cannot open local collision diagnostics");
        log << std::setprecision(17);
        #endif // COL_DIAGNOSTICS
    }
    ~local_workspace ()
    {
        if (environment) gpuFree(environment);
        gpuFree(cached);
        #ifdef COL_DIAGNOSTICS
        gpuFree(work);
        gpuFree(work_sum);
        #endif // COL_DIAGNOSTICS
        gpuFree(ids);
        gpuFree(queue_a);
        gpuFree(queue_b);
        gpuFree(error);
        gpuFree(dt);
        gpuFree(graph);
        gpuFree(change_rate);
        gpuFree(second_rate);
    }
};

// =====================================================================================================================
// host function: evolve_local_collisions
// purpose: advance all owners through one fixed-position collision operator with local power-of-two bath durations
//
// per call:
//   1 cache gas environments and, once per geometry epoch, the owner-group lists and dependency graph
//   2 publish every reservoir, compute bath-start rates and merged size bins, and request per-group durations
//   3 repeatedly advance the earliest due groups: republish and rerate them, adapt their levels,
//     screen no-event owners, run chain continuations until none remain, and audit the completed baths
// =====================================================================================================================
inline void evolve_local_collisions (
    local_workspace & local,
    bool & local_geometry_valid,
    real duration,
    real total_dust_mass,
    real & clock_dyn,
    real & dt_col,
    int & count_col,
    int col_raw_count,
    swarm *dev_particle,
    curs *dev_rngstate,
    int *dev_col_spatial,
    int *dev_col_neighbor,
    int *dev_col_events,
    int *dev_col_count,
    int *dev_col_binmap,
    int *dev_col_error,
    int *dev_col_unfinished,
    unsigned char*dev_col_active,
    unsigned char*dev_col_complete,
    real *dev_size_old,
    real *dev_numr_old,
    real *dev_col_time,
    real *dev_col_rate,
    real *dev_col_hazard,
    real *dev_col_jump1_int,
    real *dev_col_jump2_int,
    real *dev_col_jumpmax_int,
    real *dev_col_measure,
    col_rate_bin *dev_col_ratebin,
    col_audit_accum *dev_col_audit
#ifdef IMPORTGAS
    , const real *dev_gas_dens
#endif // IMPORTGAS
#ifdef COL_DIAGNOSTICS
    , real clock_sim, col_controller_summary &col_summary
#endif // COL_DIAGNOSTICS
)
{
// called by evolve_collisions after production geometry/cache construction
// all launches below use the default stream; no publication occurs while an event chain or its audit can
// still be reading the previous reservoir
#ifdef COL_DIAGNOSTICS
using local_clock = std::chrono::steady_clock;
const auto local_begin = local_clock::now();
#endif // COL_DIAGNOSTICS
#ifdef COL_QUERY_ENV_CACHE
col_env_cache <<< NB_P, TPB >>> (local.environment, dev_particle);
GPU_KERNEL_CHECK("col_env_cache");
#endif // COL_QUERY_ENV_CACHE
#ifdef COL_DIAGNOSTICS
GPU_CHECK(gpuMemset(local.work, 0, sizeof(event_work)*N_P));
#endif // COL_DIAGNOSTICS
// rebuild owner lists and the group dependency graph only after positions change
if (!local_geometry_valid)
{
    std::vector<int> spatial(N_P);
    GPU_CHECK(gpuMemcpy(spatial.data(), dev_col_spatial, sizeof(int)*N_P, gpuMemcpyDeviceToHost));
    for (auto &v : local.owners) v.clear();
    for (int i = 0; i < N_P; ++i) local.owners[spatial[i]].push_back(i);
    GPU_CHECK(gpuMemset(local.graph, 0, sizeof(unsigned int)*local.edges.size()));
    col_dep_graph <<<
        #ifdef GAMEDEV_ROCM
        (N_P*(TPB % 32 == 0 ? 32 : 1) + TPB - 1) / TPB, TPB
        #else  // !GAMEDEV_ROCM
        NB_P, TPB
        #endif // GAMEDEV_ROCM
    >>> (local.graph, dev_col_spatial, dev_col_neighbor, dev_col_active);
    GPU_KERNEL_CHECK("col_dep_graph");
    GPU_CHECK(gpuMemcpy(local.edges.data(), local.graph,
        sizeof(unsigned int)*local.edges.size(), gpuMemcpyDeviceToHost));
    local_geometry_valid = true;
}

std::vector<int> ids(N_P), counts(col_raw_count), binmap(col_raw_count);
std::iota(ids.begin(), ids.end(), 0);
GPU_CHECK(gpuMemcpy(local.ids, ids.data(), sizeof(int)*N_P, gpuMemcpyHostToDevice));
std::vector<col_rate_bin> ratebin(col_raw_count);
std::vector<col_audit_accum> audit(col_raw_count);
const real lambda0 = N_P / static_cast<real>(N_K) / total_dust_mass;

// publish the listed owners' current sizes and counts as the frozen reservoir and reset their bath clocks
auto initialize = [&](int count)
{
    #ifdef COL_PARTNER_REFRESH
    // campaign hook: only at refresh boundaries, before snapshots and cached rates
    COL_PARTNER_REFRESH(dev_col_neighbor, local.queue_a, count);
    #endif // COL_PARTNER_REFRESH
    int blocks = (count + TPB - 1) / TPB;
    col_bath_init <<< blocks, TPB >>> (local.ids, count, dev_size_old, dev_numr_old,
        dev_col_time, dev_col_events, dev_col_complete, dev_particle);
    GPU_KERNEL_CHECK("col_bath_init");
    col_comp_zero <<< blocks, TPB >>> (local.ids, count, dev_col_hazard, dev_col_jump1_int,
        dev_col_jump2_int, dev_col_jumpmax_int);
    GPU_KERNEL_CHECK("col_comp_zero");
};
// recompute bath-start rates, merge sparse size bins, and copy mass-weighted rate moments to the host
auto rates_and_bins = [&](int count)
{
    // keep each refreshed group's bounds unchanged until its collision interval and audit finish
    col_size_zero <<< (moving_groups + TPB - 1) / TPB, TPB >>> ();
    GPU_KERNEL_CHECK("col_size_zero");
    col_size_scan <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_particle, dev_col_spatial,
        dev_col_active);
    GPU_KERNEL_CHECK("col_size_scan");
    col_size_bnds <<< (moving_groups + TPB - 1) / TPB, TPB >>> ();
    GPU_KERNEL_CHECK("col_size_bnds");
    #ifdef COL_PERF_VAL
    auto rate_start = col_perf_start();
    #endif // COL_PERF_VAL
    col_bath_rate <<< count, COL_BATH_TPB >>> (local.ids, count, dev_col_rate, local.change_rate, local.second_rate,
        dev_particle, dev_col_neighbor, dev_col_measure, dev_col_active,
        dev_size_old, dev_numr_old,
        #ifdef IMPORTGAS
        dev_gas_dens,
        #endif // IMPORTGAS
        lambda0, local.environment, local.cached);
    GPU_KERNEL_CHECK("col_bath_rate");
    GPU_CHECK(gpuMemset(dev_col_count, 0, sizeof(int)*col_raw_count));
    col_count_bin <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_count,
        dev_particle, dev_col_spatial, dev_col_active);
    GPU_KERNEL_CHECK("col_count_bin");
    GPU_CHECK(gpuMemcpy(counts.data(), dev_col_count, sizeof(int)*col_raw_count, gpuMemcpyDeviceToHost));
    int merged = _build_col_binmap(counts, binmap);
    GPU_CHECK(gpuMemcpy(dev_col_binmap, binmap.data(), sizeof(int)*col_raw_count, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemset(dev_col_ratebin, 0, sizeof(col_rate_bin)*col_raw_count));
    col_rate_bins <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_ratebin,
        dev_particle, dev_col_rate, local.change_rate, local.second_rate, dev_col_spatial, dev_col_binmap,
        dev_col_active);
    GPU_KERNEL_CHECK("col_rate_bins");
    GPU_CHECK(gpuMemcpy(ratebin.data(), dev_col_ratebin, sizeof(col_rate_bin)*merged, gpuMemcpyDeviceToHost));
    #ifdef COL_PERF_VAL
    col_perf_rate_ms += col_perf_stop(rate_start);
    #endif // COL_PERF_VAL
    return merged;
};
// merged bins of group c occupy the contiguous range [binmap[c*COL_BIN_S], bin_end(c))
auto bin_end = [&](int c, int merged)
{
    return c + 1 < LOCAL_GROUPS ? binmap[(c + 1)*COL_BIN_S] : merged;
};
auto requested_step = [&](int c, int merged, int *binding = nullptr)
{
    int first = binmap[c*COL_BIN_S];
    int last = bin_end(c, merged);
    std::vector<col_rate_bin> slice(ratebin.begin() + first, ratebin.begin() + last);
    return _choose_col_bath(slice, int(slice.size()), duration, local.state[c].limit_scale, binding);
};

// start every group from the same published reservoir at tick zero
initialize(N_P);
int merged = rates_and_bins(N_P);
std::vector<double> requested(LOCAL_GROUPS);
#ifdef COL_DIAGNOSTICS
std::vector<int> binding(LOCAL_GROUPS);
for (int c = 0; c < LOCAL_GROUPS; ++c) requested[c] = requested_step(c, merged, &binding[c]);
#else  // !COL_DIAGNOSTICS
for (int c = 0; c < LOCAL_GROUPS; ++c) requested[c] = requested_step(c, merged);
#endif // COL_DIAGNOSTICS
local_schedule schedule(duration, requested, local.edges);
std::vector<real> steps(LOCAL_GROUPS);
#ifdef COL_DIAGNOSTICS
std::vector<real> published(LOCAL_GROUPS, 0);
#endif // COL_DIAGNOSTICS
for (int c = 0; c < LOCAL_GROUPS; ++c) steps[c] = schedule.seconds(schedule.step[c]);
GPU_CHECK(gpuMemcpy(local.dt, steps.data(), sizeof(real)*LOCAL_GROUPS, gpuMemcpyHostToDevice));
std::vector<bool> passed(LOCAL_GROUPS, true);
std::vector<double> current_request = requested;
#ifdef COL_DIAGNOSTICS
std::vector<local_group_stats> stats(LOCAL_GROUPS);
std::vector<int> initial_level = schedule.level;
std::vector<std::uint64_t> coarsened(LOCAL_GROUPS, 0), refined(LOCAL_GROUPS, 0),
    constrained(LOCAL_GROUPS, 0);
std::vector<double> min_request = requested, max_request = requested;
std::uint64_t owner_updates = 0, chain_blocks = 0, waves = 0, launches = 0;
double chain_seconds = 0.0;
double audit_seconds = 0.0;
col_summary.operator_count++;
int operator_index = col_summary.operator_count;
int batch_index = 0;

#endif // COL_DIAGNOSTICS

while (schedule.time() < schedule.end)
{
    auto tick = schedule.time();
    auto groups = schedule.due(tick);
    real time = schedule.seconds(tick);
    ids.clear();
    for (int c : groups)
    {
        #ifdef COL_DIAGNOSTICS
        published[c] = time;
        #endif // COL_DIAGNOSTICS
        ids.insert(ids.end(), local.owners[c].begin(), local.owners[c].end());
    }
    int count = static_cast<int>(ids.size());
    if (count == 0)
    {
        schedule.advance(groups);
        continue;
    }
    // at tick zero the complete population was already initialized and rated
    if (tick != 0)
    {
        GPU_CHECK(gpuMemcpy(local.ids, ids.data(), sizeof(int)*count, gpuMemcpyHostToDevice));
        initialize(count);
        merged = rates_and_bins(count);
    }
    if (tick != 0)
    {
        for (int c : groups)
        {
            current_request[c] = requested_step(c, merged);
            #ifdef COL_DIAGNOSTICS
            min_request[c] = std::min(min_request[c], current_request[c]);
            max_request[c] = std::max(max_request[c], current_request[c]);
            #endif // COL_DIAGNOSTICS
        }
        #ifdef COL_DIAGNOSTICS
        auto previous = schedule.level;
        #endif // COL_DIAGNOSTICS
        schedule.adapt(tick, groups, current_request, passed);
        for (int c : groups)
        {
            #ifdef COL_DIAGNOSTICS
            coarsened[c] += schedule.level[c] < previous[c];
            refined[c] += schedule.level[c] > previous[c];
            #endif // COL_DIAGNOSTICS
            steps[c] = schedule.seconds(schedule.step[c]);
            #ifdef COL_DIAGNOSTICS
            constrained[c] += steps[c] > current_request[c];
            #endif // COL_DIAGNOSTICS
        }
        GPU_CHECK(gpuMemcpy(local.dt, steps.data(), sizeof(real)*LOCAL_GROUPS, gpuMemcpyHostToDevice));
    }
    #ifdef COL_DIAGNOSTICS
    for (int c : groups)
    {
        if (local.owners[c].empty()) continue;
        for (int d = 0; d < LOCAL_GROUPS; ++d)
        {
            if (local.edges[c*LOCAL_WORDS + d / 32] & (1u << (d % 32)))
            {
                stats[c].max_age = std::max(stats[c].max_age, time - published[d]);
            }
        }
        stats[c].max_requested_ratio = std::max(stats[c].max_requested_ratio,
            steps[c] / requested_step(c, merged));
    }

    #endif // COL_DIAGNOSTICS
    #ifdef COL_PERF_VAL
    auto event_start = col_perf_start();
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    const auto chain_begin = local_clock::now();
    #endif // COL_DIAGNOSTICS
    // screen owners with no event in the interval, then relaunch the chain on the shrinking unfinished queue
    GPU_CHECK(gpuMemset(local.error, 0, sizeof(int)));
    GPU_CHECK(gpuMemset(dev_col_unfinished, 0, sizeof(int)));
    col_skip_scan <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_rngstate,
        dev_col_active, dev_col_measure, dev_col_spatial, local.dt, local.cached,
        dev_col_time, dev_col_events, dev_col_complete, dev_col_hazard,
        dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, local.queue_a, dev_col_unfinished);
    GPU_KERNEL_CHECK("col_skip_scan");
    int unfinished = 0;
    int continuations = 0;
    GPU_CHECK(gpuMemcpy(&unfinished, dev_col_unfinished, sizeof(int), gpuMemcpyDeviceToHost));
    const int *input = local.queue_a;
    int *output = local.queue_b;
    while (unfinished > 0)
    {
        if (++continuations > 1000000)
        {
            throw std::runtime_error("local collision continuation limit exceeded");
        }
        GPU_CHECK(gpuMemset(dev_col_unfinished, 0, sizeof(int)));
        #ifdef COL_DIAGNOSTICS
        chain_blocks += unfinished;
        #endif // COL_DIAGNOSTICS
        col_chain_run <<< unfinished, COL_BATH_TPB >>> (input, unfinished,
            dev_particle, dev_rngstate, dev_col_error, dev_col_unfinished,
            dev_col_time, dev_col_events, dev_col_complete, dev_col_hazard,
            dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, dev_col_neighbor,
            dev_col_measure, dev_col_active, dev_size_old, dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif // IMPORTGAS
            lambda0, local.dt, dev_col_spatial, output, local.error, local.work, local.environment, local.cached);
        GPU_KERNEL_CHECK("col_chain_run");
        GPU_CHECK(gpuMemcpy(&unfinished, dev_col_unfinished, sizeof(int), gpuMemcpyDeviceToHost));
        input = output;
        output = (output == local.queue_a) ? local.queue_b : local.queue_a;
    }
    int error = 0;
    GPU_CHECK(gpuMemcpy(&error, local.error, sizeof(int), gpuMemcpyDeviceToHost));
    if (error) throw std::runtime_error("local collision chain error "+std::to_string(error));
    #ifdef COL_DIAGNOSTICS
    chain_seconds += std::chrono::duration<double>(local_clock::now() - chain_begin).count();
    #endif // COL_DIAGNOSTICS

    #ifdef COL_PERF_VAL
    col_perf_event_ms += col_perf_stop(event_start);
    auto audit_start = col_perf_start();
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    const auto audit_begin = local_clock::now();
    #endif // COL_DIAGNOSTICS
    // compare realized activity and size change with the predicted envelopes and adapt each group's safety factor
    GPU_CHECK(gpuMemset(dev_col_audit, 0, sizeof(col_audit_accum)*col_raw_count));
    col_audit_bin <<< (count + TPB - 1) / TPB, TPB >>> (local.ids, count, dev_col_audit,
        dev_particle, dev_size_old, dev_numr_old, dev_col_rate, dev_col_hazard,
        dev_col_jump1_int, dev_col_jump2_int, dev_col_jumpmax_int, dev_col_events,
        dev_col_spatial, dev_col_binmap, dev_col_active, local.dt);
    GPU_KERNEL_CHECK("col_audit_bin");
    GPU_CHECK(gpuMemcpy(audit.data(), dev_col_audit,
        sizeof(col_audit_accum)*merged, gpuMemcpyDeviceToHost));
    for (int c : groups)
    {
        if (local.owners[c].empty()) continue;
        int first = binmap[c*COL_BIN_S];
        int last = bin_end(c, merged);
        real mass = 0.0;
        for (int b = first; b < last; ++b)
        {
            if (audit[b].invalid_count) throw std::runtime_error("invalid local audit state");
            mass += audit[b].mass;
        }
        if (!(mass > 0)) continue;
        #ifdef COL_DIAGNOSTICS
        real before = local.state[c].limit_scale;
        #endif // COL_DIAGNOSTICS
        std::vector<col_audit_accum> slice(audit.begin() + first, audit.begin() + last);
        auto result = _finish_col_bath(slice, int(slice.size()), local.state[c]);
        passed[c] = !(result.activity_overshoot || result.distribution_overshoot || result.persistent_overshoot);
        #ifdef COL_DIAGNOSTICS
        auto &s = stats[c];
        ++s.updates;
        s.overshoots += result.activity_overshoot || result.distribution_overshoot;
        s.persistent += result.persistent_overshoot;
        s.max_growth = std::max(s.max_growth, result.max_g);
        s.max_activity = std::max(s.max_activity, result.max_f);
        // audit feedback controls the next due update; pending neighbors stay fixed
        col_bath_record record;
        record.group_index = c;
        record.operator_index = operator_index;
        record.bath_index=++batch_index;
        record.merged_bins = last - first;
        record.duration = steps[c];
        record.limit_before = before;
        record.limit_after = local.state[c].limit_scale;
        record.result = result;
        _record_col_bath(col_summary, record);
        #endif // COL_DIAGNOSTICS
    }
    // count physical launches once per wave, not once per group in that wave
    #ifdef COL_DIAGNOSTICS
    col_summary.continuation_launches += continuations;
    audit_seconds += std::chrono::duration<double>(local_clock::now() - audit_begin).count();
    #endif // COL_DIAGNOSTICS
    #ifdef COL_PERF_VAL
    col_perf_audit_ms += col_perf_stop(audit_start);
    ++col_perf_batches;
    col_perf_launches += continuations;
    #endif // COL_PERF_VAL
    #ifdef COL_DIAGNOSTICS
    ++col_summary.wave_count;
    owner_updates += count;
    launches += continuations;
    ++waves;
    #endif // COL_DIAGNOSTICS
    ++count_col;
    dt_col = duration;
    for (int c : groups) dt_col = std::min(dt_col, steps[c]);
    schedule.advance(groups);
    clock_dyn = schedule.seconds(schedule.time());
}
// every pending endpoint has reached the horizon; transport may read all particle states
clock_dyn = duration;
#ifdef COL_DIAGNOSTICS
local.log << "{\"schema\":1,\"method\":\"local_cached_collision\",\"operator\":" << ++local.operator_id
    << ",\"clock_sim\":" << clock_sim << ",\"duration\":" << duration
    << ",\"waves\":" << waves << ",\"owner_updates\":" << owner_updates
    << ",\"chain_blocks\":" << chain_blocks << ",\"chain_launches\":" << launches
    << ",\"chain_seconds\":" << chain_seconds << ",\"audit_seconds\":" << audit_seconds
    << ",\"scheduler_wall_seconds\":"
    << std::chrono::duration<double>(local_clock::now() - local_begin).count()
    << ",\"finest_ticks\":" << schedule.end << ",\"groups\":[";
for (int c = 0; c < LOCAL_GROUPS; ++c)
{
    if (c) local.log << ',';
    const auto &s = stats[c];
    local.log << "{\"id\":" << c << ",\"owners\":" << local.owners[c].size()
        << ",\"level\":" << schedule.level[c] << ",\"dt\":" << steps[c]
        << ",\"initial_level\":" << initial_level[c]
        << ",\"coarsened_updates\":" << coarsened[c]
        << ",\"refined_updates\":" << refined[c]
        << ",\"neighbor_constrained_updates\":" << constrained[c]
        << ",\"minimum_requested_dt\":" << min_request[c]
        << ",\"maximum_requested_dt\":" << max_request[c]
        << ",\"final_requested_dt\":" << current_request[c]
        << ",\"initial_requested_dt\":" << requested[c]
        << ",\"initial_binding_constraint\":" << binding[c]
        << ",\"updates\":" << s.updates << ",\"overshoots\":" << s.overshoots
        << ",\"persistent_overshoots\":" << s.persistent
        << ",\"max_snapshot_age_at_start\":" << s.max_age
        << ",\"max_requested_dt_ratio\":" << s.max_requested_ratio
        << ",\"max_growth\":" << s.max_growth
        << ",\"max_predicted_activity\":" << s.max_activity << '}';
}
local.log << "],\"event_counts\":[";
col_event_sum <<< EVENT_CATEGORIES, TPB >>> (local.work, local.work_sum);
GPU_KERNEL_CHECK("col_event_sum");
event_work totals;
GPU_CHECK(gpuMemcpy(&totals, local.work_sum, sizeof(event_work), gpuMemcpyDeviceToHost));
for (int k = 0; k < EVENT_CATEGORIES; ++k)
{
    if (k) local.log<<',';
    local.log << totals.count[k];
}
local.log << "],\"event_log_mass_sums\":[";
for (int k = 0; k < EVENT_CATEGORIES; ++k)
{
    if (k) local.log<<',';
    local.log << totals.log_mass[k];
}
local.log << "]}\n";
local.log.flush();
if (!local.log) throw std::runtime_error("cannot write local collision diagnostics");
#endif // COL_DIAGNOSTICS
}

#endif // COLLISION && !BERNOULLI

#endif // GAMEDEV_SWARM_COL_CHAIN_CUH
