#ifndef GAMEDEV_SWARM_COL_BOUND_CUH
#define GAMEDEV_SWARM_COL_BOUND_CUH

// host bath controller: size-bin merging, bath durations, post-bath audit, and archives

#ifdef COLLISION
#include <algorithm> // std::max, std::min
#include <cmath>     // std::abs, std::isfinite, std::log, std::sqrt
#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream
#include <iomanip>   // std::setprecision
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

#include <_col_types.cuh>

// A is the mass-weighted mean absolute log-size jump rate
// B is the mass-weighted second log-size jump moment rate, not the variance of the mean
// frozen rates give a mean accumulated absolute change h*A and a compound-Poisson fluctuation scale sqrt(h*B);
// bound each by epsilon
inline change_bound change_limit (double A, double B, double epsilon, double horizon)
{
    if (!std::isfinite(A) || A < 0 || !std::isfinite(B) || B < 0
        || !std::isfinite(epsilon) || !(epsilon > 0)
        || !std::isfinite(horizon) || !(horizon > 0))
        throw std::runtime_error("invalid change-based refresh inputs");
    change_bound result{horizon, 0};
    if (A > 0 && epsilon / A < result.duration) result = {epsilon / A, 1};
    if (B > 0 && epsilon*epsilon / B < result.duration) result = {epsilon*epsilon / B, 2};
    if (!(result.duration > 0)) throw std::runtime_error("change-based timestep underflow");
    return result;
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
        {
            if (raw_count[idx_spatial*COL_BIN_S + idx_size] > 0) last_occupied = idx_size;
        }
        if (last_occupied < 0)
        {
            int idx_merged = merged_count++;
            for (int idx_size = 0; idx_size < COL_BIN_S; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
            continue;
        }

        int idx_begin = 0;
        int idx_previous = -1;
        while (idx_begin <= last_occupied)
        {
            int idx_end = idx_begin;
            int count = 0;
            while (idx_end <= last_occupied && count < COL_BIN_MIN)
            {
                count += raw_count[idx_spatial*COL_BIN_S + idx_end++];
            }
            bool merge_tail = idx_end > last_occupied && count < COL_BIN_MIN
                && idx_previous >= 0;
            int idx_merged = merge_tail ? idx_previous : merged_count++;
            for (int idx_size = idx_begin; idx_size < idx_end; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
            idx_previous = idx_merged;
            idx_begin = idx_end;
        }
        if (last_occupied + 1 < COL_BIN_S)
        {
            int idx_merged = merged_count++;
            for (int idx_size = last_occupied + 1; idx_size < COL_BIN_S; idx_size++)
            {
                raw_to_merged[idx_spatial*COL_BIN_S + idx_size] = idx_merged;
            }
        }
    }
    return merged_count;
}

// choose a bath duration from the mean absolute log-size change and compound-Poisson variance constraints
inline
real _choose_col_bath (const std::vector<col_rate_bin> &bin, int bin_count,
    real remaining, real limit_scale, int *binding = nullptr)
{
    real duration = std::min(remaining, COL_BATH_MAX);
    real tolerance = COL_BATH_EPS*limit_scale;
    int reason = 0;
    for (int b = 0; b < bin_count; ++b)
    {
        const auto &v = bin[b];
        if (v.invalid_count) throw std::runtime_error("invalid change-based rate moments");
        if (!(v.mass > 0)) continue;
        auto bound = change_limit(v.weighted_change / v.mass, v.weighted_second / v.mass,
                                tolerance, duration);
        if (bound.duration < duration)
        {
            duration = bound.duration;
            reason = bound.reason;
        }
    }
    if (binding) *binding = reason;
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
        {
            throw std::runtime_error("collision bath controller returned an invalid audit state");
        }
        if (value.mass > 0.0) active_count++;
        total_mass += value.mass;
    }
    if (!(total_mass > 0.0))
    {
        throw std::runtime_error("collision bath controller found no active represented mass");
    }
    real confidence_log = std::log(2.0*static_cast<real>(active_count) / COL_BATH_ALPHA);
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
        // independent weight sums can put a fully touched fraction a few ulps above one
        // keep the raw diagnostic, but enforce its physical ceiling in the overshoot test
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
         << "  \"schema\": 2,\n"
         << "  \"operator_count\": " << summary.operator_count << ",\n"
         << "  \"bath_count\": " << summary.bath_count << ",\n"
         << "  \"wave_count\": " << summary.wave_count << ",\n"
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

#endif // COLLISION

#endif // GAMEDEV_SWARM_COL_BOUND_CUH
