#ifndef GAMEDEV_SWARM_COL_TYPES_CUH
#define GAMEDEV_SWARM_COL_TYPES_CUH

// plain records shared by the collision-chain kernels and their host controller

#if defined(COLLISION) && !defined(BERNOULLI)

#include <limits> // std::numeric_limits
#include <vector> // std::vector

#include <const_defs.cuh>

// owner-position gas coefficients reused by every pair evaluated for that owner
// Z height, omega Keplerian frequency, vn pressure-drift speed, radial and strat Stokes scalings,
// cs sound speed, re_inv_sqrt inverse square-root Reynolds number, vg_sq turbulent velocity scale squared
struct query_environment
{
    real Z, omega, vn, radial, strat, cs, re_inv_sqrt, vg_sq;
};
#if !defined(IMPORTGAS) && !defined(CODE_UNIT) && !defined(CONST_ST)
// analytic physical-unit gas allows the environment to be cached once per collision half-operator
#define COL_QUERY_ENV_CACHE
#endif // !IMPORTGAS && !CODE_UNIT && !CONST_ST

// bath-start total rate, first and second log-size jump-rate moments, and largest single jump for one owner
// a negative rate marks an invalid start state that must take the full chain path
struct cached_rate_moments { real rate, first, second, maximum; };

// requested refresh duration and its binding constraint: 0 horizon, 1 mean change, 2 fluctuation
struct change_bound { double duration; int reason; };

// per-owner event counts and log-mass changes by outcome category, recorded only with COL_DIAGNOSTICS
constexpr int EVENT_CATEGORIES = 7;
struct event_work { unsigned long long count[EVENT_CATEGORIES]; real log_mass[EVENT_CATEGORIES]; };

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
    int wave_count = 0;
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

// spatial controller groups and the 32-bit words of one bit-packed group dependency row
constexpr int LOCAL_GROUPS = ((N_X > 1) ? COL_BIN_X : 1)*COL_BIN_Y*((N_Z > 1) ? COL_BIN_Z : 1);
constexpr int LOCAL_WORDS = (LOCAL_GROUPS + 31) / 32;
constexpr int moving_groups = LOCAL_GROUPS;

#endif // COLLISION && !BERNOULLI

#endif // GAMEDEV_SWARM_COL_TYPES_CUH
