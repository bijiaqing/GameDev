// Included inside evolve_collisions after production geometry/cache construction.
// All launches below use the default stream. No publication occurs while an
// event chain or its audit can still be reading the previous reservoir.
using local_clock = std::chrono::steady_clock;
const auto local_begin=local_clock::now();
cache_query_environments<<<NB_P,TPB>>>(local.environment,dev_particle);
LOCAL_KERNEL("cache_query_environments");
LOCAL_CHECK(localZero(local.work,0,sizeof(event_work)*N_P));
if (!local_geometry_valid) {
    std::vector<int> spatial(N_P);
    LOCAL_CHECK(localCopy(spatial.data(),dev_col_spatial,sizeof(int)*N_P,localD2H));
    for (auto &v:local.owners) v.clear();
    for (int i=0;i<N_P;++i) local.owners[spatial[i]].push_back(i);
    LOCAL_CHECK(localZero(local.graph,0,sizeof(unsigned int)*local.edges.size()));
    local_graph<<<NB_P,TPB>>>(local.graph,dev_col_spatial,dev_col_neighbor,dev_col_active);
    LOCAL_KERNEL("local_graph");
    LOCAL_CHECK(localCopy(local.edges.data(),local.graph,
        sizeof(unsigned int)*local.edges.size(),localD2H));
    local_geometry_valid=true;
}

std::vector<int> ids(N_P), counts(col_raw_count), binmap(col_raw_count);
std::iota(ids.begin(),ids.end(),0);
LOCAL_CHECK(localCopy(local.ids,ids.data(),sizeof(int)*N_P,localH2D));
std::vector<col_rate_bin> ratebin(col_raw_count);
std::vector<col_audit_accum> audit(col_raw_count);
const real lambda0=N_P/static_cast<real>(N_K)/total_dust_mass;

auto initialize = [&](int count) {
    int blocks=(count+TPB-1)/TPB;
    col_bath_init<<<blocks,TPB>>>(local.ids,count,dev_size_old,dev_numr_old,
        dev_col_time,dev_col_events,dev_col_complete,dev_particle);
    LOCAL_KERNEL("local_publish_and_init");
    local_reset<<<blocks,TPB>>>(local.ids,count,dev_col_hazard,dev_col_jump1_int,
        dev_col_jump2_int,dev_col_jumpmax_int);
    LOCAL_KERNEL("local_reset");
};
auto rates_and_bins = [&](int count) {
    col_bath_rate<<<count,COL_BATH_TPB>>>(local.ids,count,dev_col_rate,local.change_rate,local.second_rate,
        dev_particle,dev_col_neighbor,dev_col_measure,dev_col_active,
        dev_size_old,dev_numr_old,
        #ifdef IMPORTGAS
        dev_gas_dens,
        #endif
        lambda0,local.environment,local.cached);
    LOCAL_KERNEL("local_rates");
    LOCAL_CHECK(localZero(dev_col_count,0,sizeof(int)*col_raw_count));
    col_count_bin<<<(count+TPB-1)/TPB,TPB>>>(local.ids,count,dev_col_count,
        dev_particle,dev_col_spatial,dev_col_active);
    LOCAL_KERNEL("local_count");
    LOCAL_CHECK(localCopy(counts.data(),dev_col_count,sizeof(int)*col_raw_count,localD2H));
    int merged=_build_col_binmap(counts,binmap);
    LOCAL_CHECK(localCopy(dev_col_binmap,binmap.data(),sizeof(int)*col_raw_count,localH2D));
    LOCAL_CHECK(localZero(dev_col_ratebin,0,sizeof(col_rate_bin)*col_raw_count));
    col_rate_bins<<<(count+TPB-1)/TPB,TPB>>>(local.ids,count,dev_col_ratebin,
        dev_particle,dev_col_rate,local.change_rate,local.second_rate,dev_col_spatial,dev_col_binmap,dev_col_active);
    LOCAL_KERNEL("local_rate_bins");
    LOCAL_CHECK(localCopy(ratebin.data(),dev_col_ratebin,sizeof(col_rate_bin)*merged,localD2H));
    return merged;
};
auto bin_end = [&](int c,int merged) {
    return c+1<LOCAL_GROUPS ? binmap[(c+1)*COL_BIN_S] : merged;
};
auto requested_step = [&](int c,int merged,int *binding=nullptr) {
    int first=binmap[c*COL_BIN_S], last=bin_end(c,merged);
    std::vector<col_rate_bin> slice(ratebin.begin()+first,ratebin.begin()+last);
    return _choose_col_bath(slice,int(slice.size()),duration,local.state[c].limit_scale,binding);
};

initialize(N_P);
int merged=rates_and_bins(N_P);
std::vector<double> requested(LOCAL_GROUPS);
std::vector<int> binding(LOCAL_GROUPS);
for (int c=0;c<LOCAL_GROUPS;++c) requested[c]=requested_step(c,merged,&binding[c]);
local_schedule schedule(duration,requested,local.edges);
std::vector<real> steps(LOCAL_GROUPS), published(LOCAL_GROUPS,0);
for (int c=0;c<LOCAL_GROUPS;++c) steps[c]=schedule.seconds(schedule.step[c]);
LOCAL_CHECK(localCopy(local.dt,steps.data(),sizeof(real)*LOCAL_GROUPS,localH2D));
std::vector<local_group_stats> stats(LOCAL_GROUPS);
std::vector<bool> passed(LOCAL_GROUPS,true);
std::vector<int> initial_level=schedule.level;
std::vector<std::uint64_t> coarsened(LOCAL_GROUPS,0), refined(LOCAL_GROUPS,0),
    constrained(LOCAL_GROUPS,0);
std::vector<double> current_request=requested, min_request=requested, max_request=requested;
std::uint64_t owner_updates=0, chain_blocks=0, waves=0, launches=0;
double chain_seconds=0, audit_seconds=0;
col_summary.operator_count++;
int operator_index=col_summary.operator_count;
int batch_index=0;

while (schedule.time()<schedule.end) {
    auto tick=schedule.time();
    auto groups=schedule.due(tick);
    real time=schedule.seconds(tick);
    ids.clear();
    for (int c:groups) {
        published[c]=time;
        ids.insert(ids.end(),local.owners[c].begin(),local.owners[c].end());
    }
    int count=static_cast<int>(ids.size());
    if (count==0) { schedule.advance(groups); continue; }
    // At tick zero the complete population was already initialized and rated.
    if (tick!=0) {
        LOCAL_CHECK(localCopy(local.ids,ids.data(),sizeof(int)*count,localH2D));
        initialize(count);
        merged=rates_and_bins(count);
    }
    if (tick!=0) {
        for (int c:groups) {
            current_request[c]=requested_step(c,merged);
            min_request[c]=std::min(min_request[c],current_request[c]);
            max_request[c]=std::max(max_request[c],current_request[c]);
        }
        auto previous=schedule.level;
        schedule.adapt(tick,groups,current_request,passed);
        for (int c:groups) {
            coarsened[c]+=schedule.level[c]<previous[c];
            refined[c]+=schedule.level[c]>previous[c];
            steps[c]=schedule.seconds(schedule.step[c]);
            constrained[c]+=steps[c]>current_request[c];
        }
        LOCAL_CHECK(localCopy(local.dt,steps.data(),sizeof(real)*LOCAL_GROUPS,localH2D));
    }
    for (int c:groups) {
        if (local.owners[c].empty()) continue;
        for (int d=0;d<LOCAL_GROUPS;++d)
            if (local.edges[c*LOCAL_WORDS+d/32] & (1u<<(d%32)))
                stats[c].max_age=std::max(stats[c].max_age,time-published[d]);
        stats[c].max_requested_ratio=std::max(stats[c].max_requested_ratio,
            steps[c]/requested_step(c,merged));
    }

    const auto chain_begin=local_clock::now();
    LOCAL_CHECK(localZero(local.error,0,sizeof(int)));
    int unfinished=count, continuations=0;
    const int *input=local.ids;
    int *output=local.queue_a;
    while (unfinished>0) {
        if (++continuations>1000000)
            throw std::runtime_error("local collision continuation limit exceeded");
        LOCAL_CHECK(localZero(dev_col_unfinished,0,sizeof(int)));
        chain_blocks+=unfinished;
        col_chain_run<<<unfinished,COL_BATH_TPB>>>(input,unfinished,
            dev_particle,dev_rngstate,dev_col_error,dev_col_unfinished,
            dev_col_time,dev_col_events,dev_col_complete,dev_col_hazard,
            dev_col_jump1_int,dev_col_jump2_int,dev_col_jumpmax_int,dev_col_neighbor,
            dev_col_measure,dev_col_active,dev_size_old,dev_numr_old,
            #ifdef IMPORTGAS
            dev_gas_dens,
            #endif
            lambda0,local.dt,dev_col_spatial,output,local.error,local.work,local.environment,local.cached);
        LOCAL_KERNEL("local_chain");
        LOCAL_CHECK(localCopy(&unfinished,dev_col_unfinished,sizeof(int),localD2H));
        input=output;
        output=(output==local.queue_a)?local.queue_b:local.queue_a;
    }
    int error=0;
    LOCAL_CHECK(localCopy(&error,local.error,sizeof(int),localD2H));
    if (error) throw std::runtime_error("local collision chain error "+std::to_string(error));
    chain_seconds+=std::chrono::duration<double>(local_clock::now()-chain_begin).count();

    const auto audit_begin=local_clock::now();
    LOCAL_CHECK(localZero(dev_col_audit,0,sizeof(col_audit_accum)*col_raw_count));
    col_audit_bin<<<(count+TPB-1)/TPB,TPB>>>(local.ids,count,dev_col_audit,
        dev_particle,dev_size_old,dev_numr_old,dev_col_rate,dev_col_hazard,
        dev_col_jump1_int,dev_col_jump2_int,dev_col_jumpmax_int,dev_col_events,
        dev_col_spatial,dev_col_binmap,dev_col_active,local.dt);
    LOCAL_KERNEL("local_audit");
    LOCAL_CHECK(localCopy(audit.data(),dev_col_audit,
        sizeof(col_audit_accum)*merged,localD2H));
    for (int c:groups) {
        if (local.owners[c].empty()) continue;
        int first=binmap[c*COL_BIN_S], last=bin_end(c,merged);
        real mass=0;
        for (int b=first;b<last;++b) {
            if (audit[b].invalid_count) throw std::runtime_error("invalid local audit state");
            mass+=audit[b].mass;
        }
        if (!(mass>0)) continue;
        real before=local.state[c].limit_scale;
        std::vector<col_audit_accum> slice(audit.begin()+first,audit.begin()+last);
        auto result=_finish_col_bath(slice,int(slice.size()),local.state[c]);
        passed[c]=!(result.activity_overshoot || result.distribution_overshoot || result.persistent_overshoot);
        auto &s=stats[c];
        ++s.updates;
        s.overshoots+=result.activity_overshoot || result.distribution_overshoot;
        s.persistent+=result.persistent_overshoot;
        s.max_growth=std::max(s.max_growth,result.max_g);
        s.max_activity=std::max(s.max_activity,result.max_f);
        // Audit feedback controls the next due update; pending neighbors stay fixed.
        col_bath_record record;
        record.group_index=c;
        record.operator_index=operator_index; record.bath_index=++batch_index;
        record.merged_bins=last-first; record.duration=steps[c];
        record.limit_before=before; record.limit_after=local.state[c].limit_scale;
        record.result=result;
        _record_col_bath(col_summary,record);
    }
    // Count physical launches once per wave, not once per group in that wave.
    col_summary.continuation_launches+=continuations;
    audit_seconds+=std::chrono::duration<double>(local_clock::now()-audit_begin).count();
    owner_updates+=count; launches+=continuations; ++waves;
    ++count_col;
    dt_col=duration;
    for (int c:groups) dt_col=std::min(dt_col,steps[c]);
    schedule.advance(groups);
    clock_dyn=schedule.seconds(schedule.time());
}
// Every pending endpoint has now reached H; transport may read all particle states.
clock_dyn=duration;
local.log << "{\"schema\":1,\"method\":\"erosion_cached_rates\",\"operator\":" << ++local.operator_id
    << ",\"clock_sim\":" << clock_sim << ",\"duration\":" << duration
    << ",\"waves\":" << waves << ",\"owner_updates\":" << owner_updates
    << ",\"chain_blocks\":" << chain_blocks << ",\"chain_launches\":" << launches
    << ",\"chain_seconds\":" << chain_seconds << ",\"audit_seconds\":" << audit_seconds
    << ",\"scheduler_wall_seconds\":"
    << std::chrono::duration<double>(local_clock::now()-local_begin).count()
    << ",\"finest_ticks\":" << schedule.end << ",\"groups\":[";
for (int c=0;c<LOCAL_GROUPS;++c) {
    if (c) local.log << ',';
    const auto &s=stats[c];
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
summarize_event_work<<<EVENT_CATEGORIES,TPB>>>(local.work,local.work_sum);
LOCAL_KERNEL("summarize_event_work");
event_work totals;
LOCAL_CHECK(localCopy(&totals,local.work_sum,sizeof(event_work),localD2H));
for (int k=0;k<EVENT_CATEGORIES;++k) {if(k) local.log<<',';local.log<<totals.count[k];}
local.log << "],\"event_log_mass_sums\":[";
for (int k=0;k<EVENT_CATEGORIES;++k) {if(k) local.log<<',';local.log<<totals.log_mass[k];}
local.log << "]}\n";
local.log.flush();
if (!local.log) throw std::runtime_error("cannot write local collision diagnostics");
