#ifndef LAB_EVENT_WORK_CUH
#define LAB_EVENT_WORK_CUH
constexpr int EVENT_CATEGORIES=7;
struct event_work { unsigned long long count[EVENT_CATEGORIES]; real log_mass[EVENT_CATEGORIES]; };
__host__ __device__ inline void record_event_work(event_work &work, int category, real log_mass) {
    ++work.count[category];
    work.log_mass[category]+=log_mass;
}
#endif
