#ifndef COLLISION_LOCAL_GPU_CUH
#define COLLISION_LOCAL_GPU_CUH
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

// Integer endpoints avoid rounding drift between power-of-two timestep levels.
struct local_schedule {
    double horizon;
    std::vector<std::pair<int,int>> links;
    std::vector<int> level;
    std::vector<std::uint64_t> step, next;
    std::uint64_t end;

    local_schedule(double h, const std::vector<double>& requested,
                   const std::vector<unsigned int>& edges)
        : horizon(h), level(requested.size(), 0), step(requested.size()),
          next(requested.size(), 0) {
        if (!(h > 0) || !std::isfinite(h))
            throw std::runtime_error("invalid local collision horizon");
        const int n = static_cast<int>(level.size()), words = (n+31)/32;
        for (int c=0; c<n; ++c) {
            if (!(requested[c] > 0) || !std::isfinite(requested[c]))
                throw std::runtime_error("invalid local collision timestep");
            double dt = h;
            while (dt > requested[c]) {
                if (++level[c] > 52)
                    throw std::runtime_error("local collision timestep exceeds time resolution");
                dt *= 0.5;
            }
        }
        for (int c=0;c<n;++c) for (int d=c+1;d<n;++d)
            if ((edges[c*words+d/32] & (1u<<(d%32))) ||
                (edges[d*words+c/32] & (1u<<(c%32)))) links.emplace_back(c,d);
        // Symmetric compatibility on every directed KNN dependency: ratio <= 16.
        bool changed;
        do {
            changed = false;
            for (int c=0; c<n; ++c) for (int d=0; d<n; ++d) {
                if (!(edges[c*words+d/32] & (1u << (d%32)))) continue;
                if (level[c] < level[d]-4) { level[c]=level[d]-4; changed=true; }
                if (level[d] < level[c]-4) { level[d]=level[c]-4; changed=true; }
            }
        } while (changed);
        // A fixed lattice permits later refinement without moving pending endpoints.
        int finest = 52;
        end = std::uint64_t(1) << finest;
        for (int c=0; c<n; ++c) step[c] = std::uint64_t(1) << (finest-level[c]);
    }
    std::uint64_t time() const { return *std::min_element(next.begin(),next.end()); }
    double seconds(std::uint64_t tick) const { return horizon*(double(tick)/double(end)); }
    std::vector<int> due(std::uint64_t tick) const {
        std::vector<int> out;
        for (int c=0; c<int(next.size()); ++c) if (next[c]==tick) out.push_back(c);
        return out;
    }
    // Only due groups may change: other groups already hold computed endpoints.
    void adapt(std::uint64_t tick, const std::vector<int>& groups,
               const std::vector<double>& requested, const std::vector<bool>& passed) {
        const int n=level.size();
        std::vector<bool> active(n,false);
        for (int c:groups) active[c]=true;
        // Largest permitted level, propagated from immutable pending neighbors.
        std::vector<int> cap(n,52), candidate=level;
        for (int c=0;c<n;++c) if (!active[c]) cap[c]=level[c];
        bool changed;
        do {
            changed=false;
            for (const auto &edge:links) {
                int c=edge.first,d=edge.second;
                if (active[c] && cap[c]>cap[d]+4) {cap[c]=cap[d]+4;changed=true;}
                if (active[d] && cap[d]>cap[c]+4) {cap[d]=cap[c]+4;changed=true;}
            }
        } while (changed);
        for (int c:groups) {
            if (!(requested[c]>0) || !std::isfinite(requested[c]))
                throw std::runtime_error("invalid adaptive collision timestep");
            int want=0;
            double h=horizon;
            while (h>requested[c]) {
                if (++want>52) throw std::runtime_error("adaptive collision time resolution exceeded");
                h*=0.5;
            }
            // After a passing audit, recover directly; alignment and neighbors still constrain it.
            candidate[c]=passed[c] ? want : std::max(want,level[c]);
            while (tick % (std::uint64_t(1)<<(52-candidate[c]))) ++candidate[c];
            candidate[c]=std::min(candidate[c],cap[c]);
        }
        // Refine due groups until all directed dependencies satisfy ratio <= 16.
        // Caps above ensure this never requires changing an in-flight endpoint.
        do {
            changed=false;
            for (const auto &edge:links) {
                int c=edge.first,d=edge.second;
                if (active[c] && candidate[c]<candidate[d]-4) {
                    candidate[c]=candidate[d]-4;changed=true;
                }
                if (active[d] && candidate[d]<candidate[c]-4) {
                    candidate[d]=candidate[c]-4;changed=true;
                }
            }
        } while (changed);
        for (int c:groups) {
            level[c]=candidate[c];
            step[c]=std::uint64_t(1)<<(52-level[c]);
        }
    }
    void advance(const std::vector<int>& groups) {
        for (int c:groups) next[c] += step[c];
    }
};
#include <chrono>
#include <numeric>

#ifdef GAMEDEV_CUDA
#define LOCAL_CHECK CUDA_CHECK
#define LOCAL_KERNEL CUDA_KERNEL_CHECK
#define localMalloc cudaMalloc
#define localFree cudaFree
#define localCopy cudaMemcpy
#define localZero cudaMemset
#define localH2D cudaMemcpyHostToDevice
#define localD2H cudaMemcpyDeviceToHost
#else
#define LOCAL_CHECK HIP_CHECK
#define LOCAL_KERNEL HIP_KERNEL_CHECK
#define localMalloc hipMalloc
#define localFree hipFree
#define localCopy hipMemcpy
#define localZero hipMemset
#define localH2D hipMemcpyHostToDevice
#define localD2H hipMemcpyDeviceToHost
#endif

constexpr int LOCAL_GROUPS = ((N_X>1)?COL_BIN_X:1)*COL_BIN_Y*((N_Z>1)?COL_BIN_Z:1);
constexpr int LOCAL_WORDS = (LOCAL_GROUPS+31)/32;

#ifdef COL_QUERY_ENV_CACHE
__global__ void cache_query_environments(query_environment *env,const swarm *particle) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N_P && particle[i].position.y>0.0) env[i]=cache_query_environment(particle[i]);
}

#endif // COL_QUERY_ENV_CACHE

// One geometry-epoch pass. Reduce duplicate neighbor-group edges within each owner
// before issuing atomics; no per-neighbor transfer to the CPU is needed.
__global__ void local_graph(unsigned int *edges, const int *spatial,
    const int *neighbors, const unsigned char *active) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N_P || !active[i]) return;
    unsigned int bits[LOCAL_WORDS] = {};
    for (int k=0;k<N_K;++k) {
        int entry=neighbors[_get_col_offset(i,k)];
        if (entry<0) continue;
        int j=_get_col_idx_old(entry), c=spatial[j];
        bits[c/32] |= 1u << (c%32);
    }
    for (int w=0;w<LOCAL_WORDS;++w)
        if (bits[w]) atomicOr(edges+spatial[i]*LOCAL_WORDS+w,bits[w]);
}

__global__ void local_reset(const int *ids, int count, real *hazard,
    real *jump1, real *jump2, real *jumpmax) {
    int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if (slot>=count) return;
    int i=ids[slot];
    hazard[i]=jump1[i]=jump2[i]=jumpmax[i]=0;
}

// One block per category reduces per-owner diagnostics once per collision half-step.
__global__ void summarize_event_work(const event_work *work, event_work *sum) {
    const int k=blockIdx.x, t=threadIdx.x;
    __shared__ unsigned long long counts[TPB];
    __shared__ real growth[TPB];
    unsigned long long n=0; real g=0;
    for (int i=t;i<N_P;i+=TPB) { n+=work[i].count[k]; g+=work[i].log_mass[k]; }
    counts[t]=n;growth[t]=g;__syncthreads();
    for (int stride=TPB/2;stride>0;stride/=2) {
        if (t<stride) {counts[t]+=counts[t+stride];growth[t]+=growth[t+stride];}
        __syncthreads();
    }
    if (t==0) {sum->count[k]=counts[0];sum->log_mass[k]=growth[0];}
}
static_assert(TPB>0 && (TPB & (TPB-1))==0,"event reduction needs power-of-two TPB");

struct local_group_stats {
    std::uint64_t updates=0;
    int overshoots=0, persistent=0;
    real max_age=0, max_growth=0, max_activity=0, max_requested_ratio=0;
};

// Scratch allocations survive all operator calls. Particle IDs retain their RNG
// identity even when continuation queues are reordered by atomic append.
struct local_workspace {
    query_environment *environment=nullptr;
    cached_rate_moments *cached=nullptr;
    event_work *work=nullptr, *work_sum=nullptr;
    int *ids=nullptr, *queue_a=nullptr, *queue_b=nullptr, *error=nullptr;
    real *dt=nullptr, *change_rate=nullptr, *second_rate=nullptr;
    unsigned int *graph=nullptr;
    std::vector<std::vector<int>> owners;
    std::vector<unsigned int> edges;
    std::vector<col_bath_state> state;
#ifdef COL_DIAGNOSTICS
    std::ofstream log;
    std::uint64_t operator_id=0;
#endif
    local_workspace(const std::string &path): owners(LOCAL_GROUPS), edges(LOCAL_GROUPS*LOCAL_WORDS),
        state(LOCAL_GROUPS) {
        #ifdef COL_QUERY_ENV_CACHE
        LOCAL_CHECK(localMalloc((void**)&environment,sizeof(query_environment)*N_P));
        #endif
        LOCAL_CHECK(localMalloc((void**)&cached,sizeof(cached_rate_moments)*N_P));
#ifdef COL_DIAGNOSTICS
        LOCAL_CHECK(localMalloc((void**)&work,sizeof(event_work)*N_P));
        LOCAL_CHECK(localMalloc((void**)&work_sum,sizeof(event_work)));
#endif
        LOCAL_CHECK(localMalloc((void**)&ids,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_a,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_b,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&error,sizeof(int)));
        LOCAL_CHECK(localMalloc((void**)&dt,sizeof(real)*LOCAL_GROUPS));
        LOCAL_CHECK(localMalloc((void**)&change_rate,sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&second_rate,sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&graph,sizeof(unsigned int)*edges.size()));
#ifdef COL_DIAGNOSTICS
        auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        log.open(path+"collision_local_"+std::to_string(stamp)+".jsonl");
        if (!log) throw std::runtime_error("cannot open local collision diagnostics");
        log << std::setprecision(17);
#endif
    }
    ~local_workspace() {
        if (environment) localFree(environment);
        localFree(cached);
#ifdef COL_DIAGNOSTICS
        localFree(work); localFree(work_sum);
#endif
        localFree(ids); localFree(queue_a); localFree(queue_b);
        localFree(error); localFree(dt); localFree(graph);
        localFree(change_rate); localFree(second_rate);
    }
};
#endif
