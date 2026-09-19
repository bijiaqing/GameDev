// Lab runtime: root transport with local collision scheduling.
#include <cmath>            // std::fabs, std::fmin, std::sin
#include <cstdlib>          // EXIT_FAILURE, std::exit
#include <filesystem>       // std::filesystem::create_directories
#include <iostream>         // std::cout, std::endl
#include <limits>           // std::numeric_limits
#include <sstream>          // std::stringstream
#include <stdexcept>        // std::runtime_error
#include <string>           // std::string, std::to_string
#include <vector>           // std::vector

#if defined(TRANSPORT) || defined(COLLISION)
#include <thrust/device_ptr.h>  // thrust::device_ptr
#include <thrust/extrema.h>     // thrust::max_element
#endif // TRANSPORT || COLLISION

#include <swarm_host.cuh>
#include <swarm_kern.cuh>

#if defined(COLLISION) && !defined(BERNOULLI)
#include <_col_chain.cuh>
#ifndef COLLISION_LOCAL_GPU_CUH
#define COLLISION_LOCAL_GPU_CUH
#ifndef COLLISION_LOCAL_SCHEDULE_HPP
#define COLLISION_LOCAL_SCHEDULE_HPP
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
#endif

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

__global__ void cache_query_environments(query_environment *env,const swarm *particle) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N_P && particle[i].position.y>0.0) env[i]=cache_query_environment(particle[i]);
}

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
    std::ofstream log;
    std::uint64_t operator_id=0;
    local_workspace(const std::string &path): owners(LOCAL_GROUPS), edges(LOCAL_GROUPS*LOCAL_WORDS),
        state(LOCAL_GROUPS) {
        LOCAL_CHECK(localMalloc((void**)&environment,sizeof(query_environment)*N_P));
        LOCAL_CHECK(localMalloc((void**)&cached,sizeof(cached_rate_moments)*N_P));
        LOCAL_CHECK(localMalloc((void**)&work,sizeof(event_work)*N_P));
        LOCAL_CHECK(localMalloc((void**)&work_sum,sizeof(event_work)));
        LOCAL_CHECK(localMalloc((void**)&ids,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_a,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&queue_b,sizeof(int)*N_P));
        LOCAL_CHECK(localMalloc((void**)&error,sizeof(int)));
        LOCAL_CHECK(localMalloc((void**)&dt,sizeof(real)*LOCAL_GROUPS));
        LOCAL_CHECK(localMalloc((void**)&change_rate,sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&second_rate,sizeof(real)*N_P));
        LOCAL_CHECK(localMalloc((void**)&graph,sizeof(unsigned int)*edges.size()));
        auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        log.open(path+"collision_local_"+std::to_string(stamp)+".jsonl");
        if (!log) throw std::runtime_error("cannot open local collision diagnostics");
        log << std::setprecision(17);
    }
    ~local_workspace() {
        localFree(environment); localFree(cached);
        localFree(work); localFree(work_sum);
        localFree(ids); localFree(queue_a); localFree(queue_b);
        localFree(error); localFree(dt); localFree(graph);
        localFree(change_rate); localFree(second_rate);
    }
};
#endif

#endif // COLLISION && !BERNOULLI

#ifdef KNN_CACHE
#include <_col_cache.cuh>
#endif // KNN_CACHE

#ifdef COLLISION_MORTON
#include <morton/morton_ghost.cuh>
#endif // COLLISION_MORTON

#ifdef GAMEDEV_ROCM
#define COAG_CHECK HIP_CHECK
#define COAG_KERNEL_CHECK HIP_KERNEL_CHECK
#define COAG_DeviceSynchronize hipDeviceSynchronize
#define COAG_Free hipFree
#define COAG_FreeHost hipHostFree
#define COAG_Malloc hipMalloc
#define COAG_MallocHost hipHostMalloc
#define COAG_Memcpy hipMemcpy
#define COAG_MemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define COAG_MemcpyDeviceToHost hipMemcpyDeviceToHost
#define COAG_MemcpyHostToDevice hipMemcpyHostToDevice
#define COAG_Memset hipMemset
#else
#define COAG_CHECK CUDA_CHECK
#define COAG_KERNEL_CHECK CUDA_KERNEL_CHECK
#define COAG_DeviceSynchronize cudaDeviceSynchronize
#define COAG_Free cudaFree
#define COAG_FreeHost cudaFreeHost
#define COAG_Malloc cudaMalloc
#define COAG_MallocHost cudaMallocHost
#define COAG_Memcpy cudaMemcpy
#define COAG_MemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define COAG_MemcpyDeviceToHost cudaMemcpyDeviceToHost
#define COAG_MemcpyHostToDevice cudaMemcpyHostToDevice
#define COAG_Memset cudaMemset
#endif

std::mt19937 rand_generator;

const std::string PATH = PATH_OUT; // convert the Makefile string literal to the output-path string used below

// =========================================================================================================================
// main program
// initialize or resume a swarm and advance enabled operators between successive output frames
//
// combined dynamics sequence:
//   1 half collision step
//   2 half spatial-diffusion step
//   3 full staggered semi-analytic transport step with optional midpoint radiation reconstruction
//   4 half spatial-diffusion step
//   5 half collision step
// =========================================================================================================================

int main (int argc, char **argv)
{
    #ifdef HALF_DISK
    if (N_Z > 1 && std::fabs(Z_MAX - 0.5*M_PI) > 16.0*std::numeric_limits<real>::epsilon())
        throw std::runtime_error("HALF_DISK requires Z_MAX = pi/2");
    #endif // HALF_DISK

    std::vector <real> mass_bank;
    initmass_calc(mass_bank);
    const real total_dust_mass = get_total_dust_mass(mass_bank);

    int idx_from;
    real clock_sim;   // total simulated time
    real clock_out;   // elapsed time in the current output interval
    real dt_out;      // duration of the current output interval

    #ifdef TRANSPORT
    int count_dyn;    // dynamics steps completed in the current output interval
    real dt_dyn;      // current dynamics timestep
    #endif // TRANSPORT
    
    #ifdef COLLISION
    int count_col;    // collision batches completed in the current dynamics interval
    real clock_dyn;   // elapsed collision time in the current dynamics interval
    real dt_col;      // current collision-batch timestep
    #endif // COLLISION
    
    // allocate the particle state and feature-dependent work arrays
    swarm *particle, *dev_particle;
    COAG_CHECK(COAG_MallocHost((void**)&particle, sizeof(swarm)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_particle, sizeof(swarm)*N_P));

    #ifdef TRANSPORT
    real *dev_dyn_rate;
    COAG_CHECK(COAG_Malloc((void**)&dev_dyn_rate, sizeof(real)*N_P));
    #endif // TRANSPORT
    
    #ifdef SAVE_DENS
    real *dustdens, *dev_dustdens;
    COAG_CHECK(COAG_MallocHost((void**)&dustdens, sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_dustdens, sizeof(real)*N_G));
    #endif // SAVE_DENS
    
    #ifdef IMPORTGAS
    real *gas_dens, *dev_gas_dens;
    COAG_CHECK(COAG_MallocHost((void**)&gas_dens,  sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_dens,  sizeof(real)*N_G));

    real *gas_velx, *dev_gas_velx;
    COAG_CHECK(COAG_MallocHost((void**)&gas_velx,  sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_velx,  sizeof(real)*N_G));

    real *gas_vely, *dev_gas_vely;
    COAG_CHECK(COAG_MallocHost((void**)&gas_vely,  sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_vely,  sizeof(real)*N_G));

    real *gas_velz, *dev_gas_velz;
    COAG_CHECK(COAG_MallocHost((void**)&gas_velz,  sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_velz,  sizeof(real)*N_G));

    real *dev_gas_dens_next, *dev_gas_velx_next, *dev_gas_vely_next, *dev_gas_velz_next;
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_dens_next, sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_velx_next, sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_vely_next, sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_gas_velz_next, sizeof(real)*N_G));
    #endif // IMPORTGAS
    
    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    COAG_CHECK(COAG_MallocHost((void**)&optdepth, sizeof(real)*N_G));
    COAG_CHECK(COAG_Malloc((void**)&dev_optdepth, sizeof(real)*N_G));
    #endif // RADIATION

    #ifdef COLLISION
    unsigned char *dev_col_active;
    COAG_CHECK(COAG_Malloc((void**)&dev_col_active, sizeof(unsigned char)*N_P));

    int *dev_bad_part;
    COAG_CHECK(COAG_Malloc((void**)&dev_bad_part, sizeof(int)));

    #ifdef COLLISION_KDTREE
    kdtree_boxf *dev_kdtree_box;
    COAG_CHECK(COAG_Malloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)));

    kdtree_node *dev_kdtree_node;
    COAG_CHECK(COAG_Malloc((void**)&dev_kdtree_node, sizeof(kdtree_node)*N_T));
    #else  // COLLISION_MORTON
    float3 *dev_morton_point;
    float *dev_morton_posx, *dev_search_dist;
    unsigned int *dev_morton_overflow;
    COAG_CHECK(COAG_Malloc((void**)&dev_morton_point, sizeof(float3)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_morton_posx, sizeof(float)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_search_dist, sizeof(float)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_morton_overflow, sizeof(unsigned int)*N_P));
    morton_ghost_index morton_owner;
    #endif // COLLISION_KDTREE

    real *dev_size_old, *dev_numr_old, *dev_col_rate;
    COAG_CHECK(COAG_Malloc((void**)&dev_size_old, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_numr_old, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_rate, sizeof(real)*N_P));

    #if !defined(BERNOULLI) || defined(KNN_CACHE)
    const std::size_t col_neighbor_count = static_cast<std::size_t>(N_P)*N_K;
    int *dev_col_neighbor;
    real *dev_col_measure;
    COAG_CHECK(COAG_Malloc((void**)&dev_col_neighbor, sizeof(int)*col_neighbor_count));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_measure, sizeof(real)*N_P));

    #endif // FROZEN_BATH || KNN_CACHE

    #ifndef BERNOULLI
    const int col_raw_count = _get_col_raw_count();
    int *dev_col_events, *dev_col_spatial;
    int *dev_col_count, *dev_col_binmap, *dev_col_error, *dev_col_unfinished;
    real *dev_col_time, *dev_col_hazard;
    real *dev_col_jump1_int, *dev_col_jump2_int, *dev_col_jumpmax_int;
    unsigned char *dev_col_complete;
    col_rate_bin *dev_col_ratebin;
    col_audit_accum *dev_col_audit;
    COAG_CHECK(COAG_Malloc((void**)&dev_col_events, sizeof(int)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_spatial, sizeof(int)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_count, sizeof(int)*col_raw_count));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_binmap, sizeof(int)*col_raw_count));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_error, sizeof(int)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_unfinished, sizeof(int)));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_time, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_hazard, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_jump1_int, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_jump2_int, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_jumpmax_int, sizeof(real)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_complete, sizeof(unsigned char)*N_P));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_ratebin, sizeof(col_rate_bin)*col_raw_count));
    COAG_CHECK(COAG_Malloc((void**)&dev_col_audit, sizeof(col_audit_accum)*col_raw_count));
    COAG_CHECK(COAG_Memset(dev_col_error, 0, sizeof(int)*N_P));
    #elif !defined(KNN_CACHE)  // DIRECT_BERNOULLI
    real *dev_col_dist;
    COAG_CHECK(COAG_Malloc((void**)&dev_col_dist, sizeof(real)*N_P));
    #endif // FROZEN_BATH / KNN_CACHE / DIRECT_BERNOULLI
    #endif // COLLISION

    #if defined(COLLISION) || defined(DIFFUSION)
    curs *dev_rngstate;
    COAG_CHECK(COAG_Malloc((void**)&dev_rngstate, sizeof(curs)*N_P));
    #endif // COLLISION || DIFFUSION

    if (argc <= 1)
	{
        // construct a fresh realization from the configured analytic or imported distribution
        
        idx_from = 0;

        real *randposx, *dev_randposx;
        COAG_CHECK(COAG_MallocHost((void**)&randposx, sizeof(real)*N_P));
        COAG_CHECK(COAG_Malloc((void**)&dev_randposx, sizeof(real)*N_P));

        real *randposy, *dev_randposy;
        COAG_CHECK(COAG_MallocHost((void**)&randposy, sizeof(real)*N_P));
        COAG_CHECK(COAG_Malloc((void**)&dev_randposy, sizeof(real)*N_P));

        real *randposz, *dev_randposz;
        COAG_CHECK(COAG_MallocHost((void**)&randposz, sizeof(real)*N_P));
        COAG_CHECK(COAG_Malloc((void**)&dev_randposz, sizeof(real)*N_P));

        #ifdef MULTISIZE
        real *randsize, *dev_randsize;
        COAG_CHECK(COAG_MallocHost((void**)&randsize, sizeof(real)*N_P));
        COAG_CHECK(COAG_Malloc((void**)&dev_randsize, sizeof(real)*N_P));

        real *dev_mass_bank;
        COAG_CHECK(COAG_Malloc((void**)&dev_mass_bank, sizeof(real)*mass_bank.size()));
        COAG_CHECK(COAG_Memcpy(dev_mass_bank, mass_bank.data(), sizeof(real)*mass_bank.size(), COAG_MemcpyHostToDevice));
        #endif // MULTISIZE

        rand_generator.seed(0); // keep initialization reproducible across runs

        #ifdef MULTISIZE
        // Every representative starts at the same monomer diameter.
        std::fill(randsize, randsize + N_P, INIT_SMIN); // fixed monomer diameter

        // correct size sampling and finite-domain containment so represented masses sum exactly to total_dust_mass
        real mass_norm = get_mass_norm(randsize, mass_bank, total_dust_mass);
        #endif // MULTISIZE

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);

        real *epsilon;
        COAG_CHECK(COAG_MallocHost((void**)&epsilon,  sizeof(real)*N_G));
        
        if (!load_epsilon(PATH, idx_from, epsilon))
        {
            std::cerr << "Error: Failed to load gas data files for frame " << idx_from << std::endl;
            return 1;
        }

        // use one imported total-dust spatial distribution for all previously sampled grain species
        rand_from_file(randposx, randposy, randposz, N_P, gas_dens, epsilon);
        
        COAG_CHECK(COAG_FreeHost(epsilon));
        #else  // NO IMPORTGAS
        #if defined(MULTISIZE) && defined(DIFFUSION)
        rand_disk_poly(randposx, randposy, randposz, randsize, N_P);
        #else  // !(MULTISIZE && DIFFUSION)
        rand_disk_mono(randposx, randposy, randposz, S_0, N_P);
        #endif // MULTISIZE && DIFFUSION
        #endif // IMPORTGAS

        COAG_CHECK(COAG_Memcpy(dev_randposx, randposx, sizeof(real)*N_P, COAG_MemcpyHostToDevice));
        COAG_CHECK(COAG_Memcpy(dev_randposy, randposy, sizeof(real)*N_P, COAG_MemcpyHostToDevice));
        COAG_CHECK(COAG_Memcpy(dev_randposz, randposz, sizeof(real)*N_P, COAG_MemcpyHostToDevice));

        #ifdef MULTISIZE
        COAG_CHECK(COAG_Memcpy(dev_randsize, randsize, sizeof(real)*N_P, COAG_MemcpyHostToDevice));
        #endif // MULTISIZE

        // convert sampled coordinates and sizes into the device particle state
        particle_init <<< NB_P, TPB >>> (dev_particle, dev_randposx, dev_randposy, dev_randposz
            #ifdef MULTISIZE
            , dev_randsize, dev_mass_bank, static_cast<int>(mass_bank.size()), mass_norm
            #endif // MULTISIZE
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        COAG_KERNEL_CHECK("particle_init");

        COAG_CHECK(COAG_FreeHost(randposx));
        COAG_CHECK(COAG_Free(dev_randposx));
        COAG_CHECK(COAG_FreeHost(randposy));
        COAG_CHECK(COAG_Free(dev_randposy));
        COAG_CHECK(COAG_FreeHost(randposz));
        COAG_CHECK(COAG_Free(dev_randposz));

        #ifdef MULTISIZE
        COAG_CHECK(COAG_FreeHost(randsize));
        COAG_CHECK(COAG_Free(dev_randsize));
        COAG_CHECK(COAG_Free(dev_mass_bank));
        #endif // MULTISIZE
        
        #if defined(COLLISION) || defined(DIFFUSION)
        rngstate_init <<< NB_P, TPB >>> (dev_rngstate);
        COAG_KERNEL_CHECK("rngstate_init");
        #endif // COLLISION || DIFFUSION
        
        // write the initial state and active configuration before evolution
        std::filesystem::create_directories(PATH);
        save_variable(PATH + "variables.txt", total_dust_mass);

        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_from, false);
        #endif // RADIATION

        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_from);
        #endif // SAVE_DENS

        SAVE_PARTICLE_TO_FILE(idx_from);

        msg_output(0);
    }
    else
    {
        // resume particle and imported-gas states from the requested output frame
        std::stringstream frame_stream{argv[1]};

        if (!(frame_stream >> idx_from))
        {
            std::cerr << "Error: Invalid resume file number: " << argv[1] << std::endl;
            return 1;
        }

        LOAD_PARTICLE_TO_VRAM(idx_from);

        #ifdef IMPORTGAS
        LOAD_GAS_DATA_TO_VRAM(idx_from);
        #endif // IMPORTGAS
        
        msg_output(idx_from);
    }

    #ifdef LOGTIMING
    clock_sim = (idx_from == 0) ? 0.0 : int_pow(LOG_BASE, idx_from)*DT_OUT;
    #else  // LOGOUTPUT or LINEAR
    clock_sim =                         static_cast<real>(idx_from)*DT_OUT;
    #endif // LOGTIMING

    #ifdef COLLISION
    bool col_geom_valid = false;
    float image_dist_min = -1.0f;
    if (X_WEDGE)
    {
        image_dist_min = _get_image_dist_min(
            static_cast<float>(X_MIN), static_cast<float>(X_MAX), static_cast<float>(Y_MIN),
            static_cast<float>(Z_MIN), static_cast<float>(Z_MAX)
        );
    }
    #if defined(COLLISION) && !defined(BERNOULLI)
    local_workspace local(PATH);
    bool local_geometry_valid = false;
    col_controller_summary col_summary;
    #endif // COLLISION && !BERNOULLI

    // invalidate the search package only when a position update ends its current geometry epoch
    auto invalidate_col_geometry = [&] ()
    {
        col_geom_valid = false;
        local_geometry_valid = false;
    };

    // evolve collisions over a fixed-position interval with the configured collision integrator
    auto evolve_collisions = [&] (real duration)
    {



        // geometry reuse must not suppress the per-operator nonfinite-state failure path
        if (col_geom_valid)
        {
            COAG_CHECK(COAG_Memset(dev_bad_part, 0, sizeof(int)));
            colstate_flag <<< NB_P, TPB >>> (dev_particle, dev_bad_part);
            COAG_KERNEL_CHECK("colstate_flag");
            int bad_part = 0;
            COAG_CHECK(COAG_Memcpy(&bad_part, dev_bad_part, sizeof(int), COAG_MemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // rebuild the complete geometric search package only after particle positions change
        if (!col_geom_valid)
        {

            COAG_CHECK(COAG_Memset(dev_bad_part, 0, sizeof(int)));
            #ifdef COLLISION_KDTREE
            col_site_init <<< NB_P, TPB >>> (
                dev_kdtree_node, dev_col_active, dev_particle, dev_bad_part
            );
            COAG_KERNEL_CHECK("col_site_init");
            int bad_part = 0;
            COAG_CHECK(COAG_Memcpy(&bad_part, dev_bad_part, sizeof(int), COAG_MemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }
            kdtree::buildTree <kdtree_node, kdtree_traits> (
                dev_kdtree_node, N_T, dev_kdtree_box
            );
            COAG_KERNEL_CHECK("kdtree::buildTree");
            #else  // COLLISION_MORTON
            col_site_init <<< NB_P, TPB >>> (
                dev_morton_point, dev_morton_posx, dev_search_dist,
                dev_col_active, dev_particle, dev_bad_part
            );
            COAG_KERNEL_CHECK("col_site_init");
            int bad_part = 0;
            COAG_CHECK(COAG_Memcpy(&bad_part, dev_bad_part, sizeof(int), COAG_MemcpyDeviceToHost));
            if (bad_part != 0)
            {
                std::cerr << "Error: non-finite particle state before collision search at particle "
                    << bad_part - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }

            thrust::device_ptr <const float> search_dist_ptr(dev_search_dist);
            float max_search_dist = *thrust::max_element(search_dist_ptr, search_dist_ptr + N_P);
            bool unique_ids = image_dist_min < 0.0f || image_dist_min > 2.0f*max_search_dist;
            morton_owner.build(
                dev_morton_point, dev_morton_posx, N_P, max_search_dist,
                static_cast<float>(X_MIN), static_cast<float>(X_MAX),
                static_cast<float>(Y_MAX), X_WEDGE, unique_ids,
                (N_Z > 1) ? 3 : 2, MORTON_LEAF_TARGET, MORTON_MAX_LEVEL
            );
            #endif // COLLISION_KDTREE

            #if !defined(BERNOULLI) || defined(KNN_CACHE)
            // retain fixed physical neighbors while collision properties continue to evolve
            #ifdef COLLISION_KDTREE
            col_cache_get <<< (N_T + kdtree_heap::threads - 1)/kdtree_heap::threads, kdtree_heap::threads >>> (
                dev_col_neighbor, dev_col_measure, dev_kdtree_node, dev_kdtree_box,
                dev_col_active, dev_particle, image_dist_min
            );
            COAG_KERNEL_CHECK("col_cache_get");
            #else  // COLLISION_MORTON
            col_cache_get <<< N_P, MORTON_TPB >>> (
                dev_col_neighbor, dev_col_measure, dev_morton_overflow, dev_morton_point,
                dev_col_active, dev_particle, morton_owner.view(), morton_owner.unique_ids()
            );
            COAG_KERNEL_CHECK("col_cache_get");
            thrust::device_ptr <const unsigned int> morton_overflow_ptr(dev_morton_overflow);
            unsigned int max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_cache_get");
            #endif // COLLISION_KDTREE
            #endif // FROZEN_BATH || KNN_CACHE


            #ifndef BERNOULLI
            col_space_bin <<< NB_P, TPB >>> (dev_col_spatial, dev_particle);
            COAG_KERNEL_CHECK("col_space_bin");
            #endif // FROZEN_BATH

            // publish validity only after every required hierarchy, cache, and guard has completed
            col_geom_valid = true;
        }

        #ifndef BERNOULLI
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

        #else  // BERNOULLI
        real elapsed = 0.0;
        while (elapsed < duration)
        {

            // freeze only the species fields changed by collisions while positions and velocities remain fixed
            col_snap_save <<< NB_P, TPB >>> (dev_size_old, dev_numr_old, dev_particle);
            COAG_KERNEL_CHECK("col_snap_save");
            #ifdef KNN_CACHE
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< NB_P, TPB >>> (
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
            #endif // COLLISION_KDTREE
                dev_col_rate, dev_particle, dev_col_neighbor, dev_col_measure,
                dev_col_active, dev_size_old, dev_numr_old,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #else  // DIRECT_BERNOULLI
            #ifdef COLLISION_KDTREE
            col_rate_calc <<< (N_T + kdtree_heap::threads - 1)/kdtree_heap::threads, kdtree_heap::threads >>> (dev_col_rate, dev_col_dist, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #else  // COLLISION_MORTON
            col_rate_calc <<< N_P, MORTON_TPB >>> (
                dev_col_rate, dev_col_dist, dev_morton_overflow, dev_particle,
                dev_col_active, dev_size_old, dev_numr_old, dev_morton_point,
                morton_owner.view(), morton_owner.unique_ids(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass
            );
            #endif // COLLISION_KDTREE
            #endif // KNN_CACHE
            COAG_KERNEL_CHECK("col_rate_calc");

            COAG_CHECK(COAG_Memset(dev_bad_part, 0, sizeof(int)));
            inf_rate_flag <<< NB_P, TPB >>> (dev_col_rate,
                #ifdef KNN_CACHE
                dev_col_measure,
                #else  // DIRECT_BERNOULLI
                dev_col_dist,
                #endif // KNN_CACHE
                dev_bad_part
            );
            COAG_KERNEL_CHECK("inf_rate_flag");
            int bad_result = 0;
            COAG_CHECK(COAG_Memcpy(&bad_result, dev_bad_part, sizeof(int), COAG_MemcpyDeviceToHost));
            if (bad_result != 0)
            {
                std::cerr << "Error: non-finite collision result at particle "
                    << bad_result - 1 << std::endl;
                std::exit(EXIT_FAILURE);
            }

            #if defined(COLLISION_MORTON) && !defined(KNN_CACHE)
            thrust::device_ptr <const unsigned int> morton_overflow_ptr(dev_morton_overflow);
            unsigned int max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_rate_calc");
            #endif // COLLISION_MORTON && !KNN_CACHE

            // use the largest total propensity to control every representative's event probability
            thrust::device_ptr <const real> col_rate_ptr(dev_col_rate);
            real max_col_rate = *thrust::max_element(col_rate_ptr, col_rate_ptr + N_P);
            real remaining = duration - elapsed;

            if (!(max_col_rate > 0.0))
            {
                // consume the remaining interval when no collision channel is active
                dt_col = remaining;
                elapsed = duration;
                break;
            }

            // keep the fastest frozen propensity below CFL_COL before sampling one event at most
            dt_col = fmin(CFL_COL / max_col_rate, remaining);
            #ifdef KNN_CACHE
            #ifdef COLLISION_KDTREE
            col_event_run <<< NB_P, TPB >>> (
            #else  // COLLISION_MORTON
            col_event_run <<< N_P, MORTON_TPB >>> (
            #endif // COLLISION_KDTREE
                dev_particle, dev_rngstate, dev_col_rate, dev_col_neighbor, dev_col_measure,
                dev_col_active, dev_size_old, dev_numr_old,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #else  // DIRECT_BERNOULLI
            #ifdef COLLISION_KDTREE
            col_event_run <<< (N_T + kdtree_heap::threads - 1)/kdtree_heap::threads, kdtree_heap::threads >>> (dev_particle, dev_rngstate, dev_col_rate, dev_col_dist,
                dev_col_active, dev_size_old, dev_numr_old, dev_kdtree_node, dev_kdtree_box,
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                image_dist_min,
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #else  // COLLISION_MORTON
            col_event_run <<< N_P, MORTON_TPB >>> (
                dev_particle, dev_rngstate, dev_col_rate, dev_col_dist, dev_morton_overflow,
                dev_col_active, dev_size_old, dev_numr_old, dev_morton_point,
                morton_owner.view(), morton_owner.unique_ids(),
                #ifdef IMPORTGAS
                dev_gas_dens,
                #endif // IMPORTGAS
                N_P / static_cast<real>(N_K) / total_dust_mass,
                dt_col
            );
            #endif // COLLISION_KDTREE
            #endif // KNN_CACHE
            COAG_KERNEL_CHECK("col_event_run");
            COAG_CHECK(COAG_DeviceSynchronize());

            #if defined(COLLISION_MORTON) && !defined(KNN_CACHE)
            max_morton_overflow = *thrust::max_element(
                morton_overflow_ptr, morton_overflow_ptr + N_P
            );
            if (max_morton_overflow != 0)
                throw std::runtime_error("Morton traversal stack overflow in col_event_run");
            #endif // COLLISION_MORTON && !KNN_CACHE


            real elapsed_old = elapsed;
            elapsed += dt_col;
            if (!(elapsed > elapsed_old))
                throw std::runtime_error("collision timestep cannot advance the operator clock");
            clock_dyn = elapsed;
            count_col++;
        }
        #endif // FROZEN_BATH

    };
    #endif // COLLISION

    for (int idx_file = idx_from + 1; idx_file <= SAVE_MAX; idx_file++)
    {
        // preload the next external frame and advance exactly one output interval
        dt_out = _get_dt_out(idx_file);

        #ifdef IMPORTGAS
        LOAD_GAS_NEXT_TO_VRAM(idx_file);
        real gas_frac = 0.0;
        #endif // IMPORTGAS
        
        clock_out = 0.0;
        
        #ifdef TRANSPORT
        count_dyn = 0;
        #endif // TRANSPORT

        #if defined(COLLISION) && !defined(BERNOULLI)
        // restart controller memory at checkpoint boundaries while retaining it across split operators
        std::fill(local.state.begin(), local.state.end(), col_bath_state{});
        col_summary = col_controller_summary{};
        #endif // COLLISION && !BERNOULLI

        PRINT_TITLE_TO_SCREEN();
        
        do
        {
            #ifdef TRANSPORT
            // reduce all local inverse rates to a globally valid dynamics timestep
            dyn_rate_calc <<< NB_P, TPB >>> (dev_dyn_rate, dev_particle
                #ifdef IMPORTGAS
                , dev_gas_velx, dev_gas_vely, dev_gas_velz
                , dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next
                , dev_gas_dens, dev_gas_dens_next
                #endif // IMPORTGAS
            );
            COAG_KERNEL_CHECK("dyn_rate_calc");
            thrust::device_ptr <const real> dt_rate_ptr(dev_dyn_rate);
            real max_dt_rate = *thrust::max_element(dt_rate_ptr, dt_rate_ptr + N_P);
            dt_dyn = fmin(DT_MAX, fmin(1.0 / max_dt_rate, dt_out - clock_out));

            #ifdef IMPORTGAS
            // interpolate the working gas fields to the midpoint time of this dynamics step
            real gas_target = (clock_out + 0.5*dt_dyn) / dt_out;
            real gas_blend = (gas_target - gas_frac) / (1.0 - gas_frac);
            gas_lerp_calc <<< NB_G, TPB >>> (
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, gas_blend
            );
            COAG_KERNEL_CHECK("gas_lerp_calc");
            gas_frac = gas_target;
            #endif // IMPORTGAS

            #ifdef COLLISION
            // begin the symmetric composition with half a collision interval
            count_col = 0;
            clock_dyn = 0.0;
            evolve_collisions(0.5*dt_dyn);
            #endif // COLLISION

            #ifdef DIFFUSION
            // apply the first half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn
#ifdef IMPORTGAS
                , dev_gas_dens
#endif
            );
            COAG_KERNEL_CHECK("diffusion_pos");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // DIFFUSION

            #ifdef RADIATION
            // drift to midpoint positions and reconstruct the optical depth used by the force solve
            ssa_substep_1 <<< NB_P, TPB >>> (dev_particle, dt_dyn);
            COAG_KERNEL_CHECK("ssa_substep_1");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            optdepth_init <<< NB_G, TPB >>> (dev_optdepth);
            COAG_KERNEL_CHECK("optdepth_init");
            optdepth_depo <<< NB_P, TPB >>> (dev_optdepth, dev_particle, total_dust_mass);
            COAG_KERNEL_CHECK("optdepth_depo");
            optdepth_calc <<< NB_G, TPB >>> (dev_optdepth);
            COAG_KERNEL_CHECK("optdepth_calc");
            optdepth_csum <<< NB_Y, TPB >>> (dev_optdepth);
            COAG_KERNEL_CHECK("optdepth_csum");

            real taper = (T_BETA > 0.0) ? (clock_sim + 0.5*dt_dyn) / T_BETA : 1.0;
            taper = fmin(fmax(taper, 0.0), 1.0);
            real beta_taper = taper*taper*(3.0 - 2.0*taper);

            ssa_substep_2 <<< NB_P, TPB >>> (dev_particle, dev_optdepth,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                beta_taper,
                dt_dyn
            );
            COAG_KERNEL_CHECK("ssa_substep_2");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #else  // NO RADIATION
            // complete transport in one launch when no midpoint radiation field is required
            ssa_transport <<< NB_P, TPB >>> (dev_particle,
                #ifdef IMPORTGAS
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                #endif // IMPORTGAS
                dt_dyn
            );
            COAG_KERNEL_CHECK("ssa_transport");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // RADIATION

            #ifdef DIFFUSION
            // apply the second half of the spatial diffusion operator
            diffusion_pos <<< NB_P, TPB >>> (dev_particle, dev_rngstate, 0.5*dt_dyn
#ifdef IMPORTGAS
                , dev_gas_dens
#endif
            );
            COAG_KERNEL_CHECK("diffusion_pos");
            #ifdef COLLISION
            invalidate_col_geometry();
            #endif // COLLISION
            #endif // DIFFUSION

            #ifdef COLLISION
            // close the symmetric composition with half a collision interval
            evolve_collisions(0.5*dt_dyn);
            #endif // COLLISION

            COAG_CHECK(COAG_DeviceSynchronize());
            clock_sim += dt_dyn;
            clock_out += dt_dyn;
            count_dyn++;
            PRINT_VALUE_TO_SCREEN();
            #endif // TRANSPORT

            #if defined(COLLISION) && !defined(TRANSPORT)
            // collision-only runs evolve directly across the complete output interval
            count_col = 0;
            clock_dyn = 0.0;
            real duration = dt_out - clock_out;
            
            #ifdef IMPORTGAS
            real gas_target = (clock_out + 0.5*duration) / dt_out;
            real gas_blend = (gas_target - gas_frac) / (1.0 - gas_frac);
            gas_lerp_calc <<< NB_G, TPB >>> (
                dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
                dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, gas_blend
            );
            COAG_KERNEL_CHECK("gas_lerp_calc");
            gas_frac = gas_target;
            #endif // IMPORTGAS
            
            evolve_collisions(duration);
            clock_out += duration;
            clock_sim += duration;
            PRINT_VALUE_TO_SCREEN();
            #endif // COLLISION && !TRANSPORT
        } while (clock_out < dt_out);

        #ifdef IMPORTGAS
        // replace the incrementally blended working fields by the exact endpoint snapshot
        COAG_CHECK(COAG_Memcpy(dev_gas_dens, dev_gas_dens_next, sizeof(real)*N_G, COAG_MemcpyDeviceToDevice));
        COAG_CHECK(COAG_Memcpy(dev_gas_velx, dev_gas_velx_next, sizeof(real)*N_G, COAG_MemcpyDeviceToDevice));
        COAG_CHECK(COAG_Memcpy(dev_gas_vely, dev_gas_vely_next, sizeof(real)*N_G, COAG_MemcpyDeviceToDevice));
        COAG_CHECK(COAG_Memcpy(dev_gas_velz, dev_gas_velz_next, sizeof(real)*N_G, COAG_MemcpyDeviceToDevice));
        #endif // IMPORTGAS

        // reconstruct requested mesh fields and save particle frames under the configured output cadence
        #ifdef RADIATION
        SAVE_OPTDEPTH_TO_FILE(idx_file, false);
        #endif // RADIATION
    
        #ifdef SAVE_DENS
        SAVE_DUSTDENS_TO_FILE(idx_file);
        #endif // SAVE_DENS

        #ifdef LOGTIMING
        SAVE_PARTICLE_TO_FILE(idx_file);
        #elif defined(LOGOUTPUT)
        if (is_log_power(idx_file)) SAVE_PARTICLE_TO_FILE(idx_file);
        #else  // LINEAR_OUTPUT
        if (idx_file % LIN_BASE == 0) SAVE_PARTICLE_TO_FILE(idx_file);
        #endif // LOGTIMING / LOGOUTPUT / LINEAR_OUTPUT

        #if defined(COLLISION) && !defined(BERNOULLI)
        std::string controller_file = PATH + "collision_chain_" + frame_num(idx_file) + ".json";
        if (!save_col_controller(controller_file, col_summary))
        {
            std::cerr << "Error: Failed to save file: " << controller_file << std::endl;
            return 1;
        }
        #endif // COLLISION && !BERNOULLI



        msg_output(idx_file);
    }

    return 0;
}

// =========================================================================================================================
