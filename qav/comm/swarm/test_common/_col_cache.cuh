#ifndef SWARM_COL_CACHE_CUH
#define SWARM_COL_CACHE_CUH

// QAV header override: retain reduced cache-distribution diagnostics outside inc/

#if defined(COLLISION) && (!defined(BERNOULLI) || defined(KNN_CACHE))

#include <climits>  // INT_MAX
#include <cstddef>  // std::size_t

#ifdef COL_CACHE_QAV
#include <algorithm>  // std::sort
#include <fstream>    // std::ofstream
#include <iomanip>    // std::setprecision
#include <string>     // std::string
#include <vector>     // std::vector
#endif // COL_CACHE_QAV

#include <_collision.cuh>
#ifdef COLLISION_MORTON
#include <morton/morton_query.cuh>
#endif // COLLISION_MORTON

__host__ __device__ __forceinline__
std::size_t _get_col_offset (int idx_owner, int idx_neighbor)
{
    return static_cast<std::size_t>(idx_owner)*static_cast<std::size_t>(N_K)
        + static_cast<std::size_t>(idx_neighbor);
}

// cache the fixed physical top-K neighborhood for one geometry epoch
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
        {
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        }
        dev_col_measure[idx_old_i] = 0.0;
        return;
    }

    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    float search_dist = static_cast<float>(H_SEARCH*_get_hg(R)*R);

    // deduplicate overlapping wedge images using the same physical-id heap as direct search
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
        {
            dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)] = -1;
        }
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
        {
            if (work_idx_old[idx_neighbor] != INT_MAX)
                max_dist_sq = fmaxf(max_dist_sq, work_dist_sq[idx_neighbor]);
        }
        real radius = sqrt(static_cast<real>(max_dist_sq));
        real measure = _get_ball_measure(y, z, radius);
        dev_col_measure[idx_old_i] = measure;
        dev_morton_overflow[idx_old_i] = stack_overflow;
    }
}
#endif // COLLISION_KDTREE

#ifdef COL_CACHE_QAV

struct col_cache_qav_sample
{
    int valid_slots;
    int self_slots;
    real radius_hg;
    real measure;
};

// measure the realized cache occupancy without copying the full N_P by N_K neighbor table to the host
__global__
void cache_qav_get (col_cache_qav_sample *dev_sample, const int *dev_col_neighbor,
    const real *dev_col_measure, const unsigned char *dev_col_active,
    const swarm *dev_particle)
{
    int idx_old_i = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_old_i >= N_P) return;

    col_cache_qav_sample sample;
    sample.valid_slots = -1;
    sample.self_slots = 0;
    sample.radius_hg = 0.0;
    sample.measure = 0.0;
    if (dev_col_active[idx_old_i] == 0)
    {
        dev_sample[idx_old_i] = sample;
        return;
    }

    const swarm &particle_i = dev_particle[idx_old_i];
    real R_i = _get_cyl_R(particle_i.position.y, particle_i.position.z);
    real Z_i = _get_cyl_Z(particle_i.position.y, particle_i.position.z);
    real max_dist_sq = 0.0;
    sample.valid_slots = 0;
    for (int idx_neighbor = 0; idx_neighbor < N_K; idx_neighbor++)
    {
        int idx_old_j = dev_col_neighbor[_get_col_offset(idx_old_i, idx_neighbor)];
        if (idx_old_j < 0) continue;
        sample.valid_slots++;
        if (idx_old_j == idx_old_i) sample.self_slots++;

        const swarm &particle_j = dev_particle[idx_old_j];
        real R_j = _get_cyl_R(particle_j.position.y, particle_j.position.z);
        real Z_j = _get_cyl_Z(particle_j.position.y, particle_j.position.z);
        real dx = 0.0;
        if constexpr (N_X > 1)
        {
            real width = X_MAX - X_MIN;
            dx = fabs(particle_i.position.x - particle_j.position.x);
            dx = fmin(dx, fabs(width - dx));
        }
        real dist_sq = (R_i - R_j)*(R_i - R_j) + (Z_i - Z_j)*(Z_i - Z_j)
            + 2.0*R_i*R_j*(1.0 - cos(dx));
        max_dist_sq = fmax(max_dist_sq, dist_sq);
    }

    real h_g = R_i*_get_hg(R_i);
    sample.radius_hg = (h_g > 0.0) ? sqrt(max_dist_sq) / h_g : 0.0;
    sample.measure = dev_col_measure[idx_old_i];
    dev_sample[idx_old_i] = sample;
}

inline double _get_col_cache_quantile (const std::vector<double> &values, double fraction)
{
    double index = fraction*static_cast<double>(values.size() - 1);
    std::size_t idx_lo = static_cast<std::size_t>(index);
    std::size_t idx_hi = std::min(idx_lo + 1, values.size() - 1);
    double weight = index - static_cast<double>(idx_lo);
    return values[idx_lo] + weight*(values[idx_hi] - values[idx_lo]);
}

inline void _write_col_cache_distribution (std::ofstream &file, const std::string &name,
    std::vector<double> values, bool trailing_comma)
{
    std::sort(values.begin(), values.end());
    long double sum = 0.0;
    for (double value : values) sum += value;
    file
        << "  \"" << name << "\": {\n"
        << "    \"minimum\": " << values.front() << ",\n"
        << "    \"p01\": " << _get_col_cache_quantile(values, 0.01) << ",\n"
        << "    \"p05\": " << _get_col_cache_quantile(values, 0.05) << ",\n"
        << "    \"median\": " << _get_col_cache_quantile(values, 0.50) << ",\n"
        << "    \"p95\": " << _get_col_cache_quantile(values, 0.95) << ",\n"
        << "    \"p99\": " << _get_col_cache_quantile(values, 0.99) << ",\n"
        << "    \"maximum\": " << values.back() << ",\n"
        << "    \"mean\": " << static_cast<double>(sum / values.size()) << "\n"
        << "  }" << (trailing_comma ? "," : "") << "\n";
}

// archive compact occupancy and spatial-scale statistics for one QAV geometry build
inline bool save_col_cache_qav (const std::string &filename,
    const std::vector<col_cache_qav_sample> &samples, const std::string &backend)
{
    std::vector<double> valid_slots;
    std::vector<double> usable_neighbors;
    std::vector<double> radius_hg;
    std::vector<double> measures;
    std::size_t full_count = 0;
    std::size_t self_count = 0;
    std::size_t zero_measure_count = 0;
    for (const col_cache_qav_sample &sample : samples)
    {
        if (sample.valid_slots < 0) continue;
        valid_slots.push_back(static_cast<double>(sample.valid_slots));
        usable_neighbors.push_back(static_cast<double>(sample.valid_slots - sample.self_slots));
        radius_hg.push_back(sample.radius_hg);
        measures.push_back(sample.measure);
        if (sample.valid_slots == N_K) full_count++;
        if (sample.self_slots > 0) self_count++;
        if (!(sample.measure > 0.0)) zero_measure_count++;
    }
    if (valid_slots.empty()) return false;

    double active_count = static_cast<double>(valid_slots.size());
    std::ofstream file(filename);
    if (!file) return false;
    file << std::setprecision(17)
        << "{\n"
        << "  \"schema\": 1,\n"
        << "  \"backend\": \"" << backend << "\",\n"
        << "  \"search\": \""
        #ifdef COLLISION_KDTREE
        << "kdtree"
        #else  // COLLISION_MORTON
        << "morton"
        #endif // COLLISION_KDTREE
        << "\",\n"
        << "  \"particles\": " << N_P << ",\n"
        << "  \"active_particles\": " << valid_slots.size() << ",\n"
        << "  \"neighbors_capacity\": " << N_K << ",\n"
        << "  \"search_height\": " << H_SEARCH << ",\n"
        << "  \"full_cache_fraction\": " << full_count / active_count << ",\n"
        << "  \"incomplete_cache_fraction\": " << 1.0 - full_count / active_count << ",\n"
        << "  \"self_present_fraction\": " << self_count / active_count << ",\n"
        << "  \"zero_measure_fraction\": " << zero_measure_count / active_count << ",\n";
    _write_col_cache_distribution(file, "valid_slots", valid_slots, true);
    _write_col_cache_distribution(file, "usable_neighbors", usable_neighbors, true);
    _write_col_cache_distribution(file, "farthest_radius_hg", radius_hg, true);
    _write_col_cache_distribution(file, "neighborhood_measure", measures, false);
    file << "}\n";
    return static_cast<bool>(file);
}

#endif // COL_CACHE_QAV

#endif // COLLISION && (FROZEN_BATH || KNN_CACHE)

#endif // SWARM_COL_CACHE_CUH
