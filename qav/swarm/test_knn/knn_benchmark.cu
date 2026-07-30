#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <kdtree/builder.h>
#include <kdtree/knn.h>

#include <morton/morton_index.cuh>
#include "knn_types.cuh"

#ifndef QAV_KNN_K
#define QAV_KNN_K 200
#endif

namespace
{

constexpr int K = QAV_KNN_K;
constexpr int KD_THREADS = 64;
constexpr int MORTON_THREADS = 256;

using kd_box = kdtree::box_t<float3>;

struct options
{
    int particles = 100000;
    int queries = 4096;
    int repeats = 5;
    int brute_queries = 32;
    int dimension = 2;
    int seed = 17;
    int leaf_target = 128;
    int max_level = 20;
    int max_leaf_scan = 4096;
    float radius = 0.1f;
    std::string distribution = "smooth";
    std::string output;
};

struct occupancy_stats
{
    double mean = 0.0;
    int median = 0;
    int p95 = 0;
    int p99 = 0;
    int maximum = 0;
};

struct quality_stats
{
    int mismatched_queries = 0;
    long long mismatched_neighbors = 0;
    int kd_brute_mismatches = 0;
    int morton_brute_mismatches = 0;
    int disagreement_queries_checked = 0;
    int kd_disagreement_brute_mismatches = 0;
    int morton_disagreement_brute_mismatches = 0;
    unsigned int stack_overflows = 0;
    float maximum_distance_error = 0.0f;
};

__global__
void kd_point_init (kd_point *tree, const float3 *source, int point_count)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= point_count) return;
    tree[idx].cartesian = source[idx];
    tree[idx].index_old = idx;
    tree[idx].split_dim = 0;
    tree[idx].image = 0;
}

template<int TOP_K>
__global__
void kd_query (int *neighbor_idx, float *neighbor_dist, const float3 *queries, int query_count,
    const kd_point *tree, const kd_box *bounds, int point_count, float radius)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kd_heap<TOP_K> result(radius, tree);
    kdtree::cct::knn<kd_heap<TOP_K>, kd_point, kd_traits>(
        result, queries[idx_query], *bounds, tree, point_count
    );

    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_tree = result.returnIndex(idx_neighbor);
        int idx_out = idx_query*TOP_K + idx_neighbor;
        neighbor_idx[idx_out] = (idx_tree < 0) ? -1 : tree[idx_tree].index_old;
        neighbor_dist[idx_out] = (idx_tree < 0) ? CUDART_INF_F : result.returnDist2(idx_neighbor);
    }
}

template<int TOP_K>
__global__
void kd_checksum (double *checksum, const float3 *queries, int query_count,
    const kd_point *tree, const kd_box *bounds, int point_count, float radius)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kd_heap<TOP_K> result(radius, tree);
    kdtree::cct::knn<kd_heap<TOP_K>, kd_point, kd_traits>(
        result, queries[idx_query], *bounds, tree, point_count
    );

    double value = 0.0;
    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_tree = result.returnIndex(idx_neighbor);
        if (idx_tree < 0) continue;
        value += static_cast<double>(result.returnDist2(idx_neighbor))
            + 1.0e-12*static_cast<double>(tree[idx_tree].index_old);
    }
    checksum[idx_query] = value;
}

const char *next_argument (int argc, char **argv, int &idx)
{
    if (++idx >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[idx - 1]);
    return argv[idx];
}

options parse_options (int argc, char **argv)
{
    options result;
    for (int idx = 1; idx < argc; idx++)
    {
        std::string argument = argv[idx];
        if (argument == "--particles") result.particles = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--queries") result.queries = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--repeat") result.repeats = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--brute-queries") result.brute_queries = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--dim") result.dimension = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--seed") result.seed = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--leaf-target") result.leaf_target = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--max-level") result.max_level = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--max-leaf-scan") result.max_leaf_scan = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--radius") result.radius = std::stof(next_argument(argc, argv, idx));
        else if (argument == "--distribution") result.distribution = next_argument(argc, argv, idx);
        else if (argument == "--output") result.output = next_argument(argc, argv, idx);
        else throw std::invalid_argument("unknown argument: " + argument);
    }

    if (result.particles < K) throw std::invalid_argument("--particles must be at least QAV_KNN_K");
    if (result.queries <= 0 || result.repeats <= 0 || result.brute_queries < 0)
        throw std::invalid_argument("queries and repeats must be positive and brute queries nonnegative");
    if (result.dimension != 2 && result.dimension != 3) throw std::invalid_argument("--dim must be 2 or 3");
    if (result.leaf_target <= 0) throw std::invalid_argument("--leaf-target must be positive");
    if (result.max_level <= 0 || result.max_level > 20) throw std::invalid_argument("--max-level must be 1 through 20");
    if (result.max_leaf_scan <= 0) throw std::invalid_argument("--max-leaf-scan must be positive");
    if (result.radius <= 0.0f) throw std::invalid_argument("--radius must be positive");
    if (result.distribution != "smooth" && result.distribution != "ring" && result.distribution != "clump")
        throw std::invalid_argument("--distribution must be smooth, ring, or clump");
    result.queries = std::min(result.queries, result.particles);
    result.brute_queries = std::min(result.brute_queries, result.queries);
    return result;
}

std::vector<float3> generate_points (const options &config)
{
    std::mt19937 generator(config.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float3> points(config.particles);

    for (int idx = 0; idx < config.particles; idx++)
    {
        float phi = 2.0f*static_cast<float>(M_PI)*uniform(generator);
        float R;
        float Z = 0.0f;

        if (config.distribution == "clump" && uniform(generator) < 0.8f)
        {
            points[idx].x = 1.0f + 0.01f*normal(generator);
            points[idx].y = 0.01f*normal(generator);
            points[idx].z = (config.dimension == 3) ? 0.01f*normal(generator) : 0.0f;
            continue;
        }

        if (config.distribution == "ring") R = 1.0f + 0.03f*normal(generator);
        else R = std::sqrt(0.25f + 2.0f*uniform(generator));
        if (config.dimension == 3)
            Z = ((config.distribution == "ring") ? 0.02f : 0.05f)*normal(generator);

        points[idx] = make_float3(R*std::cos(phi), R*std::sin(phi), Z);
    }
    return points;
}

void get_root (const std::vector<float3> &points, int dimension, float3 &origin, float &root_width)
{
    float3 lower = points.front();
    float3 upper = points.front();
    for (const float3 &point : points)
    {
        lower.x = std::min(lower.x, point.x);
        lower.y = std::min(lower.y, point.y);
        lower.z = std::min(lower.z, point.z);
        upper.x = std::max(upper.x, point.x);
        upper.y = std::max(upper.y, point.y);
        upper.z = std::max(upper.z, point.z);
    }

    float extent_x = upper.x - lower.x;
    float extent_y = upper.y - lower.y;
    float extent_z = (dimension == 2) ? 0.0f : upper.z - lower.z;
    root_width = 1.0001f*std::max({extent_x, extent_y, extent_z});
    float center_x = 0.5f*(lower.x + upper.x);
    float center_y = 0.5f*(lower.y + upper.y);
    float center_z = (dimension == 2) ? 0.0f : 0.5f*(lower.z + upper.z);
    origin = make_float3(
        center_x - 0.5f*root_width,
        center_y - 0.5f*root_width,
        center_z - 0.5f*root_width
    );
}

template<typename Function>
double wall_time_ms (Function operation)
{
    auto start = std::chrono::steady_clock::now();
    operation();
    _morton_cuda_check(cudaDeviceSynchronize(), "synchronize timed operation");
    auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

template<typename Function>
double kernel_time_ms (Function operation, int repeats)
{
    operation();
    _morton_cuda_check(cudaDeviceSynchronize(), "synchronize benchmark warmup");

    cudaEvent_t begin;
    cudaEvent_t end;
    _morton_cuda_check(cudaEventCreate(&begin), "create benchmark start event");
    _morton_cuda_check(cudaEventCreate(&end), "create benchmark end event");
    _morton_cuda_check(cudaEventRecord(begin), "record benchmark start");
    for (int repeat = 0; repeat < repeats; repeat++)
    {
        operation();
    }
    _morton_cuda_check(cudaEventRecord(end), "record benchmark end");
    _morton_cuda_check(cudaEventSynchronize(end), "synchronize benchmark end");
    float elapsed = 0.0f;
    _morton_cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "read benchmark duration");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return static_cast<double>(elapsed) / repeats;
}

occupancy_stats get_occupancy (const morton_index &index)
{
    std::vector<int> counts = index.leaf_counts();
    std::sort(counts.begin(), counts.end());

    occupancy_stats result;
    long long sum = 0;
    for (int count : counts)
    {
        sum += count;
    }
    result.mean = static_cast<double>(sum) / counts.size();
    result.median = counts[counts.size() / 2];
    result.p95 = counts[static_cast<std::size_t>(0.95*(counts.size() - 1))];
    result.p99 = counts[static_cast<std::size_t>(0.99*(counts.size() - 1))];
    result.maximum = counts.back();
    return result;
}

quality_stats compare_neighbors (const std::vector<int> &kd_idx, const std::vector<float> &kd_dist,
    const std::vector<int> &morton_idx, const std::vector<float> &morton_dist, int query_count,
    std::vector<int> &disagreement_queries)
{
    quality_stats result;
    for (int idx_query = 0; idx_query < query_count; idx_query++)
    {
        std::vector<std::pair<float, int>> kd_neighbors;
        std::vector<std::pair<float, int>> morton_neighbors;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            int idx = idx_query*K + idx_neighbor;
            if (kd_idx[idx] >= 0) kd_neighbors.emplace_back(kd_dist[idx], kd_idx[idx]);
            if (morton_idx[idx] >= 0) morton_neighbors.emplace_back(morton_dist[idx], morton_idx[idx]);
        }
        std::sort(kd_neighbors.begin(), kd_neighbors.end());
        std::sort(morton_neighbors.begin(), morton_neighbors.end());

        bool query_mismatch = kd_neighbors.size() != morton_neighbors.size();
        std::size_t common = std::min(kd_neighbors.size(), morton_neighbors.size());
        result.mismatched_neighbors += static_cast<long long>(
            std::max(kd_neighbors.size(), morton_neighbors.size()) - common
        );
        for (std::size_t idx = 0; idx < common; idx++)
        {
            float distance_error = std::fabs(kd_neighbors[idx].first - morton_neighbors[idx].first);
            result.maximum_distance_error = std::max(result.maximum_distance_error, distance_error);
            float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(kd_neighbors[idx].first));
            if (kd_neighbors[idx].second != morton_neighbors[idx].second || distance_error > tolerance)
            {
                query_mismatch = true;
                result.mismatched_neighbors++;
            }
        }
        if (query_mismatch)
        {
            result.mismatched_queries++;
            disagreement_queries.push_back(idx_query);
        }
    }
    return result;
}

std::vector<std::pair<float, int>> get_array_neighbors (
    const std::vector<int> &indices, const std::vector<float> &distances, int idx_query)
{
    std::vector<std::pair<float, int>> neighbors;
    for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
    {
        int idx = idx_query*K + idx_neighbor;
        if (indices[idx] >= 0) neighbors.emplace_back(distances[idx], indices[idx]);
    }
    std::sort(neighbors.begin(), neighbors.end());
    return neighbors;
}

bool differs_from_brute (const std::vector<std::pair<float, int>> &actual,
    const std::vector<std::pair<float, int>> &expected, float &maximum_distance_error)
{
    if (actual.size() != expected.size()) return true;
    for (std::size_t idx = 0; idx < expected.size(); idx++)
    {
        float distance_error = std::fabs(actual[idx].first - expected[idx].first);
        maximum_distance_error = std::max(maximum_distance_error, distance_error);
        float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(expected[idx].first));
        if (actual[idx].second != expected[idx].second || distance_error > tolerance) return true;
    }
    return false;
}

void compare_brute_force (quality_stats &quality, const std::vector<float3> &points,
    const std::vector<int> &kd_idx, const std::vector<float> &kd_dist,
    const std::vector<int> &morton_idx, const std::vector<float> &morton_dist,
    int query_count, float radius)
{
    float radius_sq = radius*radius;
    for (int idx_query = 0; idx_query < query_count; idx_query++)
    {
        std::vector<std::pair<float, int>> expected;
        const float3 &query = points[idx_query];
        for (std::size_t idx_point = 0; idx_point < points.size(); idx_point++)
        {
            const float3 &candidate = points[idx_point];
            float dx = query.x - candidate.x;
            float dy = query.y - candidate.y;
            float dz = query.z - candidate.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq <= radius_sq) expected.emplace_back(dist_sq, static_cast<int>(idx_point));
        }
        std::sort(expected.begin(), expected.end());
        if (expected.size() > K) expected.resize(K);

        std::vector<std::pair<float, int>> kd_neighbors = get_array_neighbors(kd_idx, kd_dist, idx_query);
        std::vector<std::pair<float, int>> morton_neighbors = get_array_neighbors(
            morton_idx, morton_dist, idx_query
        );
        if (differs_from_brute(kd_neighbors, expected, quality.maximum_distance_error))
            quality.kd_brute_mismatches++;
        if (differs_from_brute(morton_neighbors, expected, quality.maximum_distance_error))
            quality.morton_brute_mismatches++;
    }
}

void compare_disagreements_brute_force (quality_stats &quality, const std::vector<float3> &points,
    const std::vector<int> &kd_idx, const std::vector<float> &kd_dist,
    const std::vector<int> &morton_idx, const std::vector<float> &morton_dist,
    const std::vector<int> &disagreement_queries, float radius)
{
    float radius_sq = radius*radius;
    quality.disagreement_queries_checked = static_cast<int>(disagreement_queries.size());

    for (int idx_query : disagreement_queries)
    {
        std::vector<std::pair<float, int>> expected;
        const float3 &query = points[idx_query];
        for (std::size_t idx_point = 0; idx_point < points.size(); idx_point++)
        {
            const float3 &candidate = points[idx_point];
            float dx = query.x - candidate.x;
            float dy = query.y - candidate.y;
            float dz = query.z - candidate.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq <= radius_sq) expected.emplace_back(dist_sq, static_cast<int>(idx_point));
        }
        std::sort(expected.begin(), expected.end());
        if (expected.size() > K) expected.resize(K);

        std::vector<std::pair<float, int>> kd_neighbors = get_array_neighbors(kd_idx, kd_dist, idx_query);
        std::vector<std::pair<float, int>> morton_neighbors = get_array_neighbors(
            morton_idx, morton_dist, idx_query
        );
        if (differs_from_brute(kd_neighbors, expected, quality.maximum_distance_error))
            quality.kd_disagreement_brute_mismatches++;
        if (differs_from_brute(morton_neighbors, expected, quality.maximum_distance_error))
            quality.morton_disagreement_brute_mismatches++;
    }
}

void write_json (const options &config, float root_width, int node_count, int leaf_count,
    double kd_build_ms, double morton_build_ms, double kd_query_ms, double morton_query_ms,
    std::size_t kd_bytes, std::size_t morton_bytes, const occupancy_stats &occupancy,
    const quality_stats &quality, double mean_cells, double mean_candidates)
{
    std::ostream *output = &std::cout;
    std::ofstream file;
    if (!config.output.empty())
    {
        std::filesystem::path output_path(config.output);
        if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
        file.open(output_path);
        if (!file) throw std::runtime_error("cannot open output file: " + config.output);
        output = &file;
    }

    bool quality_passed = quality.mismatched_queries == 0
        && quality.kd_brute_mismatches == 0 && quality.morton_brute_mismatches == 0
        && quality.kd_disagreement_brute_mismatches == 0
        && quality.morton_disagreement_brute_mismatches == 0
        && quality.stack_overflows == 0;

    *output << std::setprecision(10)
        << "{\n"
        << "  \"particles\": " << config.particles << ",\n"
        << "  \"quality_queries\": " << config.queries << ",\n"
        << "  \"brute_force_queries\": " << config.brute_queries << ",\n"
        << "  \"dimension\": " << config.dimension << ",\n"
        << "  \"distribution\": \"" << config.distribution << "\",\n"
        << "  \"k\": " << K << ",\n"
        << "  \"radius\": " << config.radius << ",\n"
        << "  \"root_width\": " << root_width << ",\n"
        << "  \"leaf_target\": " << config.leaf_target << ",\n"
        << "  \"max_level\": " << config.max_level << ",\n"
        << "  \"max_leaf_scan\": " << config.max_leaf_scan << ",\n"
        << "  \"tree_nodes\": " << node_count << ",\n"
        << "  \"occupied_leaves\": " << leaf_count << ",\n"
        << "  \"quality_passed\": " << (quality_passed ? "true" : "false") << ",\n"
        << "  \"mismatched_queries\": " << quality.mismatched_queries << ",\n"
        << "  \"mismatched_neighbors\": " << quality.mismatched_neighbors << ",\n"
        << "  \"kd_brute_mismatches\": " << quality.kd_brute_mismatches << ",\n"
        << "  \"morton_brute_mismatches\": " << quality.morton_brute_mismatches << ",\n"
        << "  \"disagreement_queries_checked\": " << quality.disagreement_queries_checked << ",\n"
        << "  \"kd_disagreement_brute_mismatches\": "
        << quality.kd_disagreement_brute_mismatches << ",\n"
        << "  \"morton_disagreement_brute_mismatches\": "
        << quality.morton_disagreement_brute_mismatches << ",\n"
        << "  \"stack_overflows\": " << quality.stack_overflows << ",\n"
        << "  \"maximum_distance_error\": " << quality.maximum_distance_error << ",\n"
        << "  \"kd_build_ms\": " << kd_build_ms << ",\n"
        << "  \"morton_build_ms\": " << morton_build_ms << ",\n"
        << "  \"kd_query_ms\": " << kd_query_ms << ",\n"
        << "  \"morton_query_ms\": " << morton_query_ms << ",\n"
        << "  \"query_speedup\": " << kd_query_ms / morton_query_ms << ",\n"
        << "  \"kd_persistent_bytes\": " << kd_bytes << ",\n"
        << "  \"morton_persistent_bytes\": " << morton_bytes << ",\n"
        << "  \"persistent_memory_ratio\": " << static_cast<double>(morton_bytes) / kd_bytes << ",\n"
        << "  \"leaf_occupancy\": {\n"
        << "    \"mean\": " << occupancy.mean << ",\n"
        << "    \"median\": " << occupancy.median << ",\n"
        << "    \"p95\": " << occupancy.p95 << ",\n"
        << "    \"p99\": " << occupancy.p99 << ",\n"
        << "    \"maximum\": " << occupancy.maximum << "\n"
        << "  },\n"
        << "  \"mean_leaves_visited\": " << mean_cells << ",\n"
        << "  \"mean_candidates_examined\": " << mean_candidates << "\n"
        << "}\n";
}

} // namespace

int main (int argc, char **argv)
{
    try
    {
        options config = parse_options(argc, argv);
        std::vector<float3> points = generate_points(config);
        float3 origin;
        float root_width;
        get_root(points, config.dimension, origin, root_width);

        float3 *dev_points = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_points, sizeof(float3)*points.size()), "allocate benchmark points");
        _morton_cuda_check(cudaMemcpy(dev_points, points.data(), sizeof(float3)*points.size(), cudaMemcpyHostToDevice),
            "copy benchmark points");

        kd_point *dev_kd_tree = nullptr;
        kd_box *dev_kd_bounds = nullptr;
        double kd_build_ms = wall_time_ms([&]
        {
            _morton_cuda_check(cudaMalloc((void**)&dev_kd_tree, sizeof(kd_point)*points.size()), "allocate KD tree");
            _morton_cuda_check(cudaMalloc((void**)&dev_kd_bounds, sizeof(kd_box)), "allocate KD bounds");
            int blocks = (config.particles + KD_THREADS - 1) / KD_THREADS;
            kd_point_init <<< blocks, KD_THREADS >>> (dev_kd_tree, dev_points, config.particles);
            _morton_cuda_check(cudaGetLastError(), "launch kd_point_init");
            kdtree::buildTree<kd_point, kd_traits>(dev_kd_tree, config.particles, dev_kd_bounds);
        });

        morton_index morton;
        double morton_build_ms = wall_time_ms([&]
        {
            morton.build(
                dev_points, config.particles, origin, root_width,
                config.dimension, config.leaf_target, config.max_level
            );
        });
        occupancy_stats occupancy = get_occupancy(morton);
        if (occupancy.maximum > config.max_leaf_scan)
        {
            throw std::runtime_error(
                "adaptive Morton maximum leaf occupancy " + std::to_string(occupancy.maximum)
                + " exceeds --max-leaf-scan=" + std::to_string(config.max_leaf_scan)
                + "; increase --max-level or inspect coincident particles"
            );
        }

        std::size_t quality_size = static_cast<std::size_t>(config.queries)*K;
        int *dev_kd_idx = nullptr;
        int *dev_morton_idx = nullptr;
        float *dev_kd_dist = nullptr;
        float *dev_morton_dist = nullptr;
        unsigned int *dev_cell_visits = nullptr;
        unsigned int *dev_candidate_visits = nullptr;
        unsigned int *dev_quality_overflows = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_idx, sizeof(int)*quality_size), "allocate KD quality indices");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_idx, sizeof(int)*quality_size), "allocate Morton quality indices");
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_dist, sizeof(float)*quality_size), "allocate KD quality distances");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_dist, sizeof(float)*quality_size), "allocate Morton quality distances");
        _morton_cuda_check(cudaMalloc((void**)&dev_cell_visits, sizeof(unsigned int)*config.queries), "allocate cell visits");
        _morton_cuda_check(cudaMalloc((void**)&dev_candidate_visits, sizeof(unsigned int)*config.queries),
            "allocate candidate visits");
        _morton_cuda_check(cudaMalloc((void**)&dev_quality_overflows, sizeof(unsigned int)*config.queries),
            "allocate quality stack-overflow flags");

        int kd_quality_blocks = (config.queries + KD_THREADS - 1) / KD_THREADS;
        kd_query<K> <<< kd_quality_blocks, KD_THREADS >>> (
            dev_kd_idx, dev_kd_dist, dev_points, config.queries,
            dev_kd_tree, dev_kd_bounds, config.particles, config.radius
        );
        morton_query<K> <<< config.queries, MORTON_THREADS >>> (
            dev_morton_idx, dev_morton_dist, dev_cell_visits, dev_candidate_visits, dev_quality_overflows,
            dev_points, config.queries, morton.view(), config.radius
        );
        _morton_cuda_check(cudaDeviceSynchronize(), "run KNN quality queries");

        std::vector<int> kd_idx(quality_size);
        std::vector<int> morton_idx(quality_size);
        std::vector<float> kd_dist(quality_size);
        std::vector<float> morton_dist(quality_size);
        std::vector<unsigned int> cell_visits(config.queries);
        std::vector<unsigned int> candidate_visits(config.queries);
        std::vector<unsigned int> quality_overflows(config.queries);
        _morton_cuda_check(cudaMemcpy(kd_idx.data(), dev_kd_idx, sizeof(int)*quality_size, cudaMemcpyDeviceToHost),
            "copy KD quality indices");
        _morton_cuda_check(cudaMemcpy(morton_idx.data(), dev_morton_idx, sizeof(int)*quality_size, cudaMemcpyDeviceToHost),
            "copy Morton quality indices");
        _morton_cuda_check(cudaMemcpy(kd_dist.data(), dev_kd_dist, sizeof(float)*quality_size, cudaMemcpyDeviceToHost),
            "copy KD quality distances");
        _morton_cuda_check(cudaMemcpy(morton_dist.data(), dev_morton_dist, sizeof(float)*quality_size, cudaMemcpyDeviceToHost),
            "copy Morton quality distances");
        _morton_cuda_check(cudaMemcpy(cell_visits.data(), dev_cell_visits, sizeof(unsigned int)*config.queries,
            cudaMemcpyDeviceToHost), "copy cell visits");
        _morton_cuda_check(cudaMemcpy(candidate_visits.data(), dev_candidate_visits,
            sizeof(unsigned int)*config.queries, cudaMemcpyDeviceToHost), "copy candidate visits");
        _morton_cuda_check(cudaMemcpy(quality_overflows.data(), dev_quality_overflows,
            sizeof(unsigned int)*config.queries, cudaMemcpyDeviceToHost), "copy quality stack-overflow flags");

        std::vector<int> disagreement_queries;
        quality_stats quality = compare_neighbors(
            kd_idx, kd_dist, morton_idx, morton_dist, config.queries, disagreement_queries
        );
        compare_brute_force(
            quality, points, kd_idx, kd_dist, morton_idx, morton_dist,
            config.brute_queries, config.radius
        );
        compare_disagreements_brute_force(
            quality, points, kd_idx, kd_dist, morton_idx, morton_dist,
            disagreement_queries, config.radius
        );
        for (unsigned int overflow : quality_overflows)
        {
            quality.stack_overflows += overflow;
        }

        double *dev_kd_checksum = nullptr;
        double *dev_morton_checksum = nullptr;
        unsigned int *dev_performance_overflows = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_checksum, sizeof(double)*config.particles), "allocate KD checksum");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_checksum, sizeof(double)*config.particles),
            "allocate Morton checksum");
        _morton_cuda_check(cudaMalloc((void**)&dev_performance_overflows, sizeof(unsigned int)*config.particles),
            "allocate performance stack-overflow flags");

        int kd_blocks = (config.particles + KD_THREADS - 1) / KD_THREADS;
        double kd_query_ms = kernel_time_ms([&]
        {
            kd_checksum<K> <<< kd_blocks, KD_THREADS >>> (
                dev_kd_checksum, dev_points, config.particles,
                dev_kd_tree, dev_kd_bounds, config.particles, config.radius
            );
        }, config.repeats);
        double morton_query_ms = kernel_time_ms([&]
        {
            morton_checksum<K> <<< config.particles, MORTON_THREADS >>> (
                dev_morton_checksum, dev_performance_overflows,
                dev_points, config.particles, morton.view(), config.radius
            );
        }, config.repeats);

        std::vector<unsigned int> performance_overflows(config.particles);
        _morton_cuda_check(cudaMemcpy(performance_overflows.data(), dev_performance_overflows,
            sizeof(unsigned int)*config.particles, cudaMemcpyDeviceToHost),
            "copy performance stack-overflow flags");
        for (unsigned int overflow : performance_overflows)
        {
            quality.stack_overflows += overflow;
        }
        double mean_cells = 0.0;
        double mean_candidates = 0.0;
        for (int idx = 0; idx < config.queries; idx++)
        {
            mean_cells += cell_visits[idx];
            mean_candidates += candidate_visits[idx];
        }
        mean_cells /= config.queries;
        mean_candidates /= config.queries;

        std::size_t kd_bytes = sizeof(kd_point)*static_cast<std::size_t>(config.particles) + sizeof(kd_box);
        write_json(
            config, root_width, morton.node_count(), morton.leaf_count(), kd_build_ms, morton_build_ms,
            kd_query_ms, morton_query_ms, kd_bytes, morton.persistent_bytes(), occupancy,
            quality, mean_cells, mean_candidates
        );

        cudaFree(dev_kd_checksum);
        cudaFree(dev_morton_checksum);
        cudaFree(dev_performance_overflows);
        cudaFree(dev_kd_idx);
        cudaFree(dev_morton_idx);
        cudaFree(dev_kd_dist);
        cudaFree(dev_morton_dist);
        cudaFree(dev_cell_visits);
        cudaFree(dev_candidate_visits);
        cudaFree(dev_quality_overflows);
        cudaFree(dev_kd_tree);
        cudaFree(dev_kd_bounds);
        cudaFree(dev_points);

        bool quality_passed = quality.mismatched_queries == 0
            && quality.kd_brute_mismatches == 0 && quality.morton_brute_mismatches == 0
            && quality.kd_disagreement_brute_mismatches == 0
            && quality.morton_disagreement_brute_mismatches == 0
            && quality.stack_overflows == 0;
        return quality_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
