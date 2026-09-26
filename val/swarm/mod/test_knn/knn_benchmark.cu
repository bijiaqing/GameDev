#include <algorithm>      // std::min, std::max, std::sort
#include <chrono>         // std::chrono timing
#include <cmath>          // std::cos, std::sin, std::sqrt
#include <cstddef>        // std::size_t
#include <cstdint>        // std::uint64_t
#include <cstdlib>        // EXIT_SUCCESS, EXIT_FAILURE
#include <filesystem>     // std::filesystem::create_directories
#include <fstream>        // std::ofstream
#include <iomanip>        // std::setprecision
#include <iostream>       // std::cout, std::cerr
#include <limits>         // std::numeric_limits
#include <random>         // std::mt19937, probability distributions
#include <stdexcept>      // std::invalid_argument, std::runtime_error
#include <string>         // std::string, std::stoi
#include <utility>        // std::pair
#include <vector>         // std::vector

#include <gpu.cuh> // CUDA allocation, events, copies, and kernel launches

#include <kdtree/builder.h>
#include <kdtree/knn.h>

#include <morton/morton_index.cuh>
#include "knn_types.cuh"

// compare exact nonperiodic KD-tree and adaptive-Morton top-K searches on matched point clouds
// validate both backends against brute force before reporting build, query, and persistent-memory costs

#ifndef VAL_KNN_K
#define VAL_KNN_K 200
#endif // !VAL_KNN_K

namespace
{

constexpr int K = VAL_KNN_K;
constexpr int KDTREE_TPB = 64;
constexpr int MORTON_TPB = 64;

using kdtree_boxf = kdtree::box_t<float3>;

struct options
{
    int particles = 100000;
    int queries = 4096;
    int repeats = 5;
    int brute_queries = 32;
    int dim = 2;
    int seed = 17;
    int leaf_target = 128;
    int max_level = 20;
    int max_leaf_scan = 4096;
    float radius = 0.1f;
    bool quality_only = false;
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
    int kdtree_brute_mismatches = 0;
    int morton_brute_mismatches = 0;
    int disagreement_queries_checked = 0;
    int kdtree_disagreement_brute_mismatches = 0;
    int morton_disagreement_brute_mismatches = 0;
    int morton_record_mismatches = 0;
    int record_geometry_mismatches = 0;
    int first_failure_query = -1;
    unsigned int stack_overflows = 0;
    float maximum_distance_error = 0.0f;
};

__global__
void kdtree_point_init (kdtree_point *dev_kdtree_node, const float3 *dev_source_point, int point_count)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= point_count) return;
    dev_kdtree_node[idx].cartesian = dev_source_point[idx];
    dev_kdtree_node[idx].idx_old = idx;
    dev_kdtree_node[idx].split_dim = 0;
    dev_kdtree_node[idx].image = 0;
}

template<int TOP_K>
__global__
void kdtree_query (int *dev_near_idx_old, float *dev_near_dist_sq,
    const float3 *dev_query_point, int query_count,
    const kdtree_point *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    int point_count, float search_dist)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kdtree_heap<TOP_K> near_result(search_dist, dev_kdtree_node);
    kdtree::cct::knn<kdtree_heap<TOP_K>, kdtree_point, kdtree_traits>(
        near_result, dev_query_point[idx_query], *dev_kdtree_box, dev_kdtree_node, point_count
    );

    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_old = near_result.returnIndex(idx_neighbor);
        int idx_out = idx_query*TOP_K + idx_neighbor;
        dev_near_idx_old[idx_out] = idx_old;
        #ifdef GAMEDEV_ROCM
        dev_near_dist_sq[idx_out] = (idx_old < 0) ? MORTON_INF_F : near_result.returnDist2(idx_neighbor);
        #else  // !GAMEDEV_ROCM
        dev_near_dist_sq[idx_out] = (idx_old < 0) ? CUDART_INF_F : near_result.returnDist2(idx_neighbor);
        #endif // GAMEDEV_ROCM
    }
}

template<int TOP_K>
__global__
void kdtree_checksum (double *dev_checksum, const float3 *dev_query_point, int query_count,
    const kdtree_point *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    int point_count, float search_dist)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kdtree_heap<TOP_K> near_result(search_dist, dev_kdtree_node);
    kdtree::cct::knn<kdtree_heap<TOP_K>, kdtree_point, kdtree_traits>(
        near_result, dev_query_point[idx_query], *dev_kdtree_box, dev_kdtree_node, point_count
    );

    double value = 0.0;
    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_old = near_result.returnIndex(idx_neighbor);
        if (idx_old < 0) continue;
        value += static_cast<double>(near_result.returnDist2(idx_neighbor))
            + 1.0e-12*static_cast<double>(idx_old);
    }
    dev_checksum[idx_query] = value;
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
        else if (argument == "--dim") result.dim = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--seed") result.seed = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--leaf-target") result.leaf_target = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--max-level") result.max_level = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--max-leaf-scan") result.max_leaf_scan = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--radius") result.radius = std::stof(next_argument(argc, argv, idx));
        else if (argument == "--quality-only") result.quality_only = true;
        else if (argument == "--distribution") result.distribution = next_argument(argc, argv, idx);
        else if (argument == "--output") result.output = next_argument(argc, argv, idx);
        else throw std::invalid_argument("unknown argument: " + argument);
    }

    if (result.particles < K) throw std::invalid_argument("--particles must be at least VAL_KNN_K");
    if (result.queries <= 0 || result.repeats <= 0 || result.brute_queries < 0)
    {
        throw std::invalid_argument("queries and repeats must be positive and brute queries nonnegative");
    }
    if (result.dim != 2 && result.dim != 3) throw std::invalid_argument("--dim must be 2 or 3");
    if (result.leaf_target <= 0) throw std::invalid_argument("--leaf-target must be positive");
    if (result.max_level <= 0 || result.max_level > 20) throw std::invalid_argument("--max-level must be 1 through 20");
    if (result.max_leaf_scan <= 0) throw std::invalid_argument("--max-leaf-scan must be positive");
    if (result.radius <= 0.0f) throw std::invalid_argument("--radius must be positive");
    if (result.distribution != "smooth" && result.distribution != "ring"
        && result.distribution != "clump" && result.distribution != "radial")
        throw std::invalid_argument("--distribution must be smooth, ring, clump, or radial");
    result.queries = std::min(result.queries, result.particles);
    result.brute_queries = std::min(result.brute_queries, result.queries);
    return result;
}

// generate smooth, ring, or clumped Cartesian point clouds from one reproducible seed
std::vector<float3> generate_points (const options &config)
{
    std::mt19937 generator(config.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float3> points(config.particles);

    for (int idx = 0; idx < config.particles; idx++)
    {
        if (config.distribution == "radial")
        {
            float R = 0.5f + uniform(generator);
            points[idx] = make_float3(R, 0.0f, 0.0f);
            continue;
        }

        float phi = 2.0f*static_cast<float>(M_PI)*uniform(generator);
        float R;
        float Z = 0.0f;

        if (config.distribution == "clump" && uniform(generator) < 0.8f)
        {
            points[idx].x = 1.0f + 0.01f*normal(generator);
            points[idx].y = 0.01f*normal(generator);
            points[idx].z = (config.dim == 3) ? 0.01f*normal(generator) : 0.0f;
            continue;
        }

        if (config.distribution == "ring") R = 1.0f + 0.03f*normal(generator);
        else R = std::sqrt(0.25f + 2.0f*uniform(generator));
        if (config.dim == 3)
        {
            Z = ((config.distribution == "ring") ? 0.02f : 0.05f)*normal(generator);
        }

        points[idx] = make_float3(R*std::cos(phi), R*std::sin(phi), Z);
    }
    return points;
}

// enclose all generated points in the cubic Morton root used by both build and traversal
void get_root (const std::vector<float3> &points, int dim, float3 &root_origin, float &root_width)
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
    float extent_z = (dim == 2) ? 0.0f : upper.z - lower.z;
    root_width = 1.0001f*std::max({extent_x, extent_y, extent_z});
    float center_x = 0.5f*(lower.x + upper.x);
    float center_y = 0.5f*(lower.y + upper.y);
    float center_z = (dim == 2) ? 0.0f : 0.5f*(lower.z + upper.z);
    root_origin = make_float3(
        center_x - 0.5f*root_width,
        center_y - 0.5f*root_width,
        center_z - 0.5f*root_width
    );
}

template<typename Function>
double wall_time_ms (Function operation)
{
    auto time_start = std::chrono::steady_clock::now();
    operation();
    _morton_gpu_check(gpuDeviceSynchronize(), "synchronize timed operation");
    auto time_stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(time_stop - time_start).count();
}

template<typename Function>
double kernel_time_ms (Function operation, int repeats)
{
    operation();
    _morton_gpu_check(gpuDeviceSynchronize(), "synchronize benchmark warmup");

    gpuEvent_t event_start;
    gpuEvent_t event_stop;
    _morton_gpu_check(gpuEventCreate(&event_start), "create benchmark start event");
    _morton_gpu_check(gpuEventCreate(&event_stop), "create benchmark end event");
    _morton_gpu_check(gpuEventRecord(event_start), "record benchmark start");
    for (int repeat = 0; repeat < repeats; repeat++)
    {
        operation();
    }
    _morton_gpu_check(gpuEventRecord(event_stop), "record benchmark end");
    _morton_gpu_check(gpuEventSynchronize(event_stop), "synchronize benchmark end");
    float elapsed_ms = 0.0f;
    _morton_gpu_check(gpuEventElapsedTime(&elapsed_ms, event_start, event_stop), "read benchmark duration");
    #ifdef GAMEDEV_ROCM
    _morton_gpu_check(gpuEventDestroy(event_start), "destroy benchmark start event");
    _morton_gpu_check(gpuEventDestroy(event_stop), "destroy benchmark end event");
    #else  // !GAMEDEV_ROCM
    gpuEventDestroy(event_start);
    gpuEventDestroy(event_stop);
    #endif // GAMEDEV_ROCM
    return static_cast<double>(elapsed_ms) / repeats;
}

occupancy_stats get_occupancy (const morton_index &morton_owner)
{
    std::vector<int> counts = morton_owner.leaf_counts();
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

// compare complete sorted top-K lists and retain enough diagnostics to localize disagreements
quality_stats compare_neighbors (const std::vector<int> &kdtree_idx_old, const std::vector<float> &kdtree_dist_sq,
    const std::vector<int> &morton_idx_old, const std::vector<float> &morton_dist_sq, int query_count,
    std::vector<int> &disagreement_queries)
{
    quality_stats result;
    for (int idx_query = 0; idx_query < query_count; idx_query++)
    {
        std::vector<std::pair<float, int>> kdtree_neighbors;
        std::vector<std::pair<float, int>> morton_neighbors;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            int idx = idx_query*K + idx_neighbor;
            if (kdtree_idx_old[idx] >= 0) kdtree_neighbors.emplace_back(kdtree_dist_sq[idx], kdtree_idx_old[idx]);
            if (morton_idx_old[idx] >= 0) morton_neighbors.emplace_back(morton_dist_sq[idx], morton_idx_old[idx]);
        }
        #ifdef GAMEDEV_ROCM
        auto by_identifier = [](const auto &left, const auto &right)
        {
            return left.second < right.second;
        };
        std::sort(kdtree_neighbors.begin(), kdtree_neighbors.end(), by_identifier);
        std::sort(morton_neighbors.begin(), morton_neighbors.end(), by_identifier);
        #else  // !GAMEDEV_ROCM
        std::sort(kdtree_neighbors.begin(), kdtree_neighbors.end());
        std::sort(morton_neighbors.begin(), morton_neighbors.end());
        #endif // GAMEDEV_ROCM

        bool query_mismatch = kdtree_neighbors.size() != morton_neighbors.size();
        std::size_t common = std::min(kdtree_neighbors.size(), morton_neighbors.size());
        result.mismatched_neighbors += static_cast<long long>(
            std::max(kdtree_neighbors.size(), morton_neighbors.size()) - common
        );
        for (std::size_t idx = 0; idx < common; idx++)
        {
            float distance_error = std::fabs(kdtree_neighbors[idx].first - morton_neighbors[idx].first);
            result.maximum_distance_error = std::max(result.maximum_distance_error, distance_error);
            float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(kdtree_neighbors[idx].first));
            if (kdtree_neighbors[idx].second != morton_neighbors[idx].second || distance_error > tolerance)
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
    #ifdef GAMEDEV_ROCM

    std::vector<std::pair<float, int>> actual_by_id = actual;
    std::vector<std::pair<float, int>> expected_by_id = expected;
    auto by_identifier = [](const auto &left, const auto &right)
    #else  // !GAMEDEV_ROCM
    for (std::size_t idx = 0; idx < expected.size(); idx++)
    #endif // GAMEDEV_ROCM
    {
        #ifdef GAMEDEV_ROCM
        return left.second < right.second;
    };
    std::sort(actual_by_id.begin(), actual_by_id.end(), by_identifier);
    std::sort(expected_by_id.begin(), expected_by_id.end(), by_identifier);

    for (std::size_t idx = 0; idx < expected_by_id.size(); idx++)
    {
        float distance_error = std::fabs(actual_by_id[idx].first - expected_by_id[idx].first);
        #else  // !GAMEDEV_ROCM
        float distance_error = std::fabs(actual[idx].first - expected[idx].first);
        #endif // GAMEDEV_ROCM
        maximum_distance_error = std::max(maximum_distance_error, distance_error);
        #ifdef GAMEDEV_ROCM
        float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(expected_by_id[idx].first));
        if (actual_by_id[idx].second != expected_by_id[idx].second || distance_error > tolerance) return true;
        #else  // !GAMEDEV_ROCM
        float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(expected[idx].first));
        if (actual[idx].second != expected[idx].second || distance_error > tolerance) return true;
        #endif // GAMEDEV_ROCM
    }
    return false;
}

void report_difference (const char *label, int idx_query,
    const std::vector<std::pair<float, int>> &actual,
    const std::vector<std::pair<float, int>> &expected)
{
    std::vector<std::pair<int, float>> actual_by_idx;
    std::vector<std::pair<int, float>> expected_by_idx;
    for (const auto &neighbor : actual) actual_by_idx.emplace_back(neighbor.second, neighbor.first);
    for (const auto &neighbor : expected) expected_by_idx.emplace_back(neighbor.second, neighbor.first);
    std::sort(actual_by_idx.begin(), actual_by_idx.end());
    std::sort(expected_by_idx.begin(), expected_by_idx.end());

    std::cerr << "  " << label << " query " << idx_query
        << " actual-count=" << actual.size() << " expected-count=" << expected.size();
    int reported = 0;
    std::size_t idx_actual = 0;
    std::size_t idx_expected = 0;
    while (idx_actual < actual_by_idx.size() || idx_expected < expected_by_idx.size())
    {
        if (idx_expected == expected_by_idx.size()
            || (idx_actual < actual_by_idx.size()
                && actual_by_idx[idx_actual].first < expected_by_idx[idx_expected].first))
        {
            if (reported++ < 4) std::cerr << " extra=(" << actual_by_idx[idx_actual].first
                << ',' << actual_by_idx[idx_actual].second << ')';
            idx_actual++;
            continue;
        }
        if (idx_actual == actual_by_idx.size()
            || expected_by_idx[idx_expected].first < actual_by_idx[idx_actual].first)
        {
            if (reported++ < 4) std::cerr << " missing=(" << expected_by_idx[idx_expected].first
                << ',' << expected_by_idx[idx_expected].second << ')';
            idx_expected++;
            continue;
        }
        idx_actual++;
        idx_expected++;
    }
    if (!actual.empty()) std::cerr << " actual-kth=" << actual.back().first;
    if (!expected.empty()) std::cerr << " expected-kth=" << expected.back().first;
    std::cerr << std::endl;
}

std::vector<std::pair<float, int>> get_record_neighbors (
    const std::vector<morton_point> &records, const float3 &query, float radius)
{
    std::vector<std::pair<float, int>> result;
    float radius_sq = radius*radius;
    for (const morton_point &record : records)
    {
        float dx = query.x - record.cartesian.x;
        float dy = query.y - record.cartesian.y;
        float dz = query.z - record.cartesian.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq <= radius_sq) result.emplace_back(dist_sq, record.idx_old);
    }
    std::sort(result.begin(), result.end());
    if (result.size() > K) result.resize(K);
    return result;
}

void diagnose_morton_failure (quality_stats &quality, int idx_query,
    const std::vector<morton_point> &records, const float3 &query,
    const std::vector<std::pair<float, int>> &actual,
    const std::vector<std::pair<float, int>> &physical_brute, float radius)
{
    if (quality.first_failure_query < 0) quality.first_failure_query = idx_query;
    report_difference("Morton", idx_query, actual, physical_brute);
    std::vector<std::pair<float, int>> record = get_record_neighbors(
        records, query, radius
    );
    if (differs_from_brute(actual, record, quality.maximum_distance_error))
    {
        quality.morton_record_mismatches++;
        report_difference("Morton traversal versus stored records", idx_query, actual, record);
    }
    if (differs_from_brute(record, physical_brute, quality.maximum_distance_error))
    {
        quality.record_geometry_mismatches++;
        report_difference("stored records versus physical brute force", idx_query, record, physical_brute);
    }
}

void compare_brute_force (quality_stats &quality, const std::vector<float3> &points,
    const std::vector<morton_point> &records,
    const std::vector<int> &kdtree_idx_old, const std::vector<float> &kdtree_dist_sq,
    const std::vector<int> &morton_idx_old, const std::vector<float> &morton_dist_sq,
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

        std::vector<std::pair<float, int>> kdtree_neighbors = get_array_neighbors(kdtree_idx_old, kdtree_dist_sq,
            idx_query);
        std::vector<std::pair<float, int>> morton_neighbors = get_array_neighbors(
            morton_idx_old, morton_dist_sq, idx_query
        );
        if (differs_from_brute(kdtree_neighbors, expected, quality.maximum_distance_error))
        {
            quality.kdtree_brute_mismatches++;
        }
        if (differs_from_brute(morton_neighbors, expected, quality.maximum_distance_error))
        {
            quality.morton_brute_mismatches++;
            diagnose_morton_failure(
                quality, idx_query, records, query, morton_neighbors, expected, radius
            );
        }
    }
}

// arbitrate backend disagreements with a double-precision brute-force reference
void compare_disagreements_brute_force (quality_stats &quality, const std::vector<float3> &points,
    const std::vector<morton_point> &records,
    const std::vector<int> &kdtree_idx_old, const std::vector<float> &kdtree_dist_sq,
    const std::vector<int> &morton_idx_old, const std::vector<float> &morton_dist_sq,
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

        std::vector<std::pair<float, int>> kdtree_neighbors = get_array_neighbors(kdtree_idx_old, kdtree_dist_sq,
            idx_query);
        std::vector<std::pair<float, int>> morton_neighbors = get_array_neighbors(
            morton_idx_old, morton_dist_sq, idx_query
        );
        if (differs_from_brute(kdtree_neighbors, expected, quality.maximum_distance_error))
        {
            quality.kdtree_disagreement_brute_mismatches++;
        }
        if (differs_from_brute(morton_neighbors, expected, quality.maximum_distance_error))
        {
            quality.morton_disagreement_brute_mismatches++;
            diagnose_morton_failure(
                quality, idx_query, records, query, morton_neighbors, expected, radius
            );
        }
    }
}

void write_json (const options &config, float root_width, int node_count, int leaf_count,
    double kdtree_build_ms, double morton_build_ms, double kdtree_query_ms, double morton_query_ms,
    std::size_t kdtree_bytes, std::size_t morton_bytes, const occupancy_stats &occupancy,
    const quality_stats &quality, double mean_cells, double mean_candidates)
{
    double query_ratio_kd_morton =
        (morton_query_ms > 0.0) ? kdtree_query_ms / morton_query_ms : 0.0;
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
        && quality.kdtree_brute_mismatches == 0 && quality.morton_brute_mismatches == 0
        && quality.kdtree_disagreement_brute_mismatches == 0
        && quality.morton_disagreement_brute_mismatches == 0
        && quality.stack_overflows == 0;

    *output << std::setprecision(10)
        << "{\n"
        << "  \"particles\": " << config.particles << ",\n"
        << "  \"quality_queries\": " << config.queries << ",\n"
        << "  \"brute_force_queries\": " << config.brute_queries << ",\n"
        << "  \"dimension\": " << config.dim << ",\n"
        << "  \"physical_dimension\": " << ((config.distribution == "radial") ? 1 : config.dim) << ",\n"
        << "  \"distribution\": \"" << config.distribution << "\",\n"
        << "  \"k\": " << K << ",\n"
        << "  \"radius\": " << config.radius << ",\n"
        << "  \"quality_only\": " << (config.quality_only ? "true" : "false") << ",\n"
        << "  \"root_width\": " << root_width << ",\n"
        << "  \"leaf_target\": " << config.leaf_target << ",\n"
        << "  \"max_level\": " << config.max_level << ",\n"
        << "  \"max_leaf_scan\": " << config.max_leaf_scan << ",\n"
        << "  \"tree_nodes\": " << node_count << ",\n"
        << "  \"occupied_leaves\": " << leaf_count << ",\n"
        << "  \"quality_passed\": " << (quality_passed ? "true" : "false") << ",\n"
        << "  \"passed\": " << (quality_passed ? "true" : "false") << ",\n"
        << "  \"mismatched_queries\": " << quality.mismatched_queries << ",\n"
        << "  \"mismatched_neighbors\": " << quality.mismatched_neighbors << ",\n"
        << "  \"kd_brute_mismatches\": " << quality.kdtree_brute_mismatches << ",\n"
        << "  \"morton_brute_mismatches\": " << quality.morton_brute_mismatches << ",\n"
        << "  \"disagreement_queries_checked\": " << quality.disagreement_queries_checked << ",\n"
        << "  \"kd_disagreement_brute_mismatches\": "
        << quality.kdtree_disagreement_brute_mismatches << ",\n"
        << "  \"morton_disagreement_brute_mismatches\": "
        << quality.morton_disagreement_brute_mismatches << ",\n"
        << "  \"morton_record_mismatches\": " << quality.morton_record_mismatches << ",\n"
        << "  \"record_geometry_mismatches\": " << quality.record_geometry_mismatches << ",\n"
        << "  \"first_failure_query\": " << quality.first_failure_query << ",\n"
        << "  \"stack_overflows\": " << quality.stack_overflows << ",\n"
        << "  \"maximum_distance_error\": " << quality.maximum_distance_error << ",\n"
        << "  \"kd_build_ms\": " << kdtree_build_ms << ",\n"
        << "  \"morton_build_ms\": " << morton_build_ms << ",\n"
        << "  \"kd_query_ms\": " << kdtree_query_ms << ",\n"
        << "  \"morton_query_ms\": " << morton_query_ms << ",\n"
        << "  \"query_time_ratio_kd_morton\": " << query_ratio_kd_morton << ",\n"
        << "  \"kd_persistent_bytes\": " << kdtree_bytes << ",\n"
        << "  \"morton_persistent_bytes\": " << morton_bytes << ",\n"
        << "  \"memory_ratio_morton_kd\": "
        << static_cast<double>(morton_bytes) / kdtree_bytes << ",\n"
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
        float3 root_origin;
        float root_width;
        get_root(points, config.dim, root_origin, root_width);

        float3 *dev_point = nullptr;
        _morton_gpu_check(gpuMalloc((void**)&dev_point, sizeof(float3)*points.size()), "allocate benchmark points");
        _morton_gpu_check(gpuMemcpy(dev_point, points.data(), sizeof(float3)*points.size(), gpuMemcpyHostToDevice),
            "copy benchmark points");

        // build both indexes from the same physical records and include allocation in build timing
        kdtree_point *dev_kdtree_node = nullptr;
        kdtree_boxf *dev_kdtree_box = nullptr;
        double kdtree_build_ms = wall_time_ms([&]
        {
            _morton_gpu_check(gpuMalloc((void**)&dev_kdtree_node, sizeof(kdtree_point)*points.size()),
            "allocate KD tree");
            _morton_gpu_check(gpuMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)), "allocate KD bounds");
            int block_count = (config.particles + KDTREE_TPB - 1) / KDTREE_TPB;
            kdtree_point_init <<< block_count, KDTREE_TPB >>> (dev_kdtree_node, dev_point, config.particles);
            _morton_gpu_check(gpuGetLastError(), "launch kdtree_point_init");
            kdtree::buildTree<kdtree_point, kdtree_traits>(dev_kdtree_node, config.particles, dev_kdtree_box);
        });

        morton_index morton_owner;
        double morton_build_ms = wall_time_ms([&]
        {
            morton_owner.build(
                dev_point, config.particles, root_origin, root_width,
                config.dim, config.leaf_target, config.max_level
            );
        });
        occupancy_stats occupancy = get_occupancy(morton_owner);
        if (occupancy.maximum > config.max_leaf_scan)
        {
            throw std::runtime_error(
                "adaptive Morton maximum leaf occupancy " + std::to_string(occupancy.maximum)
                + " exceeds --max-leaf-scan=" + std::to_string(config.max_leaf_scan)
                + "; increase --max-level or inspect coincident particles"
            );
        }

        // retain complete neighbor lists for correctness comparison and traversal diagnostics
        std::size_t quality_size = static_cast<std::size_t>(config.queries)*K;
        int *dev_kdtree_idx_old = nullptr;
        int *dev_morton_idx_old = nullptr;
        float *dev_kdtree_dist_sq = nullptr;
        float *dev_morton_dist_sq = nullptr;
        unsigned int *dev_leaf_visit_count = nullptr;
        unsigned int *dev_candidate_count = nullptr;
        unsigned int *dev_quality_stack_overflow = nullptr;
        _morton_gpu_check(gpuMalloc((void**)&dev_kdtree_idx_old, sizeof(int)*quality_size),
            "allocate KD quality indices");
        _morton_gpu_check(gpuMalloc((void**)&dev_morton_idx_old, sizeof(int)*quality_size),
            "allocate Morton quality indices");
        _morton_gpu_check(gpuMalloc((void**)&dev_kdtree_dist_sq, sizeof(float)*quality_size),
            "allocate KD quality distances");
        _morton_gpu_check(gpuMalloc((void**)&dev_morton_dist_sq, sizeof(float)*quality_size),
            "allocate Morton quality distances");
        _morton_gpu_check(gpuMalloc((void**)&dev_leaf_visit_count, sizeof(unsigned int)*config.queries),
            "allocate cell visits");
        _morton_gpu_check(gpuMalloc((void**)&dev_candidate_count, sizeof(unsigned int)*config.queries),
            "allocate candidate visits");
        _morton_gpu_check(gpuMalloc((void**)&dev_quality_stack_overflow, sizeof(unsigned int)*config.queries),
            "allocate quality stack-overflow flags");

        kdtree_query<K> <<< (config.queries + kdtree_heap<K>::threads - 1)
            / kdtree_heap<K>::threads, kdtree_heap<K>::threads >>> (
            dev_kdtree_idx_old, dev_kdtree_dist_sq, dev_point, config.queries,
            dev_kdtree_node, dev_kdtree_box, config.particles, config.radius
        );
        morton_search<K, MORTON_TPB> <<< config.queries, MORTON_TPB >>> (
            dev_morton_idx_old, dev_morton_dist_sq, dev_leaf_visit_count, dev_candidate_count,
            dev_quality_stack_overflow,
            dev_point, config.queries, morton_owner.view(), config.radius
        );
        _morton_gpu_check(gpuDeviceSynchronize(), "run KNN quality queries");

        std::vector<int> kdtree_idx_old(quality_size);
        std::vector<int> morton_idx_old(quality_size);
        std::vector<float> kdtree_dist_sq(quality_size);
        std::vector<float> morton_dist_sq(quality_size);
        std::vector<unsigned int> leaf_visit_count(config.queries);
        std::vector<unsigned int> candidate_count(config.queries);
        std::vector<unsigned int> quality_stack_overflow(config.queries);
        _morton_gpu_check(gpuMemcpy(kdtree_idx_old.data(), dev_kdtree_idx_old, sizeof(int)*quality_size,
            gpuMemcpyDeviceToHost),
            "copy KD quality indices");
        _morton_gpu_check(gpuMemcpy(morton_idx_old.data(), dev_morton_idx_old, sizeof(int)*quality_size,
            gpuMemcpyDeviceToHost),
            "copy Morton quality indices");
        _morton_gpu_check(gpuMemcpy(kdtree_dist_sq.data(), dev_kdtree_dist_sq, sizeof(float)*quality_size,
            gpuMemcpyDeviceToHost),
            "copy KD quality distances");
        _morton_gpu_check(gpuMemcpy(morton_dist_sq.data(), dev_morton_dist_sq, sizeof(float)*quality_size,
            gpuMemcpyDeviceToHost),
            "copy Morton quality distances");
        _morton_gpu_check(gpuMemcpy(leaf_visit_count.data(), dev_leaf_visit_count, sizeof(unsigned int)*config.queries,
            gpuMemcpyDeviceToHost), "copy cell visits");
        _morton_gpu_check(gpuMemcpy(candidate_count.data(), dev_candidate_count,
            sizeof(unsigned int)*config.queries, gpuMemcpyDeviceToHost), "copy candidate visits");
        _morton_gpu_check(gpuMemcpy(quality_stack_overflow.data(), dev_quality_stack_overflow,
            sizeof(unsigned int)*config.queries, gpuMemcpyDeviceToHost), "copy quality stack-overflow flags");
        morton_view morton_data = morton_owner.view();
        std::vector<morton_point> morton_records(morton_data.point_count);
        _morton_gpu_check(gpuMemcpy(morton_records.data(), morton_data.dev_point,
            sizeof(morton_point)*morton_data.point_count, gpuMemcpyDeviceToHost),
            "copy Morton records for quality diagnosis");

        std::vector<int> disagreement_queries;
        // use brute force on a fixed prefix and on every backend disagreement
        quality_stats quality = compare_neighbors(
            kdtree_idx_old, kdtree_dist_sq, morton_idx_old, morton_dist_sq, config.queries, disagreement_queries
        );
        compare_brute_force(
            quality, points, morton_records, kdtree_idx_old, kdtree_dist_sq, morton_idx_old, morton_dist_sq,
            config.brute_queries, config.radius
        );
        compare_disagreements_brute_force(
            quality, points, morton_records, kdtree_idx_old, kdtree_dist_sq, morton_idx_old, morton_dist_sq,
            disagreement_queries, config.radius
        );
        for (unsigned int overflow : quality_stack_overflow)
        {
            quality.stack_overflows += overflow;
        }

        double *dev_kdtree_checksum = nullptr;
        double *dev_morton_checksum = nullptr;
        unsigned int *dev_performance_stack_overflow = nullptr;
        double kdtree_query_ms = 0.0;
        double morton_query_ms = 0.0;
        if (!config.quality_only)
        {
            // time all-particle queries through checksums to avoid output-transfer cost
            _morton_gpu_check(gpuMalloc((void**)&dev_kdtree_checksum, sizeof(double)*config.particles),
                "allocate KD checksum");
            _morton_gpu_check(gpuMalloc((void**)&dev_morton_checksum, sizeof(double)*config.particles),
                "allocate Morton checksum");
            _morton_gpu_check(gpuMalloc((void**)&dev_performance_stack_overflow,
                sizeof(unsigned int)*config.particles), "allocate performance stack-overflow flags");

            kdtree_query_ms = kernel_time_ms([&]
            {
                kdtree_checksum<K> <<< (config.particles + kdtree_heap<K>::threads - 1)
                / kdtree_heap<K>::threads, kdtree_heap<K>::threads >>> (
                    dev_kdtree_checksum, dev_point, config.particles,
                    dev_kdtree_node, dev_kdtree_box, config.particles, config.radius
                );
            }, config.repeats);
            morton_query_ms = kernel_time_ms([&]
            {
                morton_digest<K, MORTON_TPB> <<< config.particles, MORTON_TPB >>> (
                    dev_morton_checksum, dev_performance_stack_overflow,
                    dev_point, config.particles, morton_owner.view(), config.radius
                );
            }, config.repeats);

            std::vector<unsigned int> performance_stack_overflow(config.particles);
            _morton_gpu_check(gpuMemcpy(performance_stack_overflow.data(), dev_performance_stack_overflow,
                sizeof(unsigned int)*config.particles, gpuMemcpyDeviceToHost),
                "copy performance stack-overflow flags");
            for (unsigned int overflow : performance_stack_overflow)
            {
                quality.stack_overflows += overflow;
            }
        }
        double mean_cells = 0.0;
        double mean_candidates = 0.0;
        for (int idx = 0; idx < config.queries; idx++)
        {
            mean_cells += leaf_visit_count[idx];
            mean_candidates += candidate_count[idx];
        }
        mean_cells /= config.queries;
        mean_candidates /= config.queries;

        std::size_t kdtree_bytes = sizeof(kdtree_point)*static_cast<std::size_t>(config.particles)
            + sizeof(kdtree_boxf);
        write_json(
            config, root_width, morton_owner.node_count(), morton_owner.leaf_count(), kdtree_build_ms, morton_build_ms,
            kdtree_query_ms, morton_query_ms, kdtree_bytes, morton_owner.persistent_bytes(), occupancy,
            quality, mean_cells, mean_candidates
        );

        #ifdef GAMEDEV_ROCM
        if (dev_kdtree_checksum)
        {
            _morton_gpu_check(gpuFree(dev_kdtree_checksum), "release KD checksum");
        }
        if (dev_morton_checksum)
        {
            _morton_gpu_check(gpuFree(dev_morton_checksum), "release Morton checksum");
        }
        if (dev_performance_stack_overflow)
        {
            _morton_gpu_check(gpuFree(dev_performance_stack_overflow), "release performance overflow flags");
        }
        _morton_gpu_check(gpuFree(dev_kdtree_idx_old), "release KD neighbor identifiers");
        _morton_gpu_check(gpuFree(dev_morton_idx_old), "release Morton neighbor identifiers");
        _morton_gpu_check(gpuFree(dev_kdtree_dist_sq), "release KD neighbor distances");
        _morton_gpu_check(gpuFree(dev_morton_dist_sq), "release Morton neighbor distances");
        _morton_gpu_check(gpuFree(dev_leaf_visit_count), "release leaf counters");
        _morton_gpu_check(gpuFree(dev_candidate_count), "release candidate counters");
        _morton_gpu_check(gpuFree(dev_quality_stack_overflow), "release quality overflow flags");
        _morton_gpu_check(gpuFree(dev_kdtree_node), "release KD nodes");
        _morton_gpu_check(gpuFree(dev_kdtree_box), "release KD bounds");
        _morton_gpu_check(gpuFree(dev_point), "release benchmark points");
        #else  // !GAMEDEV_ROCM
        if (dev_kdtree_checksum) gpuFree(dev_kdtree_checksum);
        if (dev_morton_checksum) gpuFree(dev_morton_checksum);
        if (dev_performance_stack_overflow) gpuFree(dev_performance_stack_overflow);
        gpuFree(dev_kdtree_idx_old);
        gpuFree(dev_morton_idx_old);
        gpuFree(dev_kdtree_dist_sq);
        gpuFree(dev_morton_dist_sq);
        gpuFree(dev_leaf_visit_count);
        gpuFree(dev_candidate_count);
        gpuFree(dev_quality_stack_overflow);
        gpuFree(dev_kdtree_node);
        gpuFree(dev_kdtree_box);
        gpuFree(dev_point);
        #endif // GAMEDEV_ROCM

        bool quality_passed = quality.mismatched_queries == 0
            && quality.kdtree_brute_mismatches == 0 && quality.morton_brute_mismatches == 0
            && quality.kdtree_disagreement_brute_mismatches == 0
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
