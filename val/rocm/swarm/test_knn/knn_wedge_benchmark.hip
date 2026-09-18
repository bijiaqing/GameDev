#include <algorithm>      // std::min, std::max, std::sort
#include <chrono>         // std::chrono timing
#include <cmath>          // std::cos, std::sin, std::sqrt
#include <cstddef>        // std::size_t
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

#include <hip/hip_runtime.h> // HIP allocation, events, copies, and kernel launches

#include <kdtree/builder.h>
#include <kdtree/knn.h>

#include <morton/morton_ghost.cuh>
#include <morton/morton_query.cuh>

#include "knn_types.cuh"

// compare periodic-wedge KD-tree images with query-image and compact-ghost Morton searches
// stress ordinary, interior-clump, and seam-clump distributions in two and three dimensions

#ifndef VAL_KNN_K
#define VAL_KNN_K 200
#endif

namespace
{

constexpr int K = VAL_KNN_K;
constexpr int MORTON_TPB = 64;
using kdtree_boxf = kdtree::box_t<float3>;

struct options
{
    int particles = 100000;
    int queries = 4096;
    int brute_queries = 32;
    int repeats = 5;
    int dim = 2;
    int seed = 17;
    int leaf_target = 128;
    int max_level = 20;
    float radius = 0.1f;
    float x_min = -0.25f*static_cast<float>(M_PI);
    float x_max =  0.25f*static_cast<float>(M_PI);
    std::string distribution = "smooth";
    std::string output;
};

__global__
void kdtree_wedge_init (kdtree_point *dev_kdtree_node, const float3 *dev_source_point,
    int point_count, float width)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= 3*point_count) return;

    int idx_old = idx_tree % point_count;
    int image = idx_tree / point_count;
    float3 point = dev_source_point[idx_old];
    if (image != 0)
    {
        float angle = (image == 1) ? -width : width;
        float sin_angle;
        float cos_angle;
        sincosf(angle, &sin_angle, &cos_angle);
        point = make_float3(
            cos_angle*point.x - sin_angle*point.y,
            sin_angle*point.x + cos_angle*point.y,
            point.z
        );
    }
    dev_kdtree_node[idx_tree].cartesian = point;
    dev_kdtree_node[idx_tree].idx_old = idx_old;
    dev_kdtree_node[idx_tree].split_dim = 0;
    dev_kdtree_node[idx_tree].image = image;
}

template<int TOP_K>
__global__
void morton_wedge_query (int *dev_near_idx_old, float *dev_near_dist_sq, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data,
    float search_dist, bool unique_ids)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[1024];
    __shared__ int work_idx_old[1024];
    __shared__ int idx_node_stack[256];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;

    _morton_ghost_topk<TOP_K, MORTON_TPB, 1024, 256>(
        morton_data, dev_query_point[idx_query], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < TOP_K; idx_neighbor += blockDim.x)
    {
        int idx_out = idx_query*TOP_K + idx_neighbor;
        dev_near_idx_old[idx_out] = (work_idx_old[idx_neighbor] == INT_MAX)
            ? -1 : work_idx_old[idx_neighbor];
        dev_near_dist_sq[idx_out] = work_dist_sq[idx_neighbor];
    }
    if (threadIdx.x == 0) dev_stack_overflow[idx_query] = stack_overflow;
}

template<int TOP_K>
__global__
void morton_wedge_checksum (double *dev_checksum, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data,
    float search_dist, bool unique_ids)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[1024];
    __shared__ int work_idx_old[1024];
    __shared__ int idx_node_stack[256];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_visit_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int stack_overflow;

    _morton_ghost_topk<TOP_K, MORTON_TPB, 1024, 256>(
        morton_data, dev_query_point[idx_query], search_dist, unique_ids,
        work_dist_sq, work_idx_old, idx_node_stack, stack_count, idx_node, batch_count,
        leaf_visit_count, candidate_count, stack_overflow
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
        {
            if (work_idx_old[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(work_dist_sq[idx_neighbor])
                + 1.0e-12*static_cast<double>(_get_col_idx_old(work_idx_old[idx_neighbor]));
        }
        dev_checksum[idx_query] = value;
        dev_stack_overflow[idx_query] = stack_overflow;
    }
}

template<int TOP_K>
__global__
void kdtree_wedge_query (int *dev_near_idx_old, float *dev_near_dist_sq,
    const float3 *dev_query_point, int query_count,
    const kdtree_point *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    int node_count, float search_dist, bool dedup_needed)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kdtree_heap<TOP_K> near_result(search_dist, dev_kdtree_node, dedup_needed);
    kdtree::cct::knn<kdtree_heap<TOP_K>, kdtree_point, kdtree_traits>(
        near_result, dev_query_point[idx_query], *dev_kdtree_box, dev_kdtree_node, node_count
    );
    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_old = near_result.returnIndex(idx_neighbor);
        int idx_out = idx_query*TOP_K + idx_neighbor;
        dev_near_idx_old[idx_out] = (idx_old < 0) ? -1 : near_result.returnNeighbor(idx_neighbor);
        dev_near_dist_sq[idx_out] = (idx_old < 0) ? MORTON_INF_F : near_result.returnDist2(idx_neighbor);
    }
}

template<int TOP_K>
__global__
void kdtree_wedge_checksum (double *dev_checksum, const float3 *dev_query_point, int query_count,
    const kdtree_point *dev_kdtree_node, const kdtree_boxf *dev_kdtree_box,
    int node_count, float search_dist, bool dedup_needed)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kdtree_heap<TOP_K> near_result(search_dist, dev_kdtree_node, dedup_needed);
    kdtree::cct::knn<kdtree_heap<TOP_K>, kdtree_point, kdtree_traits>(
        near_result, dev_query_point[idx_query], *dev_kdtree_box, dev_kdtree_node, node_count
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
        else if (argument == "--brute-queries") result.brute_queries = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--repeat") result.repeats = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--dim") result.dim = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--seed") result.seed = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--leaf-target") result.leaf_target = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--max-level") result.max_level = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--radius") result.radius = std::stof(next_argument(argc, argv, idx));
        else if (argument == "--x-min") result.x_min = std::stof(next_argument(argc, argv, idx));
        else if (argument == "--x-max") result.x_max = std::stof(next_argument(argc, argv, idx));
        else if (argument == "--distribution") result.distribution = next_argument(argc, argv, idx);
        else if (argument == "--output") result.output = next_argument(argc, argv, idx);
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (result.particles < K || result.queries <= 0 || result.repeats <= 0 || result.brute_queries < 0)
        throw std::invalid_argument("invalid benchmark sizes");
    if (result.dim != 2 && result.dim != 3) throw std::invalid_argument("--dim must be 2 or 3");
    if (result.x_max <= result.x_min
        || result.x_max - result.x_min >= 6.28318530717958647692f - 1.0e-6f)
        throw std::invalid_argument("wedge must satisfy 0 < x_max-x_min < 2*pi-1e-6");
    if (result.distribution != "smooth" && result.distribution != "ring"
        && result.distribution != "interior_clump" && result.distribution != "seam_clump")
        throw std::invalid_argument("unknown wedge distribution");
    result.queries = std::min(result.queries, result.particles);
    result.brute_queries = std::min(result.brute_queries, result.queries);
    return result;
}

// generate production-like disk distributions and retain azimuth for seam-image selection
void generate_points (const options &config, std::vector<float3> &points, std::vector<float> &azimuth)
{
    std::mt19937 generator(config.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    points.resize(config.particles);
    azimuth.resize(config.particles);
    float width = config.x_max - config.x_min;

    for (int idx = 0; idx < config.particles; idx++)
    {
        float x = config.x_min + width*uniform(generator);
        float R = std::sqrt(0.25f + 2.0f*uniform(generator));
        float Z = (config.dim == 2) ? 0.0f : 0.05f*normal(generator);
        bool clump = (config.distribution == "interior_clump" || config.distribution == "seam_clump")
            && uniform(generator) < 0.8f;
        if (config.distribution == "ring") R = 1.0f + 0.03f*normal(generator);
        if (clump)
        {
            R = 1.0f + 0.01f*normal(generator);
            Z = (config.dim == 2) ? 0.0f : 0.01f*normal(generator);
            if (config.distribution == "interior_clump") x = 0.01f*normal(generator);
            else
            {
                float offset = std::fabs(0.01f*normal(generator));
                x = (uniform(generator) < 0.5f) ? config.x_min + offset : config.x_max - offset;
                x = std::min(config.x_max, std::max(config.x_min, x));
            }
        }
        azimuth[idx] = x;
        points[idx] = make_float3(R*std::cos(x), R*std::sin(x), Z);
    }

    if (width >= 6.0f)
    {
        // isolate the wide-wedge eligibility counterexample from the sampled disk
        constexpr float radius = 0.25f;
        float x_owner = 0.5f*(config.x_min + config.x_max);
        float x_partner = x_owner + 0.06f;
        azimuth[0] = x_owner;
        azimuth[1] = x_partner;
        points[0] = make_float3(radius*std::cos(x_owner), radius*std::sin(x_owner), 0.0f);
        points[1] = make_float3(radius*std::cos(x_partner), radius*std::sin(x_partner), 0.0f);
    }
}

template<typename Function>
double wall_time_ms (Function operation)
{
    auto time_start = std::chrono::steady_clock::now();
    operation();
    _morton_gpu_check(hipDeviceSynchronize(), "synchronize build");
    auto time_stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(time_stop - time_start).count();
}

template<typename Function>
double kernel_time_ms (Function operation, int repeats)
{
    operation();
    _morton_gpu_check(hipDeviceSynchronize(), "synchronize warmup");
    hipEvent_t event_start;
    hipEvent_t event_stop;
    _morton_gpu_check(hipEventCreate(&event_start), "create timing event");
    _morton_gpu_check(hipEventCreate(&event_stop), "create timing event");
    _morton_gpu_check(hipEventRecord(event_start), "record timing start");
    for (int repeat = 0; repeat < repeats; repeat++)
    {
        operation();
    }
    _morton_gpu_check(hipEventRecord(event_stop), "record timing end");
    _morton_gpu_check(hipEventSynchronize(event_stop), "synchronize timing end");
    float elapsed_ms = 0.0f;
    _morton_gpu_check(hipEventElapsedTime(&elapsed_ms, event_start, event_stop), "read timing");
    _morton_gpu_check(hipEventDestroy(event_start), "destroy benchmark start event");
    _morton_gpu_check(hipEventDestroy(event_stop), "destroy benchmark end event");
    return static_cast<double>(elapsed_ms) / repeats;
}

std::vector<std::pair<float, int>> get_neighbors (
    const std::vector<int> &near_idx_old, const std::vector<float> &near_dist_sq, int idx_query)
{
    std::vector<std::pair<float, int>> result;
    for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
    {
        int idx = idx_query*K + idx_neighbor;
        if (near_idx_old[idx] >= 0)
            result.emplace_back(near_dist_sq[idx], _get_col_idx_old(near_idx_old[idx]));
    }
    std::sort(result.begin(), result.end());
    return result;
}

int count_image_geometry_mismatches (const std::vector<int> &neighbor,
    const std::vector<float> &dist_sq, const std::vector<float3> &point,
    const std::vector<float3> &query, int query_count, float width)
{
    int mismatches = 0;
    for (int idx_query = 0; idx_query < query_count; idx_query++)
    {
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            int idx_out = idx_query*K + idx_neighbor;
            if (neighbor[idx_out] < 0) continue;
            int idx_old = _get_col_idx_old(neighbor[idx_out]);
            int image = _get_col_image(neighbor[idx_out]);
            if (idx_old < 0 || idx_old >= static_cast<int>(point.size())
                || image < 0 || image >= COL_IMAGE_COUNT)
            {
                mismatches++;
                continue;
            }
            float angle = static_cast<float>(_get_col_image_shift(image))*width;
            float sin_angle = std::sin(angle);
            float cos_angle = std::cos(angle);
            float3 image_point = make_float3(
                cos_angle*point[idx_old].x - sin_angle*point[idx_old].y,
                sin_angle*point[idx_old].x + cos_angle*point[idx_old].y,
                point[idx_old].z
            );
            float dx = query[idx_query].x - image_point.x;
            float dy = query[idx_query].y - image_point.y;
            float dz = query[idx_query].z - image_point.z;
            float expected = dx*dx + dy*dy + dz*dz;
            float tolerance = 2.0e-6f*std::max(1.0f, std::fabs(expected));
            if (std::fabs(dist_sq[idx_out] - expected) > tolerance) mismatches++;
        }
    }
    return mismatches;
}

bool lists_differ (const std::vector<std::pair<float, int>> &a,
    const std::vector<std::pair<float, int>> &b, float &maximum_error)
{
    if (a.size() != b.size()) return true;
    for (std::size_t idx = 0; idx < a.size(); idx++)
    {
        float error = std::fabs(a[idx].first - b[idx].first);
        maximum_error = std::max(maximum_error, error);
        if (a[idx].second != b[idx].second || error > 2.0e-6f) return true;
    }
    return false;
}

bool differs_from_brute (const std::vector<std::pair<float, int>> &actual,
    const std::vector<std::pair<float, int>> &expected, float &maximum_error,
    int &tie_equivalent_neighbors)
{
    if (actual.size() != expected.size()) return true;

    std::vector<std::pair<int, float>> actual_by_idx;
    std::vector<std::pair<int, float>> expected_by_idx;
    actual_by_idx.reserve(actual.size());
    expected_by_idx.reserve(expected.size());
    for (const auto &neighbor : actual)
    {
        actual_by_idx.emplace_back(neighbor.second, neighbor.first);
    }
    for (const auto &neighbor : expected)
    {
        expected_by_idx.emplace_back(neighbor.second, neighbor.first);
    }
    std::sort(actual_by_idx.begin(), actual_by_idx.end());
    std::sort(expected_by_idx.begin(), expected_by_idx.end());

    bool same_indices = true;
    for (std::size_t idx = 0; idx < actual_by_idx.size(); idx++)
    {
        if (actual_by_idx[idx].first == expected_by_idx[idx].first) continue;
        same_indices = false;
        break;
    }
    if (same_indices)
    {
        for (std::size_t idx = 0; idx < actual_by_idx.size(); idx++)
        {
            float error = std::fabs(actual_by_idx[idx].second - expected_by_idx[idx].second);
            maximum_error = std::max(maximum_error, error);
            if (error > 2.0e-6f) return true;
        }
        return false;
    }

    for (std::size_t idx = 0; idx < actual.size(); idx++)
    {
        float error = std::fabs(actual[idx].first - expected[idx].first);
        maximum_error = std::max(maximum_error, error);
        if (error > 2.0e-6f) return true;
        if (actual[idx].second == expected[idx].second) continue;

        // host and GPU rotations may reverse candidates whose squared distances differ below
        // single-precision resolution; record these substitutions without treating them as errors
        if (error <= 1.0e-7f)
        {
            tie_equivalent_neighbors++;
            continue;
        }
        return true;
    }
    return false;
}

void report_brute_difference (const char *backend, int idx_query,
    const std::vector<std::pair<float, int>> &actual,
    const std::vector<std::pair<float, int>> &expected)
{
    std::vector<std::pair<int, float>> actual_by_idx;
    std::vector<std::pair<int, float>> expected_by_idx;
    for (const auto &neighbor : actual)
    {
        actual_by_idx.emplace_back(neighbor.second, neighbor.first);
    }
    for (const auto &neighbor : expected)
    {
        expected_by_idx.emplace_back(neighbor.second, neighbor.first);
    }
    std::sort(actual_by_idx.begin(), actual_by_idx.end());
    std::sort(expected_by_idx.begin(), expected_by_idx.end());

    std::cerr << "  " << backend << " query " << idx_query
        << " actual-count=" << actual.size() << " expected-count=" << expected.size();
    if (actual_by_idx.size() != expected_by_idx.size())
    {
        std::cerr << " count mismatch" << std::endl;
        return;
    }

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
    if (reported > 0)
    {
        if (!actual.empty()) std::cerr << " actual-kth=" << actual.back().first;
        if (!expected.empty()) std::cerr << " expected-kth=" << expected.back().first;
        std::cerr << std::endl;
        return;
    }

    float max_error = 0.0f;
    int idx_error = -1;
    for (std::size_t idx = 0; idx < actual_by_idx.size(); idx++)
    {
        float error = std::fabs(actual_by_idx[idx].second - expected_by_idx[idx].second);
        if (error <= max_error) continue;
        max_error = error;
        idx_error = static_cast<int>(idx);
    }
    std::cerr << " distance mismatch idx_old=" << actual_by_idx[idx_error].first
        << " actual=" << actual_by_idx[idx_error].second
        << " expected=" << expected_by_idx[idx_error].second
        << " error=" << max_error << std::endl;
}

float seam_distance (const float3 &point, float x_offset)
{
    float R = std::hypot(point.x, point.y);
    return (std::cos(x_offset) >= 0.0f) ? R*std::fabs(std::sin(x_offset)) : R;
}

float3 rotate_point (const float3 &point, float angle)
{
    float sin_angle = std::sin(angle);
    float cos_angle = std::cos(angle);
    return make_float3(
        cos_angle*point.x - sin_angle*point.y,
        sin_angle*point.x + cos_angle*point.y,
        point.z
    );
}

float distance_sq (const float3 &a, const float3 &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// construct the physical top-K set admitted by one search representation
std::vector<std::pair<float, int>> candidate_neighbors (
    const std::vector<float3> &points, const std::vector<float> &azimuth,
    const float3 &query, float x_min, float x_max, float radius, bool morton_eligibility)
{
    std::vector<std::pair<float, int>> result;
    float radius_sq = radius*radius;
    float width = x_max - x_min;
    for (std::size_t idx = 0; idx < points.size(); idx++)
    {
        float best = distance_sq(query, points[idx]);
        bool lower_eligible = !morton_eligibility
            || seam_distance(points[idx], azimuth[idx] - x_min) <= radius;
        bool upper_eligible = !morton_eligibility
            || seam_distance(points[idx], x_max - azimuth[idx]) <= radius;
        if (lower_eligible)
            best = std::min(best, distance_sq(query, rotate_point(points[idx], width)));
        if (upper_eligible)
            best = std::min(best, distance_sq(query, rotate_point(points[idx], -width)));
        if (best <= radius_sq) result.emplace_back(best, static_cast<int>(idx));
    }
    std::sort(result.begin(), result.end());
    if (result.size() > K) result.resize(K);
    return result;
}

} // namespace

int main (int argc, char **argv)
{
    try
    {
        options config = parse_options(argc, argv);
        std::vector<float3> points;
        std::vector<float> azimuth;
        generate_points(config, points, azimuth);
        float width = config.x_max - config.x_min;

        float3 *dev_point = nullptr;
        float *dev_azimuth = nullptr;
        _morton_gpu_check(hipMalloc((void**)&dev_point, sizeof(float3)*config.particles), "allocate wedge points");
        _morton_gpu_check(hipMalloc((void**)&dev_azimuth, sizeof(float)*config.particles), "allocate wedge azimuths");
        _morton_gpu_check(hipMemcpy(dev_point, points.data(), sizeof(float3)*config.particles,
            hipMemcpyHostToDevice), "copy wedge points");
        _morton_gpu_check(hipMemcpy(dev_azimuth, azimuth.data(), sizeof(float)*config.particles,
            hipMemcpyHostToDevice), "copy wedge azimuths");

        // compare the three-copy KD representation with the compact Morton ghost representation
        int kdtree_node_count = 3*config.particles;
        kdtree_point *dev_kdtree_node = nullptr;
        kdtree_boxf *dev_kdtree_box = nullptr;
        double kdtree_build_ms = wall_time_ms([&]
        {
            _morton_gpu_check(hipMalloc((void**)&dev_kdtree_node, sizeof(kdtree_point)*kdtree_node_count), "allocate wedge KD tree");
            _morton_gpu_check(hipMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)), "allocate wedge KD bounds");
            int block_count = (kdtree_node_count + 255) / 256;
            kdtree_wedge_init <<< block_count, 256 >>> (dev_kdtree_node, dev_point, config.particles, width);
            kdtree::buildTree<kdtree_point, kdtree_traits>(dev_kdtree_node, kdtree_node_count, dev_kdtree_box);
        });

        // at the equatorial inner edge, the image separation reduces to sin(wedge_width/2)
        float image_dist_min = std::fabs(std::sin(0.5f*(config.x_max - config.x_min)));
        bool unique_ids = image_dist_min > 2.0f*config.radius;
        morton_ghost_index morton_owner;
        double morton_build_ms = wall_time_ms([&]
        {
            morton_owner.build(
                dev_point, dev_azimuth, config.particles, config.radius,
                config.x_min, config.x_max, 1.5f, true, unique_ids,
                config.dim, config.leaf_target, config.max_level
            );
        });
        bool kdtree_dedup_needed = image_dist_min <= 2.0f*config.radius;
        if (kdtree_dedup_needed == morton_owner.unique_ids())
        {
            throw std::runtime_error("KD-tree and Morton periodic-image dedup decisions disagree");
        }

        // retain complete physical-id neighbor lists for backend and brute-force comparison
        std::size_t quality_count = static_cast<std::size_t>(config.queries)*K;
        int *dev_kdtree_idx_old = nullptr;
        int *dev_morton_idx_old = nullptr;
        float *dev_kdtree_dist_sq = nullptr;
        float *dev_morton_dist_sq = nullptr;
        unsigned int *dev_quality_stack_overflow = nullptr;
        _morton_gpu_check(hipMalloc((void**)&dev_kdtree_idx_old, sizeof(int)*quality_count), "allocate KD quality indices");
        _morton_gpu_check(hipMalloc((void**)&dev_morton_idx_old, sizeof(int)*quality_count), "allocate Morton quality indices");
        _morton_gpu_check(hipMalloc((void**)&dev_kdtree_dist_sq, sizeof(float)*quality_count), "allocate KD quality distances");
        _morton_gpu_check(hipMalloc((void**)&dev_morton_dist_sq, sizeof(float)*quality_count), "allocate Morton quality distances");
        _morton_gpu_check(hipMalloc((void**)&dev_quality_stack_overflow, sizeof(unsigned int)*config.queries),
            "allocate quality overflows");

        kdtree_wedge_query<K> <<< (config.queries + kdtree_heap<K>::threads - 1)/kdtree_heap<K>::threads, kdtree_heap<K>::threads >>> (
            dev_kdtree_idx_old, dev_kdtree_dist_sq, dev_point, config.queries,
            dev_kdtree_node, dev_kdtree_box, kdtree_node_count, config.radius, kdtree_dedup_needed
        );
        morton_wedge_query<K> <<< config.queries, MORTON_TPB >>> (
            dev_morton_idx_old, dev_morton_dist_sq, dev_quality_stack_overflow,
            dev_point, config.queries, morton_owner.view(), config.radius,
            morton_owner.unique_ids()
        );
        _morton_gpu_check(hipDeviceSynchronize(), "run wedge quality queries");

        std::vector<int> kdtree_idx_old(quality_count);
        std::vector<int> morton_idx_old(quality_count);
        std::vector<float> kdtree_dist_sq(quality_count);
        std::vector<float> morton_dist_sq(quality_count);
        std::vector<unsigned int> quality_stack_overflow(config.queries);
        _morton_gpu_check(hipMemcpy(kdtree_idx_old.data(), dev_kdtree_idx_old, sizeof(int)*quality_count,
            hipMemcpyDeviceToHost), "copy KD quality indices");
        _morton_gpu_check(hipMemcpy(morton_idx_old.data(), dev_morton_idx_old, sizeof(int)*quality_count,
            hipMemcpyDeviceToHost), "copy Morton quality indices");
        _morton_gpu_check(hipMemcpy(kdtree_dist_sq.data(), dev_kdtree_dist_sq, sizeof(float)*quality_count,
            hipMemcpyDeviceToHost), "copy KD quality distances");
        _morton_gpu_check(hipMemcpy(morton_dist_sq.data(), dev_morton_dist_sq, sizeof(float)*quality_count,
            hipMemcpyDeviceToHost), "copy Morton quality distances");
        _morton_gpu_check(hipMemcpy(quality_stack_overflow.data(), dev_quality_stack_overflow,
            sizeof(unsigned int)*config.queries, hipMemcpyDeviceToHost), "copy quality overflows");

        int mismatched_queries = 0;
        int kdtree_reference_mismatches = 0;
        int morton_reference_mismatches = 0;
        int disagreement_queries_checked = 0;
        int kdtree_tie_equivalent_neighbors = 0;
        int morton_tie_equivalent_neighbors = 0;
        int kdtree_image_geometry_mismatches = count_image_geometry_mismatches(
            kdtree_idx_old, kdtree_dist_sq, points, points, config.queries, width
        );
        int morton_image_geometry_mismatches = count_image_geometry_mismatches(
            morton_idx_old, morton_dist_sq, points, points, config.queries, width
        );
        bool wide_counterexample_distinction = false;
        float maximum_error = 0.0f;
        // arbitrate every backend disagreement and a fixed query prefix against
        // the exact candidate set admitted by that backend's representation
        for (int idx_query = 0; idx_query < config.queries; idx_query++)
        {
            std::vector<std::pair<float, int>> kdtree_neighbors =
                get_neighbors(kdtree_idx_old, kdtree_dist_sq, idx_query);
            std::vector<std::pair<float, int>> morton_neighbors =
                get_neighbors(morton_idx_old, morton_dist_sq, idx_query);
            bool morton_differs = lists_differ(kdtree_neighbors, morton_neighbors, maximum_error);
            if (morton_differs) mismatched_queries++;
            if (idx_query == 0 && morton_differs) wide_counterexample_distinction = true;
            if (idx_query >= config.brute_queries && !morton_differs) continue;
            if (morton_differs) disagreement_queries_checked++;

            std::vector<std::pair<float, int>> kdtree_reference = candidate_neighbors(
                points, azimuth, points[idx_query], config.x_min, config.x_max,
                config.radius, false
            );
            std::vector<std::pair<float, int>> morton_reference = candidate_neighbors(
                points, azimuth, points[idx_query], config.x_min, config.x_max,
                config.radius, true
            );
            if (differs_from_brute(
                kdtree_neighbors, kdtree_reference, maximum_error,
                kdtree_tie_equivalent_neighbors
            ))
            {
                kdtree_reference_mismatches++;
                report_brute_difference("KD", idx_query, kdtree_neighbors, kdtree_reference);
            }
            if (differs_from_brute(
                morton_neighbors, morton_reference, maximum_error,
                morton_tie_equivalent_neighbors
            ))
            {
                morton_reference_mismatches++;
                report_brute_difference("Morton", idx_query, morton_neighbors, morton_reference);
            }
        }

        double *dev_kdtree_checksum = nullptr;
        double *dev_morton_checksum = nullptr;
        unsigned int *dev_performance_stack_overflow = nullptr;
        _morton_gpu_check(hipMalloc((void**)&dev_kdtree_checksum, sizeof(double)*config.particles),
            "allocate KD checksums");
        _morton_gpu_check(hipMalloc((void**)&dev_morton_checksum, sizeof(double)*config.particles),
            "allocate Morton checksums");
        _morton_gpu_check(hipMalloc((void**)&dev_performance_stack_overflow, sizeof(unsigned int)*config.particles),
            "allocate performance overflows");

        // time all-particle searches through checksums after correctness has been established
        double kdtree_query_ms = kernel_time_ms([&]
        {
            kdtree_wedge_checksum<K> <<< (config.particles + kdtree_heap<K>::threads - 1)/kdtree_heap<K>::threads, kdtree_heap<K>::threads >>> (
                dev_kdtree_checksum, dev_point, config.particles,
                dev_kdtree_node, dev_kdtree_box, kdtree_node_count, config.radius, kdtree_dedup_needed
            );
        }, config.repeats);
        double morton_query_ms = kernel_time_ms([&]
        {
            morton_wedge_checksum<K> <<< config.particles, MORTON_TPB >>> (
                dev_morton_checksum, dev_performance_stack_overflow,
                dev_point, config.particles, morton_owner.view(), config.radius,
                morton_owner.unique_ids()
            );
        }, config.repeats);

        std::vector<unsigned int> performance_stack_overflow(config.particles);
        _morton_gpu_check(hipMemcpy(performance_stack_overflow.data(), dev_performance_stack_overflow,
            sizeof(unsigned int)*config.particles, hipMemcpyDeviceToHost), "copy performance overflows");
        unsigned long long stack_overflows = 0;
        for (int idx = 0; idx < config.particles; idx++)
        {
            stack_overflows += performance_stack_overflow[idx];
        }
        // permit independently built trees to exchange distance-equivalent cutoff neighbors
        bool wide_candidate_distinction = width < 6.0f || wide_counterexample_distinction;
        bool quality_passed = kdtree_reference_mismatches == 0 && morton_reference_mismatches == 0
            && kdtree_image_geometry_mismatches == 0
            && morton_image_geometry_mismatches == 0 && stack_overflows == 0
            && wide_candidate_distinction;

        std::ostream *output = &std::cout;
        std::ofstream file;
        if (!config.output.empty())
        {
            std::filesystem::path path(config.output);
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            file.open(path);
            if (!file) throw std::runtime_error("cannot open output: " + config.output);
            output = &file;
        }
        std::size_t kdtree_bytes = sizeof(kdtree_point)*static_cast<std::size_t>(kdtree_node_count) + sizeof(kdtree_boxf);
        *output << std::setprecision(10)
            << "{\n"
            << "  \"particles\": " << config.particles << ",\n"
            << "  \"dimension\": " << config.dim << ",\n"
            << "  \"distribution\": \"" << config.distribution << "\",\n"
            << "  \"x_min\": " << config.x_min << ",\n"
            << "  \"x_max\": " << config.x_max << ",\n"
            << "  \"kd_deduplicate\": " << (kdtree_dedup_needed ? "true" : "false") << ",\n"
            << "  \"duplicate_safe\": " << (morton_owner.unique_ids() ? "true" : "false") << ",\n"
            << "  \"quality_passed\": " << (quality_passed ? "true" : "false") << ",\n"
            << "  \"passed\": " << (quality_passed ? "true" : "false") << ",\n"
            << "  \"mismatched_queries\": " << mismatched_queries << ",\n"
            << "  \"kd_reference_mismatches\": " << kdtree_reference_mismatches << ",\n"
            << "  \"morton_reference_mismatches\": " << morton_reference_mismatches << ",\n"
            << "  \"reference_queries_checked\": " << config.brute_queries << ",\n"
            << "  \"disagreement_queries_checked\": " << disagreement_queries_checked << ",\n"
            << "  \"kd_tie_equivalent_neighbors\": " << kdtree_tie_equivalent_neighbors << ",\n"
            << "  \"morton_tie_equivalent_neighbors\": " << morton_tie_equivalent_neighbors << ",\n"
            << "  \"wide_candidate_distinction\": "
            << (wide_candidate_distinction ? "true" : "false") << ",\n"
            << "  \"kd_image_geometry_mismatches\": " << kdtree_image_geometry_mismatches << ",\n"
            << "  \"morton_image_geometry_mismatches\": " << morton_image_geometry_mismatches << ",\n"
            << "  \"stack_overflows\": " << stack_overflows << ",\n"
            << "  \"maximum_distance_error\": " << maximum_error << ",\n"
            << "  \"morton_records\": " << morton_owner.record_count() << ",\n"
            << "  \"record_ratio_morton_per_particle\": "
            << static_cast<double>(morton_owner.record_count()) / config.particles << ",\n"
            << "  \"kd_build_ms\": " << kdtree_build_ms << ",\n"
            << "  \"morton_build_ms\": " << morton_build_ms << ",\n"
            << "  \"kd_query_ms\": " << kdtree_query_ms << ",\n"
            << "  \"morton_query_ms\": " << morton_query_ms << ",\n"
            << "  \"query_time_ratio_kd_morton\": "
            << kdtree_query_ms / morton_query_ms << ",\n"
            << "  \"kd_persistent_bytes\": " << kdtree_bytes << ",\n"
            << "  \"morton_persistent_bytes\": " << morton_owner.persistent_bytes() << ",\n"
            << "  \"memory_ratio_morton_kd\": "
            << static_cast<double>(morton_owner.persistent_bytes()) / kdtree_bytes << "\n"
            << "}\n";

        _morton_gpu_check(hipFree(dev_performance_stack_overflow), "release performance overflow flags");
        _morton_gpu_check(hipFree(dev_morton_checksum), "release Morton checksum");
        _morton_gpu_check(hipFree(dev_kdtree_checksum), "release KD checksum");
        _morton_gpu_check(hipFree(dev_quality_stack_overflow), "release quality overflow flags");
        _morton_gpu_check(hipFree(dev_morton_dist_sq), "release Morton neighbor distances");
        _morton_gpu_check(hipFree(dev_kdtree_dist_sq), "release KD neighbor distances");
        _morton_gpu_check(hipFree(dev_morton_idx_old), "release Morton neighbor identifiers");
        _morton_gpu_check(hipFree(dev_kdtree_idx_old), "release KD neighbor identifiers");
        _morton_gpu_check(hipFree(dev_kdtree_box), "release KD bounds");
        _morton_gpu_check(hipFree(dev_kdtree_node), "release KD nodes");
        _morton_gpu_check(hipFree(dev_azimuth), "release particle azimuths");
        _morton_gpu_check(hipFree(dev_point), "release benchmark points");
        return quality_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "wedge-benchmark failure: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
