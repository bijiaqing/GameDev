#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

#include <morton/morton_ghost.cuh>
#include <morton/morton_query.cuh>

#include "knn_types.cuh"
#include "periodic_query.cuh"

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
    int brute_queries = 32;
    int repeats = 5;
    int dimension = 2;
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
void kd_wedge_init (kd_point *tree, const float3 *source, int point_count, float width)
{
    int idx_tree = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_tree >= 3*point_count) return;

    int idx_old = idx_tree % point_count;
    int image = idx_tree / point_count;
    float3 point = source[idx_old];
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
    tree[idx_tree].cartesian = point;
    tree[idx_tree].index_old = idx_old;
    tree[idx_tree].split_dim = 0;
    tree[idx_tree].image = image;
}

template<int TOP_K>
__global__
void ghost_morton_query (int *neighbor_idx, float *neighbor_dist, unsigned int *stack_overflows,
    const float3 *queries, int query_count, morton_view view, float radius, bool duplicate_safe)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist[1024];
    __shared__ int work_idx[1024];
    __shared__ int node_stack[256];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;

    _morton_ghost_topk<TOP_K, 256, 1024, 256>(
        view, queries[idx_query], radius, duplicate_safe,
        work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < TOP_K; idx_neighbor += blockDim.x)
    {
        int idx_out = idx_query*TOP_K + idx_neighbor;
        neighbor_idx[idx_out] = (work_idx[idx_neighbor] == INT_MAX) ? -1 : work_idx[idx_neighbor];
        neighbor_dist[idx_out] = work_dist[idx_neighbor];
    }
    if (threadIdx.x == 0) stack_overflows[idx_query] = stack_overflow;
}

template<int TOP_K>
__global__
void ghost_morton_checksum (double *checksum, unsigned int *stack_overflows,
    const float3 *queries, int query_count, morton_view view, float radius, bool duplicate_safe)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist[1024];
    __shared__ int work_idx[1024];
    __shared__ int node_stack[256];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;

    _morton_ghost_topk<TOP_K, 256, 1024, 256>(
        view, queries[idx_query], radius, duplicate_safe,
        work_dist, work_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
        {
            if (work_idx[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(work_dist[idx_neighbor])
                + 1.0e-12*static_cast<double>(work_idx[idx_neighbor]);
        }
        checksum[idx_query] = value;
        stack_overflows[idx_query] = stack_overflow;
    }
}

template<int TOP_K>
__global__
void kd_wedge_query (int *neighbor_idx, float *neighbor_dist, const float3 *queries, int query_count,
    const kd_point *tree, const kd_box *bounds, int tree_count, float radius)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kd_heap<TOP_K, true> result(radius, tree);
    kdtree::cct::knn<kd_heap<TOP_K, true>, kd_point, kd_traits>(
        result, queries[idx_query], *bounds, tree, tree_count
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
void kd_wedge_checksum (double *checksum, const float3 *queries, int query_count,
    const kd_point *tree, const kd_box *bounds, int tree_count, float radius)
{
    int idx_query = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_query >= query_count) return;

    kd_heap<TOP_K, true> result(radius, tree);
    kdtree::cct::knn<kd_heap<TOP_K, true>, kd_point, kd_traits>(
        result, queries[idx_query], *bounds, tree, tree_count
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
        else if (argument == "--brute-queries") result.brute_queries = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--repeat") result.repeats = std::stoi(next_argument(argc, argv, idx));
        else if (argument == "--dim") result.dimension = std::stoi(next_argument(argc, argv, idx));
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
    if (result.dimension != 2 && result.dimension != 3) throw std::invalid_argument("--dim must be 2 or 3");
    if (result.x_max <= result.x_min || result.x_max - result.x_min >= 2.0f*static_cast<float>(M_PI))
        throw std::invalid_argument("wedge must satisfy 0 < x_max-x_min < 2*pi");
    if (result.distribution != "smooth" && result.distribution != "ring"
        && result.distribution != "interior_clump" && result.distribution != "seam_clump")
        throw std::invalid_argument("unknown wedge distribution");
    result.queries = std::min(result.queries, result.particles);
    result.brute_queries = std::min(result.brute_queries, result.queries);
    return result;
}

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
        float Z = (config.dimension == 2) ? 0.0f : 0.05f*normal(generator);
        bool clump = (config.distribution == "interior_clump" || config.distribution == "seam_clump")
            && uniform(generator) < 0.8f;
        if (config.distribution == "ring") R = 1.0f + 0.03f*normal(generator);
        if (clump)
        {
            R = 1.0f + 0.01f*normal(generator);
            Z = (config.dimension == 2) ? 0.0f : 0.01f*normal(generator);
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
}

void get_root (const std::vector<float3> &points, int dimension, float3 &origin, float &width)
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
    float extent_z = (dimension == 2) ? 0.0f : upper.z - lower.z;
    width = 1.0001f*std::max({upper.x - lower.x, upper.y - lower.y, extent_z});
    origin = make_float3(
        0.5f*(lower.x + upper.x - width),
        0.5f*(lower.y + upper.y - width),
        (dimension == 2) ? -0.5f*width : 0.5f*(lower.z + upper.z - width)
    );
}

template<typename Function>
double wall_time_ms (Function operation)
{
    auto begin = std::chrono::steady_clock::now();
    operation();
    _morton_cuda_check(cudaDeviceSynchronize(), "synchronize build");
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template<typename Function>
double kernel_time_ms (Function operation, int repeats)
{
    operation();
    _morton_cuda_check(cudaDeviceSynchronize(), "synchronize warmup");
    cudaEvent_t begin;
    cudaEvent_t end;
    _morton_cuda_check(cudaEventCreate(&begin), "create timing event");
    _morton_cuda_check(cudaEventCreate(&end), "create timing event");
    _morton_cuda_check(cudaEventRecord(begin), "record timing start");
    for (int repeat = 0; repeat < repeats; repeat++)
    {
        operation();
    }
    _morton_cuda_check(cudaEventRecord(end), "record timing end");
    _morton_cuda_check(cudaEventSynchronize(end), "synchronize timing end");
    float elapsed = 0.0f;
    _morton_cuda_check(cudaEventElapsedTime(&elapsed, begin, end), "read timing");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return static_cast<double>(elapsed) / repeats;
}

std::vector<std::pair<float, int>> get_neighbors (
    const std::vector<int> &indices, const std::vector<float> &distances, int idx_query)
{
    std::vector<std::pair<float, int>> result;
    for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
    {
        int idx = idx_query*K + idx_neighbor;
        if (indices[idx] >= 0) result.emplace_back(distances[idx], indices[idx]);
    }
    std::sort(result.begin(), result.end());
    return result;
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
    for (std::size_t idx = 0; idx < actual.size(); idx++)
    {
        float error = std::fabs(actual[idx].first - expected[idx].first);
        maximum_error = std::max(maximum_error, error);
        if (error > 2.0e-6f) return true;
        if (actual[idx].second == expected[idx].second) continue;

        // Host and GPU rotations may reverse candidates whose squared distances differ below
        // single-precision resolution; record these substitutions without treating them as errors.
        if (error <= 1.0e-7f)
        {
            tie_equivalent_neighbors++;
            continue;
        }
        return true;
    }
    return false;
}

std::vector<std::pair<float, int>> brute_neighbors (
    const std::vector<float3> &points, const float3 &query, float width, float radius)
{
    float sin_width = std::sin(width);
    float cos_width = std::cos(width);
    std::vector<std::pair<float, int>> result;
    float radius_sq = radius*radius;
    for (std::size_t idx = 0; idx < points.size(); idx++)
    {
        float best = std::numeric_limits<float>::infinity();
        for (int image = -1; image <= 1; image++)
        {
            float3 query_image = query;
            if (image != 0)
            {
                float sin_angle = image*sin_width;
                query_image = make_float3(
                    cos_width*query.x - sin_angle*query.y,
                    sin_angle*query.x + cos_width*query.y,
                    query.z
                );
            }
            float dx = query_image.x - points[idx].x;
            float dy = query_image.y - points[idx].y;
            float dz = query_image.z - points[idx].z;
            best = std::min(best, dx*dx + dy*dy + dz*dz);
        }
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
        float3 origin;
        float root_width;
        get_root(points, config.dimension, origin, root_width);
        float width = config.x_max - config.x_min;

        float3 *dev_points = nullptr;
        float *dev_azimuth = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_points, sizeof(float3)*config.particles), "allocate wedge points");
        _morton_cuda_check(cudaMalloc((void**)&dev_azimuth, sizeof(float)*config.particles), "allocate wedge azimuths");
        _morton_cuda_check(cudaMemcpy(dev_points, points.data(), sizeof(float3)*config.particles,
            cudaMemcpyHostToDevice), "copy wedge points");
        _morton_cuda_check(cudaMemcpy(dev_azimuth, azimuth.data(), sizeof(float)*config.particles,
            cudaMemcpyHostToDevice), "copy wedge azimuths");

        int tree_count = 3*config.particles;
        kd_point *dev_kd_tree = nullptr;
        kd_box *dev_kd_bounds = nullptr;
        double kd_build_ms = wall_time_ms([&]
        {
            _morton_cuda_check(cudaMalloc((void**)&dev_kd_tree, sizeof(kd_point)*tree_count), "allocate wedge KD tree");
            _morton_cuda_check(cudaMalloc((void**)&dev_kd_bounds, sizeof(kd_box)), "allocate wedge KD bounds");
            int blocks = (tree_count + 255) / 256;
            kd_wedge_init <<< blocks, 256 >>> (dev_kd_tree, dev_points, config.particles, width);
            kdtree::buildTree<kd_point, kd_traits>(dev_kd_tree, tree_count, dev_kd_bounds);
        });

        morton_index morton;
        double morton_build_ms = wall_time_ms([&]
        {
            morton.build(dev_points, config.particles, origin, root_width,
                config.dimension, config.leaf_target, config.max_level);
        });

        morton_ghost_index ghost_morton;
        double ghost_build_ms = wall_time_ms([&]
        {
            ghost_morton.build(
                dev_points, dev_azimuth, config.particles, config.radius,
                config.x_min, config.x_max, 0.5f, 1.5f,
                0.5f*static_cast<float>(M_PI), 0.5f*static_cast<float>(M_PI),
                true, config.dimension, config.leaf_target, config.max_level
            );
        });

        std::size_t quality_count = static_cast<std::size_t>(config.queries)*K;
        int *dev_kd_idx = nullptr;
        int *dev_morton_idx = nullptr;
        int *dev_ghost_idx = nullptr;
        float *dev_kd_dist = nullptr;
        float *dev_morton_dist = nullptr;
        float *dev_ghost_dist = nullptr;
        unsigned int *dev_quality_overflow = nullptr;
        unsigned int *dev_ghost_quality_overflow = nullptr;
        unsigned int *dev_quality_images = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_idx, sizeof(int)*quality_count), "allocate KD quality indices");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_idx, sizeof(int)*quality_count), "allocate Morton quality indices");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_idx, sizeof(int)*quality_count),
            "allocate ghost-Morton quality indices");
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_dist, sizeof(float)*quality_count), "allocate KD quality distances");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_dist, sizeof(float)*quality_count), "allocate Morton quality distances");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_dist, sizeof(float)*quality_count),
            "allocate ghost-Morton quality distances");
        _morton_cuda_check(cudaMalloc((void**)&dev_quality_overflow, sizeof(unsigned int)*config.queries),
            "allocate quality overflows");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_quality_overflow, sizeof(unsigned int)*config.queries),
            "allocate ghost-Morton quality overflows");
        _morton_cuda_check(cudaMalloc((void**)&dev_quality_images, sizeof(unsigned int)*config.queries),
            "allocate quality image counts");

        kd_wedge_query<K> <<< (config.queries + KD_THREADS - 1) / KD_THREADS, KD_THREADS >>> (
            dev_kd_idx, dev_kd_dist, dev_points, config.queries,
            dev_kd_tree, dev_kd_bounds, tree_count, config.radius
        );
        periodic_morton_query<K> <<< config.queries, MORTON_THREADS >>> (
            dev_morton_idx, dev_morton_dist, dev_quality_overflow, dev_quality_images,
            dev_points, dev_azimuth, config.queries, morton.view(), config.radius,
            config.x_min, config.x_max
        );
        ghost_morton_query<K> <<< config.queries, MORTON_THREADS >>> (
            dev_ghost_idx, dev_ghost_dist, dev_ghost_quality_overflow,
            dev_points, config.queries, ghost_morton.view(), config.radius,
            ghost_morton.duplicate_safe()
        );
        _morton_cuda_check(cudaDeviceSynchronize(), "run wedge quality queries");

        std::vector<int> kd_idx(quality_count);
        std::vector<int> morton_idx(quality_count);
        std::vector<int> ghost_idx(quality_count);
        std::vector<float> kd_dist(quality_count);
        std::vector<float> morton_dist(quality_count);
        std::vector<float> ghost_dist(quality_count);
        std::vector<unsigned int> quality_overflow(config.queries);
        std::vector<unsigned int> ghost_quality_overflow(config.queries);
        std::vector<unsigned int> quality_images(config.queries);
        _morton_cuda_check(cudaMemcpy(kd_idx.data(), dev_kd_idx, sizeof(int)*quality_count,
            cudaMemcpyDeviceToHost), "copy KD quality indices");
        _morton_cuda_check(cudaMemcpy(morton_idx.data(), dev_morton_idx, sizeof(int)*quality_count,
            cudaMemcpyDeviceToHost), "copy Morton quality indices");
        _morton_cuda_check(cudaMemcpy(ghost_idx.data(), dev_ghost_idx, sizeof(int)*quality_count,
            cudaMemcpyDeviceToHost), "copy ghost-Morton quality indices");
        _morton_cuda_check(cudaMemcpy(kd_dist.data(), dev_kd_dist, sizeof(float)*quality_count,
            cudaMemcpyDeviceToHost), "copy KD quality distances");
        _morton_cuda_check(cudaMemcpy(morton_dist.data(), dev_morton_dist, sizeof(float)*quality_count,
            cudaMemcpyDeviceToHost), "copy Morton quality distances");
        _morton_cuda_check(cudaMemcpy(ghost_dist.data(), dev_ghost_dist, sizeof(float)*quality_count,
            cudaMemcpyDeviceToHost), "copy ghost-Morton quality distances");
        _morton_cuda_check(cudaMemcpy(quality_overflow.data(), dev_quality_overflow,
            sizeof(unsigned int)*config.queries, cudaMemcpyDeviceToHost), "copy quality overflows");
        _morton_cuda_check(cudaMemcpy(ghost_quality_overflow.data(), dev_ghost_quality_overflow,
            sizeof(unsigned int)*config.queries, cudaMemcpyDeviceToHost), "copy ghost-Morton quality overflows");
        _morton_cuda_check(cudaMemcpy(quality_images.data(), dev_quality_images,
            sizeof(unsigned int)*config.queries, cudaMemcpyDeviceToHost), "copy quality image counts");

        int mismatched_queries = 0;
        int ghost_mismatched_queries = 0;
        int kd_brute_mismatches = 0;
        int morton_brute_mismatches = 0;
        int ghost_brute_mismatches = 0;
        int disagreement_queries_checked = 0;
        int kd_disagreement_brute_mismatches = 0;
        int morton_disagreement_brute_mismatches = 0;
        int ghost_disagreement_brute_mismatches = 0;
        int kd_tie_equivalent_neighbors = 0;
        int morton_tie_equivalent_neighbors = 0;
        int ghost_tie_equivalent_neighbors = 0;
        float maximum_error = 0.0f;
        for (int idx_query = 0; idx_query < config.queries; idx_query++)
        {
            std::vector<std::pair<float, int>> kd = get_neighbors(kd_idx, kd_dist, idx_query);
            std::vector<std::pair<float, int>> mt = get_neighbors(morton_idx, morton_dist, idx_query);
            std::vector<std::pair<float, int>> gh = get_neighbors(ghost_idx, ghost_dist, idx_query);
            bool morton_differs = lists_differ(kd, mt, maximum_error);
            bool ghost_differs = lists_differ(kd, gh, maximum_error);
            if (morton_differs) mismatched_queries++;
            if (ghost_differs) ghost_mismatched_queries++;
            if (idx_query >= config.brute_queries && (morton_differs || ghost_differs))
            {
                disagreement_queries_checked++;
                std::vector<std::pair<float, int>> brute = brute_neighbors(
                    points, points[idx_query], width, config.radius
                );
                if (differs_from_brute(kd, brute, maximum_error, kd_tie_equivalent_neighbors))
                    kd_disagreement_brute_mismatches++;
                if (differs_from_brute(mt, brute, maximum_error, morton_tie_equivalent_neighbors))
                    morton_disagreement_brute_mismatches++;
                if (differs_from_brute(gh, brute, maximum_error, ghost_tie_equivalent_neighbors))
                    ghost_disagreement_brute_mismatches++;
            }
            if (idx_query >= config.brute_queries) continue;
            std::vector<std::pair<float, int>> brute = brute_neighbors(
                points, points[idx_query], width, config.radius
            );
            if (differs_from_brute(kd, brute, maximum_error, kd_tie_equivalent_neighbors))
                kd_brute_mismatches++;
            if (differs_from_brute(mt, brute, maximum_error, morton_tie_equivalent_neighbors))
                morton_brute_mismatches++;
            if (differs_from_brute(gh, brute, maximum_error, ghost_tie_equivalent_neighbors))
                ghost_brute_mismatches++;
        }

        double *dev_kd_checksum = nullptr;
        double *dev_morton_checksum = nullptr;
        double *dev_ghost_checksum = nullptr;
        unsigned int *dev_perf_overflow = nullptr;
        unsigned int *dev_perf_images = nullptr;
        unsigned int *dev_ghost_perf_overflow = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_kd_checksum, sizeof(double)*config.particles),
            "allocate KD checksums");
        _morton_cuda_check(cudaMalloc((void**)&dev_morton_checksum, sizeof(double)*config.particles),
            "allocate Morton checksums");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_checksum, sizeof(double)*config.particles),
            "allocate ghost-Morton checksums");
        _morton_cuda_check(cudaMalloc((void**)&dev_perf_overflow, sizeof(unsigned int)*config.particles),
            "allocate performance overflows");
        _morton_cuda_check(cudaMalloc((void**)&dev_perf_images, sizeof(unsigned int)*config.particles),
            "allocate performance image counts");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_perf_overflow, sizeof(unsigned int)*config.particles),
            "allocate ghost-Morton performance overflows");

        double kd_query_ms = kernel_time_ms([&]
        {
            kd_wedge_checksum<K> <<< (config.particles + KD_THREADS - 1) / KD_THREADS, KD_THREADS >>> (
                dev_kd_checksum, dev_points, config.particles,
                dev_kd_tree, dev_kd_bounds, tree_count, config.radius
            );
        }, config.repeats);
        double morton_query_ms = kernel_time_ms([&]
        {
            periodic_morton_checksum<K> <<< config.particles, MORTON_THREADS >>> (
                dev_morton_checksum, dev_perf_overflow, dev_perf_images,
                dev_points, dev_azimuth, config.particles, morton.view(), config.radius,
                config.x_min, config.x_max
            );
        }, config.repeats);
        double ghost_query_ms = kernel_time_ms([&]
        {
            ghost_morton_checksum<K> <<< config.particles, MORTON_THREADS >>> (
                dev_ghost_checksum, dev_ghost_perf_overflow,
                dev_points, config.particles, ghost_morton.view(), config.radius,
                ghost_morton.duplicate_safe()
            );
        }, config.repeats);

        std::vector<unsigned int> perf_overflow(config.particles);
        std::vector<unsigned int> perf_images(config.particles);
        std::vector<unsigned int> ghost_perf_overflow(config.particles);
        _morton_cuda_check(cudaMemcpy(perf_overflow.data(), dev_perf_overflow,
            sizeof(unsigned int)*config.particles, cudaMemcpyDeviceToHost), "copy performance overflows");
        _morton_cuda_check(cudaMemcpy(perf_images.data(), dev_perf_images,
            sizeof(unsigned int)*config.particles, cudaMemcpyDeviceToHost), "copy performance image counts");
        _morton_cuda_check(cudaMemcpy(ghost_perf_overflow.data(), dev_ghost_perf_overflow,
            sizeof(unsigned int)*config.particles, cudaMemcpyDeviceToHost),
            "copy ghost-Morton performance overflows");
        unsigned long long stack_overflows = 0;
        unsigned long long ghost_stack_overflows = 0;
        double mean_images = 0.0;
        int boundary_queries = 0;
        for (int idx = 0; idx < config.particles; idx++)
        {
            stack_overflows += perf_overflow[idx];
            ghost_stack_overflows += ghost_perf_overflow[idx];
            mean_images += perf_images[idx];
            if (perf_images[idx] > 1) boundary_queries++;
        }
        mean_images /= config.particles;
        double boundary_fraction = static_cast<double>(boundary_queries) / config.particles;
        bool quality_passed = mismatched_queries == 0 && kd_brute_mismatches == 0
            && morton_brute_mismatches == 0 && ghost_mismatched_queries == 0
            && ghost_brute_mismatches == 0 && stack_overflows == 0 && ghost_stack_overflows == 0;

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
        std::size_t kd_bytes = sizeof(kd_point)*static_cast<std::size_t>(tree_count) + sizeof(kd_box);
        *output << std::setprecision(10)
            << "{\n"
            << "  \"particles\": " << config.particles << ",\n"
            << "  \"dimension\": " << config.dimension << ",\n"
            << "  \"distribution\": \"" << config.distribution << "\",\n"
            << "  \"x_min\": " << config.x_min << ",\n"
            << "  \"x_max\": " << config.x_max << ",\n"
            << "  \"duplicate_safe\": " << (ghost_morton.duplicate_safe() ? "true" : "false") << ",\n"
            << "  \"quality_passed\": " << (quality_passed ? "true" : "false") << ",\n"
            << "  \"mismatched_queries\": " << mismatched_queries << ",\n"
            << "  \"ghost_mismatched_queries\": " << ghost_mismatched_queries << ",\n"
            << "  \"kd_brute_mismatches\": " << kd_brute_mismatches << ",\n"
            << "  \"morton_brute_mismatches\": " << morton_brute_mismatches << ",\n"
            << "  \"ghost_brute_mismatches\": " << ghost_brute_mismatches << ",\n"
            << "  \"disagreement_queries_checked\": " << disagreement_queries_checked << ",\n"
            << "  \"kd_disagreement_brute_mismatches\": "
            << kd_disagreement_brute_mismatches << ",\n"
            << "  \"morton_disagreement_brute_mismatches\": "
            << morton_disagreement_brute_mismatches << ",\n"
            << "  \"ghost_disagreement_brute_mismatches\": "
            << ghost_disagreement_brute_mismatches << ",\n"
            << "  \"kd_tie_equivalent_neighbors\": " << kd_tie_equivalent_neighbors << ",\n"
            << "  \"morton_tie_equivalent_neighbors\": " << morton_tie_equivalent_neighbors << ",\n"
            << "  \"ghost_tie_equivalent_neighbors\": " << ghost_tie_equivalent_neighbors << ",\n"
            << "  \"stack_overflows\": " << stack_overflows << ",\n"
            << "  \"ghost_stack_overflows\": " << ghost_stack_overflows << ",\n"
            << "  \"maximum_distance_error\": " << maximum_error << ",\n"
            << "  \"mean_query_images\": " << mean_images << ",\n"
            << "  \"boundary_query_fraction\": " << boundary_fraction << ",\n"
            << "  \"ghost_records\": " << ghost_morton.record_count() << ",\n"
            << "  \"ghost_record_ratio\": "
            << static_cast<double>(ghost_morton.record_count()) / config.particles << ",\n"
            << "  \"kd_build_ms\": " << kd_build_ms << ",\n"
            << "  \"morton_build_ms\": " << morton_build_ms << ",\n"
            << "  \"ghost_build_ms\": " << ghost_build_ms << ",\n"
            << "  \"kd_query_ms\": " << kd_query_ms << ",\n"
            << "  \"morton_query_ms\": " << morton_query_ms << ",\n"
            << "  \"ghost_query_ms\": " << ghost_query_ms << ",\n"
            << "  \"query_speedup\": " << kd_query_ms / morton_query_ms << ",\n"
            << "  \"ghost_query_speedup\": " << kd_query_ms / ghost_query_ms << ",\n"
            << "  \"kd_persistent_bytes\": " << kd_bytes << ",\n"
            << "  \"morton_persistent_bytes\": " << morton.persistent_bytes() << ",\n"
            << "  \"ghost_persistent_bytes\": " << ghost_morton.persistent_bytes() << ",\n"
            << "  \"persistent_memory_ratio\": "
            << static_cast<double>(morton.persistent_bytes()) / kd_bytes << ",\n"
            << "  \"ghost_memory_ratio\": "
            << static_cast<double>(ghost_morton.persistent_bytes()) / kd_bytes << "\n"
            << "}\n";

        cudaFree(dev_ghost_perf_overflow);
        cudaFree(dev_ghost_checksum);
        cudaFree(dev_perf_images);
        cudaFree(dev_perf_overflow);
        cudaFree(dev_morton_checksum);
        cudaFree(dev_kd_checksum);
        cudaFree(dev_quality_images);
        cudaFree(dev_ghost_quality_overflow);
        cudaFree(dev_quality_overflow);
        cudaFree(dev_morton_dist);
        cudaFree(dev_ghost_dist);
        cudaFree(dev_kd_dist);
        cudaFree(dev_morton_idx);
        cudaFree(dev_ghost_idx);
        cudaFree(dev_kd_idx);
        cudaFree(dev_kd_bounds);
        cudaFree(dev_kd_tree);
        cudaFree(dev_azimuth);
        cudaFree(dev_points);
        return quality_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "wedge-benchmark failure: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
