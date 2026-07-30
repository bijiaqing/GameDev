#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <morton/morton_index.cuh>

namespace
{

constexpr int K = 3;
constexpr int QUERY_THREADS = 256;

struct edge_case
{
    std::string name;
    int dimension;
    float radius;
    int leaf_target;
    std::vector<float3> points;
    std::vector<float3> queries;
};

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

    float extent_x = upper.x - lower.x;
    float extent_y = upper.y - lower.y;
    float extent_z = (dimension == 2) ? 0.0f : upper.z - lower.z;
    width = 1.0001f*std::max({extent_x, extent_y, extent_z});
    if (width <= 0.0f) width = 1.0f;

    float center_x = 0.5f*(lower.x + upper.x);
    float center_y = 0.5f*(lower.y + upper.y);
    float center_z = (dimension == 2) ? 0.0f : 0.5f*(lower.z + upper.z);
    origin = make_float3(
        center_x - 0.5f*width,
        center_y - 0.5f*width,
        center_z - 0.5f*width
    );
}

std::vector<std::pair<float, int>> brute_neighbors (
    const std::vector<float3> &points, const float3 &query, float radius)
{
    float radius_sq = radius*radius;
    std::vector<std::pair<float, int>> result;
    for (std::size_t idx = 0; idx < points.size(); idx++)
    {
        float dx = query.x - points[idx].x;
        float dy = query.y - points[idx].y;
        float dz = query.z - points[idx].z;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq <= radius_sq) result.emplace_back(dist_sq, static_cast<int>(idx));
    }
    std::sort(result.begin(), result.end());
    if (result.size() > K) result.resize(K);
    return result;
}

void append_anchors (std::vector<float3> &points, int dimension)
{
    points.push_back(make_float3(-2.0f, -2.0f, (dimension == 2) ? 0.0f : -2.0f));
    points.push_back(make_float3( 2.0f,  2.0f, (dimension == 2) ? 0.0f :  2.0f));
}

edge_case make_ties (int dimension)
{
    edge_case test{"equal_distance_" + std::to_string(dimension) + "d", dimension, 1.0f, 2, {}, {}};
    test.points = {
        make_float3( 1.0f,  0.0f, 0.0f),
        make_float3(-1.0f,  0.0f, 0.0f),
        make_float3( 0.0f,  1.0f, 0.0f),
        make_float3( 0.0f, -1.0f, 0.0f),
    };
    if (dimension == 3)
    {
        test.points.push_back(make_float3(0.0f, 0.0f,  1.0f));
        test.points.push_back(make_float3(0.0f, 0.0f, -1.0f));
    }
    append_anchors(test.points, dimension);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_duplicates (int dimension)
{
    edge_case test{"coincident_" + std::to_string(dimension) + "d", dimension, 0.25f, 128, {}, {}};

    // More than one CUDA block of coincident candidates exercises chunked leaf buffering
    for (int idx = 0; idx < 300; idx++)
    {
        test.points.push_back(make_float3(0.0f, 0.0f, 0.0f));
    }
    append_anchors(test.points, dimension);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_radius_boundary (int dimension)
{
    edge_case test{"radius_boundary_" + std::to_string(dimension) + "d", dimension, 1.0f, 2, {}, {}};
    test.points = {
        make_float3( 1.0f, 0.0f, 0.0f),
        make_float3(-1.0f, 0.0f, 0.0f),
        make_float3( 0.0f, 1.0f, 0.0f),
        make_float3(std::nextafter(1.0f, 2.0f), 0.0f, 0.0f),
    };
    append_anchors(test.points, dimension);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_sparse (int dimension)
{
    edge_case test{"fewer_than_k_" + std::to_string(dimension) + "d", dimension, 0.75f, 2, {}, {}};
    test.points = {
        make_float3(0.25f, 0.0f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f),
    };
    append_anchors(test.points, dimension);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_split_planes (int dimension)
{
    edge_case test{"split_planes_" + std::to_string(dimension) + "d", dimension, 0.9f, 4, {}, {}};
    const float values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    for (float x : values)
    {
        for (float y : values)
        {
            if (dimension == 2)
            {
                test.points.push_back(make_float3(x, y, 0.0f));
                continue;
            }
            for (float z : values)
            {
                test.points.push_back(make_float3(x, y, z));
            }
        }
    }
    test.queries = {
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.25f, -0.25f, (dimension == 2) ? 0.0f : 0.25f),
    };
    return test;
}

bool run_case (const edge_case &test)
{
    float3 origin;
    float root_width;
    get_root(test.points, test.dimension, origin, root_width);

    float3 *dev_points = nullptr;
    float3 *dev_queries = nullptr;
    int *dev_indices = nullptr;
    float *dev_distances = nullptr;
    unsigned int *dev_overflows = nullptr;
    std::size_t output_count = test.queries.size()*K;

    _morton_cuda_check(cudaMalloc((void**)&dev_points, sizeof(float3)*test.points.size()), "allocate edge points");
    _morton_cuda_check(cudaMalloc((void**)&dev_queries, sizeof(float3)*test.queries.size()), "allocate edge queries");
    _morton_cuda_check(cudaMalloc((void**)&dev_indices, sizeof(int)*output_count), "allocate edge indices");
    _morton_cuda_check(cudaMalloc((void**)&dev_distances, sizeof(float)*output_count), "allocate edge distances");
    _morton_cuda_check(cudaMalloc((void**)&dev_overflows, sizeof(unsigned int)*test.queries.size()),
        "allocate edge overflow flags");
    _morton_cuda_check(cudaMemcpy(dev_points, test.points.data(), sizeof(float3)*test.points.size(),
        cudaMemcpyHostToDevice), "copy edge points");
    _morton_cuda_check(cudaMemcpy(dev_queries, test.queries.data(), sizeof(float3)*test.queries.size(),
        cudaMemcpyHostToDevice), "copy edge queries");

    morton_index index;
    index.build(
        dev_points, static_cast<int>(test.points.size()), origin, root_width,
        test.dimension, test.leaf_target, 20
    );
    morton_query<K> <<< static_cast<int>(test.queries.size()), QUERY_THREADS >>> (
        dev_indices, dev_distances, nullptr, nullptr, dev_overflows,
        dev_queries, static_cast<int>(test.queries.size()), index.view(), test.radius
    );
    _morton_cuda_check(cudaDeviceSynchronize(), "run edge queries");

    std::vector<int> indices(output_count);
    std::vector<float> distances(output_count);
    std::vector<unsigned int> overflows(test.queries.size());
    _morton_cuda_check(cudaMemcpy(indices.data(), dev_indices, sizeof(int)*output_count, cudaMemcpyDeviceToHost),
        "copy edge indices");
    _morton_cuda_check(cudaMemcpy(distances.data(), dev_distances, sizeof(float)*output_count,
        cudaMemcpyDeviceToHost), "copy edge distances");
    _morton_cuda_check(cudaMemcpy(overflows.data(), dev_overflows, sizeof(unsigned int)*test.queries.size(),
        cudaMemcpyDeviceToHost), "copy edge overflow flags");

    bool passed = true;
    for (std::size_t idx_query = 0; idx_query < test.queries.size(); idx_query++)
    {
        std::vector<std::pair<float, int>> expected = brute_neighbors(
            test.points, test.queries[idx_query], test.radius
        );
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            std::size_t idx_out = idx_query*K + idx_neighbor;
            int expected_idx = (idx_neighbor < static_cast<int>(expected.size()))
                ? expected[idx_neighbor].second : -1;
            float expected_dist = (idx_neighbor < static_cast<int>(expected.size()))
                ? expected[idx_neighbor].first : std::numeric_limits<float>::infinity();
            float error = std::fabs(distances[idx_out] - expected_dist);
            bool distance_matches = (std::isinf(distances[idx_out]) && std::isinf(expected_dist))
                || error <= 2.0e-6f*std::max(1.0f, std::fabs(expected_dist));
            if (indices[idx_out] == expected_idx && distance_matches) continue;

            passed = false;
            std::cerr << "  query " << idx_query << " neighbor " << idx_neighbor
                << " expected=(" << expected_idx << ',' << expected_dist << ")"
                << " actual=(" << indices[idx_out] << ',' << distances[idx_out] << ')' << std::endl;
        }
        if (overflows[idx_query] != 0)
        {
            passed = false;
            std::cerr << "  query " << idx_query << " overflowed the traversal stack" << std::endl;
        }
    }

    cudaFree(dev_overflows);
    cudaFree(dev_distances);
    cudaFree(dev_indices);
    cudaFree(dev_queries);
    cudaFree(dev_points);

    std::cout << (passed ? "PASS  " : "FAIL  ") << test.name << std::endl;
    return passed;
}

} // namespace

int main ()
{
    try
    {
        std::vector<edge_case> tests;
        for (int dimension : {2, 3})
        {
            tests.push_back(make_ties(dimension));
            tests.push_back(make_duplicates(dimension));
            tests.push_back(make_radius_boundary(dimension));
            tests.push_back(make_sparse(dimension));
            tests.push_back(make_split_planes(dimension));
        }

        int failed = 0;
        for (const edge_case &test : tests)
        {
            if (!run_case(test)) failed++;
        }
        std::cout << "edge tests: " << tests.size() - failed << '/' << tests.size() << " passed" << std::endl;
        return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "edge-test failure: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
