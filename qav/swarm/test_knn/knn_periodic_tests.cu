#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "periodic_query.cuh"

namespace
{

constexpr int K = 8;
constexpr int QUERY_THREADS = 256;

struct periodic_case
{
    std::string name;
    int dimension;
    float radius;
    float x_min;
    float x_max;
    std::vector<float3> points;
    std::vector<float3> queries;
    std::vector<float> query_x;
    std::vector<unsigned int> expected_images;
};

float3 make_point (float R, float x, float Z = 0.0f)
{
    return make_float3(R*std::cos(x), R*std::sin(x), Z);
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

float3 rotate_z (const float3 &point, float angle)
{
    float sin_angle = std::sin(angle);
    float cos_angle = std::cos(angle);
    return make_float3(
        cos_angle*point.x - sin_angle*point.y,
        sin_angle*point.x + cos_angle*point.y,
        point.z
    );
}

float get_dist_sq (const float3 &a, const float3 &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

std::vector<std::pair<float, int>> periodic_brute (
    const periodic_case &test, std::size_t idx_query)
{
    float width = test.x_max - test.x_min;
    float radius_sq = test.radius*test.radius;
    bool periodic = width < 2.0f*static_cast<float>(M_PI) - 1.0e-6f;
    std::vector<std::pair<float, int>> result;

    for (std::size_t idx_point = 0; idx_point < test.points.size(); idx_point++)
    {
        float dist_sq = get_dist_sq(test.queries[idx_query], test.points[idx_point]);
        if (periodic)
        {
            dist_sq = std::min(dist_sq, get_dist_sq(
                rotate_z(test.queries[idx_query], width), test.points[idx_point]
            ));
            dist_sq = std::min(dist_sq, get_dist_sq(
                rotate_z(test.queries[idx_query], -width), test.points[idx_point]
            ));
        }
        if (dist_sq <= radius_sq) result.emplace_back(dist_sq, static_cast<int>(idx_point));
    }
    std::sort(result.begin(), result.end());
    if (result.size() > K) result.resize(K);
    return result;
}

std::vector<float3> make_wedge_points (int dimension, float x_min, float x_max)
{
    float Z = (dimension == 2) ? 0.0f : 0.03f;
    return {
        make_point(1.00f, x_min + 0.01f,  Z),
        make_point(1.00f, x_max - 0.01f,  Z),
        make_point(0.98f, x_min + 0.03f, -Z),
        make_point(1.02f, x_max - 0.03f, -Z),
        make_point(1.00f, 0.0f,              Z),
        make_point(1.10f, 0.1f,             -Z),
        make_point(0.90f, -0.1f,             Z),
        make_point(1.30f, 0.0f,             -Z),
        make_point(0.70f, 0.0f,              Z),
    };
}

periodic_case make_lower_seam (int dimension)
{
    float x_min = -0.25f*static_cast<float>(M_PI);
    float x_max =  0.25f*static_cast<float>(M_PI);
    float Z = (dimension == 2) ? 0.0f : 0.03f;
    return {
        "lower_seam_" + std::to_string(dimension) + "d", dimension, 0.09f, x_min, x_max,
        make_wedge_points(dimension, x_min, x_max),
        {make_point(1.0f, x_min + 0.01f, Z)}, {x_min + 0.01f}, {2}
    };
}

periodic_case make_upper_seam (int dimension)
{
    float x_min = -0.25f*static_cast<float>(M_PI);
    float x_max =  0.25f*static_cast<float>(M_PI);
    float Z = (dimension == 2) ? 0.0f : 0.03f;
    return {
        "upper_seam_" + std::to_string(dimension) + "d", dimension, 0.09f, x_min, x_max,
        make_wedge_points(dimension, x_min, x_max),
        {make_point(1.0f, x_max - 0.01f, Z)}, {x_max - 0.01f}, {2}
    };
}

periodic_case make_interior (int dimension)
{
    float x_min = -0.25f*static_cast<float>(M_PI);
    float x_max =  0.25f*static_cast<float>(M_PI);
    float Z = (dimension == 2) ? 0.0f : 0.03f;
    return {
        "interior_" + std::to_string(dimension) + "d", dimension, 0.05f, x_min, x_max,
        make_wedge_points(dimension, x_min, x_max),
        {make_point(1.0f, 0.0f, Z)}, {0.0f}, {1}
    };
}

periodic_case make_narrow_wedge (int dimension)
{
    float x_min = -0.1f;
    float x_max =  0.1f;
    float Z = (dimension == 2) ? 0.0f : 0.02f;
    std::vector<float3> points;
    for (int idx = 0; idx < 16; idx++)
    {
        float fraction = static_cast<float>(idx) / 15.0f;
        float x = x_min + (x_max - x_min)*fraction;
        points.push_back(make_point(0.98f + 0.002f*idx, x, (idx % 2 == 0) ? Z : -Z));
    }
    return {
        "narrow_dedup_" + std::to_string(dimension) + "d", dimension, 0.22f, x_min, x_max,
        points, {make_point(1.0f, 0.0f, Z)}, {0.0f}, {3}
    };
}

periodic_case make_full_disk (int dimension)
{
    float x_min = -static_cast<float>(M_PI);
    float x_max =  static_cast<float>(M_PI);
    float Z = (dimension == 2) ? 0.0f : 0.02f;
    std::vector<float3> points = {
        make_point(1.0f, x_min + 0.01f, Z),
        make_point(1.0f, x_max - 0.01f, Z),
        make_point(1.0f, 0.0f, Z),
        make_point(0.8f, 1.0f, -Z),
        make_point(1.2f, -1.0f, -Z),
    };
    return {
        "full_disk_" + std::to_string(dimension) + "d", dimension, 0.1f, x_min, x_max,
        points, {make_point(1.0f, x_min + 0.01f, Z)}, {x_min + 0.01f}, {1}
    };
}

bool run_case (const periodic_case &test)
{
    float3 origin;
    float root_width;
    get_root(test.points, test.dimension, origin, root_width);

    float3 *dev_points = nullptr;
    float3 *dev_queries = nullptr;
    float *dev_query_x = nullptr;
    int *dev_indices = nullptr;
    float *dev_distances = nullptr;
    unsigned int *dev_overflows = nullptr;
    unsigned int *dev_image_counts = nullptr;
    std::size_t output_count = test.queries.size()*K;

    _morton_cuda_check(cudaMalloc((void**)&dev_points, sizeof(float3)*test.points.size()), "allocate periodic points");
    _morton_cuda_check(cudaMalloc((void**)&dev_queries, sizeof(float3)*test.queries.size()), "allocate periodic queries");
    _morton_cuda_check(cudaMalloc((void**)&dev_query_x, sizeof(float)*test.query_x.size()), "allocate query azimuths");
    _morton_cuda_check(cudaMalloc((void**)&dev_indices, sizeof(int)*output_count), "allocate periodic indices");
    _morton_cuda_check(cudaMalloc((void**)&dev_distances, sizeof(float)*output_count), "allocate periodic distances");
    _morton_cuda_check(cudaMalloc((void**)&dev_overflows, sizeof(unsigned int)*test.queries.size()),
        "allocate periodic overflow flags");
    _morton_cuda_check(cudaMalloc((void**)&dev_image_counts, sizeof(unsigned int)*test.queries.size()),
        "allocate periodic image counts");
    _morton_cuda_check(cudaMemcpy(dev_points, test.points.data(), sizeof(float3)*test.points.size(),
        cudaMemcpyHostToDevice), "copy periodic points");
    _morton_cuda_check(cudaMemcpy(dev_queries, test.queries.data(), sizeof(float3)*test.queries.size(),
        cudaMemcpyHostToDevice), "copy periodic queries");
    _morton_cuda_check(cudaMemcpy(dev_query_x, test.query_x.data(), sizeof(float)*test.query_x.size(),
        cudaMemcpyHostToDevice), "copy query azimuths");

    morton_index index;
    index.build(
        dev_points, static_cast<int>(test.points.size()), origin, root_width,
        test.dimension, 4, 20
    );
    periodic_morton_query<K> <<< static_cast<int>(test.queries.size()), QUERY_THREADS >>> (
        dev_indices, dev_distances, dev_overflows, dev_image_counts,
        dev_queries, dev_query_x, static_cast<int>(test.queries.size()), index.view(),
        test.radius, test.x_min, test.x_max
    );
    _morton_cuda_check(cudaDeviceSynchronize(), "run periodic queries");

    std::vector<int> indices(output_count);
    std::vector<float> distances(output_count);
    std::vector<unsigned int> overflows(test.queries.size());
    std::vector<unsigned int> image_counts(test.queries.size());
    _morton_cuda_check(cudaMemcpy(indices.data(), dev_indices, sizeof(int)*output_count, cudaMemcpyDeviceToHost),
        "copy periodic indices");
    _morton_cuda_check(cudaMemcpy(distances.data(), dev_distances, sizeof(float)*output_count,
        cudaMemcpyDeviceToHost), "copy periodic distances");
    _morton_cuda_check(cudaMemcpy(overflows.data(), dev_overflows, sizeof(unsigned int)*test.queries.size(),
        cudaMemcpyDeviceToHost), "copy periodic overflow flags");
    _morton_cuda_check(cudaMemcpy(image_counts.data(), dev_image_counts,
        sizeof(unsigned int)*test.queries.size(), cudaMemcpyDeviceToHost), "copy periodic image counts");

    bool passed = true;
    for (std::size_t idx_query = 0; idx_query < test.queries.size(); idx_query++)
    {
        std::vector<std::pair<float, int>> expected = periodic_brute(test, idx_query);
        std::vector<std::pair<float, int>> actual;
        std::set<int> unique_indices;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            std::size_t idx_out = idx_query*K + idx_neighbor;
            if (indices[idx_out] >= 0 && !unique_indices.insert(indices[idx_out]).second)
            {
                passed = false;
                std::cerr << "  query " << idx_query << " repeats stable index "
                    << indices[idx_out] << std::endl;
            }
            if (indices[idx_out] >= 0)
            {
                actual.emplace_back(distances[idx_out], indices[idx_out]);
                continue;
            }
            if (indices[idx_out] == -1 && std::isinf(distances[idx_out])) continue;
            passed = false;
            std::cerr << "  query " << idx_query << " invalid slot " << idx_neighbor
                << " has index " << indices[idx_out] << " and distance " << distances[idx_out] << std::endl;
        }
        std::sort(actual.begin(), actual.end());
        if (actual.size() != expected.size())
        {
            passed = false;
            std::cerr << "  query " << idx_query << " expected " << expected.size()
                << " neighbors but received " << actual.size() << std::endl;
        }
        std::size_t common = std::min(actual.size(), expected.size());
        for (std::size_t idx_neighbor = 0; idx_neighbor < common; idx_neighbor++)
        {
            float error = std::fabs(actual[idx_neighbor].first - expected[idx_neighbor].first);
            bool distance_matches = error <= 2.0e-6f
                *std::max(1.0f, std::fabs(expected[idx_neighbor].first));
            if (actual[idx_neighbor].second == expected[idx_neighbor].second && distance_matches) continue;

            passed = false;
            std::cerr << "  query " << idx_query << " neighbor " << idx_neighbor
                << " expected=(" << expected[idx_neighbor].second << ',' << expected[idx_neighbor].first << ")"
                << " actual=(" << actual[idx_neighbor].second << ',' << actual[idx_neighbor].first << ')' << std::endl;
        }
        if (overflows[idx_query] != 0 || image_counts[idx_query] != test.expected_images[idx_query])
        {
            passed = false;
            std::cerr << "  query " << idx_query << " overflow=" << overflows[idx_query]
                << " images=" << image_counts[idx_query]
                << " expected_images=" << test.expected_images[idx_query] << std::endl;
        }
    }

    cudaFree(dev_image_counts);
    cudaFree(dev_overflows);
    cudaFree(dev_distances);
    cudaFree(dev_indices);
    cudaFree(dev_query_x);
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
        std::vector<periodic_case> tests;
        for (int dimension : {2, 3})
        {
            tests.push_back(make_lower_seam(dimension));
            tests.push_back(make_upper_seam(dimension));
            tests.push_back(make_interior(dimension));
            tests.push_back(make_narrow_wedge(dimension));
            tests.push_back(make_full_disk(dimension));
        }

        int failed = 0;
        for (const periodic_case &test : tests)
        {
            if (!run_case(test)) failed++;
        }
        std::cout << "periodic tests: " << tests.size() - failed << '/' << tests.size()
            << " passed" << std::endl;
        return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "periodic-test failure: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
