#include <algorithm>      // std::min, std::max, std::sort
#include <cmath>          // std::nextafter
#include <cstdlib>        // EXIT_SUCCESS, EXIT_FAILURE
#include <iostream>       // std::cout, std::cerr
#include <stdexcept>      // std::runtime_error
#include <string>         // std::string, std::to_string
#include <utility>        // std::pair
#include <vector>         // std::vector

#include <cuda_runtime.h> // CUDA allocation, copies, and kernel launches

#include <kdtree/knn.h>
#include <morton/morton_index.cuh>
#include "knn_types.cuh"

// compare Morton top-K output with brute force for ties, duplicates, sparse balls, split planes, and inactive records

namespace
{

constexpr int K = 3;
constexpr int QUERY_THREADS = 256;
using kdtree_boxf = kdtree::box_t<float3>;

struct edge_case
{
    std::string name;
    int dim;
    float radius;
    int leaf_target;
    std::vector<float3> points;
    std::vector<float3> queries;
    std::vector<unsigned char> active;
};

void get_root (const std::vector<float3> &points, int dim, float3 &root_origin, float &width)
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
    width = 1.0001f*std::max({extent_x, extent_y, extent_z});
    if (width <= 0.0f) width = 1.0f;

    float center_x = 0.5f*(lower.x + upper.x);
    float center_y = 0.5f*(lower.y + upper.y);
    float center_z = (dim == 2) ? 0.0f : 0.5f*(lower.z + upper.z);
    root_origin = make_float3(
        center_x - 0.5f*width,
        center_y - 0.5f*width,
        center_z - 0.5f*width
    );
}

// build a double-check reference ordered by squared distance then physical identifier
std::vector<std::pair<float, int>> brute_neighbors (
    const std::vector<float3> &points, const std::vector<unsigned char> &active,
    const float3 &query, float radius)
{
    float radius_sq = radius*radius;
    std::vector<std::pair<float, int>> result;
    for (std::size_t idx = 0; idx < points.size(); idx++)
    {
        if (!active.empty() && active[idx] == 0) continue;

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

void append_anchors (std::vector<float3> &points, int dim)
{
    points.push_back(make_float3(-2.0f, -2.0f, (dim == 2) ? 0.0f : -2.0f));
    points.push_back(make_float3( 2.0f,  2.0f, (dim == 2) ? 0.0f :  2.0f));
}

edge_case make_ties (int dim)
{
    edge_case test{"equal_distance_" + std::to_string(dim) + "d", dim, 1.0f, 2, {}, {}, {}};
    test.points = {
        make_float3( 1.0f,  0.0f, 0.0f),
        make_float3(-1.0f,  0.0f, 0.0f),
        make_float3( 0.0f,  1.0f, 0.0f),
        make_float3( 0.0f, -1.0f, 0.0f),
    };
    if (dim == 3)
    {
        test.points.push_back(make_float3(0.0f, 0.0f,  1.0f));
        test.points.push_back(make_float3(0.0f, 0.0f, -1.0f));
    }
    append_anchors(test.points, dim);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_duplicates (int dim)
{
    edge_case test{"coincident_" + std::to_string(dim) + "d", dim, 0.25f, 128, {}, {}, {}};

    // more than one CUDA block of coincident candidates exercises chunked leaf buffering
    for (int idx = 0; idx < 300; idx++)
    {
        test.points.push_back(make_float3(0.0f, 0.0f, 0.0f));
    }
    append_anchors(test.points, dim);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_radius_boundary (int dim)
{
    edge_case test{"radius_boundary_" + std::to_string(dim) + "d", dim, 1.0f, 2, {}, {}, {}};
    test.points = {
        make_float3( 1.0f, 0.0f, 0.0f),
        make_float3(-1.0f, 0.0f, 0.0f),
        make_float3( 0.0f, 1.0f, 0.0f),
        make_float3(std::nextafter(1.0f, 2.0f), 0.0f, 0.0f),
    };
    append_anchors(test.points, dim);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_sparse (int dim)
{
    edge_case test{"fewer_than_k_" + std::to_string(dim) + "d", dim, 0.75f, 2, {}, {}, {}};
    test.points = {
        make_float3(0.25f, 0.0f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f),
    };
    append_anchors(test.points, dim);
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

edge_case make_split_planes (int dim)
{
    edge_case test{"split_planes_" + std::to_string(dim) + "d", dim, 0.9f, 4, {}, {}, {}};
    const float values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    for (float x : values)
    {
        for (float y : values)
        {
            if (dim == 2)
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
        make_float3(0.25f, -0.25f, (dim == 2) ? 0.0f : 0.25f),
    };
    return test;
}

edge_case make_radial_line ()
{
    edge_case test{"radial_line_1d", 2, 0.6f, 2, {}, {}, {}};
    test.points = {
        make_float3(0.50f, 0.0f, 0.0f),
        make_float3(0.75f, 0.0f, 0.0f),
        make_float3(1.00f, 0.0f, 0.0f),
        make_float3(1.25f, 0.0f, 0.0f),
        make_float3(1.50f, 0.0f, 0.0f),
    };
    test.queries = {
        make_float3(0.50f, 0.0f, 0.0f),
        make_float3(1.00f, 0.0f, 0.0f),
        make_float3(1.50f, 0.0f, 0.0f),
    };
    return test;
}

edge_case make_inactive_filter (int dim)
{
    edge_case test{"inactive_filter_" + std::to_string(dim) + "d", dim, 1.0f, 2, {}, {}, {}};
    for (int idx = 1; idx <= 7; idx++)
    {
        test.points.push_back(make_float3(0.1f*idx, 0.0f, 0.0f));
        test.active.push_back(static_cast<unsigned char>(idx > 4));
    }
    test.queries.push_back(make_float3(0.0f, 0.0f, 0.0f));
    return test;
}

template<int TOP_K>
__global__
void kdtree_active_query (int *dev_near_idx_old, float *dev_near_dist_sq,
    float3 query_point, const kdtree_point *dev_kdtree_node,
    const kdtree_boxf *dev_kdtree_box, int point_count, float search_dist,
    const unsigned char *dev_active)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    kdtree_heap<TOP_K> near_result(search_dist, dev_kdtree_node, false, dev_active);
    kdtree::cct::knn<kdtree_heap<TOP_K>, kdtree_point, kdtree_traits>(
        near_result, query_point, *dev_kdtree_box, dev_kdtree_node, point_count
    );

    for (int idx_neighbor = 0; idx_neighbor < TOP_K; idx_neighbor++)
    {
        int idx_old = near_result.returnIndex(idx_neighbor);
        dev_near_idx_old[idx_neighbor] = idx_old;
        dev_near_dist_sq[idx_neighbor] =
            (idx_old < 0) ? CUDART_INF_F : near_result.returnDist2(idx_neighbor);
    }
}

bool run_kdtree_filter_case (const edge_case &test)
{
    std::vector<kdtree_point> kdtree_node(test.points.size());
    for (std::size_t idx = 0; idx < test.points.size(); idx++)
    {
        kdtree_node[idx].cartesian = test.points[idx];
        kdtree_node[idx].idx_old = static_cast<int>(idx);
        kdtree_node[idx].split_dim = 0;
        kdtree_node[idx].image = 0;
    }

    kdtree_point *dev_kdtree_node = nullptr;
    kdtree_boxf *dev_kdtree_box = nullptr;
    int *dev_near_idx_old = nullptr;
    float *dev_near_dist_sq = nullptr;
    unsigned char *dev_active = nullptr;
    _morton_gpu_check(cudaMalloc((void**)&dev_kdtree_node, sizeof(kdtree_point)*kdtree_node.size()),
        "allocate filtered KD tree");
    _morton_gpu_check(cudaMalloc((void**)&dev_kdtree_box, sizeof(kdtree_boxf)),
        "allocate filtered KD bounds");
    _morton_gpu_check(cudaMalloc((void**)&dev_near_idx_old, sizeof(int)*K),
        "allocate filtered KD indices");
    _morton_gpu_check(cudaMalloc((void**)&dev_near_dist_sq, sizeof(float)*K),
        "allocate filtered KD distances");
    _morton_gpu_check(cudaMalloc((void**)&dev_active, sizeof(unsigned char)*test.active.size()),
        "allocate filtered KD active flags");
    _morton_gpu_check(cudaMemcpy(dev_kdtree_node, kdtree_node.data(),
        sizeof(kdtree_point)*kdtree_node.size(), cudaMemcpyHostToDevice),
        "copy filtered KD points");
    _morton_gpu_check(cudaMemcpy(dev_active, test.active.data(),
        sizeof(unsigned char)*test.active.size(), cudaMemcpyHostToDevice),
        "copy filtered KD active flags");

    kdtree::buildTree<kdtree_point, kdtree_traits>(
        dev_kdtree_node, static_cast<int>(kdtree_node.size()), dev_kdtree_box
    );
    kdtree_active_query<K> <<< 1, 1 >>> (
        dev_near_idx_old, dev_near_dist_sq, test.queries.front(), dev_kdtree_node,
        dev_kdtree_box, static_cast<int>(kdtree_node.size()), test.radius, dev_active
    );
    _morton_gpu_check(cudaDeviceSynchronize(), "run filtered KD query");

    std::vector<int> near_idx_old(K);
    std::vector<float> near_dist_sq(K);
    _morton_gpu_check(cudaMemcpy(near_idx_old.data(), dev_near_idx_old,
        sizeof(int)*K, cudaMemcpyDeviceToHost), "copy filtered KD indices");
    _morton_gpu_check(cudaMemcpy(near_dist_sq.data(), dev_near_dist_sq,
        sizeof(float)*K, cudaMemcpyDeviceToHost), "copy filtered KD distances");

    std::vector<std::pair<float, int>> actual;
    for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
    {
        if (near_idx_old[idx_neighbor] >= 0)
            actual.emplace_back(near_dist_sq[idx_neighbor], near_idx_old[idx_neighbor]);
    }
    std::sort(actual.begin(), actual.end());
    std::vector<std::pair<float, int>> expected = brute_neighbors(
        test.points, test.active, test.queries.front(), test.radius
    );
    bool passed = actual.size() == expected.size();
    for (std::size_t idx_neighbor = 0;
        passed && idx_neighbor < expected.size(); idx_neighbor++)
    {
        float error = std::fabs(actual[idx_neighbor].first - expected[idx_neighbor].first);
        passed = actual[idx_neighbor].second == expected[idx_neighbor].second
            && error <= 2.0e-6f*std::max(1.0f, std::fabs(expected[idx_neighbor].first));
    }

    cudaFree(dev_active);
    cudaFree(dev_near_dist_sq);
    cudaFree(dev_near_idx_old);
    cudaFree(dev_kdtree_box);
    cudaFree(dev_kdtree_node);

    std::cout << (passed ? "PASS  " : "FAIL  ") << "kdtree_" << test.name << std::endl;
    return passed;
}

// execute one adversarial geometry through Morton search and compare every returned slot
bool run_case (const edge_case &test)
{
    float3 root_origin;
    float root_width;
    get_root(test.points, test.dim, root_origin, root_width);

    float3 *dev_point = nullptr;
    float3 *dev_query_point = nullptr;
    int *dev_near_idx_old = nullptr;
    float *dev_near_dist_sq = nullptr;
    unsigned int *dev_stack_overflow = nullptr;
    unsigned char *dev_active = nullptr;
    std::size_t output_count = test.queries.size()*K;

    _morton_gpu_check(cudaMalloc((void**)&dev_point, sizeof(float3)*test.points.size()), "allocate edge points");
    _morton_gpu_check(cudaMalloc((void**)&dev_query_point, sizeof(float3)*test.queries.size()), "allocate edge queries");
    _morton_gpu_check(cudaMalloc((void**)&dev_near_idx_old, sizeof(int)*output_count), "allocate edge near_idx_old");
    _morton_gpu_check(cudaMalloc((void**)&dev_near_dist_sq, sizeof(float)*output_count), "allocate edge near_dist_sq");
    _morton_gpu_check(cudaMalloc((void**)&dev_stack_overflow, sizeof(unsigned int)*test.queries.size()),
        "allocate edge overflow flags");
    if (!test.active.empty())
        _morton_gpu_check(cudaMalloc((void**)&dev_active, sizeof(unsigned char)*test.active.size()),
            "allocate edge active flags");
    _morton_gpu_check(cudaMemcpy(dev_point, test.points.data(), sizeof(float3)*test.points.size(),
        cudaMemcpyHostToDevice), "copy edge points");
    _morton_gpu_check(cudaMemcpy(dev_query_point, test.queries.data(), sizeof(float3)*test.queries.size(),
        cudaMemcpyHostToDevice), "copy edge queries");
    if (dev_active)
        _morton_gpu_check(cudaMemcpy(dev_active, test.active.data(), sizeof(unsigned char)*test.active.size(),
            cudaMemcpyHostToDevice), "copy edge active flags");

    morton_index morton_owner;
    morton_owner.build(
        dev_point, static_cast<int>(test.points.size()), root_origin, root_width,
        test.dim, test.leaf_target, 20
    );
    morton_search<K> <<< static_cast<int>(test.queries.size()), QUERY_THREADS >>> (
        dev_near_idx_old, dev_near_dist_sq, nullptr, nullptr, dev_stack_overflow,
        dev_query_point, static_cast<int>(test.queries.size()), morton_owner.view(), test.radius,
        dev_active
    );
    _morton_gpu_check(cudaDeviceSynchronize(), "run edge queries");

    std::vector<int> near_idx_old(output_count);
    std::vector<float> near_dist_sq(output_count);
    std::vector<unsigned int> stack_overflow(test.queries.size());
    _morton_gpu_check(cudaMemcpy(near_idx_old.data(), dev_near_idx_old, sizeof(int)*output_count, cudaMemcpyDeviceToHost),
        "copy edge near_idx_old");
    _morton_gpu_check(cudaMemcpy(near_dist_sq.data(), dev_near_dist_sq, sizeof(float)*output_count,
        cudaMemcpyDeviceToHost), "copy edge near_dist_sq");
    _morton_gpu_check(cudaMemcpy(stack_overflow.data(), dev_stack_overflow, sizeof(unsigned int)*test.queries.size(),
        cudaMemcpyDeviceToHost), "copy edge overflow flags");

    bool passed = true;
    for (std::size_t idx_query = 0; idx_query < test.queries.size(); idx_query++)
    {
        std::vector<std::pair<float, int>> expected = brute_neighbors(
            test.points, test.active, test.queries[idx_query], test.radius
        );
        std::vector<std::pair<float, int>> actual;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            std::size_t idx_out = idx_query*K + idx_neighbor;
            if (near_idx_old[idx_out] >= 0)
            {
                actual.emplace_back(near_dist_sq[idx_out], near_idx_old[idx_out]);
                continue;
            }
            if (near_idx_old[idx_out] == -1 && std::isinf(near_dist_sq[idx_out])) continue;
            passed = false;
            std::cerr << "  query " << idx_query << " invalid slot " << idx_neighbor
                << " has index " << near_idx_old[idx_out] << " and distance " << near_dist_sq[idx_out] << std::endl;
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
        if (stack_overflow[idx_query] != 0)
        {
            passed = false;
            std::cerr << "  query " << idx_query << " overflowed the traversal stack" << std::endl;
        }
    }

    if (dev_active) cudaFree(dev_active);
    cudaFree(dev_stack_overflow);
    cudaFree(dev_near_dist_sq);
    cudaFree(dev_near_idx_old);
    cudaFree(dev_query_point);
    cudaFree(dev_point);

    std::cout << (passed ? "PASS  " : "FAIL  ") << test.name << std::endl;
    return passed;
}

} // namespace

int main (int argc, char **argv)
{
    try
    {
        bool radial_only = false;
        for (int idx_arg = 1; idx_arg < argc; idx_arg++)
        {
            std::string argument = argv[idx_arg];
            if (argument == "--radial-only") radial_only = true;
            else throw std::invalid_argument("unknown edge-test option: " + argument);
        }

        std::vector<edge_case> tests;
        if (radial_only)
        {
            edge_case inactive = make_inactive_filter(2);
            inactive.name = "inactive_filter_1d";
            tests.push_back(make_radial_line());
            tests.push_back(std::move(inactive));
        }
        else
        {
            for (int dim : {2, 3})
            {
                tests.push_back(make_ties(dim));
                tests.push_back(make_duplicates(dim));
                tests.push_back(make_radius_boundary(dim));
                tests.push_back(make_sparse(dim));
                tests.push_back(make_split_planes(dim));
                tests.push_back(make_inactive_filter(dim));
            }
            tests.push_back(make_radial_line());
        }

        int failed = 0;
        int check_count = static_cast<int>(tests.size());
        for (const edge_case &test : tests)
        {
            if (!run_case(test)) failed++;
            if (!test.active.empty())
            {
                check_count++;
                if (!run_kdtree_filter_case(test)) failed++;
            }
        }
        std::cout << "edge tests: " << check_count - failed << '/' << check_count << " passed" << std::endl;
        return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error)
    {
        std::cerr << "edge-test failure: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
