#ifndef LAB_ADAPTIVE_MORTON_CUH
#define LAB_ADAPTIVE_MORTON_CUH

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "knn_types.cuh"

struct adaptive_morton_node
{
    float3 lower;
    float width;
    int begin;
    int count;
    int child[8];
    int child_number;
};

struct adaptive_morton_view
{
    const morton_point *points;
    const adaptive_morton_node *nodes;
    int point_count;
    int node_count;
    int dimension;
};

inline void _lab_cuda_check (cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__
void adaptive_key_init (std::uint64_t *keys, morton_point *points, const float3 *source,
    const int *source_ids, int point_count, float3 origin, float root_width, int max_level, int dimension)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= point_count) return;

    float3 point = source[idx];
    int cells_per_axis = 1 << max_level;
    float scale = static_cast<float>(cells_per_axis) / root_width;
    int ix = min(cells_per_axis - 1, max(0, static_cast<int>((point.x - origin.x)*scale)));
    int iy = min(cells_per_axis - 1, max(0, static_cast<int>((point.y - origin.y)*scale)));
    int iz = (dimension == 2)
        ? 0 : min(cells_per_axis - 1, max(0, static_cast<int>((point.z - origin.z)*scale)));

    keys[idx] = _get_morton_key(ix, iy, iz);
    points[idx].cartesian = point;
    points[idx].index_old = source_ids ? source_ids[idx] : idx;
}

class adaptive_morton_index
{
public:
    adaptive_morton_index () = default;
    adaptive_morton_index (const adaptive_morton_index &) = delete;
    adaptive_morton_index &operator= (const adaptive_morton_index &) = delete;

    ~adaptive_morton_index () { release(); }

    void build (const float3 *source, int point_count, float3 origin, float root_width,
        int dimension, int leaf_target, int max_level, const int *source_ids = nullptr)
    {
        release();
        if (point_count <= 0 || root_width <= 0.0f) throw std::invalid_argument("invalid adaptive Morton size");
        if (dimension != 2 && dimension != 3) throw std::invalid_argument("adaptive Morton dimension must be 2 or 3");
        if (leaf_target <= 0) throw std::invalid_argument("adaptive Morton leaf target must be positive");
        if (max_level <= 0 || max_level > 20) throw std::invalid_argument("adaptive Morton max level must be 1 through 20");

        point_count_ = point_count;
        dimension_ = dimension;
        leaf_target_ = leaf_target;
        max_level_ = max_level;
        origin_ = origin;
        root_width_ = root_width;

        std::uint64_t *keys = nullptr;
        _lab_cuda_check(cudaMalloc((void**)&keys, sizeof(std::uint64_t)*point_count_),
            "allocate adaptive Morton keys");
        _lab_cuda_check(cudaMalloc((void**)&points_, sizeof(morton_point)*point_count_),
            "allocate adaptive Morton points");

        constexpr int threads = 256;
        int blocks = (point_count_ + threads - 1) / threads;
        adaptive_key_init <<< blocks, threads >>> (
            keys, points_, source, source_ids, point_count_, origin_, root_width_, max_level_, dimension_
        );
        _lab_cuda_check(cudaGetLastError(), "launch adaptive_key_init");

        thrust::device_ptr<std::uint64_t> key_ptr(keys);
        thrust::device_ptr<morton_point> point_ptr(points_);
        thrust::stable_sort_by_key(thrust::device, key_ptr, key_ptr + point_count_, point_ptr);
        _lab_cuda_check(cudaDeviceSynchronize(), "sort adaptive Morton points");

        std::vector<std::uint64_t> host_keys(point_count_);
        _lab_cuda_check(cudaMemcpy(host_keys.data(), keys, sizeof(std::uint64_t)*point_count_,
            cudaMemcpyDeviceToHost), "copy adaptive Morton keys");

        std::vector<adaptive_morton_node> host_nodes;
        host_nodes.reserve(static_cast<std::size_t>(2*point_count_ / leaf_target_ + 64));
        leaf_counts_.clear();

        std::function<int(int, int, int, float3, float)> split =
            [&] (int begin, int end, int level, float3 lower, float width) -> int
        {
            int idx_node = static_cast<int>(host_nodes.size());
            adaptive_morton_node node{};
            node.lower = lower;
            node.width = width;
            node.begin = begin;
            node.count = end - begin;
            node.child_number = 0;
            for (int idx_child = 0; idx_child < 8; idx_child++) node.child[idx_child] = -1;
            host_nodes.push_back(node);

            if (end - begin <= leaf_target_ || level == max_level_)
            {
                leaf_counts_.push_back(end - begin);
                return idx_node;
            }

            int bits_per_level = 3;
            int shift = bits_per_level*(max_level_ - level - 1);
            int child_total = (dimension_ == 2) ? 4 : 8;
            int child_begin[9];
            child_begin[0] = begin;
            int cursor = begin;
            for (int child_code = 0; child_code < child_total; child_code++)
            {
                while (cursor < end && static_cast<int>((host_keys[cursor] >> shift) & 7ULL) == child_code)
                    cursor++;
                child_begin[child_code + 1] = cursor;
            }

            float child_width = 0.5f*width;
            for (int child_code = 0; child_code < child_total; child_code++)
            {
                if (child_begin[child_code] == child_begin[child_code + 1]) continue;
                float3 child_lower = lower;
                if (child_code & 1) child_lower.x += child_width;
                if (child_code & 2) child_lower.y += child_width;
                if (dimension_ == 3 && (child_code & 4)) child_lower.z += child_width;
                int idx_child = split(
                    child_begin[child_code], child_begin[child_code + 1], level + 1,
                    child_lower, child_width
                );
                host_nodes[idx_node].child[child_code] = idx_child;
                host_nodes[idx_node].child_number++;
            }
            return idx_node;
        };

        split(0, point_count_, 0, origin_, root_width_);
        node_count_ = static_cast<int>(host_nodes.size());
        leaf_count_ = static_cast<int>(leaf_counts_.size());

        _lab_cuda_check(cudaMalloc((void**)&nodes_, sizeof(adaptive_morton_node)*node_count_),
            "allocate adaptive Morton nodes");
        _lab_cuda_check(cudaMemcpy(nodes_, host_nodes.data(), sizeof(adaptive_morton_node)*node_count_,
            cudaMemcpyHostToDevice), "copy adaptive Morton nodes");
        _lab_cuda_check(cudaFree(keys), "release adaptive Morton keys");
    }

    void release () noexcept
    {
        if (points_) cudaFree(points_);
        if (nodes_) cudaFree(nodes_);
        points_ = nullptr;
        nodes_ = nullptr;
        point_count_ = 0;
        node_count_ = 0;
        leaf_count_ = 0;
        leaf_counts_.clear();
    }

    adaptive_morton_view view () const
    {
        return {points_, nodes_, point_count_, node_count_, dimension_};
    }

    std::size_t persistent_bytes () const
    {
        return sizeof(morton_point)*static_cast<std::size_t>(point_count_)
            + sizeof(adaptive_morton_node)*static_cast<std::size_t>(node_count_);
    }

    int node_count () const { return node_count_; }
    int leaf_count () const { return leaf_count_; }
    const std::vector<int> &leaf_counts () const { return leaf_counts_; }

private:
    morton_point *points_ = nullptr;
    adaptive_morton_node *nodes_ = nullptr;
    int point_count_ = 0;
    int node_count_ = 0;
    int leaf_count_ = 0;
    int dimension_ = 0;
    int leaf_target_ = 0;
    int max_level_ = 0;
    float3 origin_ = make_float3(0.0f, 0.0f, 0.0f);
    float root_width_ = 0.0f;
    std::vector<int> leaf_counts_;
};

__device__ __forceinline__
float _get_node_dist_sq (const float3 &query, const adaptive_morton_node &node, int dimension)
{
    float3 upper = make_float3(node.lower.x + node.width, node.lower.y + node.width, node.lower.z + node.width);
    float scale = fmaxf(1.0f, fmaxf(
        fmaxf(fabsf(node.lower.x), fabsf(upper.x)),
        fmaxf(fmaxf(fabsf(node.lower.y), fabsf(upper.y)),
            fmaxf(fabsf(node.lower.z), fabsf(upper.z)))
    ));

    // keep the pruning box conservative after repeated single-precision cell subdivision
    float pad = 8.0f*FLT_EPSILON*scale;
    float dx = fmaxf(fmaxf(node.lower.x - query.x - pad, 0.0f), query.x - upper.x - pad);
    float dy = fmaxf(fmaxf(node.lower.y - query.y - pad, 0.0f), query.y - upper.y - pad);
    float dz = (dimension == 2)
        ? 0.0f : fmaxf(fmaxf(node.lower.z - query.z - pad, 0.0f), query.z - upper.z - pad);
    return dx*dx + dy*dy + dz*dz;
}

template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _adaptive_pair_sort (float *distances, int *indices)
{
    for (int width = 2; width <= SORT_SIZE; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int slot = threadIdx.x; slot < SORT_SIZE; slot += BLOCK_SIZE)
            {
                int partner = slot ^ stride;
                if (partner <= slot) continue;
                bool ascending = (slot & width) == 0;
                bool swap_pair = ascending
                    ? _neighbor_less(distances[partner], indices[partner], distances[slot], indices[slot])
                    : _neighbor_less(distances[slot], indices[slot], distances[partner], indices[partner]);
                if (swap_pair)
                {
                    float dist_tmp = distances[slot];
                    distances[slot] = distances[partner];
                    distances[partner] = dist_tmp;
                    int idx_tmp = indices[slot];
                    indices[slot] = indices[partner];
                    indices[partner] = idx_tmp;
                }
            }
            __syncthreads();
        }
    }
}

template<int K, int BLOCK_SIZE, int SORT_SIZE, int STACK_SIZE>
__device__ __forceinline__
void _adaptive_topk (const adaptive_morton_view &view, const float3 &query, float radius,
    float *best_dist, int *best_idx, int *node_stack, int &stack_size, int &idx_node, int &batch_count,
    unsigned int &leaves_visited, unsigned int &candidates_examined, unsigned int &stack_overflow)
{
    static_assert(K + BLOCK_SIZE <= SORT_SIZE, "top-K merge array is too small");
    static_assert((SORT_SIZE & (SORT_SIZE - 1)) == 0, "top-K merge array must be a power of two");

    for (int slot = threadIdx.x; slot < SORT_SIZE; slot += BLOCK_SIZE)
    {
        best_dist[slot] = CUDART_INF_F;
        best_idx[slot] = INT_MAX;
    }
    if (threadIdx.x == 0)
    {
        stack_size = 1;
        node_stack[0] = 0;
        idx_node = -1;
        batch_count = 0;
        leaves_visited = 0;
        candidates_examined = 0;
        stack_overflow = 0;
    }
    __syncthreads();

    float radius_sq = radius*radius;
    while (true)
    {
        if (threadIdx.x == 0) idx_node = (stack_size > 0) ? node_stack[--stack_size] : -1;
        __syncthreads();
        if (idx_node < 0) break;

        const adaptive_morton_node &node = view.nodes[idx_node];
        float cutoff_sq = fminf(radius_sq, best_dist[K - 1]);
        if (_get_node_dist_sq(query, node, view.dimension) > cutoff_sq) continue;

        if (node.child_number == 0)
        {
            if (threadIdx.x == 0) leaves_visited++;
            int offset = 0;
            while (offset < node.count)
            {
                int batch_begin = batch_count;
                int take = min(BLOCK_SIZE - batch_begin, node.count - offset);
                for (int local = threadIdx.x; local < take; local += BLOCK_SIZE)
                {
                    const morton_point &candidate = view.points[node.begin + offset + local];
                    float dist_sq = _get_dist_sq(query, candidate.cartesian);
                    int slot = K + batch_begin + local;
                    best_dist[slot] = CUDART_INF_F;
                    best_idx[slot] = INT_MAX;
                    if (dist_sq <= radius_sq)
                    {
                        best_dist[slot] = dist_sq;
                        best_idx[slot] = candidate.index_old;
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    candidates_examined += take;
                    batch_count = batch_begin + take;
                }
                __syncthreads();

                offset += take;
                if (batch_count >= K)
                {
                    for (int slot = K + batch_count + threadIdx.x; slot < SORT_SIZE; slot += BLOCK_SIZE)
                    {
                        best_dist[slot] = CUDART_INF_F;
                        best_idx[slot] = INT_MAX;
                    }
                    __syncthreads();
                    _adaptive_pair_sort<SORT_SIZE, BLOCK_SIZE>(best_dist, best_idx);
                    if (threadIdx.x == 0) batch_count = 0;
                    __syncthreads();
                }
            }
            continue;
        }

        if (threadIdx.x == 0)
        {
            int child_idx[8];
            float child_dist[8];
            int valid = 0;
            cutoff_sq = fminf(radius_sq, best_dist[K - 1]);
            int child_total = (view.dimension == 2) ? 4 : 8;
            for (int child_code = 0; child_code < child_total; child_code++)
            {
                int child = node.child[child_code];
                if (child < 0) continue;
                float dist_sq = _get_node_dist_sq(query, view.nodes[child], view.dimension);
                if (dist_sq > cutoff_sq) continue;
                int insert = valid;
                while (insert > 0 && child_dist[insert - 1] > dist_sq)
                {
                    child_dist[insert] = child_dist[insert - 1];
                    child_idx[insert] = child_idx[insert - 1];
                    insert--;
                }
                child_dist[insert] = dist_sq;
                child_idx[insert] = child;
                valid++;
            }

            if (stack_size + valid > STACK_SIZE)
            {
                stack_overflow = 1;
                valid = 0;
            }
            for (int idx = valid - 1; idx >= 0; idx--) node_stack[stack_size++] = child_idx[idx];
        }
        __syncthreads();
        if (stack_overflow) break;
    }

    if (batch_count > 0)
    {
        for (int slot = K + batch_count + threadIdx.x; slot < SORT_SIZE; slot += BLOCK_SIZE)
        {
            best_dist[slot] = CUDART_INF_F;
            best_idx[slot] = INT_MAX;
        }
        __syncthreads();
        _adaptive_pair_sort<SORT_SIZE, BLOCK_SIZE>(best_dist, best_idx);
        if (threadIdx.x == 0) batch_count = 0;
        __syncthreads();
    }
}

template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void adaptive_morton_query (int *neighbor_idx, float *neighbor_dist, unsigned int *leaf_visits,
    unsigned int *candidate_visits, unsigned int *stack_overflows,
    const float3 *queries, int query_count, adaptive_morton_view view, float radius)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float best_dist[SORT_SIZE];
    __shared__ int best_idx[SORT_SIZE];
    __shared__ int node_stack[STACK_SIZE];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;

    _adaptive_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        view, queries[idx_query], radius, best_dist, best_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        int idx_out = idx_query*K + idx_neighbor;
        neighbor_idx[idx_out] = (best_idx[idx_neighbor] == INT_MAX) ? -1 : best_idx[idx_neighbor];
        neighbor_dist[idx_out] = best_dist[idx_neighbor];
    }
    if (threadIdx.x == 0)
    {
        if (leaf_visits) leaf_visits[idx_query] = leaves_visited;
        if (candidate_visits) candidate_visits[idx_query] = candidates_examined;
        if (stack_overflows) stack_overflows[idx_query] = stack_overflow;
    }
}

template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void adaptive_morton_checksum (double *checksum, unsigned int *stack_overflows,
    const float3 *queries, int query_count, adaptive_morton_view view, float radius)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float best_dist[SORT_SIZE];
    __shared__ int best_idx[SORT_SIZE];
    __shared__ int node_stack[STACK_SIZE];
    __shared__ int stack_size;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaves_visited;
    __shared__ unsigned int candidates_examined;
    __shared__ unsigned int stack_overflow;

    _adaptive_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        view, queries[idx_query], radius, best_dist, best_idx, node_stack, stack_size, idx_node, batch_count,
        leaves_visited, candidates_examined, stack_overflow
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            if (best_idx[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(best_dist[idx_neighbor])
                + 1.0e-12*static_cast<double>(best_idx[idx_neighbor]);
        }
        checksum[idx_query] = value;
        if (stack_overflows) stack_overflows[idx_query] = stack_overflow;
    }
}

#endif // LAB_ADAPTIVE_MORTON_CUH
