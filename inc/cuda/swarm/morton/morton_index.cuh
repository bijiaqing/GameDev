#ifndef GAMEDEV_MORTON_INDEX_CUH
#define GAMEDEV_MORTON_INDEX_CUH

#include <cfloat>                         // FLT_EPSILON
#include <climits>                        // INT_MAX
#include <cmath>                          // fabsf, fmaxf, fminf
#include <cstddef>                        // std::size_t
#include <cstdint>                        // std::uint64_t
#include <algorithm>                      // std::min, std::max
#include <stdexcept>                      // std::invalid_argument, std::runtime_error
#include <string>                         // std::string
#include <vector>                         // std::vector

#include <cuda_runtime.h>                 // CUDA allocation and kernel-launch API
#include <math_constants.h>  // CUDART_INF_F

#include <thrust/device_ptr.h>             // thrust::device_ptr
#include <thrust/execution_policy.h>       // thrust::device
#include <thrust/sort.h>                   // thrust::stable_sort_by_key
#include <thrust/device_vector.h>          // device construction scratch space
#include <thrust/scan.h>                   // child-slot offsets
#include <thrust/count.h>                  // leaf count
#include <thrust/copy.h>                   // diagnostic leaf compaction
#include <thrust/iterator/transform_iterator.h> // diagnostic leaf sizes

#include <morton/morton_types.cuh>

// convert CUDA failures into exceptions usable by the host-side index owner
inline void _morton_cuda_check (cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// quantize Cartesian points at the deepest level and emit sortable Morton keys
static __global__
void morton_keygen (std::uint64_t *dev_key, morton_point *dev_point,
    const float3 *dev_source_point, const int *dev_source_idx_old,
    int point_count, float3 root_origin, float root_width, int max_level, int dim)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= point_count) return;

    float3 point = dev_source_point[idx];
    int cells_per_axis = 1 << max_level;
    float scale = static_cast<float>(cells_per_axis) / root_width;
    int ix = min(cells_per_axis - 1, max(0, static_cast<int>((point.x - root_origin.x)*scale)));
    int iy = min(cells_per_axis - 1, max(0, static_cast<int>((point.y - root_origin.y)*scale)));
    int iz = (dim == 2)
        ? 0 : min(cells_per_axis - 1, max(0, static_cast<int>((point.z - root_origin.z)*scale)));

    dev_key[idx] = _get_morton_key(ix, iy, iz);
    dev_point[idx].cartesian = point;
    dev_point[idx].idx_old = dev_source_idx_old ? dev_source_idx_old[idx] : idx;
}

// the input range shares all key bits above this child digit
__host__ __device__ inline int morton_child_end(const std::uint64_t *keys, int begin, int end,
                                    int shift, int code)
{
    while (begin < end)
    {
        int mid = begin + (end - begin)/2;
        if (int((keys[mid] >> shift) & 7ULL) < code) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

static __device__ morton_node morton_cell(int begin, int count, float3 lower, float width)
{
    morton_node node{};
    node.idx_begin = begin;
    node.count = count;
    node.lower = lower;
    node.width = width;
    for (int code = 0; code < 8; ++code) node.idx_child[code] = -1;
    return node;
}

static __global__ void morton_origin(morton_node *nodes, int count, float3 lower, float width)
{
    nodes[0] = morton_cell(0, count, lower, width);
}

// independent sorted-key ranges need at most eight binary searches each
static __global__ void morton_counts(morton_node *nodes, const std::uint64_t *keys,
    int first, int count, int level, int max_level, int dim, int leaf_target, int *children)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx == 0) children[count] = 0;
    if (idx >= count) return;
    morton_node &node = nodes[first + idx];
    int child_count = 0;
    if (node.count > leaf_target && level < max_level)
    {
        int begin = node.idx_begin;
        int end = begin + node.count;
        int shift = 3*(max_level - level - 1);
        for (int code = 1; code <= (1 << dim); ++code)
        {
            int next = morton_child_end(keys, begin, end, shift, code);
            child_count += next > begin;
            begin = next;
        }
    }
    node.child_count = children[idx] = child_count;
}

// scan-assigned child slots preserve spatial child order without serial node numbering
static __global__ void morton_expand(morton_node *nodes, const std::uint64_t *keys,
    int first, int count, int next_first, const std::uint64_t *offsets, int level, int max_level, int dim)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= count) return;
    morton_node parent = nodes[first + idx];
    if (!parent.child_count) return;
    int begin = parent.idx_begin;
    int end = begin + parent.count;
    int slot = next_first + int(offsets[idx]);
    int shift = 3*(max_level - level - 1);
    float width = 0.5f*parent.width;
    for (int code = 0; code < (1 << dim); ++code)
    {
        int next = morton_child_end(keys, begin, end, shift, code + 1);
        if (next > begin)
        {
            float3 lower = parent.lower;
            if (code & 1) lower.x += width;
            if (code & 2) lower.y += width;
            if (dim == 3 && (code & 4)) lower.z += width;
            nodes[slot] = morton_cell(begin, next - begin, lower, width);
            nodes[first + idx].idx_child[code] = slot++;
        }
        begin = next;
    }
}

struct morton_is_leaf
{
    __host__ __device__ bool operator()(const morton_node &node) const { return node.child_count == 0; }
};
struct morton_leaf_size
{
    __host__ __device__ int operator()(const morton_node &node) const { return node.child_count ? 0 : node.count; }
};
struct morton_nonzero
{
    __host__ __device__ bool operator()(int count) const { return count != 0; }
};

class morton_index
{
public:
    morton_index() = default;
    morton_index(const morton_index &) = delete;
    morton_index &operator=(const morton_index &) = delete;
    ~morton_index() { release(); }

    void build(const float3 *source, int point_count, float3 origin, float width,
        int dim, int leaf_target, int max_level, const int *source_ids = nullptr)
    {
        release();
        if (point_count <= 0 || width <= 0.0f) throw std::invalid_argument("invalid adaptive Morton size");
        if (dim != 2 && dim != 3) throw std::invalid_argument("adaptive Morton dimension must be 2 or 3");
        if (leaf_target <= 0) throw std::invalid_argument("adaptive Morton leaf target must be positive");
        if (max_level <= 0 || max_level > 20) throw std::invalid_argument("adaptive Morton max level must be 1 through 20");
        point_count_ = point_count;
        dim_ = dim;
        max_level_ = max_level;
        thrust::device_vector<std::uint64_t> keys(point_count);
        auto dev_key = thrust::raw_pointer_cast(keys.data());
        _morton_cuda_check(cudaMalloc((void**)&dev_point_, sizeof(morton_point)*std::size_t(point_count)), "allocate Morton points");
        morton_keygen<<<(point_count - 1)/256 + 1, 256>>>(dev_key, dev_point_, source, source_ids,
            point_count, origin, width, max_level, dim);
        _morton_cuda_check(cudaGetLastError(), "launch morton_keygen");
        thrust::stable_sort_by_key(thrust::device, keys.begin(), keys.end(), thrust::device_pointer_cast(dev_point_));

        reserve_nodes(1);
        morton_origin<<<1, 1>>>(dev_node_, point_count, origin, width);
        _morton_cuda_check(cudaGetLastError(), "launch morton_origin");
        node_count_ = 1;
        int first = 0, count = 1;
        thrust::device_vector<int> children;
        thrust::device_vector<std::uint64_t> offsets;
        for (int level = 0; level <= max_level && count; ++level)
        {
            children.resize(std::size_t(count) + 1);
            offsets.resize(std::size_t(count) + 1);
            auto counts_ptr = thrust::raw_pointer_cast(children.data());
            auto offsets_ptr = thrust::raw_pointer_cast(offsets.data());
            morton_counts<<<(count - 1)/256 + 1, 256>>>(dev_node_, dev_key, first, count,
                level, max_level, dim, leaf_target, counts_ptr);
            _morton_cuda_check(cudaGetLastError(), "launch morton_counts");
            thrust::exclusive_scan(thrust::device, children.begin(), children.end(), offsets.begin(), std::uint64_t(0));
            std::uint64_t next_count = 0;
            // only allocation metadata crosses to the host; keys and nodes stay on-device
            _morton_cuda_check(cudaMemcpy(&next_count, offsets_ptr + count, sizeof(next_count), cudaMemcpyDeviceToHost), "read Morton frontier count");
            if (!next_count) break;
            if (next_count > std::uint64_t(INT_MAX - node_count_)) throw std::overflow_error("Morton node index overflow");
            reserve_nodes(node_count_ + int(next_count));
            morton_expand<<<(count - 1)/256 + 1, 256>>>(dev_node_, dev_key, first, count,
                node_count_, offsets_ptr, level, max_level, dim);
            _morton_cuda_check(cudaGetLastError(), "launch morton_expand");
            first = node_count_;
            node_count_ += int(next_count);
            count = int(next_count);
        }
        auto nodes = thrust::device_pointer_cast(dev_node_);
        leaf_count_ = int(thrust::count_if(thrust::device, nodes, nodes + node_count_, morton_is_leaf{}));
        _morton_cuda_check(cudaDeviceSynchronize(), "complete Morton hierarchy");
    }

    void release() noexcept
    {
        if (dev_point_) (void)cudaFree(dev_point_);
        if (dev_node_) (void)cudaFree(dev_node_);
        dev_point_ = nullptr;
        dev_node_ = nullptr;
        point_count_ = node_count_ = leaf_count_ = capacity_ = 0;
        leaf_counts_.clear();
    }
    morton_view view() const { return {dev_point_, dev_node_, point_count_, node_count_, dim_, max_level_}; }
    std::size_t persistent_bytes() const
    {
        // report actual retained capacity, including allocation slack
        return sizeof(morton_point)*std::size_t(point_count_) + sizeof(morton_node)*std::size_t(capacity_);
    }
    int node_count() const { return node_count_; }
    int leaf_count() const { return leaf_count_; }
    const std::vector<int> &leaf_counts() const
    {
        // host occupancy statistics are requested by benchmarks, never by construction or queries
        if (leaf_counts_.empty() && leaf_count_)
        {
            auto nodes = thrust::device_pointer_cast(dev_node_);
            auto sizes = thrust::make_transform_iterator(nodes, morton_leaf_size{});
            thrust::device_vector<int> compact(leaf_count_);
            thrust::copy_if(thrust::device, sizes, sizes + node_count_, compact.begin(), morton_nonzero{});
            leaf_counts_.resize(leaf_count_);
            _morton_cuda_check(cudaMemcpy(leaf_counts_.data(), thrust::raw_pointer_cast(compact.data()),
                sizeof(int)*std::size_t(leaf_count_), cudaMemcpyDeviceToHost), "read diagnostic Morton leaf sizes");
        }
        return leaf_counts_;
    }
private:
    void reserve_nodes(int required)
    {
        if (required <= capacity_) return;
        int capacity = int(std::max<long long>(required, std::min<long long>(INT_MAX, 2LL*capacity_)));
        morton_node *nodes = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&nodes, sizeof(morton_node)*std::size_t(capacity)), "allocate Morton nodes");
        if (dev_node_)
        {
            auto status = cudaMemcpy(nodes, dev_node_, sizeof(morton_node)*std::size_t(node_count_), cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess) { (void)cudaFree(nodes); _morton_cuda_check(status, "grow Morton nodes"); }
            status = cudaFree(dev_node_);
            dev_node_ = nodes;
            capacity_ = capacity;
            _morton_cuda_check(status, "release previous Morton nodes");
        }
        dev_node_ = nodes;
        capacity_ = capacity;
    }
    morton_point *dev_point_ = nullptr;
    morton_node *dev_node_ = nullptr;
    int point_count_ = 0, node_count_ = 0, leaf_count_ = 0, capacity_ = 0;
    int dim_ = 0, max_level_ = 0;
    mutable std::vector<int> leaf_counts_;
};

// calculate a conservative point-to-node lower distance for branch pruning
__device__ __forceinline__
float _get_morton_node_dist_sq (
    const float3 &query_point, const morton_node &node, int dim, int max_level)
{
    float3 upper = make_float3(node.lower.x + node.width, node.lower.y + node.width, node.lower.z + node.width);
    float scale = fmaxf(1.0f, fmaxf(
        fmaxf(fabsf(node.lower.x), fabsf(upper.x)),
        fmaxf(fmaxf(fabsf(node.lower.y), fabsf(upper.y)),
            fmaxf(fabsf(node.lower.z), fabsf(upper.z)))
    ));

    // cover accumulated rounding from every recursive single-precision subdivision
    float pad = 2.0f*static_cast<float>(max_level + 2)*FLT_EPSILON*scale;
    float dx = fmaxf(fmaxf(node.lower.x - query_point.x - pad, 0.0f), query_point.x - upper.x - pad);
    float dy = fmaxf(fmaxf(node.lower.y - query_point.y - pad, 0.0f), query_point.y - upper.y - pad);
    float dz = (dim == 2)
        ? 0.0f : fmaxf(fmaxf(node.lower.z - query_point.z - pad, 0.0f), query_point.z - upper.z - pad);
    return dx*dx + dy*dy + dz*dz;
}

// sort shared-memory neighbor pairs by distance and original identifier
template<int SORT_SIZE, int BLOCK_SIZE>
__device__ __forceinline__
void _morton_pair_sort (float *dist_sq, int *idx_old)
{
    for (int width = 2; width <= SORT_SIZE; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int idx_slot = threadIdx.x; idx_slot < SORT_SIZE; idx_slot += BLOCK_SIZE)
            {
                int idx_partner = idx_slot ^ stride;
                if (idx_partner <= idx_slot) continue;
                bool ascending = (idx_slot & width) == 0;
                bool swap_pair = ascending
                    ? _morton_neighbor_less(
                        dist_sq[idx_partner], idx_old[idx_partner], dist_sq[idx_slot], idx_old[idx_slot]
                    )
                    : _morton_neighbor_less(
                        dist_sq[idx_slot], idx_old[idx_slot], dist_sq[idx_partner], idx_old[idx_partner]
                    );
                if (swap_pair)
                {
                    float dist_sq_tmp = dist_sq[idx_slot];
                    dist_sq[idx_slot] = dist_sq[idx_partner];
                    dist_sq[idx_partner] = dist_sq_tmp;
                    int idx_old_tmp = idx_old[idx_slot];
                    idx_old[idx_slot] = idx_old[idx_partner];
                    idx_old[idx_partner] = idx_old_tmp;
                }
            }
            __syncthreads();
        }
    }
}

// merge one candidate tile into the retained top-K prefix
template<int K, int BLOCK_SIZE, int SORT_SIZE>
__device__ __forceinline__
void _morton_pair_merge (float *dist_sq, int *idx_old, int candidate_count)
{
    for (int idx_slot = K + candidate_count + threadIdx.x;
        idx_slot < SORT_SIZE; idx_slot += BLOCK_SIZE)
    {
        dist_sq[idx_slot] = CUDART_INF_F;
        idx_old[idx_slot] = INT_MAX;
    }
    __syncthreads();
    _morton_pair_sort<SORT_SIZE, BLOCK_SIZE>(dist_sq, idx_old);
}

// traverse the adaptive hierarchy cooperatively and retain the exact bounded top-K set
template<int K, int BLOCK_SIZE, int SORT_SIZE, int STACK_SIZE>
__device__ __forceinline__
void _morton_topk (const morton_view &morton_data, const float3 &query_point, float search_dist,
    float *near_dist_sq, int *near_idx_old, int *idx_node_stack,
    int &stack_count, int &idx_node, int &batch_count,
    unsigned int &leaf_visit_count, unsigned int &candidate_count,
    unsigned int &stack_overflow, const unsigned char *dev_active = nullptr,
    int idx_stride = 1)
{
    static_assert(K + BLOCK_SIZE <= SORT_SIZE, "top-K merge array is too small");
    static_assert((SORT_SIZE & (SORT_SIZE - 1)) == 0, "top-K merge array must be a power of two");
    constexpr int batch_capacity = (K < SORT_SIZE - K) ? K : SORT_SIZE - K;

    for (int idx_slot = threadIdx.x; idx_slot < SORT_SIZE; idx_slot += BLOCK_SIZE)
    {
        near_dist_sq[idx_slot] = CUDART_INF_F;
        near_idx_old[idx_slot] = INT_MAX;
    }
    if (threadIdx.x == 0)
    {
        stack_count = 1;
        idx_node_stack[0] = 0;
        idx_node = -1;
        batch_count = 0;
        leaf_visit_count = 0;
        candidate_count = 0;
        stack_overflow = 0;
    }
    __syncthreads();

    float search_dist_sq = search_dist*search_dist;
    while (true)
    {
        // finish all uses of the previous node before replacing the shared index
        __syncthreads();
        if (threadIdx.x == 0)
            idx_node = (stack_count > 0) ? idx_node_stack[--stack_count] : -1;
        __syncthreads();
        if (idx_node < 0) break;

        const morton_node &node = morton_data.dev_node[idx_node];
        float cutoff_sq = fminf(search_dist_sq, near_dist_sq[K - 1]);
        if (_get_morton_node_dist_sq(
            query_point, node, morton_data.dim, morton_data.max_level
        ) > cutoff_sq) continue;

        if (node.child_count == 0)
        {
            // stream arbitrarily large leaves through bounded shared-memory candidate tiles
            if (threadIdx.x == 0) leaf_visit_count++;
            int idx_offset = 0;
            while (idx_offset < node.count)
            {
                int batch_offset = batch_count;
                int take_count = min(batch_capacity - batch_offset, node.count - idx_offset);
                for (int idx_local = threadIdx.x; idx_local < take_count; idx_local += BLOCK_SIZE)
                {
                    const morton_point &candidate =
                        morton_data.dev_point[node.idx_begin + idx_offset + idx_local];
                    float candidate_dist_sq =
                        _get_morton_point_dist_sq(query_point, candidate.cartesian);
                    int idx_slot = K + batch_offset + idx_local;
                    near_dist_sq[idx_slot] = CUDART_INF_F;
                    near_idx_old[idx_slot] = INT_MAX;
                    int idx_old = candidate.idx_old / idx_stride;
                    if (candidate_dist_sq <= search_dist_sq
                        && (!dev_active || dev_active[idx_old] != 0))
                    {
                        near_dist_sq[idx_slot] = candidate_dist_sq;
                        near_idx_old[idx_slot] = candidate.idx_old;
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    candidate_count += take_count;
                    batch_count = batch_offset + take_count;
                }
                __syncthreads();

                idx_offset += take_count;
                if (batch_count >= batch_capacity)
                {
                    _morton_pair_merge<K, BLOCK_SIZE, SORT_SIZE>(
                        near_dist_sq, near_idx_old, batch_count
                    );
                    if (threadIdx.x == 0) batch_count = 0;
                    __syncthreads();
                }
            }
            continue;
        }

        if (threadIdx.x == 0)
        {
            // visit surviving children from nearest to farthest to tighten the cutoff early
            int idx_child[8];
            float child_dist_sq[8];
            int valid_count = 0;
            cutoff_sq = fminf(search_dist_sq, near_dist_sq[K - 1]);
            int child_count = (morton_data.dim == 2) ? 4 : 8;
            for (int idx_child_code = 0; idx_child_code < child_count; idx_child_code++)
            {
                int idx_child_node = node.idx_child[idx_child_code];
                if (idx_child_node < 0) continue;
                float dist_sq = _get_morton_node_dist_sq(
                    query_point, morton_data.dev_node[idx_child_node],
                    morton_data.dim, morton_data.max_level
                );
                if (dist_sq > cutoff_sq) continue;
                int idx_insert = valid_count;
                while (idx_insert > 0 && child_dist_sq[idx_insert - 1] > dist_sq)
                {
                    child_dist_sq[idx_insert] = child_dist_sq[idx_insert - 1];
                    idx_child[idx_insert] = idx_child[idx_insert - 1];
                    idx_insert--;
                }
                child_dist_sq[idx_insert] = dist_sq;
                idx_child[idx_insert] = idx_child_node;
                valid_count++;
            }

            if (stack_count + valid_count > STACK_SIZE)
            {
                stack_overflow = 1;
                valid_count = 0;
            }
            for (int idx = valid_count - 1; idx >= 0; idx--)
            {
                idx_node_stack[stack_count++] = idx_child[idx];
            }
        }
        __syncthreads();
        if (stack_overflow) break;
    }

    if (batch_count > 0)
    {
        _morton_pair_merge<K, BLOCK_SIZE, SORT_SIZE>(near_dist_sq, near_idx_old, batch_count);
        if (threadIdx.x == 0) batch_count = 0;
        __syncthreads();
    }
}

#endif // GAMEDEV_MORTON_INDEX_CUH
