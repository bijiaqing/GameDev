#ifndef GAMEDEV_MORTON_INDEX_CUH
#define GAMEDEV_MORTON_INDEX_CUH

// QAV header override: retain standalone query and digest kernels used only by test_knn

#include <cfloat>                         // FLT_EPSILON
#include <climits>                        // INT_MAX
#include <cmath>                          // fabsf, fmaxf, fminf
#include <cstddef>                        // std::size_t
#include <cstdint>                        // std::uint64_t
#include <functional>                     // std::function
#include <stdexcept>                      // std::invalid_argument, std::runtime_error
#include <string>                         // std::string
#include <vector>                         // std::vector

#include <hip/hip_runtime.h>                 // HIP allocation and kernel-launch API

#include <thrust/device_ptr.h>             // thrust::device_ptr
#include <thrust/execution_policy.h>       // thrust::device
#include <thrust/system/hip/execution_policy.h> // thrust::hip_rocprim::par
#include <thrust/sort.h>                   // thrust::stable_sort_by_key

#include <morton/morton_types.cuh>

// convert HIP failures into exceptions usable by the host-side index owner
inline void _morton_hip_check (hipError_t status, const char *operation)
{
    if (status == hipSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
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

// own one adaptive Morton index and its device-resident point and node arrays
class morton_index
{
public:
    morton_index () = default;
    morton_index (const morton_index &) = delete;
    morton_index &operator= (const morton_index &) = delete;

    ~morton_index () { release(); }

    void build (const float3 *dev_source_point, int point_count, float3 root_origin,
        float root_width, int dim, int leaf_target, int max_level,
        const int *dev_source_idx_old = nullptr)
    {
        release();
        if (point_count <= 0 || root_width <= 0.0f) throw std::invalid_argument("invalid adaptive Morton size");
        if (dim != 2 && dim != 3) throw std::invalid_argument("adaptive Morton dimension must be 2 or 3");
        if (leaf_target <= 0) throw std::invalid_argument("adaptive Morton leaf target must be positive");
        if (max_level <= 0 || max_level > 20) throw std::invalid_argument("adaptive Morton max level must be 1 through 20");

        point_count_ = point_count;
        dim_ = dim;
        leaf_target_ = leaf_target;
        max_level_ = max_level;
        root_origin_ = root_origin;
        root_width_ = root_width;

        std::uint64_t *dev_key = nullptr;
        _morton_hip_check(hipMalloc((void**)&dev_key, sizeof(std::uint64_t)*point_count_),
            "allocate adaptive Morton keys");
        _morton_hip_check(hipMalloc((void**)&dev_point_, sizeof(morton_point)*point_count_),
            "allocate adaptive Morton points");

        constexpr int thread_count = 256;
        int block_count = (point_count_ + thread_count - 1) / thread_count;
        morton_keygen <<< block_count, thread_count >>> (
            dev_key, dev_point_, dev_source_point, dev_source_idx_old,
            point_count_, root_origin_, root_width_, max_level_, dim_
        );
        _morton_hip_check(hipGetLastError(), "launch morton_keygen");

        thrust::device_ptr<std::uint64_t> key_ptr(dev_key);
        thrust::device_ptr<morton_point> point_ptr(dev_point_);
        thrust::stable_sort_by_key(
            thrust::hip_rocprim::par, key_ptr, key_ptr + point_count_, point_ptr
        );
        _morton_hip_check(hipDeviceSynchronize(), "sort adaptive Morton points");

        std::vector<std::uint64_t> host_keys(point_count_);
        _morton_hip_check(hipMemcpy(host_keys.data(), dev_key, sizeof(std::uint64_t)*point_count_,
            hipMemcpyDeviceToHost), "copy adaptive Morton keys");

        std::vector<morton_node> host_nodes;
        host_nodes.reserve(static_cast<std::size_t>(2*point_count_ / leaf_target_ + 64));
        leaf_counts_.clear();

        // split each sorted key range until its leaf occupancy or depth limit is reached
        std::function<int(int, int, int, float3, float)> split =
            [&] (int idx_begin, int idx_end, int level, float3 lower, float width) -> int
        {
            int idx_node = static_cast<int>(host_nodes.size());
            morton_node node{};
            node.lower = lower;
            node.width = width;
            node.idx_begin = idx_begin;
            node.count = idx_end - idx_begin;
            node.child_count = 0;
            for (int idx_child = 0; idx_child < 8; idx_child++)
            {
                node.idx_child[idx_child] = -1;
            }
            host_nodes.push_back(node);

            if (idx_end - idx_begin <= leaf_target_ || level == max_level_)
            {
                leaf_counts_.push_back(idx_end - idx_begin);
                return idx_node;
            }

            constexpr int bits_per_level = 3;
            int bit_shift = bits_per_level*(max_level_ - level - 1);
            int child_count = (dim_ == 2) ? 4 : 8;
            int idx_child_begin[9];
            idx_child_begin[0] = idx_begin;
            int idx_cursor = idx_begin;
            for (int idx_child_code = 0; idx_child_code < child_count; idx_child_code++)
            {
                while (idx_cursor < idx_end
                    && static_cast<int>((host_keys[idx_cursor] >> bit_shift) & 7ULL) == idx_child_code)
                {
                    idx_cursor++;
                }
                idx_child_begin[idx_child_code + 1] = idx_cursor;
            }

            float child_width = 0.5f*width;
            for (int idx_child_code = 0; idx_child_code < child_count; idx_child_code++)
            {
                if (idx_child_begin[idx_child_code] == idx_child_begin[idx_child_code + 1]) continue;
                float3 child_lower = lower;
                if (idx_child_code & 1) child_lower.x += child_width;
                if (idx_child_code & 2) child_lower.y += child_width;
                if (dim_ == 3 && (idx_child_code & 4)) child_lower.z += child_width;
                int idx_child = split(
                    idx_child_begin[idx_child_code], idx_child_begin[idx_child_code + 1], level + 1,
                    child_lower, child_width
                );
                host_nodes[idx_node].idx_child[idx_child_code] = idx_child;
                host_nodes[idx_node].child_count++;
            }
            return idx_node;
        };

        split(0, point_count_, 0, root_origin_, root_width_);
        node_count_ = static_cast<int>(host_nodes.size());
        leaf_count_ = static_cast<int>(leaf_counts_.size());

        _morton_hip_check(hipMalloc((void**)&dev_node_, sizeof(morton_node)*node_count_),
            "allocate adaptive Morton nodes");
        _morton_hip_check(hipMemcpy(dev_node_, host_nodes.data(), sizeof(morton_node)*node_count_,
            hipMemcpyHostToDevice), "copy adaptive Morton nodes");
        _morton_hip_check(hipFree(dev_key), "release adaptive Morton keys");
    }

    void release () noexcept
    {
        if (dev_point_) (void)hipFree(dev_point_);
        if (dev_node_) (void)hipFree(dev_node_);
        dev_point_ = nullptr;
        dev_node_ = nullptr;
        point_count_ = 0;
        node_count_ = 0;
        leaf_count_ = 0;
        leaf_counts_.clear();
    }

    morton_view view () const
    {
        return {dev_point_, dev_node_, point_count_, node_count_, dim_, max_level_};
    }

    std::size_t persistent_bytes () const
    {
        return sizeof(morton_point)*static_cast<std::size_t>(point_count_)
            + sizeof(morton_node)*static_cast<std::size_t>(node_count_);
    }

    int node_count () const { return node_count_; }
    int leaf_count () const { return leaf_count_; }
    const std::vector<int> &leaf_counts () const { return leaf_counts_; }

private:
    morton_point *dev_point_ = nullptr;
    morton_node *dev_node_ = nullptr;
    int point_count_ = 0;
    int node_count_ = 0;
    int leaf_count_ = 0;
    int dim_ = 0;
    int leaf_target_ = 0;
    int max_level_ = 0;
    float3 root_origin_ = make_float3(0.0f, 0.0f, 0.0f);
    float root_width_ = 0.0f;
    std::vector<int> leaf_counts_;
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
        dist_sq[idx_slot] = MORTON_INF_F;
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
    unsigned int &stack_overflow, const unsigned char *dev_active = nullptr)
{
    static_assert(K + BLOCK_SIZE <= SORT_SIZE, "top-K merge array is too small");
    static_assert((SORT_SIZE & (SORT_SIZE - 1)) == 0, "top-K merge array must be a power of two");
    constexpr int batch_capacity = (K < SORT_SIZE - K) ? K : SORT_SIZE - K;

    for (int idx_slot = threadIdx.x; idx_slot < SORT_SIZE; idx_slot += BLOCK_SIZE)
    {
        near_dist_sq[idx_slot] = MORTON_INF_F;
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
                    near_dist_sq[idx_slot] = MORTON_INF_F;
                    near_idx_old[idx_slot] = INT_MAX;
                    if (candidate_dist_sq <= search_dist_sq
                        && (!dev_active || dev_active[candidate.idx_old] != 0))
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

// expose one cooperative exact top-K query per HIP block for tests and standalone consumers
template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void morton_search (int *dev_near_idx_old, float *dev_near_dist_sq, unsigned int *dev_leaf_visit_count,
    unsigned int *dev_candidate_count, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data, float search_dist,
    const unsigned char *dev_active = nullptr)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[SORT_SIZE];
    __shared__ int work_idx_old[SORT_SIZE];
    __shared__ int idx_node_stack[STACK_SIZE];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_count;
    __shared__ unsigned int candidate_total;
    __shared__ unsigned int overflow;

    _morton_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        morton_data, dev_query_point[idx_query], search_dist, work_dist_sq, work_idx_old,
        idx_node_stack, stack_count, idx_node, batch_count,
        leaf_count, candidate_total, overflow, dev_active
    );

    for (int idx_neighbor = threadIdx.x; idx_neighbor < K; idx_neighbor += BLOCK_SIZE)
    {
        int idx_out = idx_query*K + idx_neighbor;
        dev_near_idx_old[idx_out] =
            (work_idx_old[idx_neighbor] == INT_MAX) ? -1 : work_idx_old[idx_neighbor];
        dev_near_dist_sq[idx_out] = work_dist_sq[idx_neighbor];
    }
    if (threadIdx.x == 0)
    {
        if (dev_leaf_visit_count) dev_leaf_visit_count[idx_query] = leaf_count;
        if (dev_candidate_count) dev_candidate_count[idx_query] = candidate_total;
        if (dev_stack_overflow) dev_stack_overflow[idx_query] = overflow;
    }
}

// reduce each top-K result to a deterministic checksum for timing without large output transfers
template<int K, int BLOCK_SIZE = 256, int SORT_SIZE = 512, int STACK_SIZE = 256>
__global__
void morton_digest (double *dev_checksum, unsigned int *dev_stack_overflow,
    const float3 *dev_query_point, int query_count, morton_view morton_data, float search_dist)
{
    int idx_query = blockIdx.x;
    if (idx_query >= query_count) return;

    __shared__ float work_dist_sq[SORT_SIZE];
    __shared__ int work_idx_old[SORT_SIZE];
    __shared__ int idx_node_stack[STACK_SIZE];
    __shared__ int stack_count;
    __shared__ int idx_node;
    __shared__ int batch_count;
    __shared__ unsigned int leaf_count;
    __shared__ unsigned int candidate_count;
    __shared__ unsigned int overflow;

    _morton_topk<K, BLOCK_SIZE, SORT_SIZE, STACK_SIZE>(
        morton_data, dev_query_point[idx_query], search_dist, work_dist_sq, work_idx_old,
        idx_node_stack, stack_count, idx_node, batch_count,
        leaf_count, candidate_count, overflow
    );

    if (threadIdx.x == 0)
    {
        double value = 0.0;
        for (int idx_neighbor = 0; idx_neighbor < K; idx_neighbor++)
        {
            if (work_idx_old[idx_neighbor] == INT_MAX) continue;
            value += static_cast<double>(work_dist_sq[idx_neighbor])
                + 1.0e-12*static_cast<double>(work_idx_old[idx_neighbor]);
        }
        dev_checksum[idx_query] = value;
        if (dev_stack_overflow) dev_stack_overflow[idx_query] = overflow;
    }
}

#endif // GAMEDEV_MORTON_INDEX_CUH
