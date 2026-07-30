#ifndef GAMEDEV_MORTON_GHOST_CUH
#define GAMEDEV_MORTON_GHOST_CUH

#include <cmath>                         // fabsf, fminf, sinf
#include <cstddef>                       // std::size_t
#include <stdexcept>                     // std::invalid_argument

#include <cuda_runtime.h>                // CUDA allocation and kernel-launch API
#include <math_constants.h>              // CUDART_PI_F

#include <thrust/device_ptr.h>           // thrust::device_ptr
#include <thrust/execution_policy.h>     // thrust::device
#include <thrust/scan.h>                 // thrust::exclusive_scan

#include <morton/morton_index.cuh>

static __device__ __forceinline__
float _get_morton_seam_dist (const float3 &point, float x_offset)
{
    float R = hypotf(point.x, point.y);
    float cos_offset = cosf(x_offset);
    return (cos_offset >= 0.0f) ? R*fabsf(sinf(x_offset)) : R;
}

static __global__
void morton_gcount (int *dev_ghost_count, const float3 *dev_point, const float *dev_morton_posx,
    int point_count, float search_dist, float x_min, float x_max)
{
    int idx_point = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_point >= point_count) return;

    int count = 0;
    if (_get_morton_seam_dist(dev_point[idx_point], dev_morton_posx[idx_point] - x_min) <= search_dist) count++;
    if (_get_morton_seam_dist(dev_point[idx_point], x_max - dev_morton_posx[idx_point]) <= search_dist) count++;
    dev_ghost_count[idx_point] = count;
}

static __global__
void morton_gwrite (float3 *dev_record, int *dev_idx_old, const float3 *dev_point, const float *dev_morton_posx,
    const int *dev_ghost_offset, int point_count, float search_dist, float x_min, float x_max)
{
    int idx_point = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx_point >= point_count) return;

    dev_record[idx_point] = dev_point[idx_point];
    dev_idx_old[idx_point] = idx_point;

    float width = x_max - x_min;
    float sin_width;
    float cos_width;
    sincosf(width, &sin_width, &cos_width);
    int idx_ghost = point_count + dev_ghost_offset[idx_point];

    if (_get_morton_seam_dist(dev_point[idx_point], dev_morton_posx[idx_point] - x_min) <= search_dist)
    {
        dev_record[idx_ghost] = make_float3(
            cos_width*dev_point[idx_point].x - sin_width*dev_point[idx_point].y,
            sin_width*dev_point[idx_point].x + cos_width*dev_point[idx_point].y,
            dev_point[idx_point].z
        );
        dev_idx_old[idx_ghost++] = idx_point;
    }
    if (_get_morton_seam_dist(dev_point[idx_point], x_max - dev_morton_posx[idx_point]) <= search_dist)
    {
        dev_record[idx_ghost] = make_float3(
            cos_width*dev_point[idx_point].x + sin_width*dev_point[idx_point].y,
            -sin_width*dev_point[idx_point].x + cos_width*dev_point[idx_point].y,
            dev_point[idx_point].z
        );
        dev_idx_old[idx_ghost] = idx_point;
    }
}

class morton_ghost_index
{
public:
    morton_ghost_index () = default;
    morton_ghost_index (const morton_ghost_index &) = delete;
    morton_ghost_index &operator= (const morton_ghost_index &) = delete;

    void build (const float3 *dev_point, const float *dev_morton_posx, int point_count,
        float search_dist, float x_min, float x_max, float y_min, float y_max,
        float z_min, float z_max,
        bool azimuth_active, int dim, int leaf_target, int max_level)
    {
        if (point_count <= 0) throw std::invalid_argument("Morton ghost point count must be positive");

        bool use_ghosts = azimuth_active && x_max - x_min < 2.0f*CUDART_PI_F - 1.0e-6f;
        if (!use_ghosts)
        {
            record_count_ = point_count;
            unique_ids_ = true;
            build_index(dev_point, nullptr, point_count, y_max, dim, leaf_target, max_level);
            return;
        }

        int *dev_ghost_count = nullptr;
        int *dev_ghost_offset = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_count, sizeof(int)*point_count),
            "allocate Morton ghost counts");
        _morton_cuda_check(cudaMalloc((void**)&dev_ghost_offset, sizeof(int)*point_count),
            "allocate Morton ghost offsets");

        constexpr int thread_count = 256;
        int block_count = (point_count + thread_count - 1) / thread_count;
        morton_gcount <<< block_count, thread_count >>> (
            dev_ghost_count, dev_point, dev_morton_posx, point_count, search_dist, x_min, x_max
        );
        _morton_cuda_check(cudaGetLastError(), "launch morton_gcount");

        thrust::device_ptr<int> count_ptr(dev_ghost_count);
        thrust::device_ptr<int> offset_ptr(dev_ghost_offset);
        thrust::exclusive_scan(thrust::device, count_ptr, count_ptr + point_count, offset_ptr);
        _morton_cuda_check(cudaDeviceSynchronize(), "scan Morton ghost offsets");

        int last_count = 0;
        int last_offset = 0;
        _morton_cuda_check(cudaMemcpy(&last_count, dev_ghost_count + point_count - 1,
            sizeof(int), cudaMemcpyDeviceToHost), "copy final Morton ghost count");
        _morton_cuda_check(cudaMemcpy(&last_offset, dev_ghost_offset + point_count - 1,
            sizeof(int), cudaMemcpyDeviceToHost), "copy final Morton ghost offset");
        record_count_ = point_count + last_offset + last_count;

        float3 *dev_record = nullptr;
        int *dev_idx_old = nullptr;
        _morton_cuda_check(cudaMalloc((void**)&dev_record, sizeof(float3)*record_count_),
            "allocate Morton ghost records");
        _morton_cuda_check(cudaMalloc((void**)&dev_idx_old, sizeof(int)*record_count_),
            "allocate Morton ghost identifiers");

        morton_gwrite <<< block_count, thread_count >>> (
            dev_record, dev_idx_old, dev_point, dev_morton_posx, dev_ghost_offset,
            point_count, search_dist, x_min, x_max
        );
        _morton_cuda_check(cudaGetLastError(), "launch morton_gwrite");

        build_index(dev_record, dev_idx_old, record_count_, y_max, dim, leaf_target, max_level);

        _morton_cuda_check(cudaFree(dev_record), "release Morton ghost records");
        _morton_cuda_check(cudaFree(dev_idx_old), "release Morton ghost identifiers");
        _morton_cuda_check(cudaFree(dev_ghost_count), "release Morton ghost counts");
        _morton_cuda_check(cudaFree(dev_ghost_offset), "release Morton ghost offsets");

        float image_dist_min = 2.0f*y_min*fminf(sinf(z_min), sinf(z_max))
            *fabsf(sinf(0.5f*(x_max - x_min)));
        unique_ids_ = image_dist_min > 2.0f*search_dist;
    }

    morton_view view () const { return morton_index_.view(); }
    bool unique_ids () const { return unique_ids_; }
    int record_count () const { return record_count_; }
    std::size_t persistent_bytes () const { return morton_index_.persistent_bytes(); }

private:
    void build_index (const float3 *dev_record, const int *dev_idx_old, int record_count,
        float y_max, int dim, int leaf_target, int max_level)
    {
        float root_width = 2.0002f*y_max;
        float3 root_origin = make_float3(-0.5f*root_width, -0.5f*root_width, -0.5f*root_width);
        morton_index_.build(dev_record, record_count, root_origin, root_width,
            dim, leaf_target, max_level, dev_idx_old);
    }

    morton_index morton_index_;
    int record_count_ = 0;
    bool unique_ids_ = true;
};

#endif // GAMEDEV_MORTON_GHOST_CUH
