#ifndef LAB_KNN_TYPES_CUH
#define LAB_KNN_TYPES_CUH

#include <cuda_runtime.h>

#include <cukd/builder.h>

#include "morton_types.cuh"

struct kd_point
{
    float3 cartesian;
    int index_old;
    int split_dim;
    int image;
};

struct kd_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };

    static inline __host__ __device__ const point_t &get_point (const kd_point &point) { return point.cartesian; }
    static inline __host__ __device__ float get_coord (const kd_point &point, int dim)
    {
        return cukd::get_coord(point.cartesian, dim);
    }
    static inline __host__ __device__ int get_dim (const kd_point &point) { return point.split_dim; }
    static inline __host__ __device__ void set_dim (kd_point &point, int dim) { point.split_dim = dim; }
};

#endif // LAB_KNN_TYPES_CUH
