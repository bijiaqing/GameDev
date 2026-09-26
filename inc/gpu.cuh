#ifndef GAMEDEV_GPU_CUH
#define GAMEDEV_GPU_CUH

#include <cstdlib>  // std::exit, EXIT_FAILURE
#include <iostream> // std::cerr, std::cout, std::endl

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GAMEDEV_GPU_DEVICE
#endif // __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__

// map API spellings only; hardware-dependent algorithms remain explicit at their call sites
#ifdef GAMEDEV_ROCM
#include <hip/hip_runtime.h>        // HIP runtime API and device qualifiers
#include <hiprand/hiprand_kernel.h> // hiprandState, hiprand_init, hiprand_normal_double, hiprand_uniform_double
#define GPU_BACKEND_NAME "HIP"
#define GPU_BACKEND_ID "rocm"
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuError_t hipError_t
#define gpuFree hipFree
#define gpuFreeHost hipHostFree
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuMalloc hipMalloc
#define gpuMallocHost hipHostMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemset hipMemset
#define gpuSuccess hipSuccess
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEvent_t hipEvent_t
#define gpuFreeAsync hipFreeAsync
#define gpuMallocAsync hipMallocAsync
#define gpuMallocManaged hipMallocManaged
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemsetAsync hipMemsetAsync
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStream_t hipStream_t
#define GPU_THRUST_DEVICE thrust::hip_rocprim::par
#define gpuRandState hiprandState
#define gpuRandInit hiprand_init
#define gpuRandNormalDouble hiprand_normal_double
#define gpuRandUniformDouble hiprand_uniform_double
#if defined(HIP_SYNC_TRACE)
#define GPU_SYNC_TRACE
#endif // HIP_SYNC_TRACE
#else  // !GAMEDEV_ROCM
#include <cuda_runtime.h>  // CUDA runtime API and device qualifiers
#include <curand_kernel.h> // curandState, curand_init, curand_normal_double, curand_uniform_double
#define GPU_BACKEND_NAME "CUDA"
#define GPU_BACKEND_ID "cuda"
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuError_t cudaError_t
#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost
#define gpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuMalloc cudaMalloc
#define gpuMallocHost cudaMallocHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemset cudaMemset
#define gpuSuccess cudaSuccess
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEvent_t cudaEvent_t
#define gpuFreeAsync cudaFreeAsync
#define gpuMallocAsync cudaMallocAsync
#define gpuMallocManaged cudaMallocManaged
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemsetAsync cudaMemsetAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStream_t cudaStream_t
#define GPU_THRUST_DEVICE thrust::device
#define gpuRandState curandState
#define gpuRandInit curand_init
#define gpuRandNormalDouble curand_normal_double
#define gpuRandUniformDouble curand_uniform_double
#if defined(CUDA_SYNC_TRACE)
#define GPU_SYNC_TRACE
#endif // CUDA_SYNC_TRACE
#endif // GAMEDEV_ROCM

// =====================================================================================================================
// host error handling: report the failing call with its location and stop the run

inline __host__
void gpu_fail (gpuError_t status, const char *operation, const char *file, int line)
{
    std::cerr
    << "Error: " GPU_BACKEND_NAME " failure at " << file << ":" << line
    << " during " << operation << ": " << gpuGetErrorString(status)
    << " (" << static_cast<int>(status) << ")"
    << std::endl;

    std::exit(EXIT_FAILURE);
}

#define GPU_CHECK(OPERATION)                                                        \
do {                                                                                \
    gpuError_t gpu_status_ = (OPERATION);                                           \
    if (gpu_status_ != gpuSuccess)                                                  \
    { gpu_fail(gpu_status_, #OPERATION, __FILE__, __LINE__); }                      \
} while (0)

#ifdef GPU_SYNC_TRACE
#define GPU_KERNEL_CHECK(KERNEL_NAME)                                               \
do {                                                                                \
    gpuError_t gpu_status_ = gpuGetLastError();                                     \
    if (gpu_status_ != gpuSuccess)                                                  \
    { gpu_fail(gpu_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }    \
    gpu_status_ = gpuDeviceSynchronize();                                           \
    if (gpu_status_ != gpuSuccess)                                                  \
    { gpu_fail(gpu_status_, KERNEL_NAME " kernel execution", __FILE__, __LINE__); } \
    std::cout << "  [" GPU_BACKEND_NAME "] completed " << KERNEL_NAME << std::endl; \
} while (0)
#else  // !GPU_SYNC_TRACE
#define GPU_KERNEL_CHECK(KERNEL_NAME)                                               \
do {                                                                                \
    gpuError_t gpu_status_ = gpuGetLastError();                                     \
    if (gpu_status_ != gpuSuccess)                                                  \
    { gpu_fail(gpu_status_, KERNEL_NAME " kernel launch", __FILE__, __LINE__); }    \
} while (0)
#endif // GPU_SYNC_TRACE

#endif // GAMEDEV_GPU_CUH
