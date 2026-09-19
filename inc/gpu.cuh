#ifndef GAMEDEV_GPU_COMPAT_CUH
#define GAMEDEV_GPU_COMPAT_CUH

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GAMEDEV_GPU_DEVICE
#endif

// API spelling only; hardware-dependent algorithms remain explicit at their call sites.
#ifdef GAMEDEV_ROCM
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#define GPU_BACKEND_NAME "HIP"
#define GPU_BACKEND_ID "rocm"
#define GPU_CHECK HIP_CHECK
#define GPU_KERNEL_CHECK HIP_KERNEL_CHECK
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
#endif
#else
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define GPU_BACKEND_NAME "CUDA"
#define GPU_BACKEND_ID "cuda"
#define GPU_CHECK CUDA_CHECK
#define GPU_KERNEL_CHECK CUDA_KERNEL_CHECK
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
#endif
#endif
#endif
