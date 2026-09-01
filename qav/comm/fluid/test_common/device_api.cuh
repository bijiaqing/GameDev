#ifndef QAV_DEVICE_API_CUH
#define QAV_DEVICE_API_CUH

// keep the analytical test drivers backend-neutral without introducing another simulation abstraction layer
// map every wrapper directly to one CUDA or HIP runtime call and abort at the operation that failed

#include <cstddef>
#include <cstdlib>
#include <iostream>

#ifdef GAMEDEV_CUDA
#include <cuda_runtime.h>

inline void qav_check (cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return;
    std::cerr << operation << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename T>
void qav_malloc (T **pointer, std::size_t count, const char *operation)
{ qav_check(cudaMalloc(reinterpret_cast<void **>(pointer), sizeof(T)*count), operation); }

template <typename T>
void qav_copy_h2d (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(cudaMemcpy(target, source, sizeof(T)*count, cudaMemcpyHostToDevice), operation); }

template <typename T>
void qav_copy_d2h (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(cudaMemcpy(target, source, sizeof(T)*count, cudaMemcpyDeviceToHost), operation); }

template <typename T>
void qav_copy_d2d (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(cudaMemcpy(target, source, sizeof(T)*count, cudaMemcpyDeviceToDevice), operation); }

inline void qav_free (void *pointer, const char *operation)
{ qav_check(cudaFree(pointer), operation); }

inline void qav_kernel_check (const char *kernel)
{
    // synchronize deliberately: production may defer completion, but a test must attribute a failure to the kernel just run
    qav_check(cudaGetLastError(), kernel);
    qav_check(cudaDeviceSynchronize(), kernel);
}
#else
#include <hip/hip_runtime.h>

inline void qav_check (hipError_t status, const char *operation)
{
    if (status == hipSuccess) return;
    std::cerr << operation << ": " << hipGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename T>
void qav_malloc (T **pointer, std::size_t count, const char *operation)
{ qav_check(hipMalloc(reinterpret_cast<void **>(pointer), sizeof(T)*count), operation); }

template <typename T>
void qav_copy_h2d (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(hipMemcpy(target, source, sizeof(T)*count, hipMemcpyHostToDevice), operation); }

template <typename T>
void qav_copy_d2h (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(hipMemcpy(target, source, sizeof(T)*count, hipMemcpyDeviceToHost), operation); }

template <typename T>
void qav_copy_d2d (T *target, const T *source, std::size_t count, const char *operation)
{ qav_check(hipMemcpy(target, source, sizeof(T)*count, hipMemcpyDeviceToDevice), operation); }

inline void qav_free (void *pointer, const char *operation)
{ qav_check(hipFree(pointer), operation); }

inline void qav_kernel_check (const char *kernel)
{
    // synchronize deliberately: production may defer completion, but a test must attribute a failure to the kernel just run
    qav_check(hipGetLastError(), kernel);
    qav_check(hipDeviceSynchronize(), kernel);
}
#endif

#endif
