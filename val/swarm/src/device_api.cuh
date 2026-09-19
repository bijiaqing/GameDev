#ifndef VAL_DEVICE_API_CUH
#define VAL_DEVICE_API_CUH

// keep the analytical test drivers backend-neutral without introducing another simulation abstraction layer
// map every wrapper directly to one CUDA or HIP runtime call and abort at the operation that failed

#include <cstddef>   // std::size_t
#include <cstdlib>   // std::exit, EXIT_FAILURE
#include <iostream>  // std::cerr, std::endl

#ifdef GAMEDEV_CUDA
#include <cuda_runtime.h>  // CUDA allocation, transfer, and synchronization API

inline void val_check (cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return;
    std::cerr << operation << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename T>
void val_malloc (T **pointer, std::size_t count, const char *operation)
{ val_check(cudaMalloc(reinterpret_cast<void **>(pointer), sizeof(T)*count), operation); }

template <typename T>
void val_copy_h2d (T *target, const T *source, std::size_t count, const char *operation)
{ val_check(cudaMemcpy(target, source, sizeof(T)*count, cudaMemcpyHostToDevice), operation); }

template <typename T>
void val_copy_d2h (T *target, const T *source, std::size_t count, const char *operation)
{ val_check(cudaMemcpy(target, source, sizeof(T)*count, cudaMemcpyDeviceToHost), operation); }

inline void val_free (void *pointer, const char *operation)
{ val_check(cudaFree(pointer), operation); }

inline void val_kernel_check (const char *kernel)
{
    // synchronize deliberately: production may defer completion, but a test must attribute a failure to the kernel just run
    val_check(cudaGetLastError(), kernel);
    val_check(cudaDeviceSynchronize(), kernel);
}
#else  // GAMEDEV_ROCM
#include <hip/hip_runtime.h>  // HIP allocation, transfer, and synchronization API

inline void val_check (hipError_t status, const char *operation)
{
    if (status == hipSuccess) return;
    std::cerr << operation << ": " << hipGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename T>
void val_malloc (T **pointer, std::size_t count, const char *operation)
{ val_check(hipMalloc(reinterpret_cast<void **>(pointer), sizeof(T)*count), operation); }

template <typename T>
void val_copy_h2d (T *target, const T *source, std::size_t count, const char *operation)
{ val_check(hipMemcpy(target, source, sizeof(T)*count, hipMemcpyHostToDevice), operation); }

template <typename T>
void val_copy_d2h (T *target, const T *source, std::size_t count, const char *operation)
{ val_check(hipMemcpy(target, source, sizeof(T)*count, hipMemcpyDeviceToHost), operation); }

inline void val_free (void *pointer, const char *operation)
{ val_check(hipFree(pointer), operation); }

inline void val_kernel_check (const char *kernel)
{
    // synchronize deliberately: production may defer completion, but a test must attribute a failure to the kernel just run
    val_check(hipGetLastError(), kernel);
    val_check(hipDeviceSynchronize(), kernel);
}
#endif // GAMEDEV_CUDA

#endif // VAL_DEVICE_API_CUH
