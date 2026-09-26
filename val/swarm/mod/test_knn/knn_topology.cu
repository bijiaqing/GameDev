#ifdef GAMEDEV_ROCM
#define GAMEDEV_ROCM
#else  // !GAMEDEV_ROCM
#define GAMEDEV_CUDA
#endif // GAMEDEV_ROCM
#include "../test_knn/topology.cuh"
