# Future AMD GPU and ROCm support

## Assessment

Porting GameDev from CUDA to AMD GPUs through ROCm is feasible, but a complete production-quality port is not a simple global replacement of CUDA library and function names with HIP counterparts

The fluid branch is comparatively straightforward because it primarily uses standard CUDA kernel syntax, runtime allocation and copying, shared memory, synchronization, Thrust, and cuRAND-style random initialization. HIP provides close equivalents for these interfaces. The current transport-and-radiation swarm configuration is similarly approachable. The complete swarm implementation is harder because collision handling depends on the bundled cuKD implementation, opaque vendor random-number states are written directly to restart files, and several build and test facilities are explicitly NVIDIA-specific

This assessment is based on source inspection rather than an AMD compilation because the local development system does not provide `hipcc` or an AMD GPU

## Porting difficulty

| Component | Expected difficulty | Main concern |
| --- | --- | --- |
| fluid transport | low to moderate | mostly standard CUDA language and runtime features |
| fluid diffusion and radiation | moderate | dynamic shared-memory limits and target-specific performance |
| swarm transport and radiation | low to moderate | standard kernels, interpolation, deposition, and atomics |
| swarm diffusion | moderate | random-number behavior and checkpoint portability |
| swarm collision and cuKD | high | CUDA-specific third-party code, streams, allocators, macros, and sorting |
| restart portability | high | raw vendor RNG states and compiler-dependent particle structures |
| build and verification infrastructure | moderate | NVIDIA compiler flags, diagnostics, and environment capture |

## CUDA facilities with direct HIP counterparts

Much of the production code uses CUDA features that HIP supports directly or nearly directly:

- `__global__`, `__device__`, `__host__`, `__shared__`, and `__syncthreads()`
- `threadIdx`, `blockIdx`, `blockDim`, and triple-chevron kernel launches
- `cudaMalloc`, `cudaMallocHost`, `cudaMemcpy`, and the corresponding deallocation routines
- `cudaDeviceSynchronize`, `cudaGetLastError`, and kernel error checking
- floating-point atomics such as `atomicAdd`
- Thrust algorithms such as reduction and sorting
- cuRAND-style state initialization and random draws

HIP provides corresponding runtime calls, rocThrust provides a Thrust-compatible interface, and hipRAND provides a cuRAND-compatible interface. HIPIFY can perform much of the initial mechanical translation. It should be treated as a bootstrap tool rather than a completed port because it cannot resolve the backend-specific build, macro, checkpoint, and performance issues described below

The absence of cuFFT, cuBLAS, textures, inline PTX, cooperative groups, and explicit warp-shuffle algorithms makes the fluid port substantially easier

Relevant ROCm documentation:

- [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIPIFY CUDA Runtime API mappings](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)
- [rocThrust build backends](https://rocm.docs.amd.com/projects/rocThrust/en/latest/how-to/rocThrust-build-backends.html)
- [hipRAND documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)

## Required non-mechanical work

### Build system

The `Makefile` currently assumes `nvcc`, `-arch=sm_80`, `--use_fast_math`, NVCC diagnostic controls, `-Xptxas=-v`, and CUDA separate compilation through `--device-c`. Model flag files also append flags directly to `NVCC`

A portable build should introduce an explicit backend selection such as:

```make
GPU_BACKEND ?= cuda
```

The CUDA branch can retain the validated NVCC configuration. The HIP branch should select `hipcc`, a configurable AMD target such as `--offload-arch=gfx90a`, HIP-compatible relocatable-device-code settings, and backend-specific diagnostic and resource-reporting options. The existing `.cu` files may remain if the HIP compiler is explicitly told to compile them as HIP, although this must be encoded consistently in the build rules

### Runtime portability layer

Separate CUDA and HIP source trees should be avoided because they would quickly diverge numerically. A small shared portability layer is preferable, for example:

```text
inc/share/gpu_runtime.cuh
inc/share/gpu_random.cuh
```

These headers can select the backend runtime and provide common wrappers for:

- allocation, copying, deallocation, and synchronization
- error and kernel-launch checking
- stream types and operations
- compiler and device-pass macros
- dynamic shared-memory attributes
- random-number state and draws

The numerical kernels can then remain common to both backends wherever possible

### Thread and wavefront geometry

Both branches currently use `TPB = 32`. This is natural for NVIDIA's 32-thread warp but may execute as only half of a 64-lane wavefront on many AMD Instinct devices. The GameDev kernels do not appear to rely on implicit warp synchronization, so this is primarily a performance issue rather than a correctness issue

Thread-block sizes should become backend- or kernel-tunable. Values such as 64, 128, and 256 should be benchmarked on the target AMD GPU rather than applying one global replacement

The thread-line fluid implementation also contains large thread-local line arrays. Their spilling and occupancy can differ substantially between NVCC and AMD's compiler. The block-line implementation is likely the more suitable initial AMD target, but its advantage must be measured for both advection and diffusion

### Dynamic shared memory

At a line length of 1024 cells, the block diffusion kernels request approximately:

- 32 KiB per block for the x-direction solver
- 48 KiB per block for the y- and z-direction solvers

HIP maps `cudaFuncSetAttribute` to `hipFuncSetAttribute`, but some CUDA-derived shared-memory and cache controls may be ignored or treated only as hints on AMD hardware. The available LDS capacity and permitted dynamic allocation must therefore be queried and validated for each target GPU. Larger grid-line lengths can exceed the limit even if the current fiducial setup succeeds

See the [HIP runtime global definitions](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html)

### cuKD and collision handling

The bundled `inc/swarm/cukd/` implementation is the largest portability obstacle. It contains or depends on:

- CUDA runtime and driver headers
- CUDA stream types and synchronization
- managed and asynchronous CUDA allocation
- `CUDART_VERSION` feature checks
- macros that construct CUDA runtime function names
- Thrust and cudaBitonic implementations
- host/device selection based on `__CUDA_ARCH__`

On AMD targets, `__CUDA_ARCH__` is not the portable device-compilation test; HIP provides `__HIP_DEVICE_COMPILE__`. Blind API renaming can therefore leave cuKD selecting host definitions while compiling device code

The initial HIP collision implementation should favor conservative facilities:

- backend-neutral host/device annotation macros
- rocThrust for the supported sorting path
- ordinary HIP allocation before adopting asynchronous allocators
- explicit backend-aware feature tests instead of `CUDART_VERSION`
- brute-force small-particle comparisons of KD-tree and nearest-neighbor results

Equal-key sorting and parallel construction need not produce an identical tree layout on CUDA and ROCm. Correctness should be assessed from neighbor results and collision statistics rather than the internal tree representation

### Random numbers and restart portability

The swarm branch aliases its random state to `curandState` and saves the raw device-state array to `rngstate_*.dat`. hipRAND maps the cuRAND interface onto rocRAND on AMD, but corresponding rocRAND and cuRAND generators are not guaranteed to produce the same sequence from the same seed

This has three consequences:

1. CUDA and AMD stochastic trajectories should not be expected to agree bit for bit
2. a raw CUDA RNG-state checkpoint is not a portable HIP checkpoint
3. the vendor state structure has no stable cross-backend size or binary representation

For durable cross-platform restarts, GameDev should eventually store a backend-neutral logical random state, such as an explicitly controlled counter and key, rather than an opaque vendor state structure. Merely selecting a similarly named generator through cuRAND and hipRAND does not by itself guarantee identical sequences

See the [rocRAND cuRAND compatibility notes](https://rocm.docs.amd.com/projects/rocRAND/en/latest/conceptual/curand-compatibility.html)

The particle checkpoint currently also stores raw `swarm` structures containing GPU vector types such as `double3`. Compiler-dependent alignment and padding should not be assumed portable between CUDA and HIP. A defined versioned disk structure or fixed scalar arrays would provide a safer cross-backend format. Fluid output already consists primarily of scalar arrays and is naturally easier to exchange

### Floating-point ordering

Swarm deposition uses floating-point atomic additions. CUDA and AMD hardware can process these additions in different orders, causing normal roundoff-level differences. Sorting equal keys and parallel reductions can introduce similar differences

Cross-vendor verification should therefore compare:

- conservation errors
- analytical error norms and convergence orders
- density and velocity tolerances
- collision-rate and size-distribution statistics
- physical invariants

Bytewise output equality is not an appropriate general CUDA-versus-ROCm requirement

### Verification and profiling tools

The analytical Python validators are mostly portable, but the runners and sweep models currently collect or use NVIDIA-specific information including `nvcc --version`, `nvidia-smi`, and `-Xptxas=-v`

The test infrastructure should select backend-aware equivalents such as:

- `hipcc --version`
- `rocminfo`
- `amd-smi` or `rocm-smi`
- AMD compiler resource diagnostics
- ROCm profiling tools for occupancy, memory traffic, and kernel timing

The existing fluid analytical suite should remain the primary correctness gate. The swarm branch also needs targeted tests before full ROCm support can be claimed, especially small deterministic KD-tree tests, deposition conservation tests, diffusion statistics, collision-distribution tests, and restart tests

## Recommended implementation sequence

1. add backend-aware build rules and runtime wrappers
2. port fluid transport using the simplest implementation first
3. port and benchmark the block-line fluid implementation
4. add fluid diffusion and radiation
5. run the complete analytical fluid verification suite on AMD
6. port swarm transport and radiation without collision handling
7. port swarm diffusion and introduce a portable RNG checkpoint design
8. port cuKD and the collision machinery
9. validate nearest-neighbor results and stochastic collision distributions
10. tune block size, LDS use, register pressure, local-memory traffic, and launch geometry for each target GPU

The exact fiducial production configurations are easier than supporting every compile-time feature. In particular, the current collision-disabled swarm model should not be blocked on the complete cuKD port

## Rough effort estimate

These estimates assume access to a working ROCm development system and will depend on the target GPU and compiler version:

- compile-capable fluid HIP prototype: roughly one to three focused engineering days
- validated and reasonably tuned fluid backend: several additional days to approximately two weeks
- current collision-disabled swarm configuration: several additional days after the portability layer exists
- collision-enabled swarm with cuKD and portable RNG/restart support: approximately one to several additional weeks
- mature dual-backend support with automated testing: a project measured in weeks rather than hours

## Recommendation

Preserve native CUDA as the validated reference backend and add HIP conditionally within the same source tree. Use HIPIFY to accelerate the first translation, but manually redesign the build interface, cuKD portability macros, random-state checkpoints, and backend-specific performance settings

The core fluid algorithms are good candidates for ROCm. Full cross-platform support is practical, but scientific equivalence, restart portability, and collision support require substantially more work than replacing CUDA names with corresponding HIP names
