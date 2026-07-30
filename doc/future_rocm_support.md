# Future AMD GPU and ROCm support

## Assessment

A ROCm backend is feasible, but it is not a mechanical replacement of CUDA names with HIP names. The fluid algorithms mostly use portable kernel language, runtime calls, Thrust operations, synchronization, and shared memory. Swarm transport, radiation, interpolation, and deposition are similarly approachable. Collision search, random-state checkpoints, launch tuning, build rules, and profiler evidence require deliberate redesign.

This assessment is based on source inspection and the existing CUDA verification suites. It has not been confirmed by an AMD compilation because the development system used for this review has neither `hipcc` nor an AMD GPU.

Fluid and swarm will remain independent source branches. No shared fluid–swarm headers or translation units are planned. If both acquire a portability wrapper, each branch will own its own small wrapper with the same naming convention; consistency will be maintained by tests and review rather than cross-representation file sharing.

## Difficulty by component

| Component | Difficulty | Main concern |
| --- | --- | --- |
| fluid transport | low to moderate | standard CUDA syntax and runtime calls |
| fluid diffusion | moderate | LDS capacity, block-line launch limits, and tuning |
| fluid radiation | low to moderate | standard kernels and reductions |
| swarm transport and radiation | low to moderate | interpolation, deposition, and atomics |
| swarm diffusion | moderate | random-number behavior and checkpoint format |
| swarm collisions with KD-tree | high | CUDA-specific third-party implementation |
| swarm collisions with Morton | moderate | portable primitives and compact arrays; current CUDA implementation still requires a HIP port |
| build and profiling | moderate | compiler, architecture, diagnostics, and tool differences |

## What HIP can translate directly

HIP provides close counterparts for:

- `__global__`, `__device__`, `__host__`, `__shared__`, and `__syncthreads()`
- `threadIdx`, `blockIdx`, `blockDim`, and kernel launches
- allocation, copying, deallocation, streams, and synchronization
- kernel error checking
- floating-point `atomicAdd`
- Thrust-style reductions, sorting, and scans through rocThrust
- cuRAND-style interfaces through hipRAND

HIPIFY can bootstrap these substitutions, but it cannot decide launch geometry, checkpoint compatibility, collision-backend policy, numerical tolerances, or the correct build matrix.

The absence of cuFFT, cuBLAS, textures, inline PTX, and explicit warp-shuffle algorithms makes the fluid branch comparatively portable.

## Required engineering

### Build system

The current Makefile assumes `nvcc`, an NVIDIA `sm_*` architecture, NVCC diagnostics, `-Xptxas=-v`, and CUDA relocatable device compilation. Introduce an explicit backend:

```make
GPU_BACKEND ?= cuda
```

The CUDA branch should retain the validated configuration. The HIP branch should select `hipcc`, a model-supplied AMD target such as `gfx90a`, compatible device-link settings, and AMD resource diagnostics.

Model-local source and header precedence should remain unchanged. Backend choice must not mutate the branch's canonical numerical constants.

### Branch-local runtime wrappers

Each representation should receive its own narrow wrapper, for example:

```text
inc/fluid/gpu_runtime.cuh
inc/fluid/gpu_random.cuh
inc/swarm/gpu_runtime.cuh
inc/swarm/gpu_random.cuh
```

Only files actually required by that representation should exist. The wrappers may present parallel names for allocation, copying, synchronization, error checks, streams, dynamic-memory attributes, and random draws, but neither representation should include files from the other.

### Thread and wavefront geometry

`TPB = 32` matches an NVIDIA warp but may occupy only half of a 64-lane AMD wavefront. The current kernels do not appear to depend on implicit warp synchronization, so this is mainly a performance issue.

Block sizes must become backend- and possibly kernel-tunable. Values such as 64, 128, and 256 should be profiled. The fluid block-line implementation is the better initial ROCm target because it avoids the thread-line implementation's large per-thread line arrays, but this must be validated on the target hardware.

### Fluid block-line memory

At a line length of 1024 cells, the block diffusion kernels request roughly 32 KiB for the x solver and 48 KiB for the y and z solvers. HIP exposes dynamic shared-memory controls, but AMD LDS limits and attribute behavior differ by architecture.

The port must:

- query the supported per-block LDS capacity
- reject unsupported line lengths at build or launch time
- verify every synchronization point under AMD wavefront scheduling
- profile LDS occupancy and global-memory traffic

### Collision search

The bundled KD-tree code is the largest CUDA-specific dependency. It uses CUDA headers, streams, allocation policies, `CUDART_VERSION` checks, CUDA compilation macros, and CUDA-oriented sorting paths. Blind renaming can leave incorrect host/device branches or unsupported allocator behavior.

Two realistic policies are:

1. port the KD-tree conservatively, retaining it as a CUDA-reference-compatible backend
2. port the production adaptive-Morton backend's radix sort, scan, hierarchy, compact boundary ghosts, and cooperative selection with rocThrust and rocPRIM

The Morton path is the preferable long-term ROCm target because its data are compact arrays and its operations have direct CUDA and ROCm analogues. It is selectable in the production swarm build, uses location-dependent query radii and compact periodic boundary ghosts, and its clean standalone CUDA matrices pass through $N_P=10^7$. Current end-to-end collision-runtime comparisons, radial halo bins, GPU-native hierarchy construction, multi-GPU exchange, and HIP-specific tuning remain future work.

Equal-key sorting need not create identical internal trees on CUDA and ROCm. Verification should compare exact physical neighbor identifiers outside defined distance ties, followed by collision rates and statistical evolution.

### Random numbers and checkpoints

The swarm branch saves raw vendor RNG states. hipRAND can expose a cuRAND-like API, but CUDA and ROCm state structures, sizes, and sequences are not guaranteed to match.

Consequently:

- CUDA and ROCm stochastic trajectories need not be bitwise identical
- a raw CUDA RNG-state file is not a portable HIP checkpoint
- raw `swarm` structures containing GPU vector types may also differ in alignment

A durable cross-platform restart format should eventually store fixed scalar fields and a backend-neutral logical RNG state, such as an explicitly controlled counter and key. Until then, restart files should be documented as backend-specific.

### Floating-point ordering

Atomic deposition, reductions, sorting ties, and parallel scans may execute in different orders. Cross-vendor verification should compare conservation, analytical norms, convergence orders, state tolerances, collision rates, and distribution statistics. Bytewise CUDA–ROCm equality is not a general requirement.

### Verification and profiling

The test runners should select backend-aware environment tools:

| CUDA | ROCm |
| --- | --- |
| `nvcc --version` | `hipcc --version` |
| `nvidia-smi` | `amd-smi` or `rocm-smi` |
| `-Xptxas=-v` | AMD compiler resource diagnostics |
| Nsight Compute/Systems | rocprofiler and rocTracer-family tools |

The current swarm CUDA archive contains all 25 deterministic/statistical metric files, 25 metadata files, and 10 environment files. The fluid suite is structurally complete, but its local `qav/fluid/out/` directory is currently empty and must be repopulated before it can serve as a stored CUDA baseline.

Required ROCm gates are:

- every deterministic fluid and swarm test passes its existing tolerance
- convergence orders remain consistent with the CUDA baseline
- stochastic diffusion and collision statistics pass their confidence limits
- CUDA and ROCm conserve the same quantities
- restart is tested within each backend
- resource and timing evidence is recorded for both fluid sweep modes and the selected collision backend

## Recommended sequence

1. add backend-aware compiler and linker rules
2. add branch-local fluid runtime wrappers
3. port the fluid block-line transport and diffusion path
4. run and archive the complete fluid suite on CUDA and ROCm
5. port fluid radiation and the thread-line reference path
6. add branch-local swarm wrappers
7. port swarm transport, radiation, interpolation, and deposition
8. run the current 25-case swarm suite on both backends
9. define a portable particle and RNG checkpoint format
10. port and validate production Morton search
11. port the KD-tree only if maintaining it on ROCm remains scientifically or operationally useful
12. tune launch geometry and memory use per architecture

## Effort and recommendation

A compile-capable fluid prototype is likely a matter of days once AMD hardware is available. A verified, tuned fluid backend and collision-disabled swarm backend are larger but bounded tasks. Collision-enabled swarm support, portable restart semantics, dual-backend continuous testing, and performance tuning make complete support a project measured in weeks.

Keep CUDA as the reference backend while bringing up HIP incrementally. Use HIPIFY for initial syntax conversion, but preserve branch-local ownership and validate every numerical stage. Adaptive Morton is the strongest path to portable collision support; the KD-tree should remain the independent CUDA reference while both selectable backends are maintained.

## References

- AMD, [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- AMD, [HIPIFY CUDA Runtime mappings](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)
- AMD, [rocThrust backends](https://rocm.docs.amd.com/projects/rocThrust/en/latest/how-to/rocThrust-build-backends.html)
- AMD, [hipRAND documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)
- AMD, [rocRAND cuRAND compatibility](https://rocm.docs.amd.com/projects/rocRAND/en/latest/conceptual/curand-compatibility.html)
