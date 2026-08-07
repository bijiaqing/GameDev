# Future AMD GPU and ROCm support

## Status

An isolated HIP/ROCm reproduction of GameDev now exists under `lab/codex/rocm`. It contains all
22 fluid and 21 swarm production translation units, both fluid sweep implementations, both swarm
collision-search backends, production model configurations, and local copies of the fluid and swarm
verification suites. The CUDA source remains authoritative and was not modified to create the port.

The physical equations, discretizations, state definitions, and test tolerances are unchanged.
CUDA-to-ROCm agreement is therefore expected in analytical norms, convergence order, invariants,
neighbor topology outside defined ties, and stochastic distributions rather than byte for byte.

The port passes its host-side structural check, Make dependency resolution, Python parsing, source
inventory, backend-leak checks, licensing checks, and preservation-manifest validation. The first
native ROCm 7.2 build on an MI300A exposed and led to corrections for three portability assumptions:
the standalone inventory check no longer requires the CUDA parent tree, every HIP translation unit
now force-includes `hip/hip_runtime.h` before using vector types or function qualifiers, and resource
reports use Clang's `kernel-resource-usage` analysis remark instead of the unsupported
`--resource-usage` option. Subsequent native runs compiled and executed the fluid source smoke case
and the complete quick swarm grid group successfully. The standalone KNN programs and production
KD-tree collision backend also compile and link.

The next Morton production build exposed one further compiler constraint: hipRAND's nontrivial
state type cannot be declared directly in shared storage. The Morton event kernel now keeps that
state thread-local to thread zero, which is the only thread that reads or advances it. After this
correction, both production collision backends compiled and linked in 1D, 2D, and 3D; all 15 KNN
edge tests and all 14 periodic tests passed on MI300A. The ordinary matrix then exposed a validator
artifact: separately rounded equal-distance neighbors could exchange sorted position even when the
two backends and stored records returned the same identifier set. The ROCm validator now compares
distances by matching identifier, as specified by the test policy.

The corrected native KNN campaign completed on 2026-08-03 with ROCm 7.2.4 on a `gfx942` MI300A.
All seven ordinary cases, 15 edge cases, 14 periodic cases, and ten wedge cases passed, and both
production collision backends linked successfully in 1D, 2D, and 3D.

The complete native analytical campaign is now also archived. The fluid branch produced all 85
records with the thread sweep and all 85 with the block sweep at the requested resolutions and
parameter variants. Corresponding records agree throughout; the only differences exceeding the
archive comparison's roundoff threshold are two $N=256$ radial-diffusion velocity $L_\infty$ values,
all of order $10^{-12}$. The swarm branch produced all 51 analytical records, and every one of its
24 analytical component manifests reports `passed: true`. Together with the KNN component, the
reconstructed aggregate swarm manifest reports 25 of 25 models passed. This closes the implemented
ROCm numerical matrices at that archive revision. A native restart regression and an automated
CUDA/ROCm archive comparator have since been implemented and await their final cluster archive;
deliberate nonfinite-state injection, AMD profiling, and production-scale performance
characterization remain pending.

The first $N=256$ 3D swarm grid run then exposed a latent endpoint-stencil safety defect shared by
the CUDA and HIP sources. An exact final radial or polar cell-centre coordinate could select a
formally zero-weight neighbor outside its nonperiodic boundary, and an exact final azimuthal cell
centre could select the next flattened cell instead of the periodic image. Because interpolation
loads and deposition atomics are issued even for zero weights, these were genuine out-of-bounds
accesses.
Cell-centred radial and polar endpoints now clamp inclusively, exact azimuthal cell centres select
no neighbor, and true azimuthal seam stencils remain periodic. The corrected native HIP run passes
at $N=256$: density has absolute $L_2=1.31\times10^{-14}$, optical depth has absolute
$L_2=3.21\times10^{-15}$, and the deposited-mass error is $-2.66\times10^{-15}$.

Fluid and swarm remain independent source branches. No shared fluid-swarm headers or translation
units are planned for either GPU backend.

## Implemented backend mapping

| CUDA facility | ROCm implementation |
|---|---|
| `nvcc` and `sm_*` target | `hipcc` and configurable `AMDGPU_TARGET`, initially `gfx942` |
| CUDA Runtime | HIP Runtime |
| cuRAND device API | hipRAND device API |
| CUB radix operations | hipCUB |
| Thrust device algorithms | rocThrust with explicit HIP execution policies where required |
| dynamic shared memory | AMD LDS with a runtime capacity check |
| `nvidia-smi` environment record | `amd-smi` and `rocm-smi` with unavailable fallbacks |
| PTXAS resource report | Clang `-Rpass-analysis=kernel-resource-usage` diagnostic |
| `.cu` translation units | isolated `.hip` translation units |

The port includes fluid transport, diffusion, source terms, radiation, optical depth, initialization,
and finite-state checks. It also includes swarm transport, diffusion, drag, radiation, Poynting-
Robertson drag, deposition, interpolation, initialization, collisions, KD-tree KNN, adaptive-Morton
KNN, compact periodic ghosts, checkpoint I/O, and random-state handling.

## Correctness decisions

### Floating-point semantics

ROCm correctness builds deliberately omit blanket `-ffast-math`. Clang's `-ffast-math` implies
finite-only assumptions for host and device code, which can invalidate `isfinite`, `isnan`, and
`isinf` guards. This is broader than NVCC's `--use_fast_math` behavior and would compromise fluid
state detection and KNN overflow validation.

`tools/static_check.py` rejects future use of `-ffast-math` unless it is paired with
`-fno-finite-math-only`. After the native numerical baseline passes, an optional performance build
may test device-only flags such as

```text
-Xarch_device -ffast-math -Xarch_device -fno-finite-math-only
```

Host initialization, quadrature, validation, and long-double accumulation should remain precise.
Any optimized configuration must rerun the finite-state regression and the complete numerical
suite before it is accepted.

### Wavefront geometry

`TPB=32` is retained for correctness bring-up. The kernels do not use warp masks, warp shuffles,
implicit warp synchronization, or other 32-lane assumptions, so a 64-lane AMD wavefront changes
occupancy rather than numerical meaning. Values such as 64, 128, and 256 should be benchmarked only
after the reference tests pass.

### Fluid block-line LDS

The block diffusion kernels request dynamic LDS proportional to line length. For double precision,

$$
S_x = 4N_X(8\,\mathrm{B}),
\qquad
S_y = 6N_Y(8\,\mathrm{B}),
\qquad
S_z = 6N_Z(8\,\mathrm{B}).
$$

At a line length of 1024, these are 32 KiB for the x solver and 48 KiB for the y and z solvers.
MI300A/gfx942 provides 64 KiB of LDS per compute unit and permits one workgroup to request up to
64 KiB. The HIP runtime queries `hipDeviceAttributeMaxSharedMemoryPerBlock` and aborts before launch
when a requested line exceeds the reported limit. For the six-array solvers, the 64-KiB ceiling is
$N\leq1365$ in double precision.

Unlike the CUDA implementation, the HIP path does not request a CUDA-style dynamic-shared-memory
opt-in attribute. A native launch using more than 48 KiB must confirm that this policy works on the
actual gfx942 runtime.

### Fluid thread-line stack limit

The legacy thread-line X advection kernel stores 21 line-sized double-precision arrays in one
thread stack frame. At $N_X=1024$, the native gfx942 linker reports a 172,048-byte frame, exceeding
its 131,056-byte limit. This is the original local-memory scaling problem, not a change in the HIP
translation. The block-line implementation removes those per-thread arrays and remains the
production path at $1024^2$.

ROCm therefore uses a $512^2$ thread-versus-block equivalence test by default, safely below the
linker ceiling, while retaining the $128^3$ comparison. The reduced comparison size still tests
operation equivalence; it is not a production performance benchmark. Production-scale HIP timing
and correctness runs should use the block sweep.

The native $512^2$ comparison passes after 630 accepted steps in both implementations. At frame 10,
the relative $L_2$ differences between block and thread are $2.01\times10^{-14}$ in density,
$2.24\times10^{-16}$ in azimuthal velocity, and $3.51\times10^{-15}$ in radial velocity; vertical
velocity is byte-identical. Both runs have the same mass drift,
$-3.61\times10^{-9}$, and their final masses are identical. The measured wall times were 34.29 s
for thread and 21.88 s for block, giving $t_{\mathrm{block}}/t_{\mathrm{thread}}=0.638$, or a
preliminary 1.57-fold block speedup on that run. A separate $1024^2$ block-only smoke test compiled,
linked, and advanced to $t=0.1$ in one accepted step; the equivalent thread build remains impossible
because of the linker stack limit.

The native $128^3$ diffusion comparison also passes, with 5,392 accepted steps in both
implementations. At frame 10, the relative $L_2$ differences are
$2.70\times10^{-12}$ in density, $5.16\times10^{-16}$ in azimuthal velocity,
$3.08\times10^{-13}$ in radial velocity, and $2.82\times10^{-14}$ in polar velocity. The final
mass mismatch is $-3.89\times10^{-16}$. On this run, the thread and block wall times were 223.45 s
and 579.63 s, respectively, so the block method was 2.59 times slower. This single measurement
shows that the block rewrite is numerically equivalent in 3D but does not establish a universal
performance advantage; sweep choice should remain dimension- and resolution-dependent until
kernel-level profiling explains the 3D result.

ROCm QA metadata and transcripts now use JSON consistently. Analytical drivers write
`meta_N*.json`; sweep runs write `build.json`, `run.json`, `variables.json`, `timing.json`, and the
aggregate `sweep_comparison_*.json`. The raw compiler and runtime streams are retained as escaped
strings inside the corresponding JSON objects, while accepted steps and parameters are also stored
as typed fields. `tools/convert_qa_text.py` migrates older downloaded QA text artifacts without
discarding their contents.
Sweep artifacts include resolution, output count, and output interval in a configuration tag, so
quick and full runs coexist rather than silently replacing one another.

### Collision search

Both exact search backends are present:

- the bundled KD-tree has a conservative HIP port with synchronous `hipMalloc`/`hipFree` resource
  management
- the adaptive-Morton hierarchy uses hipCUB, rocThrust, explicit HIP execution policies,
  block-cooperative top-$K$ selection, and compact periodic ghosts

The synchronous KD-tree allocator favors predictable error propagation during bring-up. An
asynchronous allocator may be benchmarked later, but it requires its own correctness and lifetime
regression.

CUDA and ROCm sorting may resolve equal-distance or equal-key cases differently. KNN validation
must therefore compare physical neighbor identifiers outside defined floating-point ties and use
independent brute force for every backend disagreement. Collision-rate agreement and statistical
size evolution remain the end-to-end scientific criteria.

The optional `KDTREE_STATS` path is disabled. Its header-defined `g_traversalStats` device variable
does not have internal linkage; if statistics are enabled later, it must become an `extern`
declaration with one translation-unit definition before relocatable device linking is used.

#### Native MI300A KNN result

The corrected quick KNN suite used $10^5$ particles per benchmark case. Its final summary was

```text
knn: PASS ordinary=7/7 edge=15/15 periodic=14/14 wedge=10/10 production links=6/6
```

The reported timing ratio is $t_{\mathrm{KD}}/t_{\mathrm{Morton}}$, so values greater than one
favor Morton. In the ordinary smooth, ring, clump, and radial cases, this ratio was
$0.421$--$0.866$: KD-tree queries were faster at this particle count. Morton persistent memory was
$0.718$--$0.756$ of the KD-tree allocation, a reduction of about $24$--$28\%$.

For standard periodic-wedge cases, the timing ratio was $0.669$--$0.845$, while Morton memory was
$0.247$--$0.451$ of KD-tree memory. In the deliberately difficult narrow-seam cases, Morton became
faster: the ratios were $1.216$ in 2D and $1.879$ in 3D, with memory ratios $0.485$ and $0.491$.
The compact periodic-ghost path therefore provides its clearest benefit when most queries interact
with the periodic seam, while the KD-tree remains faster for the tested ordinary $10^5$-particle
workloads.

These measurements establish portability and a preliminary regime trend, not production
performance. They use one particle count and one native campaign; larger particle counts, repeated
timing statistics, end-to-end collision cost, and AMD profiler evidence are still required.

### Random numbers and checkpoints

`hiprandState` provides the required XORWOW-style initialization and sampling interface, but raw
vendor RNG structures are not cross-platform data formats. Consequently,

- CUDA and ROCm stochastic trajectories need not be bitwise identical
- CUDA `rngstate_*.dat` files must not be loaded by the HIP build, or vice versa
- restart must be tested independently within each backend
- physical particle fields remain portable when scalar type, model configuration, and field format
  agree

A backend-neutral counter-based RNG checkpoint remains a possible future improvement, not a
prerequisite for native ROCm validation.

The test-local `test_restart_2d` regression now exercises the production checkpoint functions. It
evolves one reference state through two diffusion steps and a second state through one step,
checkpoint output, deliberate device-memory clobbering, checkpoint reload, and one more step. The
test requires the reloaded device RNG bytes to match the checkpoint file and the final particle
positions to match the uninterrupted trajectory byte for byte; it also requires velocity
differences introduced by linear-velocity file serialization to remain below
$2\times10^{-14}$. This is deliberately a native ROCm restart test and does not attempt to load a
CUDA RNG checkpoint.

### CUDA-versus-ROCm archive comparison

`qav/compare_backends.py` compares already archived backend results without rerunning either
executable. For the fluid branch it requires the complete 85-record matrix and compares every
deterministic metric. For the common swarm matrix it requires all 51 CUDA-comparable records;
`test_restart_2d` is excluded because it validates the backend-specific RNG representation.
Deterministic values use $10^{-6}$ relative and $10^{-11}$ absolute tolerances by default.
Stochastic diffusion records are not compared pathwise: both backend validators must pass their
own analytical mean and variance bounds with matching case and resolution metadata. KNN comparison
requires both native suites to pass and their ordinary, edge, periodic, wedge, and production-link
coverage counts to agree. Performance timings are intentionally excluded from correctness.

Run the comparison from `lab/codex/rocm` after both archives have been copied into their respective
`qav/*/out/` trees:

```bash
python3 qav/compare_backends.py --component all
```

For a relocated ROCm tree, pass the CUDA archive explicitly with
`--cuda-qav /path/to/cuda-project/qav`; `CUDA_QAV` provides the equivalent environment override.

The current local archives already pass all 51 shared swarm analytical comparisons with zero
metric mismatches. The complete cross-backend gate is not yet closed locally because the CUDA
fluid archive is absent and the archived CUDA KNN manifest has ten periodic cases while the current
ROCm manifest has fourteen; the CUDA fluid and current KNN suites must be archived before the full
comparison can pass.

### Source synchronization

`tools/sync_from_cuda.py` performs the mechanical CUDA-to-HIP refresh. It replaces the generated
`inc/`, `src/`, `mod/`, and `qav/` trees only after validating and snapshotting every path listed in
`tools/rocm_preserve.txt`; registered files are restored byte for byte afterward. The manifest
currently protects 46 hand-maintained HIP adaptations, runners, build files, and ROCm-specific QA
documents.

Any new ROCm-only adaptation must be added to the manifest before the next refresh. The static
checker rejects missing and duplicate entries. Synchronization never writes to the CUDA tree.

### Licensing

The isolated port contains GameDev's MIT license. The HIP KD-tree directory also contains the full
Apache License 2.0 text; `NOTICE.md` identifies cudaKDTree and cudaBitonic, and modified upstream
headers carry file-level modification notices. The static checker verifies that both license files
are byte-identical to their canonical copies in the CUDA repository.

## Static verification completed

Run from `lab/codex/rocm`:

```bash
python3 tools/static_check.py
```

The current result is:

```text
ROCm static check: PASS
static analysis passed; native execution evidence is recorded separately by qav manifests
```

The checker verifies:

- one HIP counterpart for every CUDA production and QA translation unit
- no buildable `.cu` files or active CUDA Runtime, cuRAND, NVCC, or NVIDIA diagnostic leakage
- parseable Python drivers
- representative fluid, swarm, KD-tree, Morton, and QA dependency graphs
- safe floating-point build flags
- the MIT and Apache license copies
- the hand-maintained-file preservation manifest

This check cannot prove compiler compatibility, device execution, numerical accuracy, resource
availability, or performance.

## Required native validation

On an MI300A node, begin with:

```bash
cd lab/codex/rocm

python3 tools/static_check.py
python3 qav/fluid/test_common/run_suite.py --group source --res 8 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group grid --res 32 --quick --target gfx942
python3 qav/swarm/test_common/run_suite.py --group knn --res 32 --target gfx942
```

The first native campaign must establish all of the following:

1. HIP headers, math constants, hipRAND, rocThrust, hipCUB, KD-tree templates, and device linking
   compile successfully
2. a deliberately injected NaN or Inf is detected by the production fluid guard and KNN validator
3. a block-diffusion kernel requesting more than 48 KiB but no more than 64 KiB launches correctly
4. deterministic analytical tests satisfy their existing tolerances and convergence orders
5. stochastic diffusion and collision tests satisfy their confidence limits
6. KD-tree, Morton, and brute-force KNN comparisons satisfy the established tie policy
7. mass, momentum, and other documented invariants match the CUDA scientific baseline
8. checkpoint/restart succeeds within the HIP backend
9. compiler, driver, target, device, resource, timing, and test manifests are archived

Items 1, 4, and 6 are now established by the complete analytical and KNN campaigns, including six
production-backend links. The matched $512^2$ and $128^3$ fluid sweeps also establish the current
thread/block equivalence evidence. The restart test and cross-backend comparator are implemented
but require their final native archives. The remaining requirements concern deliberate failure-path
tests, larger-LDS launch coverage, the complete cross-backend archive, restart execution, and
profiling.

After the smoke tests pass, run the full matrices:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/fluid/test_common/run_suite.py --group sweep --target gfx942
```

## Performance work after correctness

Only after the native matrices pass should the port evaluate:

- 64, 128, and 256 threads per block on the 64-lane architecture
- thread-line versus block-line fluid sweeps
- device-only fast math with finite-value semantics restored
- removal of `-fgpu-rdc` after a successful non-RDC link and full regression
- synchronous versus asynchronous KD-tree allocation
- KD-tree versus Morton build, query, memory, and end-to-end collision cost
- LDS occupancy, bank conflicts, scratch use, HBM traffic, and kernel duration through AMD profiling
- MI300A unified-memory opportunities without changing numerical ownership or I/O semantics

No performance claim should be made from peak FLOP or bandwidth specifications alone. Production
comparison requires matched grids, particle counts, physics flags, output cadence, accepted steps,
and completed simulated time.

## Promotion into the production build

The current ROCm reproduction intentionally remains isolated. A later integration may introduce a
backend selection in the root build, but it must not merge fluid and swarm source ownership or make
the CUDA reference depend on HIP headers. Promotion requires:

- complete native ROCm test evidence
- a maintained CUDA-versus-ROCm regression policy
- documented backend-specific checkpoint restrictions
- stable compiler and target requirements
- performance evidence for the intended production configurations
- a synchronization or generation strategy that cannot silently overwrite hand-maintained code

Until those conditions are met, CUDA remains the production reference and the HIP tree remains an
experimental, independently testable reproduction.

## References

- AMD, [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- AMD, [HIP compiler and runtime documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- AMD, [MI300 ISA and LDS description](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- AMD, [rocThrust documentation](https://rocm.docs.amd.com/projects/rocThrust/en/latest/)
- AMD, [hipCUB documentation](https://rocm.docs.amd.com/projects/hipCUB/en/latest/)
- AMD, [hipRAND documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)
- LLVM, [Clang floating-point behavior](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior)
- LLVM, [Clang HIP support](https://clang.llvm.org/docs/HIPSupport.html)
