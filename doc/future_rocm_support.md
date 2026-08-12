# AMD GPU and ROCm support

## Status

The tracked HIP/ROCm reproduction of GameDev now exists under `rocm/`. It contains all
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
all of order $10^{-12}$. The swarm branch produced all 51 CUDA-comparable analytical records, and
every one of its 24 analytical component manifests reports `passed: true`. The subsequently
archived native restart regression adds one model and one record. Together with the KNN component,
the rebuilt ROCm aggregate reports 26 of 26 models and 52 analytical builds passed. This closes the
implemented native ROCm numerical matrices at that archive revision. The subsequently executed
nonfinite-state injection branch also passed natively on gfx942; AMD profiling and production-scale
performance characterization remain pending.

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
as typed fields.
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
- a simulation started with one backend must be resumed only with that same backend

GameDev will not implement cross-backend restart. Even where density, position, or velocity files
have an identical scalar layout, they are not accepted as a checkpoint for the other backend. Such
files may still be read by external analysis software, but the simulation runtime must not use them
to continue an evolution. A backend-neutral random-number checkpoint is therefore unnecessary for
the planned production interface.

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

Run the comparison from `rocm/` after both archives have been copied into their respective
`qav/*/out/` trees:

```bash
python3 qav/compare_backends.py --component all
```

For a relocated ROCm tree, pass the CUDA archive explicitly with
`--cuda-qav /path/to/cuda-project/qav`; `CUDA_QAV` provides the equivalent environment override.

The 2026-08-11 comparison has complete coverage: all 51 shared swarm records and the KNN component
pass, while 79 of the 85 fluid records satisfy the original strict comparison of derived analytical
metrics. All 77 reported values outside those metric tolerances were confined to six of the eight
`test_z_transport_3d` records. Both backends used identical accepted-step counts and recovered
effectively identical density convergence orders. The large apparent differences occurred in
velocity norms at the compact profile's numerical-vacuum edge, where the old validator selected
cells from analytical density alone even when the numerical state used a vacuum fallback velocity.
Requiring both analytical and numerical density reduced the metric mismatches from 77 to 50 and
the velocity subset from 34 to seven.

Both validators now require analytical and numerical density to exceed $10^{-12}$ of the exact
peak before including a cell in velocity norms. CUDA also supports an isolated
`--math-mode precise` QA archive. Rerun only the eight polar records and compare them against the
complete ROCm archive with:

```bash
python3 qav/fluid/test_z_transport_3d/run.py \
    --math-mode precise --cfl 0.05 --res 32 64 128 256
python3 qav/fluid/test_z_transport_3d/run.py \
    --math-mode precise --cfl 0.5 --res 32 64 128 256

python3 qav/compare_backends.py \
    --component fluid \
    --cuda-qav /path/to/cuda/qav \
    --rocm-qav /path/to/rocm/qav \
    --cuda-sweep thread_precise \
    --rocm-sweep thread \
    --allow-partial
```

The partial mode explicitly requires every CUDA record present to have a ROCm counterpart while
allowing the ROCm side to retain its additional full-matrix records. It is not a substitute for
the ordinary complete-archive gate.

For a direct solution comparison, `qav/compare_fluid_fields.py` reads the matched raw arrays before
analytical norm reduction. Initial density must be byte-identical. Final density and conserved
momenta must satisfy relative $L_2$ and $L_\infty$ differences no larger than $10^{-5}$, with an
absolute $10^{-10}$ fallback for a vanishing field. Physical velocities retain unweighted relative
and maximum differences as diagnostics, but their acceptance criterion is the mass-weighted norm

$$
E_v=
\left[
\frac{\sum_i \bar\rho_i\left(v_i^{\rm ROCm}-v_i^{\rm CUDA}\right)^2}
     {\sum_i \bar\rho_i\left(v_i^{\rm CUDA}\right)^2}
\right]^{1/2}
\le 10^{-5},
\qquad
\bar\rho_i=\frac{\rho_i^{\rm CUDA}+\rho_i^{\rm ROCm}}{2}.
$$

Only cells resolved above the common relative-density floor enter this velocity norm. This keeps
low-mass compact-support tails visible in the diagnostic columns without allowing division by
near-vacuum density to dominate the cross-backend correctness gate.

The native direct-field comparison subsequently passed all eight polar records. Initial densities
were byte-identical and every metadata record, final time, and accepted-step count agreed. Across
final density and all three conserved momenta, the worst discrepancies were
$L_{2,\mathrm{rel}}=1.93\times10^{-6}$ and
$L_{\infty,\mathrm{rel}}=5.09\times10^{-6}$, both in the coarsest $N=32$, CFL-$0.5$ run and both
below the $10^{-5}$ cross-vendor boundary. The largest mass-weighted velocity discrepancy over any
component, resolution, or CFL value was only $6.59\times10^{-11}$. Unweighted velocity differences
reached $2.06\times10^{-4}$ relatively in negligible compact-support tails, confirming why they
must remain diagnostic rather than define the physical equivalence gate.

The precise CUDA rerun produced conserved analytical error norms identical to the default
fast-math CUDA records. Therefore `--use_fast_math` was not the cause of the residual derived-norm
differences. Those differences arise from cross-vendor floating-point evolution followed by a
sensitive reduction against the analytical solution; the direct solution comparison establishes
that they are far below the discretization error. Together with the complete swarm and KNN results,
this closes the current CUDA-versus-ROCm scientific comparison.

When `qav/out/backend_field_comparison_z.json` is present and passed,
`qav/compare_backends.py` uses that raw-field result for the eight polar records while continuing
to require exact agreement of their case, resolution, final time, accepted-step count, and CFL
metadata. The other 77 fluid records retain the ordinary strict recursive metric comparison. This
allows the aggregate gate to report the scientifically relevant result without weakening global
tolerances or silently ignoring the polar case.

### Source synchronization

`tools/sync_from_cuda.py` performs the mechanical CUDA-to-HIP refresh. It replaces the generated
`inc/`, `src/`, `mod/`, and `qav/` trees only after validating and snapshotting every path listed in
`tools/rocm_preserve.txt`; registered files are restored byte for byte afterward. The manifest
currently protects 46 hand-maintained HIP adaptations, runners, build files, and ROCm-specific QA
documents.

Any new ROCm-only adaptation must be added to the manifest before the next refresh. The static
checker rejects missing and duplicate entries. Synchronization never writes to the CUDA tree.

### Licensing

The repository MIT license covers GameDev's HIP implementation. The HIP KD-tree directory retains
the full Apache License 2.0 text, the root README identifies cudaKDTree and cudaBitonic, and modified
upstream headers carry file-level modification notices. The static checker verifies that the
third-party license is byte-identical to its canonical copy in the CUDA implementation.

## Static verification completed

Run from `rocm/`:

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
cd rocm

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

All nine requirements are now established by the complete analytical, restart, KNN, failure-path,
large-LDS, and direct cross-backend fluid campaigns, including six production-backend links. The
matched $512^2$ and $128^3$ fluid sweeps also establish the current thread/block equivalence
evidence. Production-scale performance profiling remains separate from these correctness gates.

After the smoke tests pass, run the full matrices:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/fluid/test_common/run_suite.py --group sweep --target gfx942
```

## Remaining ROCm qualification procedures

The analytical and cross-backend matrices establish numerical portability, but they do not replace
explicit testing of controlled failure paths, large-LDS launches, or production performance. These
three activities must remain separate because they use different acceptance criteria: an expected
failure must terminate cleanly, a launch test must exercise a particular hardware resource boundary,
and a benchmark must measure an uninstrumented workload repeatedly.

The `failure` and `lds` interfaces described below are implemented in the tracked ROCm QA dispatcher.
The separate `qav/perf` interface is also implemented for uninstrumented timing, warm-up memory
sampling, runtime tracing, and focused hardware-counter collection. Native MI300A performance
results have not yet been archived, so the implementation is ready to run but supports no speed
claim yet.

### Deliberate nonfinite-state injection

The fluid failure driver exercises the production guard rather than merely asking Python whether a
saved array is finite. It initializes a valid test-local device state, replaces exactly one value,
and invokes the same production `inf_cell_flag`, `cfl_rate_calc`, and `get_dt_cfl` paths used by the
runtime. Separate subprocesses inject `NaN` and positive infinity into density, all three conserved
momentum components, all three primitive variables, and radiation-enabled optical depth.

Each state-guard subprocess must return nonzero, print

```text
Error: non-finite simulation state at cell (ix,iy,iz)
```

and identify the deliberately corrupted cell. A HIP illegal-memory-access message, a hang, a
nominally successful return, or a reported cell different from the injected cell is a failure. Only
one cell is corrupted per subprocess because the guard deliberately uses an atomic first-writer
record. The runner records the field, injected value, cell index, return code, matched diagnostic,
and captured stream in JSON.

The CFL rejection path needs its own case. After injecting a nonfinite primitive or momentum, the
test should run `cfl_rate_calc` and `get_dt_cfl` and verify that the affected ring receives an
infinite rate and the host terminates instead of returning a finite or zero timestep:

$$
\operatorname{CFLrate}_{\rm bad}=\infty.
$$

The swarm failure driver builds both collision-search backends. Clean controls must pass through the
selected KD-tree or Morton builder. Separate subprocesses then inject `NaN` and infinity into each
particle position, velocity, size, and represented-number field. The production `col_site_init`
records the first bad particle and the host aborts before index construction. Additional probes place
nonfinite values in collision rate and KNN radius arrays and invoke the production `inf_rate_flag`;
the collision runtime applies this check before event sampling. Silently dropping the particle or
returning an apparently valid list is therefore a failure.

The implemented interfaces are

```bash
python3 qav/fluid/test_common/run_suite.py --group failure --target gfx942
python3 qav/swarm/test_common/run_suite.py --group failure --target gfx942
```

Every expected-failure case passes only when all of the following are true:

```text
return code is nonzero
expected diagnostic is present
reported object or cell equals the injected target
no HIP runtime failure is reported
the subprocess terminates within its timeout
```

Correctness builds must continue to omit blanket `-ffast-math`, because finite-only compiler
assumptions can invalidate the very checks exercised by this branch.

The native gfx942 campaign on 2026-08-11 passed all 32 fluid cases and all 44 swarm cases. The fluid
set comprised two clean controls and 30 expected failures; the swarm set comprised two clean
controls plus 20 expected failures for each of KD-tree and Morton. Every injected case returned one,
reported the intended cell or particle, terminated within its timeout, and contained no HIP runtime
fault. The downloaded manifests independently confirm zero timeouts and complete per-case pass
status.

### Large-LDS launch coverage

The existing $N=1024$ six-array diffusion launch requests exactly 48 KiB and therefore does not
exercise the greater-than-48-KiB regime. In double precision, the dynamic LDS requests are

$$
S_x=32N_X\ \mathrm{B},
\qquad
S_y=48N_Y\ \mathrm{B},
\qquad
S_z=48N_Z\ \mathrm{B}.
$$

The launch tests must use anisotropic grids; a cubic $N=1280$ model would contain more than two
billion cells and is neither necessary nor appropriate. The following compact models exercise the
actual block diffusion kernels with substantial LDS requests:

| Kernel | Test grid | Dynamic LDS |
|---|---:|---:|
| `diffusion_xbl` | $N_X=1792$, $N_Y=8$, $N_Z=1$ | 56 KiB |
| `diffusion_ybl` | $N_X=8$, $N_Y=1280$, $N_Z=1$ | 60 KiB |
| `diffusion_zbl` | $N_X=8$, $N_Y=8$, $N_Z=1280$ | 60 KiB |

Each positive case must query the device LDS limit and the kernel attributes, launch the production
kernel, synchronize, check the HIP launch status, verify finite density and momentum, verify mass
conservation, and compare the result with either the thread solver or an analytical diffusion
reference. Its JSON record should include at least

```json
{
  "device_lds_limit": 65536,
  "kernel_static_lds": 16,
  "requested_dynamic_lds": 61440,
  "total_lds": 61456,
  "launch_succeeded": true,
  "finite": true,
  "mass_relative_error": 0.0
}
```

The shared production host check queries both kernel attributes and the active device. The block
diffusion kernels contain statically allocated shared scalars, so the rigorous launch condition is

$$
S_{\rm static}+S_{\rm dynamic}\leq S_{\rm device,max}.
$$

Before qualifying a near-limit launch, obtain `sharedSizeBytes` and
`maxDynamicSharedSizeBytes` with `hipFuncGetAttributes` and require both the per-function dynamic
limit and the static-plus-dynamic device limit to be satisfied. This distinction matters most for
`diffusion_xbl`, which has more static shared scalars than the radial and polar solvers.

A negative radial or polar case should use $N=1366$, for which

$$
48N=65568\ \mathrm{B}>64\ \mathrm{KiB}.
$$

The host capacity check must reject this configuration before launch with a controlled diagnostic.
A later `hipErrorInvalidValue`, illegal access, or device fault does not pass the test. A near-limit
$N=1365$ case may also be retained to verify the exact static-plus-dynamic accounting, but the
60-KiB cases above provide the safer primary coverage.

The implemented interface is

```bash
python3 qav/fluid/test_common/run_suite.py --group lds --target gfx942
```

The three positive models launch the production kernels on a uniform density and momentum state.
This is an exact zero-gradient diffusion solution, so the runner requires finite output, a maximum
stationary-state error no larger than $10^{-11}$, and relative finite-volume mass error no larger
than $10^{-12}$. The negative $N_Y=1366$ model must return nonzero with the controlled host-capacity
diagnostic and without a HIP runtime fault. Each record stores the device limit, compiled static LDS,
kernel dynamic limit, requested dynamic LDS, total LDS, process status, and conservation metrics in
JSON.

The native gfx942 campaign on 2026-08-11 passed all four cases without compiler diagnostics,
timeouts, or HIP runtime faults. The measured results were

| Kernel | Static LDS | Dynamic LDS | Total LDS | Maximum stationary error | Relative mass error |
|---|---:|---:|---:|---:|---:|
| `diffusion_xbl` | 80 B | 57344 B | 57424 B | $2.22\times10^{-16}$ | $1.39\times10^{-16}$ |
| `diffusion_ybl` | 16 B | 61440 B | 61456 B | $4.44\times10^{-16}$ | $2.26\times10^{-17}$ |
| `diffusion_zbl` | 16 B | 61440 B | 61456 B | $4.44\times10^{-16}$ | $1.79\times10^{-17}$ |

The device reported a 65536-byte per-block LDS limit. The compiled X kernel reported a 65456-byte
dynamic limit after its 80-byte static allocation; Y and Z each reported 65520 dynamic bytes after
16 static bytes. The negative radial case requested 65568 dynamic and 65584 total bytes, returned
one through the controlled host diagnostic, and never attempted a kernel launch. This completes the
native large-LDS correctness qualification.

Successful launch does not imply good performance. A 60-KiB workgroup severely constrains
LDS-based residency on a 64-KiB compute unit, so the positive cases should subsequently be profiled
for occupancy, LDS traffic, and bank conflicts.

### AMD profiling workflow

Profiling should begin only after an uninstrumented benchmark has established a stable timing
baseline. ROCm profilers add overhead, and ROCm Compute Profiler can replay kernels or the
application to collect incompatible counter sets. Profiled wall time must therefore never be used
as the reported production speed.

Inside an exclusively allocated MI300A compute node, first record the available tools and counters:

```bash
module load rocm/7.2

command -v rocprofv3
command -v rocprof-compute
rocprofv3 --version
rocprof-compute --version
rocprofv3 --list-avail
rocprof-compute profile --list-available-metrics
```

Use `rocprofv3` on a short production-like run to identify dominant kernels and distinguish kernel,
copy, allocation, synchronization, and scratch costs:

```bash
mkdir -p qav/perf/out/fluid_3d_trace

rocprofv3 \
    --runtime-trace \
    --stats \
    --output-format json \
    --output-directory qav/perf/out/fluid_3d_trace \
    --output-file fluid_3d \
    -- qav/fluid/test_sweep_block_3d/gamedev
```

The trace should be used to rank kernels by total duration, average duration, call count, scratch
traffic, and transfer or synchronization overhead. Likely candidates include the directional
advection and diffusion kernels, `source_update`, `col_rate_calc`, `col_event_run`, and the KD-tree
or Morton search kernels, but selection must follow the measured trace rather than this expectation.

After identifying a hot kernel, use ROCm Compute Profiler with a kernel-name filter on a deliberately
short run. For example,

```bash
ROCPROFCOMPUTE_COLOR=0 rocprof-compute profile \
    --name gamedev_diffusion_ybl \
    --no-roof \
    --kernel diffusion_ybl \
    -- qav/fluid/test_sweep_block_3d/gamedev

rocprof-compute analyze \
    --path workloads/gamedev_diffusion_ybl/MI300A
```

Use the tool's available-metric listing rather than assuming that report-block numbers are stable
between ROCm releases. For fluid block kernels, collect wavefront occupancy, VGPR and SGPR pressure,
scratch traffic, HBM and cache traffic, LDS utilization, LDS bank conflicts, instruction mix, and
VALU utilization. The main question is whether the reduction in thread-private memory traffic
compensates for additional LDS use, barriers, serial line work, and lower residency. This evidence
is particularly important for explaining why the existing $128^3$ block sweep is slower than the
thread sweep even though the block formulation removes the large thread-local arrays.

For collision searches, separate tree construction, periodic-ghost construction, neighbor query,
collision-rate evaluation, and event application. Report both isolated kernel costs and the complete
collision-operator time; a faster search build does not imply a faster physical collision step.

Profiler-native CSV, JSON, or ROCprofiler database files may be retained in their native formats.
The GameDev QA layer should additionally write a compact JSON manifest containing the command,
tool versions, selected kernels, source artifact paths, and principal derived metrics.

### Production-scale performance matrix

The unprofiled benchmark runner should execute stable, production-like models with radiation
disabled for the primary timing baseline. Radiation may be measured separately with a physically
stable setup, but an instability-induced CFL collapse is not a performance benchmark. Each case
should save only its initial and final state and should advance at least 100 accepted steps so that
initialization and file output are amortized.

Recommended scale points are

| Model | Baseline | Production | Large, if affordable |
|---|---:|---:|---:|
| fluid 2D | $512^2$ | $1024^2$ | $2048^2$ |
| fluid 3D | $64^3$ | $128^3$ | $256^3$ |
| swarm | $10^5$ particles | $10^6$ particles | $10^7$ particles |
| collision search | KD tree and Morton | KD tree and Morton | same |

Run one untimed warm-up followed by at least five measured repetitions from the same initial state
on an otherwise idle APU. The grid, particle count, physics flags, output cadence, accepted-step
count, completed simulated time, compiler options, and GPU target must match within each comparison.
Report the median and interquartile range, and repeat a noisy campaign when the five-run wall-time
spread exceeds roughly five percent.

For a fluid calculation, normalize evolution cost as

$$
t_{\rm cell,step}
=
\frac{t_{\rm evolution}}
{N_{\rm step}N_XN_YN_Z},
\qquad
\mathcal{R}_{\rm sim}
=
\frac{\Delta t_{\rm simulated}}{t_{\rm wall}}.
$$

For a swarm calculation, additionally report time per particle per dynamical step. Collision runs
must record

$$
t_{\rm build},\qquad
t_{\rm KNN},\qquad
t_{\rm rate},\qquad
t_{\rm event},\qquad
\frac{N_{\rm collision}}{t_{\rm wall}},
$$

together with persistent search memory and Morton ghost-record ratio. Initialization, evolution,
output, and total wall time should be stored separately. Every result must also record the ROCm and
compiler versions, GPU identity, all physics flags, line-sweep or KNN backend, resolution or particle
count, accepted steps, completed simulated time, and peak device memory.

The implemented uninstrumented interfaces are

```bash
python3 qav/perf/run_suite.py --group all --scale baseline --target gfx942 --repeat 5
python3 qav/perf/run_suite.py --group all --scale production --target gfx942 --repeat 5
```

The optional `--scale large` branch selects $2048^2$, $256^3$, and $10^7$-particle cases. Each case
is compiled once, run once as an excluded warm-up, and then repeated from the same deterministic
initial condition. It saves only the initial and final state and rejects any run that completes
fewer than 100 accepted fluid steps or frozen-rate collision batches. The warm-up alone is sampled
with `amd-smi`; measured timing runs have no memory-sampling process. Case JSON records the resolved
HIP compile commands, source revision and dirty state when available, a standalone-tree SHA-256
fingerprint, tool versions, target, native output path,
initialization/evolution/output times, accepted work, simulated time, median, quartiles, spread, and
normalized cost. A spread above five percent is labelled noisy.

Profiler collection is invoked independently, for example,

```bash
python3 qav/perf/profile.py --case fluid_3d_production --tool trace --target gfx942
ROCPROFCOMPUTE_COLOR=0 python3 qav/perf/profile.py \
    --case fluid_3d_production --tool compute --kernel diffusion_ybl --target gfx942
```

The profiler manifest explicitly marks the run as instrumented and unsuitable as a timing
baseline. Profiler-native artifacts remain in their native formats and are referenced, rather than
rewritten, by the JSON manifest. Collision build, query, rate, and event breakdowns are derived from
these short profiler collections; the uninstrumented collision benchmark reports only the complete
operator cost so event instrumentation cannot bias the production timing.

### Native uninstrumented results on MI300A

The first gfx942 campaign on 2026-08-12 completed the full baseline matrix and both production fluid
cases. All completed cases returned normally, repeated an identical accepted-work count, and stayed
well below the five-percent noise threshold:

| Case | Size | Work units | Median wall time | Median evolution time | Spread | Normalized evolution cost |
|---|---:|---:|---:|---:|---:|---:|
| fluid 2D baseline | $512^2$ | 126 steps | 4.604 s | 4.298 s | 0.340% | $1.301\times10^{-7}$ s cell$^{-1}$ step$^{-1}$ |
| fluid 2D production | $1024^2$ | 126 steps | 11.456 s | 11.116 s | 0.851% | $8.414\times10^{-8}$ s cell$^{-1}$ step$^{-1}$ |
| fluid 3D baseline | $64^3$ | 491 steps | 8.536 s | 8.271 s | 0.118% | $6.426\times10^{-8}$ s cell$^{-1}$ step$^{-1}$ |
| fluid 3D production | $128^3$ | 1090 steps | 118.228 s | 117.926 s | 0.511% | $5.159\times10^{-8}$ s cell$^{-1}$ step$^{-1}$ |
| swarm KD-tree baseline | $10^5$ particles | 6631 batches | 60.838 s | 60.514 s | 1.361% | $9.126\times10^{-8}$ s particle$^{-1}$ batch$^{-1}$ |
| swarm Morton baseline | $10^5$ particles | 6637 batches | 83.404 s | 83.044 s | 0.045% | $1.251\times10^{-7}$ s particle$^{-1}$ batch$^{-1}$ |

The larger fluid cases have lower normalized cell-step costs, consistent with improved accelerator
saturation; this is a throughput observation, not a convergence claim. At $10^5$ particles the
Morton collision operator costs about 1.37 times the KD-tree operator end to end, so no speed benefit
is established for Morton at this scale. Its lower-memory and decomposition advantages remain
separate questions for the profiler and larger cases.

The original $10^6$-particle collision interval of $2\times10^{-3}$ was excessive: it produced
23020 batches and required roughly 55.5--56.2 minutes for each of the two completed KD-tree repeats,
then the Slurm allocation expired during the third. These partial repeats are not combined with the
five-repeat results. The production interval is now $2\times10^{-5}$, which extrapolates to about
230 batches while preserving the same particle state, $N_K=200$, `CFL_COL=0.01`, and search kernels.

Process-memory sampling was attempted during every excluded warm-up, but ROCm 7.2 `amd-smi process`
reported no per-process memory value on this MI300A APU. Timing evidence is complete; peak-memory
evidence remains unavailable rather than being interpreted as zero. The runner now records the
first unparsed AMD SMI response and an explicit availability flag for diagnosis in the next run.

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

## Integration into one production tree

The final project should select CUDA or ROCm as one build-time GPU backend rather than maintain two
complete copies of GameDev. Fluid and swarm remain independent physical representations; merging
the GPU backends must not reintroduce shared fluid-swarm source ownership. The integration instead
removes duplication between the CUDA and HIP implementations of the same fluid file or the same
swarm file.

### Proposed repository structure

The target structure is

```text
.
├── Makefile
├── inc/
│   ├── fluid/
│   │   ├── *.cuh                 # backend-neutral fluid equations and algorithms
│   │   ├── cuda/                 # CUDA-only fluid adapters when unavoidable
│   │   └── rocm/                 # ROCm-only fluid adapters when unavoidable
│   └── swarm/
│       ├── *.cuh                 # backend-neutral swarm equations and algorithms
│       ├── cuda/                 # CUDA-only swarm, RNG, and library adapters
│       └── rocm/                 # ROCm-only swarm, RNG, and library adapters
├── src/
│   ├── fluid/
│   │   ├── shared numerical translation units
│   │   ├── cuda/                 # irreducibly CUDA-specific implementations
│   │   └── rocm/                 # irreducibly ROCm-specific implementations
│   └── swarm/
│       ├── shared numerical translation units
│       ├── cuda/
│       └── rocm/
├── mod/                          # one model configuration tree
├── qav/                          # one analytical and regression test tree
├── obj/<model>/<repr>/<backend>/ # generated backend-separated objects
├── bin/<model>/<backend>/gamedev # generated backend-separated executables
└── out/<model>/<backend>/        # generated backend-separated simulation state
```

Here, `shared numerical translation units` means source shared only between CUDA and ROCm for one
representation. It does not mean sharing source between the fluid and swarm models. A numerical
kernel should have one implementation when both compilers accept the same code and produce the
validated behavior. A backend-specific copy should remain only when compiler syntax, runtime
semantics, random-state types, device-library interfaces, or performance-critical implementations
genuinely differ.

The present `rocm/inc`, `rocm/src`, `rocm/mod`, and `rocm/qav` copies are temporary migration
inputs. Once every required HIP difference has a home in the merged structure, the duplicated
model and QA trees, synchronization script, preservation manifest, and standalone ROCm Makefile can
be removed.

### Backend selection

The merged Makefile should expose exactly one variable:

```make
GPU_BACKEND := cuda
```

or

```make
GPU_BACKEND := rocm
```

For production models, `mod/<model>/flags.mk` should define this value explicitly. The ordinary
build command then remains concise:

```bash
make MODEL=my_model
```

During initial model setup and QA, the backend may instead be selected explicitly on the command
line:

```bash
make MODEL=my_model GPU_BACKEND=cuda CUDA_ARCH=sm_80
make MODEL=my_model GPU_BACKEND=rocm AMDGPU_TARGET=gfx942
```

`GPU_BACKEND` must contain exactly one word and must be either `cuda` or `rocm`; there is no `auto`,
`both`, or runtime-switching mode. A production model should not silently default to whichever
compiler happens to be installed. The Makefile should fail before compilation when the selection is
missing or invalid.

The selection controls all backend-dependent build inputs:

| Build property | `GPU_BACKEND=cuda` | `GPU_BACKEND=rocm` |
|---|---|---|
| compiler | `nvcc` | `hipcc` |
| architecture | `CUDA_ARCH`, for example `sm_80` | `AMDGPU_TARGET`, for example `gfx942` |
| runtime API | CUDA Runtime | HIP Runtime |
| random-number API | cuRAND | hipRAND |
| device primitives | CUB and Thrust | hipCUB and rocThrust |
| source suffix or language mode | CUDA translation unit | HIP translation unit or `-x hip` |
| predefined backend macro | `GAMEDEV_CUDA` | `GAMEDEV_ROCM` |

Exactly one of `GAMEDEV_CUDA` and `GAMEDEV_ROCM` must be defined. Backend adapter headers should
check this invariant at preprocessing time. CUDA builds must not include HIP headers, and ROCm
builds must not include CUDA headers. The selected object directory, executable, output directory,
dependency files, and QA archive must all include the backend name so changing the selection cannot
reuse incompatible artifacts.

### Model and checkpoint backend lock

Backend selection is permanent for one simulation history. GameDev will neither support nor test

```text
CUDA checkpoint -> ROCm continuation
ROCm checkpoint -> CUDA continuation
```

Every output series should contain a small JSON run manifest with at least

```json
{
  "gpu_backend": "rocm",
  "compiler": "hipcc",
  "gpu_target": "gfx942",
  "model": "my_model",
  "dust_representation": "swarm"
}
```

The restart configuration signature should also contain `gpu_backend=cuda` or
`gpu_backend=rocm`. Before reading any density, momentum, particle, or RNG checkpoint array, the
runtime must compare the stored backend with the compiled backend and terminate on disagreement.
This is a policy check rather than an attempt to determine whether individual arrays happen to be
binary-compatible.

No cross-backend conversion utility, automatic fallback, or partial checkpoint import should be
provided. To move a scientific setup to another platform, the user must start a new simulation from
the model's initial conditions. CUDA and ROCm runs may still be compared as independently initialized
verification or scientific experiments; that comparison does not make their checkpoint histories
interchangeable.

### One QA interface

The final `qav/` tree should contain one definition of every analytical or statistical test. The
same `GPU_BACKEND` selection should choose the compiler and runtime used by the test:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --backend cuda
python3 qav/fluid/test_common/run_suite.py --group all --backend rocm

python3 qav/swarm/test_common/run_suite.py --group all --backend cuda
python3 qav/swarm/test_common/run_suite.py --group all --backend rocm
```

The runners should translate `--backend` into `GPU_BACKEND` and store results below distinct
`qav/*/out/<backend>/` roots. A model-local test definition, analytical validator, tolerance, and
result schema must not be copied merely to support another backend. Backend-specific test source is
permitted only for API behavior that has no common interface, such as raw vendor RNG-state restart.

Cross-backend archive comparison may remain as a development qualification tool, but it is not a
simulation mode and is not required for ordinary production runs. Each backend must first pass its
own analytical, conservation, statistical, boundary, restart, and KNN gates.

### Integration sequence

The lowest-risk order is

1. add `GPU_BACKEND` validation and backend-separated object, executable, output, and QA paths
2. make the root Makefile reproduce the present CUDA build without changing numerical source
3. add HIP compiler and library selection while still compiling the existing ROCm source copies
4. move backend adapters into the corresponding `inc/fluid`, `inc/swarm`, `src/fluid`, and
   `src/swarm` subdirectories
5. consolidate source files only after CUDA and ROCm versions are shown to be mathematically and
   operationally equivalent
6. make the single root QA tree run both selections and rerun all native backend suites
7. add and test the backend field in run manifests and restart signatures
8. remove the duplicated `rocm/mod`, `rocm/qav`, synchronization machinery, and standalone build
9. remove the remaining top-level `rocm/` staging tree only after no build or document references it

Promotion still requires stable compiler and target requirements, production performance evidence,
and completion of the integration sequence above. Native large-LDS qualification and same-backend
restart regression have passed for the current HIP tree. Until integration is complete, CUDA remains
the production reference and the HIP tree remains an independently testable implementation.

## References

- AMD, [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- AMD, [HIP compiler and runtime documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- AMD, [MI300 ISA and LDS description](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- AMD, [rocThrust documentation](https://rocm.docs.amd.com/projects/rocThrust/en/latest/)
- AMD, [hipCUB documentation](https://rocm.docs.amd.com/projects/hipCUB/en/latest/)
- AMD, [hipRAND documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)
- AMD, [ROCprofiler-SDK and `rocprofv3` usage](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- AMD, [ROCm Compute Profiler usage](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html)
- AMD, [ROCm Compute Profiler analysis](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/cli.html)
- AMD, [ROCm Compute Profiler LDS metrics](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.1.0/conceptual/local-data-share.html)
- AMD, [HIP function attributes](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_func_attributes.html)
- LLVM, [Clang floating-point behavior](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior)
- LLVM, [Clang HIP support](https://clang.llvm.org/docs/HIPSupport.html)
