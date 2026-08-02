# ROCm/HIP port audit: `lab/codex/rocm`

Date: 2026-08-01
Scope: complete audit of the experimental HIP/ROCm reproduction of the GameDev fluid and swarm
models in `lab/codex/rocm` (289 files) against the current CUDA working tree. No files in the
ROCm tree or the main tree were modified by this audit.

## Method

- Full-tree comparison: every ROCm file paired with its CUDA counterpart (`.hip` ↔ `.cu`).
  240 files were identical or differed only by mechanical CUDA→HIP token substitution; the 45
  files with substantive changes (plus the 4 ROCm-only files) were reviewed individually.
- Targeted hazard checks: wavefront width (32-lane CUDA warp vs 64-lane AMD wavefront), FP64
  atomics, dynamic shared-memory (LDS) handling, hipRAND state semantics, Thrust/hipCUB execution
  policies, allocator behavior, build-flag propagation through `flags.mk` and every QA runner.
- `tools/static_check.py` (the tree's own pre-transfer lint) run locally: **PASS**.
- Limitation: no ROCm toolchain on the audit machine, so nothing was compiled or executed;
  compile/link items are marked "verify on cluster" below.

**Headline verdict.** The port is careful and faithful: physical equations and discretizations
are untouched, all 24+24 production translation units are present, the QA tree (validators,
runners, KNN benchmarks) is reproduced with identical tolerances, and the build plumbing is
correct. One real semantic bug (build flags) and one workflow hazard (re-sync) were found, plus
a small number of verify-on-hardware items.

---

## 1. REAL BUG — `-ffast-math` silently deletes every NaN/Inf check in the tree

`lab/codex/rocm/Makefile:18` and `lab/codex/rocm/qav/swarm/test_knn/Makefile` compile everything
with blanket `-ffast-math`. Under clang this includes `-ffinite-math-only`, which lets the
optimizer fold `isfinite`/`isnan`/`isinf` to compile-time constants — in **host and device**
code alike. This silently disables:

| Location | Function |
|---|---|
| `src/fluid/inf_cell_flag.hip:18–23` | device-side nonfinite state flagging |
| `src/fluid/cfl_rate_calc.hip:59–70,114` | ring poisoning and `INFINITY` rate guard |
| `inc/fluid/fluid_host.cuh:257` | host abort on nonfinite CFL rate (`get_dt_cfl`) |
| `inc/swarm/swarm_host.cuh` (`std::isfinite(max_log_mass)`) | zero-mass initialization guard |
| `qav/swarm/test_knn/knn_periodic_tests.hip:301` | KNN validator overflow detection (`std::isinf`) |
| `qav/swarm/test_knn/knn_edge_tests.hip:379` | same, edge tests |

The CUDA build's `--use_fast_math` does **not** do this: nvcc's flag only substitutes fast
division, square root, and transcendentals in device code, and finite checks keep working. The
port therefore degrades the documented nonfinite-state detection semantics relative to the CUDA
original — a NaN propagates as garbage instead of aborting, and the KNN overflow checks become
dead code, so the QA suite can report false confidence.

**Fix:** append `-fno-finite-math-only` after `-ffast-math` in both Makefiles (clang supports
this combination), keeping the fast math while restoring Inf/NaN semantics.

**Related drift:** `-ffast-math` also reaches host code (initialization quadratures, `erfc`/`exp`
/`log`, long-double accumulations), whereas nvcc's flag touched only device code. Small host-side
numeric differences vs CUDA are therefore expected beyond the "reduction order" caveat the
README already states.

## 2. Workflow hazard — re-running `sync_from_cuda.py` destroys the hand port

`tools/sync_from_cuda.py --force-overwrite` wipes and regenerates `inc/`, `src/`, `mod/`, `qav/`
from the CUDA tree, but its mechanical token map does not reproduce the hand-maintained ROCm
adaptations. Lost on re-sync:

- `src/fluid/fluid_runtime.hip` — the device-reported LDS check and the removed
  `hipFuncSetAttribute` opt-in (regenerates the CUDA pattern).
- `inc/swarm/kdtree/helpers.h` — the synchronous `DeviceMemoryResource`; regenerates to
  `hipMallocAsync`, exactly what the port deliberately avoided pending a native regression.
- `inc/swarm/kdtree/builder_thrust.h`, `inc/swarm/morton/morton_index.cuh`,
  `inc/swarm/morton/morton_ghost.cuh` — the explicit `thrust::hip_rocprim::par` execution
  policies.
- `qav/swarm/test_knn/Makefile` — reverts to `nvcc`/`sm_80`; the KNN build breaks.
- `qav/*/run_model.py`, `run_suite.py`, `run_sweep.py`, `test_knn/run*.py` — the `--target`
  option and AMD environment recording revert to `nvidia-smi` versions.

The script's docstring acknowledges the manual re-application step, but no manifest records
which files are hand-maintained. Recommend one of: a preserve-list inside `sync_from_cuda.py`,
a documented hand-maintained manifest in `NOTICE.md`, or a sync mode that refuses to clobber
files whose content differs from the pure-mechanical transformation.

## 3. Verify on hardware (sound by design, unprovable without a ROCm node)

1. **LDS opt-in removal** (`src/fluid/fluid_runtime.hip`): the host check against
   `hipDeviceAttributeMaxSharedMemoryPerBlock` is the right AMD pattern. MI300A reports 64 KiB
   per block and has no legacy 48 KiB split, so block-diffusion kernels within budget should
   launch without any opt-in. Confirm one launch with >48 KiB dynamic LDS on gfx942. Note the
   new hard abort when `6·N·8 B > 64 KiB` (i.e. N ≳ 1365): a tighter ceiling than CUDA-H100's
   227 KiB, by design and clearly messaged.
2. **`hip/hip_math_constants.h` with `HIPRT_PI_F`/`HIPRT_INF_F`**: matches hipify conventions;
   expected in ROCm ≥ 5.x. The first native compile will confirm.
3. **`hiprandState` restart layout**: `hiprandState` is the XORWOW family like `curandState`,
   and `hiprand_init(seed, idx, 0, …)` matches the cuRAND signature. Saved `rngstate_*.dat`
   checkpoints remain platform-specific, which the documentation already covers.

## 4. Checked clean — no action required

- **Wavefront width:** no `__shfl`, `__ballot`, `__syncwarp`, warp masks, or warp-synchronous
  logic anywhere in the tree. The 32-vs-64-lane difference is performance-only, consistent with
  the README's documented `TPB=32` bring-up choice.
- **Atomics and math intrinsics:** `atomicAdd(double)` is native on gfx942; `__double2int_rn`,
  `sincosf`, `expm1`, `erf`/`erfc`, `fmax`/`fmin` all exist in HIP device math.
- **Library backends:** `thrust::hip_rocprim::par` policies, `hipcub::DeviceRadixSort`,
  rocThrust containers, and header-only hipRAND device API are all correct; no extra link
  libraries are needed.
- **`g_traversalStats`** (`__constant__` in a header): internal linkage per TU and behind the
  default-off `KDTREE_STATS` macro — no `-fgpu-rdc` duplicate-symbol hazard. (`-fgpu-rdc` is in
  fact unnecessary — all device code is header-inline — and could be dropped for link time.)
- **Build plumbing:** every `flags.mk` routes feature defines through `HIPFLAGS`;
  `AMDGPU_TARGET` propagates correctly through make environment and all QA runners; model
  resolution, `vpath`, object tagging by sweep/backend/arch, and `.d` dependencies mirror the
  CUDA Makefile; `mod/` trees are complete (flags only, runtime via `vpath`).
- **QA fidelity:** `validate_case.py` validators are byte-identical to the CUDA tree (same
  analytical checks and tolerances); environment recording correctly switches to
  `hipconfig`/`amd-smi`/`rocm-smi` with `unavailable` fallbacks.
- **Struct layout:** all I/O goes through field-wise `real` arrays, so CUDA/HIP `double3`
  layout differences are harmless.
- **Memory allocator:** the `DeviceMemoryResource` in `kdtree/helpers.h` (stream-sync +
  `hipMalloc`/`hipFree`) is functionally correct and improves error propagation; it is a
  deliberate performance compromise pending a native `hipMallocAsync` regression.
- No `out/` artifacts exist in the tree — as expected, no native AMD run has been archived yet.

## Bottom line

Port structure and fidelity are good. Before running the QA suite on the MI300A:

1. add `-fno-finite-math-only` to both Makefiles (otherwise the fluid nonfinite detection and
   the KNN overflow checks are dead code);
2. protect the hand-maintained files from `sync_from_cuda.py` re-runs;
3. confirm one >48 KiB dynamic-LDS launch on gfx942.
