# Merged fluid–swarm code audit: `inc/`, `src/`, `tst/` (2026-07-27)

Date: 2026-07-27

Scope: line-by-line review of the current merged tree after the thread/block sweep consolidation —
`inc/fluid/`, `inc/swarm/`, `src/fluid/`, `src/swarm/`, the root `Makefile`, and the complete
`tst/fluid/` harness (analytical suite, thread/block sweep branch, Python validators, mock
checks, recorded outputs). The vendored `inc/swarm/cukd/` library was checked only at call sites.
`legacy/` and `mod/` were treated as frozen/configuration material. The review asked: (1) whether
the calculations are numerically, mathematically, and physically correct; (2) whether the
thread/block merge and the renaming rework introduced code errors; (3) whether code and `doc/`
agree; (4) whether the tests are complete and test the right things.

Post-audit harness correction: analytical outputs are now namespaced under
`tst/fluid/out/thread/` and `tst/fluid/out/block/`. This fixes an artifact-management omission in
the audited version, where a block analytical run would have overwritten the archived thread
fields, metrics, and environment records. The production calculations were unaffected.

Method: source inspection plus independent algebraic re-derivation of every changed numerical
formula, and direct comparison of the merged block kernels against the previously validated
`mod/vram_optimize` implementations. No CUDA compiler is available on this machine; runtime
claims rely on the recorded artifacts under `tst/fluid/out/thread/`, per the authority order in
`doc/README.md`. Recorded artifacts were spot-checked against the documents (85 metrics, 22
environments, source-case machine-epsilon errors, zero archived sweep-comparison JSONs — all
consistent).

## Executive summary

The tree is in excellent shape. The thread/block sweep consolidation is implemented correctly:
the thread kernels are the fiducial reference implementations with mechanical renames, the block
kernels are the previously validated workspace/shared-memory implementations (including the
z-kernel invariant-bounds fix), the driver selects them cleanly at build time through
`FLUID_SWEEP`, and the test harness can exercise both variants through the same analytical cases.
The archived 85-record analytical baseline currently covers the thread variant; the block run is
still pending. The swarm branch is unchanged from its twice-verified state, with the `IMPORTGAS`+`CONST_ST`
combination now rejected at compile time. All findings from the previous two reviews
(documentation paths, the FARGO CFL sentence, the `CONST_ST` flag hole, the source-validator
cancellation, the thread-local-memory performance note) are incorporated into the canonical
documents and the code.

**No new correctness defects were found in production code in this pass.** Observations are
limited to a handful of minor notes (section D).

## A. Verified correct in this pass

### A1. The thread/block consolidation

- **Makefile**: `FLUID_SWEEP` defaults to `thread`, is validated as `thread|block`, defines
  `FLUID_BLOCK_SWEEP`, and isolates objects in per-sweep directories, so the two implementations
  can coexist and be benchmarked interchangeably. `MODEL_PARENT` support and the
  `const_defs.cuh` prerequisite tracking are correct. The `TEST_*` passthrough
  (`-DTEST_RES/CFL/POWER/SHIFT/SAVE_MAX/DT_OUT`) matches the verification constants.
- **`inc/fluid/_transport.cuh`** consolidates both variants' helpers: `_thread_ppm_faces` /
  `_thread_ppm_state` (line-at-once face storage), `_block_ppm_face` / `_block_ppm_state`
  (on-demand, identical expressions and rounding), `_block_field` workspace addressing, and the
  unchanged PPM/HLL/invariant-domain primitives verified earlier.
- **Block kernels** (`advection_[xyz]bl`, `diffusion_[xyz]bl`): checked against the validated
  `mod/vram_optimize` sources. The FARGO mean/shift, PPM reconstruction, HLL high/low fluxes,
  serial low-order update, face-ordered invariant-domain correction (now with the correct
  `_local_bounds(lx, iz+1, ...)` everywhere, verified in all six kernels), SSPRK(3,3) staging,
  outflow/HALFDISK boundaries, CN coefficients, Thomas/Sherman–Morrison recurrences, time-centred
  mass flux, donor momentum closure, barrier-separated read/update phases, and the per-substep
  `cn_diag` restore are all intact. Shared-memory provisioning (`4·N_X`, `6·N_Y`, `6·N_Z` with
  `cudaFuncSetAttribute`) is correct.
- **Thread kernels** (`advection_[xyz]th`, `diffusion_[xyz]th`): the fiducial reference
  implementations with mechanical renames only (FARGO, SSPRK, FV geometry, CN, positivity
  subcycling verified against the previously audited formulas).
- **`src/fluid/fluid_runtime.cu`**: identical palindromic driver composition for both sweeps;
  block launches use one-block-per-line grids with correct dynamic shared sizes; the workspace
  (11·N_G fields) is allocated only for the block sweep; the `CUDA_SYNC_TRACE` debug hooks are
  inert by default.
- **`_get_init_Rmin()`** (new in both `param_grid.cuh` copies): the convolved initial profile
  and its interpolation now span `[Y_MIN·min(sin Z_MIN, sin Z_MAX), Y_MAX]`, so 3D cells whose
  cylindrical radius falls below `Y_MIN` interpolate the tapered profile instead of being clipped
  to hard zero. Applied consistently in `inc/fluid/fluid_host.cuh`, `src/fluid/init_rho_calc.cu`,
  and `inc/swarm/swarm_host.cuh` (profile construction and all `initdens_lerp` call sites).
  Physically sensible and cross-representation consistent.

### A2. The swarm branch

- Unchanged from the twice-verified state; `dyn_rate_calc`/`dev_dt_rate` renames are consistent
  across declaration, definition, and call sites (`dt_rate_ptr`, `max_dt_rate`).
- `swarm_kern.cuh` now rejects `IMPORTGAS`+`CONST_ST` at compile time, matching
  `numerics_swarm.md`'s "analytic-gas-only option" statement.
- All collision (2πR axisymmetric measure, `H_g` vertical overlap, additive kernel `m_i+m_j`),
  IMPORTGAS Stokes anchoring, rngstate checkpoint/restore, and well-mixed 2D closures remain as
  previously verified.

### A3. Naming-rework consistency

No rename-induced defects found. Verified across declarations, definitions, call sites, the
Makefile, and the test harness: `advection_[xyz]{th,bl}`, `diffusion_[xyz]{th,bl}`,
`fluid_runtime`/`swarm_runtime`, `dyn_rate_calc`/`dev_dt_rate`, `dev_cfl_rate`,
`_thread_ppm_faces/_thread_ppm_state`, `_block_ppm_face/_block_ppm_state`,
`_thread_dr_cent_i/o` / `_block_dr_cent_i/o`, `BlockAdvField`/`BLOCK_*`, `dev_adv_work`,
`FLUID_SWEEP`/`FLUID_BLOCK_SWEEP`, and the `TEST_*` macro family. The 13-character kernel-name
convention holds. The known `PARAM_GRID_CUH`/`PARAM_PHYS_CUH` guard-name overlap between the two
representation include trees remains harmless under one-branch-per-build.

## B. Code–document consistency

Everything checked agrees:

- `doc/numerics_fluid.md` now states the FARGO residual bound correctly (`CFL_DYN + 0.5 ≤ 1`),
  documents both sweep implementations honestly (including "block faster at 1024², slower at
  128³"), and its thread-local-footprint discussion matches `advection_xth` (21 arrays × N_X).
- `doc/verification_fluid.md`: the recorded source-case fix is real — `validate_case.py`'s
  `source_exact` now mirrors the CUDA branch (`-expm1`, series for `h < 1e-4`, closed-form
  weights otherwise), algebraically exact; the recorded `momz L∞` is `2.22e-16` as documented.
  The stated evidence boundary (no archived sweep benchmark/profiler artifacts) matches the
  repository (zero `sweep_comparison_*.json` present). The 85-record table, step sequences, mass
  figures, and environment records remain consistent with `tst/fluid/out/thread/`.
- `doc/ref_naming_variables.md`: every applied spelling in its record tables matches the code
  (`_thread_dr_cent_i/o`, `_block_dr_cent_i/o`, `_thread_ppm_faces/_thread_ppm_state`,
  `dev_cfl_rate`, `dev_dt_rate`, the `{th,bl}` kernel names, the 13-character rule).
- `doc/ref_file_architecture.md`: `inc/share`/`src/share` correctly described as reserved-empty;
  the two `optdepth_csum.cu` copies correctly described as branch-local.
- `doc/README.md` authority statement (85 records + 22 environments present; sweep benchmark
  artifacts absent) matches the tree.

## C. Test-suite assessment

### The sweep branch is well designed

- `run_sweep.py` runs matched thread/block pairs at fixed work (1024² transport-only 2D, 128³
  diffusion 3D, radiation off), with per-model build/run/environment/timing logs — the right
  protocol.
- `compare_sweeps.py` enforces the meaningful equivalence checks: **byte-identical frame 0**
  (proves identical initialization), finite fields, nonnegative density, finite-volume mass
  mismatch `< 1e-10`, density relative L2 `< 1e-7`, velocity `< 1e-5` (tolerances appropriate for
  chaotic amplification), and accepted-step diagnostics. Its `cell_volumes` reproduce the
  production mesh measures exactly (log radial faces, `d=2/3`, polar cosine measure, `dx` weight).
- `verification_main.cu` runs the **entire analytical suite** against either implementation via
  `FLUID_SWEEP=block` (lambda-wrapped operator selection, workspace and shared-memory attributes
  provisioned). This is the strongest available validation of the block kernels and is correctly
  wired through `run_model.py`/`run_suite.py` (`RES=` → `-DTEST_RES=` mapping verified).
- The validator's `source_exact` fix (B above) is algebraically identical to the CUDA kernel,
  not an approximation of it.

### Remaining gaps (all already recorded in `doc/verification_fluid.md`)

The "verification still required" list is accurate and complete with respect to my previous
findings: full-3D MMS, boundary-crossing cases and a `HALFDISK` midplane case, restart-tolerance
and flag-matrix regressions, a CUDA case that actually triggers CN positivity subcycling, finite
attenuation, archived sweep benchmark artifacts, fast_math-off reruns, and automated pass
thresholds. `tst/swarm/` remains empty, so the swarm branch still has source-review evidence
only — the largest validation asymmetry, unchanged and documented.

Two small additions to that list, in priority order:

1. The sweep comparison exercises thread-vs-block at one resolution per dimension with radiation
   disabled; when its artifacts are next regenerated, a radiation-enabled 2D pair would also
   cover the block kernels against the optical-depth/source composition (cheap, same harness).
2. The `NB_* = n/TPB + 1` one-extra-block launch formulas remain (harmless, index-guarded);
   `ref_file_architecture.md` already records the ceiling-division cleanup.

## D. Minor observations (not defects)

- `advection_xbl`'s FARGO-mean loop passes global momentum elements to `_recover_dust_state` by
  reference, so vacuum cells are repaired in global memory one kernel earlier than
  `momentum_getv` would do it. The repair is idempotent, contained to the block's own ring, and
  the kernel's final write-back plus `momentum_getv` produce the same observable state as the
  thread implementation — consistent, but worth knowing when diffing intermediate device states.
- `verification_main.cu` duplicates the block workspace/attribute setup from
  `fluid_runtime.cu` (acceptable for a test harness; flagged only as future consolidation
  material if a shared driver helper ever exists).
- The historical development numbers quoted in `verification_fluid.md` (1.71× faster at 1024²,
  3.41× slower at 128³) are not reproducible from the repository (no committed logs); the
  document already says so. The automated sweep branch is the correct replacement once run on
  the target GPU.

## E. Recommendations (priority order)

1. Run `python3 tst/fluid/verify_common/run_suite.py --group sweep --sweep-dim all` once on the
   target GPU and commit `sweep_comparison_*.json`, build logs, `ptxas -v` reports for the
   thread kernels, and a profiler local-traffic comparison — this closes the only evidence gap
   the documents themselves flag.
2. Run the full analytical suite once with `FLUID_SWEEP=block` so the block implementation has
   the same 85-record convergence evidence as the thread implementation. Its artifacts will be
   retained separately under `tst/fluid/out/block/`.
3. Create `tst/swarm/` (orbit/stiff-drag, cylindrical diffusion, optical-depth/β attenuation,
   constant-kernel collision) — still the largest missing validation surface.
4. Work through the already-documented verification list (boundary-crossing, restart tolerance,
   CN subcycling trigger, finite attenuation) when the MMS harness lands.

## Review limitations

- No CUDA compilation or execution was possible here; runtime evidence is the recorded
  `tst/fluid/out/thread/` artifact set, verified for internal consistency and against the documents but
  not regenerated.
- `inc/swarm/cukd/` internals were not re-audited (unchanged).
- `mod/fluid_fiducial` (thread sweep, RADIATION) and `mod/swarm_fiducial` (TRANSPORT, RADIATION,
  SAVE_DENS, CODE_UNIT) were sanity-checked for flag compatibility only.
