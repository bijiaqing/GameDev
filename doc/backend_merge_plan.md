# Planned CUDA and ROCm source ownership

## 1. Purpose and decision rule

This document records the proposed file ownership before the standalone `rocm/` tree is merged into
the repository root. It is a layout plan, not a description of the active build. No production or
QA file has been moved according to this plan yet.

The first merge stage uses a conservative whole-file rule:

1. share a file only when the complete file is backend-neutral
2. do not split a file into common and vendor fragments merely to increase the shared line count
3. if any required part of a file depends on CUDA Runtime, HIP Runtime, cuRAND, hipRAND, CUB,
   hipCUB, vendor-specific launch limits, or a backend-specific numerical implementation, keep the
   complete CUDA and ROCm files
4. do not hide substantial backend differences behind a large `#if GAMEDEV_CUDA` block in an
   otherwise nominally shared file
5. promote a currently different pair into a shared path only after the two complete files have the
   same intended behavior and both native test suites pass

This rule deliberately accepts some duplication in host/runtime files. It reduces the risk that a
backend change silently alters numerical code used by the other backend. Fluid and swarm remain
independent representations; this plan shares files between CUDA and ROCm, never between fluid and
swarm.

The target backend selector remains

```make
GPU_BACKEND := cuda
```

or

```make
GPU_BACKEND := rocm
```

with exactly one of `GAMEDEV_CUDA` and `GAMEDEV_ROCM` defined by the build.

## 2. Important findings before relocation

### 2.1 Production source agreement

The current production inventory contains 22 fluid translation-unit pairs and 21 swarm pairs.
Twenty fluid pairs and 15 swarm pairs are byte-identical apart from the `.cu` versus `.hip`
filename. These 35 files are immediate shared-source candidates. The remaining eight pairs contain
runtime, RNG, search-library, launch-policy, or behavioral differences and must stay backend-owned
at the first stage.

Shared translation units should retain the `.cu` suffix. CUDA compiles them normally; ROCm must
compile the same path explicitly as HIP language, for example through `hipcc -x hip`. This avoids a
second `.hip` copy whose only distinction is its extension.

### 2.2 Collision-safety parity blocker

Three current swarm pairs are not merely vendor spellings:

- ROCm `col_site_init` rejects nonfinite particle position, velocity, size, and number fields and
  records `dev_bad_part`; CUDA currently lacks this guard
- ROCm `col_rate_calc` defines `inf_rate_flag` for nonfinite collision rates or KNN radii; CUDA
  currently lacks it
- the corresponding ROCm `swarm_kern.cuh` declarations and `swarm_runtime` allocation/check path
  include `dev_bad_part`, while CUDA does not

Therefore `col_site_init`, `col_rate_calc`, and `swarm_kern.cuh` must remain paired backend files in
the initial layout. The preferred later resolution is to add the same safety behavior to CUDA,
rerun CUDA collision/failure QA, and only then promote `col_site_init`, `col_rate_calc`, and
`swarm_kern.cuh` to shared ownership. Silently choosing either current copy during the merge would
change one backend's behavior.

### 2.3 Differences that are genuinely backend-specific

The following differences should remain backend-owned rather than be normalized away:

- CUDA Runtime versus HIP Runtime allocation, transfer, synchronization, function-attribute, and
  error-reporting calls
- cuRAND versus hipRAND state types, initialization, and sampling
- CUDA and ROCm KD-tree implementations and their CUB/hipCUB dependencies
- CUDA and ROCm Morton builders, sorting primitives, infinity/constants, and resource ownership
- fluid block workgroup width: CUDA currently uses 32 work-items and ROCm uses 64
- ROCm LDS admission queries and rejection diagnostics versus CUDA dynamic-shared-memory attribute
  requests
- the Morton collision-event RNG state: CUDA can use shared storage for its state type, whereas the
  validated HIP implementation keeps `hiprandState` thread-local

## 3. Proposed production tree

The backend directory precedes the representation directory. This matches the requested
`inc/cuda`, `inc/rocm`, `src/cuda`, and `src/rocm` ownership and keeps all backend selection at one
level.

```text
inc/
├── fluid/                              # complete backend-neutral fluid headers
│   ├── _transport.cuh
│   ├── param_grid.cuh
│   └── param_phys.cuh
├── swarm/                              # complete backend-neutral swarm headers
│   ├── _collision.cuh
│   ├── _diffusion.cuh
│   ├── _transport.cuh
│   ├── param_grid.cuh
│   ├── param_phys.cuh
│   └── swarm_grid.cuh
├── cuda/
│   ├── fluid/
│   │   ├── const_defs.cuh              # TPB = 64 and CUDA configuration
│   │   ├── fluid_host.cuh
│   │   └── fluid_kern.cuh
│   └── swarm/
│       ├── const_defs.cuh              # TPB = 64, cuRAND, and CUDA configuration
│       ├── swarm_host.cuh
│       ├── swarm_kern.cuh              # parity blocker before later promotion
│       ├── kdtree/                     # complete current CUDA KD-tree tree
│       │   ├── Apache-2.0.txt
│       │   ├── box.h
│       │   ├── builder.h
│       │   ├── builder_bitonic.h
│       │   ├── builder_common.h
│       │   ├── builder_host.h
│       │   ├── builder_inplace.h
│       │   ├── builder_thrust.h
│       │   ├── common.h
│       │   ├── data.h
│       │   ├── fcp.h
│       │   ├── helpers.h
│       │   ├── index_heap.cuh
│       │   ├── kdtree.h
│       │   ├── knn.h
│       │   ├── math.h
│       │   ├── spatial-kdtree.h
│       │   ├── traverse-cct.h
│       │   ├── traverse-default-stack-based.h
│       │   ├── traverse-sf-imp.h
│       │   ├── traverse-stack-free.h
│       │   └── cubit/
│       │       ├── common.h
│       │       ├── cubit.h
│       │       ├── cubit_maxval.h
│       │       └── cubit_zip.h
│       └── morton/
│           ├── morton_ghost.cuh
│           ├── morton_index.cuh
│           ├── morton_query.cuh
│           └── morton_types.cuh
└── rocm/
    ├── fluid/
    │   ├── const_defs.cuh              # TPB = 64 and ROCm launch policy
    │   ├── fluid_host.cuh
    │   └── fluid_kern.cuh
    └── swarm/
        ├── const_defs.cuh              # TPB = 64, hipRAND, and ROCm launch policy
        ├── swarm_host.cuh
        ├── swarm_kern.cuh              # parity blocker before later promotion
        ├── kdtree/                     # complete current HIP KD-tree tree
        │   ├── Apache-2.0.txt
        │   ├── box.h
        │   ├── builder.h
        │   ├── builder_bitonic.h
        │   ├── builder_common.h
        │   ├── builder_host.h
        │   ├── builder_inplace.h
        │   ├── builder_thrust.h
        │   ├── common.h
        │   ├── data.h
        │   ├── fcp.h
        │   ├── helpers.h
        │   ├── index_heap.cuh
        │   ├── kdtree.h
        │   ├── knn.h
        │   ├── math.h
        │   ├── spatial-kdtree.h
        │   ├── traverse-cct.h
        │   ├── traverse-default-stack-based.h
        │   ├── traverse-sf-imp.h
        │   ├── traverse-stack-free.h
        │   └── cubit/
        │       ├── common.h
        │       ├── cubit.h
        │       ├── cubit_maxval.h
        │       └── cubit_zip.h
        └── morton/
            ├── morton_ghost.cuh
            ├── morton_index.cuh
            ├── morton_query.cuh
            └── morton_types.cuh
```

Both backends use `TPB = 64` for ordinary pointwise and flat-index kernels. This fills one native
ROCm wave and launches two ordinary CUDA warps. The specialized block-sweep policy remains
`TPB_BLOCK = 32` on CUDA and `TPB_BLOCK = 64` on ROCm until matched profiling justifies another
choice.

Even though the ordinary `TPB` value and physical constants are identical, retain complete fluid
`const_defs.cuh` files under both backend paths. This keeps fluid and swarm configuration lookup
structurally symmetric and leaves room for future backend launch or type policy without moving the
header again.

The same `TPB = 64` baseline applies to swarm. Its `const_defs.cuh` files are independently
backend-owned because they select cuRAND or hipRAND state types. This does not alter the
cooperative Morton width: `MORTON_TPB = 256` is an independent algorithmic launch parameter on both
backends.

Although the two `Apache-2.0.txt` files are identical, retain one inside each complete backend
KD-tree directory so that the third-party license remains adjacent to every distributed copy of the
derived implementation.

The complete source tree should be

```text
src/
├── fluid/                              # 20 byte-identical numerical kernels
│   ├── advection_xbl.cu
│   ├── advection_xth.cu
│   ├── advection_ybl.cu
│   ├── advection_yth.cu
│   ├── advection_zbl.cu
│   ├── advection_zth.cu
│   ├── cfl_rate_calc.cu
│   ├── diffusion_xbl.cu
│   ├── diffusion_xth.cu
│   ├── diffusion_ybl.cu
│   ├── diffusion_yth.cu
│   ├── diffusion_zbl.cu
│   ├── diffusion_zth.cu
│   ├── inf_cell_flag.cu
│   ├── init_vel_calc.cu
│   ├── momentum_getv.cu
│   ├── momentum_setv.cu
│   ├── optdepth_calc.cu
│   ├── optdepth_csum.cu
│   └── source_update.cu
├── swarm/                              # 15 byte-identical numerical kernels
│   ├── col_snap_save.cu
│   ├── dustdens_calc.cu
│   ├── dustdens_depo.cu
│   ├── dustdens_init.cu
│   ├── dyn_rate_calc.cu
│   ├── gas_lerp_calc.cu
│   ├── optdepth_calc.cu
│   ├── optdepth_csum.cu
│   ├── optdepth_depo.cu
│   ├── optdepth_init.cu
│   ├── optdepth_mean.cu
│   ├── particle_init.cu
│   ├── ssa_substep_1.cu
│   ├── ssa_substep_2.cu
│   └── ssa_transport.cu
├── cuda/
│   ├── fluid/
│   │   ├── fluid_runtime.cu
│   │   └── init_rho_calc.cu            # cuRAND initialization noise
│   └── swarm/
│       ├── col_event_run.cu            # cuRAND and CUDA shared-state behavior
│       ├── col_rate_calc.cu             # parity blocker before later promotion
│       ├── col_site_init.cu             # parity blocker before later promotion
│       ├── diffusion_pos.cu             # cuRAND stochastic increments
│       ├── rngstate_init.cu             # cuRAND state initialization
│       └── swarm_runtime.cu
└── rocm/
    ├── fluid/
    │   ├── fluid_runtime.hip
    │   └── init_rho_calc.hip           # hipRAND initialization noise
    └── swarm/
        ├── col_event_run.hip           # hipRAND and thread-local state behavior
        ├── col_rate_calc.hip            # parity blocker before later promotion
        ├── col_site_init.hip            # parity blocker before later promotion
        ├── diffusion_pos.hip            # hipRAND stochastic increments
        ├── rngstate_init.hip             # hipRAND state initialization
        └── swarm_runtime.hip
```

The shared kernels keep their current include names. Build-dependent include search order chooses the
appropriate complete host and declaration headers:

```text
model-local include directory
qav/<backend>/<representation>/<model>       # QA only
qav/<representation>/<model>                 # QA only
inc/<backend>/<representation>
inc/<representation>
inc/<backend>/<representation>/kdtree        # collision search only
inc/<backend>/<representation>/morton        # collision search only
```

Backend directories must precede shared directories only for distinct, explicitly selected files.
The build should reject duplicate same-stem production translation units instead of silently choosing
one by incidental directory order.

The current production models `mod/fluid_rpi_2d` and `mod/fluid_rpi_3d` provide complete
`const_defs.cuh` overrides that also contain `TPB`. Under the same whole-file rule, the initial
merged layout needs selected copies such as `mod/<model>/cuda/const_defs.cuh` and
`mod/<model>/rocm/const_defs.cuh`; all physical values and `TPB = 64` remain matched. Extracting
launch constants into a smaller header could remove this duplication later, but that would violate
the present no-file-splitting rule.

## 4. Proposed QA tree

QA needs the same ownership rule, but generated output must never be mixed with test definitions.
The proposed structure is

```text
qav/
├── README.md
├── compare_backends.py                  # host-only cross-backend archive comparison
├── compare_fluid_fields.py              # host-only raw-field comparison
├── fluid/                               # shared models, references, and criteria
│   ├── README.md
│   ├── mock/
│   │   ├── README.md
│   │   └── algorithm_checks.py
│   ├── test_common/
│   │   ├── README.md
│   │   ├── TEST_CASES.md
│   │   ├── compare_sweeps.py
│   │   ├── param_phys.cuh
│   │   └── validate_case.py
│   └── test_*/                         # shared analytical model definitions listed below
├── swarm/                               # shared models, references, and criteria
│   ├── README.md
│   ├── test_common/
│   │   ├── TEST_CASES.md
│   │   └── validate_case.py             # adopt the restart-capable superset before sharing
│   ├── test_knn/
│   │   ├── analyze_results.py
│   │   └── analyze_wedge.py
│   └── test_*/                         # shared analytical model definitions listed below
├── cuda/
│   ├── fluid/
│   │   ├── test_common/
│   │   │   ├── const_defs.cuh
│   │   │   ├── run_model.py
│   │   │   ├── run_suite.py
│   │   │   ├── run_sweep.py
│   │   │   └── verification_main.cu
│   │   └── test_*/                     # CUDA run.py and whole CUDA runtime overrides
│   └── swarm/
│       ├── test_common/
│       │   ├── const_defs.cuh
│       │   ├── run_model.py
│       │   ├── run_suite.py
│       │   └── verification_main.cu
│       ├── test_knn/                    # complete CUDA KNN executable harness
│       └── test_*/                     # CUDA run.py and whole CUDA runtime overrides
├── rocm/
│   ├── fluid/
│   │   ├── test_common/
│   │   │   ├── const_defs.cuh
│   │   │   ├── lds_runtime.hip
│   │   │   ├── run_lds.py
│   │   │   ├── run_model.py
│   │   │   ├── run_suite.py
│   │   │   ├── run_sweep.py
│   │   │   └── verification_main.hip
│   │   ├── test_failure_2d/
│   │   ├── test_lds_reject/
│   │   ├── test_lds_x/
│   │   ├── test_lds_y/
│   │   ├── test_lds_z/
│   │   └── test_*/                     # HIP run.py and whole HIP runtime overrides
│   ├── swarm/
│   │   ├── test_common/
│   │   │   ├── const_defs.cuh
│   │   │   ├── run_model.py
│   │   │   ├── run_suite.py
│   │   │   └── verification_main.hip
│   │   ├── test_failure_knn/
│   │   ├── test_perf_collision_2d/
│   │   ├── test_restart_2d/
│   │   ├── test_knn/                    # complete HIP KNN executable harness
│   │   └── test_*/                     # HIP run.py and whole HIP runtime overrides
│   └── perf/
│       ├── README.md
│       ├── analyze_counters.py
│       ├── profile.py
│       └── run_suite.py
└── out/
    ├── cross_backend/
    ├── fluid/
    │   ├── cuda/
    │   └── rocm/
    ├── swarm/
    │   ├── cuda/
    │   └── rocm/
    └── perf/
        └── rocm/
```

### 4.1 Shared fluid model definitions

The following model identities exist on both backends and should have one shared directory for
backend-neutral analytical configuration, documentation, and host validation:

```text
test_optdepth
test_ring_all_2d
test_ring_diffusion_2d
test_ring_radiation_2d
test_ring_transport_2d
test_source_drag
test_sweep_block_2d
test_sweep_block_3d
test_sweep_thread_2d
test_sweep_thread_3d
test_x_diffusion_2d
test_x_transport_2d
test_y_diffusion_cyl
test_y_diffusion_sph
test_y_transport_cyl
test_y_transport_sph
test_z_diffusion_3d
test_z_transport_3d
```

`test_mms_3d` currently contains documentation only and stays shared. Complete fluid
`const_defs.cuh` files are not shared when they contain the launch policy. The four sweep
definitions retain CUDA and ROCm copies; the ROCm copies keep the optional `TEST_CFL` branch and
use `TPB = 64`.

All ordinary fluid `flags.mk` files differ only because one writes `NVCC +=` and the other writes
`HIPFLAGS +=`. The merged Makefile should provide one backend-neutral model variable such as
`GPU_FLAGS +=`; each complete `flags.mk` then becomes shared without conditional code. The
one-line `fluid_runtime.cu`/`.hip` wrappers and the two complete `verification_main` implementations
remain backend-owned at this stage. The shared `test_source_drag/source_update.cu` is already
byte-identical and can remain a shared model override compiled as CUDA or HIP language.

Any full QA `const_defs.cuh` that contains `TPB` also remains backend-owned under the whole-file
rule. CUDA and ROCm QA both use `TPB = 64`. In particular, this applies to both
`test_common/const_defs.cuh` files and the four complete fluid sweep definitions. Small
model-local wrappers that only define a test case and include the selected backend's
`test_common/const_defs.cuh` remain shareable.

### 4.2 Shared swarm model definitions

The following model identities exist on both backends and should have one shared directory for
backend-neutral case selection and analytical configuration:

```text
test_boundary_1d
test_boundary_2d
test_boundary_3d
test_boundary_half
test_collision_1d
test_collision_2d
test_collision_3d
test_diffusion_1d
test_diffusion_2d
test_diffusion_3d
test_drag_1d
test_drag_2d
test_grid_1d
test_grid_2d
test_grid_3d
test_import_1d
test_initial_3d
test_orbit_1d
test_orbit_2d
test_prdrag_1d
test_prdrag_2d
test_radiation_1d
test_radiation_2d
test_viscflow_1d
```

Their model-local `const_defs.cuh` and `run.py` files are currently byte-identical. Their
`flags.mk` files become shareable after the same `GPU_FLAGS` normalization. Nevertheless, the
model-local `run.py` and one-line runtime wrappers should remain under the backend QA paths during
the first stage because each runner imports a backend-specific `test_common/run_model.py` by
relative path. Moving those wrappers into shared ownership before redesigning that import contract
would create a fragile implicit dependency.

### 4.3 KNN QA

Only `test_knn/analyze_results.py` and `test_knn/analyze_wedge.py` are currently byte-identical and
fully host-only. Keep them shared. The KNN `Makefile`, executable sources, `knn_types.cuh`,
`periodic_query.cuh`, and Python launchers contain CUDA/HIP compiler, runtime, constant, or binary
selection logic, so the complete CUDA and ROCm KNN harnesses stay in `qav/cuda/swarm/test_knn` and
`qav/rocm/swarm/test_knn`.

### 4.4 QA files that require reconciliation before sharing

The ROCm QA tree currently contains a superset of the CUDA tree:

- fluid nonfinite failure injection
- ROCm LDS acceptance/rejection tests
- swarm KNN failure injection
- swarm same-backend RNG restart
- the production collision performance model

The LDS tests are intrinsically ROCm-specific and remain under `qav/rocm`. The failure policies and
same-backend restart policy are not intrinsically ROCm-only. Before the final unified `--group all`
interface is declared complete, CUDA needs equivalent failure and restart coverage, even if their
runtime drivers remain backend-specific. The collision performance model's physical constants can
eventually be shared, while the profiler/timing driver remains backend-owned.

The fluid `validate_case.py` pair differs only in CUDA/HIP wording and should become one
backend-neutral validator. The swarm ROCm validator additionally implements `restart_2d`; adopt that
superset as the shared host validator after a CUDA restart driver exists. The Python `run_model`,
`run_suite`, and `run_sweep` files contain substantial environment, compiler, archive, failure-test,
and progress-report differences. Under the whole-file rule, keep complete backend copies for the
first merge stage rather than creating one large conditional runner.

## 5. Build and lookup invariants

The merged Makefile should implement these rules explicitly:

1. require `GPU_BACKEND=cuda` or `GPU_BACKEND=rocm`
2. choose `nvcc` plus `CUDA_ARCH` or `hipcc` plus `AMDGPU_TARGET`
3. compile shared `.cu` translation units with the selected language mode
4. add only the selected `inc/<backend>/<representation>` tree to the include path
5. add only the selected `src/<backend>/<representation>` files to the source list
6. reject duplicate source stems across model, QA, backend, and shared levels unless the higher
   level is an intentional documented override
7. separate generated paths by backend:

```text
obj/<model>/<representation>/<backend>/
bin/<model>/<backend>/gamedev
out/<model>/<backend>/
qav/out/<component>/<backend>/
```

8. never use a CUDA object in a ROCm link or vice versa
9. never resume a checkpoint created by the other backend

## 6. Recommended relocation order

No mass move should be attempted in one patch. The safe order is

1. add backend selection and backend-separated generated paths while leaving source in place
2. make `GPU_BACKEND=cuda` reproduce the current CUDA source list exactly
3. make `GPU_BACKEND=rocm` reproduce the current standalone ROCm source list exactly
4. move the 35 byte-identical production translation units into shared `src/fluid` and `src/swarm`
5. move complete backend host/runtime, RNG, KD-tree, Morton, and launch-policy files into
   `inc/<backend>` and `src/<backend>`
6. reconcile the CUDA collision nonfinite guards and rerun CUDA collision/failure tests
7. promote `col_rate_calc`, `col_site_init`, and `swarm_kern.cuh` only after parity is demonstrated
8. move shared QA model definitions and host validators, keeping complete backend runners and
   runtime drivers under `qav/cuda` and `qav/rocm`
9. run both full native suites, cross-backend field comparison, KNN matrix, restart, failure, LDS,
   and performance smoke tests
10. remove the standalone `rocm/` tree only after no build, test, or document references it

This ordering preserves a working CUDA and ROCm build at every meaningful checkpoint and makes each
promotion decision reviewable at whole-file granularity.
