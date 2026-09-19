# GameDev validation

The validation tree contains the compact test suite used to support scientific claims about GameDev. It is
not a catalogue of every helper function or guard in the code. The retained cases test complete
numerical operators against analytical solutions, statistically specified stochastic solutions, or
an independent exact-neighbor reference.

Detailed mathematics, acceptance criteria, and coverage limits are documented in
[`doc/fluid_testset.md`](../doc/fluid_testset.md) and
[`doc/swarm_testset.md`](../doc/swarm_testset.md). This file describes only how to run, archive, and
compare the tests.

## Directory layout

```text
val/
├── fluid/                shared fluid models, drivers, and validators
├── swarm/                shared particle models, drivers, and validators
├── paper/                 publication models
├── *.json                cross-component campaign and comparison summaries
└── *.py                  numerical campaign, archive, and comparison tools
```

Both `fluid/` and `swarm/` contain `mod/<model>/` for models, `src/` for shared
code, `out/<model>/<backend>/` for results, and `obj/<model>/<backend>/` for executables
and object files. Fluid results add the sweep directory. Suite summaries use
`out/_suite/<backend>/`. Build products are disposable; result files are not.

## Native publication campaign

`run_all.py` runs the native numerical suites and archive checks.

Run the full CUDA campaign with

```bash
python3 val/run_all.py \
    --backend cuda \
    --target sm_80 \
    --res 32 64 128 256
```

Run the matching ROCm campaign with

```bash
python3 val/run_all.py \
    --backend rocm \
    --target gfx942 \
    --res 32 64 128 256
```

Without `--compare`, each command writes only its native backend archive. Add `--compare` on the
second machine after copying the first machine's `val/fluid/out/`, `val/swarm/out/`, and `val/*.json` records into the project. The order is
irrelevant: CUDA can be copied to ROCm or ROCm to CUDA. The comparator itself requires Python but no
GPU.

Use `--quick` only to check that the build and archive workflow functions. A quick run is not the
publication convergence record.

### Block fluid only

Thread/block selects a fluid implementation; swarm does not have this sweep choice. After the
complete thread campaigns, run just the block fluid matrix in each backend's GPU allocation:

```bash
FLUID_SWEEP=block GPU_BACKEND=cuda python3 val/fluid/src/run_suite.py \
    --group all --target sm_80 --res 32 64 128 256 &&
python3 val/check_archive.py \
    --backend cuda --component fluid --fluid-sweep block_precise \
    --output check_archive_cuda_block.json
```

```bash
FLUID_SWEEP=block GPU_BACKEND=rocm python3 val/fluid/src/run_suite.py \
    --group all --target gfx942 --res 32 64 128 256 &&
python3 val/check_archive.py \
    --backend rocm --component fluid --fluid-sweep block \
    --output check_archive_rocm_block.json
```

These commands preserve thread and swarm records. CUDA block fluid manifests use
`val/fluid/out/<model>/cuda/block_precise/`; ROCm uses `val/fluid/out/<model>/rocm/block/`. They do not create a `run_all` aggregate or record a source
fingerprint; retain the source snapshot separately as required by the evidence contract below.

## Morton integration check

After changes to the Morton builder, run the optional structural oracle and the existing KNN,
physical-collision, and collision-chain groups in each native GPU allocation. The topology target
checks 36 cases against an independent CPU oracle. It is not an automatic static-check stage and
does not change the publication matrix or its tolerances.

CUDA:

```bash
make -f val/swarm/mod/test_knn/Makefile GPU_BACKEND=cuda topology GPU_TARGET=sm_80 &&
mkdir -p val/swarm/out/test_knn/cuda/groups/knn &&
val/swarm/obj/test_knn/cuda/knn_topology > val/swarm/out/test_knn/cuda/groups/knn/topology.json
for group in knn collision chain; do
    GPU_BACKEND=cuda python3 val/swarm/src/run_suite.py --group "$group" --target sm_80 || break
done
```

ROCm:

```bash
make -f val/swarm/mod/test_knn/Makefile GPU_BACKEND=rocm topology GPU_TARGET=gfx942 &&
mkdir -p val/swarm/out/test_knn/rocm/groups/knn &&
val/swarm/obj/test_knn/rocm/knn_topology > val/swarm/out/test_knn/rocm/groups/knn/topology.json
for group in knn collision chain; do
    GPU_BACKEND=rocm python3 val/swarm/src/run_suite.py --group "$group" --target gfx942 || break
done
```

Return `topology.json` and the complete `groups/{knn,collision,chain}/` archives for each backend.
CUDA and ROCm outputs have separate paths. The integrated Morton builder passed these focused
native checks on 2026-09-14 for CUDA sm_80 and ROCm gfx942: 36 topology cases, 48 KNN cases,
three physical-collision models, and four collision-chain models with all ten variants on each
backend. The downloaded JSON results contain no failed pass flags or missing referenced metrics
or environment files. These results cover the tested source and configurations. Block-source
provenance and separate native-build qualification remain pending until supported by their own
evidence; these focused swarm runs do not settle them.

## Focused groups

The shared suite runners accept physical groups. Examples are

```bash
GPU_BACKEND=cuda python3 val/fluid/src/run_suite.py \
    --group diffusion \
    --res 32 64 128 256 \
    --target sm_80

GPU_BACKEND=rocm python3 val/swarm/src/run_suite.py \
    --group transport \
    --res 32 64 128 256 \
    --target gfx942

GPU_BACKEND=cuda python3 val/swarm/src/run_suite.py \
    --group chain \
    --target sm_80
```

A focused group writes below `groups/GROUP/` and cannot overwrite the canonical `all` archive.
Direct model invocations write below `groups/manual/` unless `VAL_SCOPE` is set explicitly.

## Evidence contract

The canonical archives are

```text
val/fluid/out/MODEL/BACKEND/SWEEP/
val/swarm/out/MODEL/BACKEND/
val/fluid/out/_suite/BACKEND/SWEEP/
val/swarm/out/_suite/BACKEND/
val/{fluid,swarm}/obj/MODEL/BACKEND/
val/run_all_BACKEND_SWEEP.json
val/check_archive_BACKEND_SWEEP.json
val/backend_comparison.json
```

The `obj/` subtrees are disposable and are not part of the scientific archive. Model result
directories contain numerical metrics and, where required, compact binary fields. Suite manifests
record the expected cases, resolutions, completion state, backend, and target. The top-level
`run_all.py` campaign records the initial and final source fingerprints; standalone suite manifests
do not. `val/check_archive.py` rejects an incomplete native archive.

Campaign and archive-check summaries use the backend and sweep in their filenames,
for example `run_all_cuda_thread.json` and `check_archive_rocm_block.json`.
The sweep is `thread` or `block`; CUDA's `_precise` suffix remains in the data paths.
Running the same backend/sweep again replaces its summary. A second full campaign
also reruns swarm and overwrites its backend-specific model records.
Matching hashes in a failed or different-sweep campaign do not
establish the provenance of a standalone run. The comparator's hash check alone does not verify
campaign success or sweep identity. Associate each result with its actual source snapshot and
review later source changes before carrying its qualification forward. Model-local overrides must
be preserved separately: the campaign fingerprint covers Makefile and source files under `inc/`,
`src/`, and `val/`, not production model directories under `mod/`.

Cross-backend comparison excludes quantities that intentionally depend on native vendor random
streams. In particular, `test_startup_3d` compares its normalized polar-balance residual and
convergence, while its cuRAND- or hipRAND-dependent absolute initial mass remains a native
positive-finite check.

Use the same statistical distinction for wedge diffusion: crossing counts and sample errors depend
on vendor RNG streams, while each run's analytical acceptance and deterministic velocity mapping
remain required. Physical-collision formula errors and float-search geometry errors also have
separate native limits; generic comparison of absolute residuals is not equivalent to comparing
physical accuracy. The current comparator does not yet handle these new diagnostics appropriately;
the dated swarm assessment records its outstanding failures. Resolve such cases individually,
retaining native tolerances and failed reports rather than loosening the global comparison limits.

When transferring evidence between machines, copy the result and campaign records but exclude
all `obj/` directories; executables, objects, and compiler stamps are backend-local and reproducible.

Validate archives without rerunning the simulations:

```bash
python3 val/check_archive.py --backend cuda --component all
python3 val/check_archive.py --backend rocm --component all
python3 val/compare_backends.py --component all
```

When the two archives live in separate copied validation roots, pass `--cuda-root` and `--rocm-root` to
`compare_backends.py`.

## Production build qualification

Analytical models may override the production runtime or initialization. Their native passes
therefore do not establish every production build path. When changing build routing, executable
selection, or feature guards, retain these focused checks on CUDA and ROCm:

- Add and remove a model-local override with populated object/dependency caches; verify the selected
  source and the model's prescribed initialization amplitude, allowing different vendor RNG samples.
- Build configuration A → B → A without cleaning, such as fluid thread → block → thread; verify
  that the final executable is A while unchanged A objects remain cached.
- Compile the non-collision production swarm runtime and retain a collision-enabled frozen-bath
  runtime check. Controller declarations, reset, and output must require
  `COLLISION && !BERNOULLI`.

Save selected source paths, build commands/results, configuration evidence, and numerical checks
as named JSON records. These are targeted build regressions, not additional permanent scientific
models. A host mock compiler establishes Make dependency behavior only; native builds establish
compiler/API compatibility and the selected physical initialization.

The 2026-09-05 numerical archives do not contain the separate native override-amplitude, cached
configuration-return, or non-collision production-runtime evidence. Use an existing production
model and its configured amplitude for those checks; the repaired parameterized initializers use
`1e-10`. Verification-runtime builds do not close these production-build requirements.

## Scope

The suite intentionally excludes micro-tests that merely repeat a local formula, failure-injection
checks, restart plumbing, implementation-to-implementation sweep comparisons, and performance
benchmarks. Those can be developed outside the publication archive when a concrete defect or
performance claim requires them. Their absence is not evidence that every configuration, flag
combination, or hardware limit has been tested.

## Shared validation source

`val/fluid/` and `val/swarm/` replace the former `comm/`, `cuda/`, and `rocm/`
source trees. Each model has one definition, one validator, and one driver;
ROCm compiles the shared `.cu` files with `hipcc -x hip`. Genuine backend
requirements remain explicit branches, including fluid shared-memory checks.
Analytical references and acceptance thresholds were retained.

`run_all.py --backend cuda|rocm` selects the backend for the full campaign.
For a direct model or suite runner, set `GPU_BACKEND=cuda` or `GPU_BACKEND=rocm`.
Suite and KNN runners accept `--target`; direct analytical models inherit
`CUDA_ARCH` or `AMDGPU_TARGET`. KNN Make commands use `GPU_BACKEND` and `GPU_TARGET`.
Executables and archives retain separate backend paths. Existing component archives were relocated into `out/` without changing file contents.
CUDA fluid archives retain the `_precise` suffix to separate them from historical
fast-math results; the campaign and archive tools use that suffix consistently.

## Retained validation code

- `mod/` and `src/`: numerical test setups, GPU drivers, analytical references,
  error metrics, and accuracy acceptance criteria.
- `val_config.py`: shared case lists, paths, backend selection, and evidence metadata.
- `run_all.py`: native campaign runner.
- `check_archive.py`: checks that numerical results and required cases are complete and passed.
- `compare_backends.py`: compares CUDA and ROCm numerical results.
- `paper/`: publication models and their analysis/plotting scripts.

Standalone source sanity checks, CPU emulations, migration snapshots, and tests of
analysis helpers were removed. They are not needed to run or score the numerical cases.
Native output archives remain intact. Publication build products now live under
each suite's `obj/` directory; no build uses `val/temp/`.

Every retained validation Python script disables bytecode writing before importing
local or third-party modules. The runners also pass `PYTHONDONTWRITEBYTECODE=1`
to child processes. Existing `__pycache__` directories were removed.
