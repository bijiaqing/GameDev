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
├── comm/                 backend-neutral model definitions and validators
│   ├── fluid/
│   └── swarm/
├── cuda/                 CUDA drivers and backend-specific test code
│   ├── fluid/
│   └── swarm/
├── rocm/                 ROCm drivers and backend-specific test code
│   ├── fluid/
│   └── swarm/
├── tool/                 archive, comparison, and all-in-one runners
├── logs/                 ignored numerical evidence and terminal records
└── temp/                 ignored executables, objects, and compiler stamps
```

Every generated validation result is written below `val/logs/`, regardless of format. This includes
JSON manifests, terminal captures, and binary numerical fields. Disposable executables, object and
dependency files, and compiler stamps are written below `val/temp/`. Test source directories
therefore remain source-only after a build.

## Native publication campaign

`run_all.py` runs the native numerical suites and archive checks. The standalone
`val/tool/static_check.py` is manual-only and is not a campaign stage.

Run the full CUDA campaign with

```bash
python3 val/tool/run_all.py \
    --backend cuda \
    --target sm_80 \
    --res 32 64 128 256
```

Run the matching ROCm campaign with

```bash
python3 val/tool/run_all.py \
    --backend rocm \
    --target gfx942 \
    --res 32 64 128 256
```

Without `--compare`, each command writes only its native backend archive. Add `--compare` on the
second machine after copying the first machine's `val/logs/` tree into the project. The order is
irrelevant: CUDA can be copied to ROCm or ROCm to CUDA. The comparator itself requires Python but no
GPU.

Use `--quick` only to check that the build and archive workflow functions. A quick run is not the
publication convergence record.

### Block fluid only

Thread/block selects a fluid implementation; swarm does not have this sweep choice. After the
complete thread campaigns, run just the block fluid matrix in each backend's GPU allocation:

```bash
FLUID_SWEEP=block python3 val/cuda/fluid/test_common/run_suite.py \
    --group all --target sm_80 --res 32 64 128 256 &&
python3 val/tool/check_archive.py \
    --backend cuda --component fluid --fluid-sweep block \
    --output archive_check_cuda_block.json
```

```bash
FLUID_SWEEP=block python3 val/rocm/fluid/test_common/run_suite.py \
    --group all --target gfx942 --res 32 64 128 256 &&
python3 val/tool/check_archive.py \
    --backend rocm --component fluid --fluid-sweep block \
    --output archive_check_rocm_block.json
```

These commands preserve thread and swarm records and write the block fluid manifests below
`val/logs/fluid/BACKEND/block/`. They do not create a `run_all` aggregate or record a source
fingerprint; retain the source snapshot separately as required by the evidence contract below.

## Morton integration check

After changes to the Morton builder, run the optional structural oracle and the existing KNN,
physical-collision, and collision-chain groups in each native GPU allocation. The topology target
checks 36 cases against an independent CPU oracle. It is not an automatic static-check stage and
does not change the publication matrix or its tolerances.

CUDA:

```bash
make -f val/cuda/swarm/test_knn/Makefile topology ARCH=sm_80 &&
mkdir -p val/logs/swarm/cuda/groups/knn &&
val/temp/cuda/swarm/test_knn/knn_topology > val/logs/swarm/cuda/groups/knn/topology.json
for group in knn collision chain; do
    python3 val/cuda/swarm/test_common/run_suite.py --group "$group" --target sm_80 || break
done
```

ROCm:

```bash
make -f val/rocm/swarm/test_knn/Makefile topology AMDGPU_TARGET=gfx942 &&
mkdir -p val/logs/swarm/rocm/groups/knn &&
val/temp/rocm/swarm/test_knn/knn_topology > val/logs/swarm/rocm/groups/knn/topology.json
for group in knn collision chain; do
    python3 val/rocm/swarm/test_common/run_suite.py --group "$group" --target gfx942 || break
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

The backend-specific suite runners accept physical groups. Examples are

```bash
python3 val/cuda/fluid/test_common/run_suite.py \
    --group diffusion \
    --res 32 64 128 256 \
    --target sm_80

python3 val/rocm/swarm/test_common/run_suite.py \
    --group transport \
    --res 32 64 128 256 \
    --target gfx942

python3 val/cuda/swarm/test_common/run_suite.py \
    --group chain \
    --target sm_80
```

A focused group writes below `groups/GROUP/` and cannot overwrite the canonical `all` archive.
Direct model invocations write below `groups/manual/` unless `VAL_SCOPE` is set explicitly.

## Evidence contract

The canonical archives are

```text
val/logs/
├── fluid/
│   ├── cuda/SWEEP/MODEL/
│   └── rocm/SWEEP/MODEL/
├── swarm/
│   ├── cuda/MODEL/
│   └── rocm/MODEL/
├── archive_check_cuda.json
├── archive_check_rocm.json
├── backend_comparison.json
├── run_all_cuda.json
└── run_all_rocm.json

val/temp/
├── bin/BACKEND/REPRESENTATION/MODEL/
├── cuda/swarm/test_knn/
├── rocm/swarm/test_knn/
└── obj/MODEL/REPRESENTATION/BACKEND/
```

The `temp/` subtree is disposable and is not part of the scientific archive. Model result
directories contain numerical metrics and, where required, compact binary fields. Suite manifests
record the expected cases, resolutions, completion state, backend, and target. The top-level
`run_all.py` campaign records the initial and final source fingerprints; standalone suite manifests
do not. `val/tool/check_archive.py` rejects an incomplete native archive.

Keep a successful campaign record for each tested backend/sweep before a later run overwrites the
canonical `run_all_BACKEND.json`, for example by copying it immediately to
`run_all_cuda_thread.json` or `run_all_rocm_thread.json` after the respective thread pass. A second
full `run_all.py` campaign also reruns swarm and overwrites its backend-specific records.
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
`val/temp/`; executables, objects, and compiler stamps are backend-local and reproducible.

Validate archives without rerunning the simulations:

```bash
python3 val/tool/check_archive.py --backend cuda --component all
python3 val/tool/check_archive.py --backend rocm --component all
python3 val/tool/compare_backends.py --component all
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
