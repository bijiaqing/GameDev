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
record the expected cases, resolutions, completion state, backend, target, and source fingerprint.
`val/tool/check_archive.py` rejects an incomplete native archive. The cross-backend comparator
requires matching source fingerprints unless explicitly told otherwise.

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

## Scope

The suite intentionally excludes micro-tests that merely repeat a local formula, failure-injection
checks, restart plumbing, implementation-to-implementation sweep comparisons, and performance
benchmarks. Those can be developed outside the publication archive when a concrete defect or
performance claim requires them. Their absence is not evidence that every configuration, flag
combination, or hardware limit has been tested.
