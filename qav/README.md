# GameDev quality assurance

The QAV tree separates test source from generated evidence:

- `comm/fluid/` and `comm/swarm/` contain backend-neutral model flags, criteria, and analyzers
- `cuda/` and `rocm/` contain native drivers and backend-owned test source
- `tool/` contains the shared matrix registry, archive checks, and cross-backend comparators
- `logs/` contains generated JSON and binary fields, is divided by backend, and is ignored by Git

`qav/tool/qav_config.py` is the authoritative common test matrix. Both native suite runners and the
cross-backend comparator import it, so model names, fixed-resolution rules, and expected metric
counts cannot drift independently.

## Evidence tiers

No test is discarded merely because it is too narrow for a paper. Every manifest records the
minimum tier of the evidence it contains:

- `publication` supports a scientific or numerical claim about the model
- `release` protects a special branch, parameter regime, boundary rule, or previously fixed defect
- `qualification` checks implementation equivalence, restart, deliberate failure handling, large
  shared memory, profiling, or performance rather than a physical equation

The common `all` campaign is the release gate and contains both `publication` and `release`
records. At four resolutions its 85 fluid metrics partition into 57 publication and 28
release-only records; its 51 swarm metrics partition into 33 publication and 18 release-only
records, followed by the publication-tier compact KNN matrix. `--knn-full` adds qualification-tier
large-particle benchmarks. Qualification-only fluid sweep, ROCm failure/LDS, swarm restart/failure,
and performance groups remain explicit and are not silently added to `all`.

## Backend-neutral transfer workflow

The comparison scripts require only Python; `compare_fluid_fields.py` additionally requires NumPy.
They do not call CUDA, ROCm, a compiler, or a GPU. Either backend may therefore run first, and the
final comparison may run on either cluster or on a third CPU-only system once both archives are
present.

An ordinary invocation creates only a checked native archive. For a CUDA-first campaign, use

```bash
python3 qav/tool/run_all.py \
    --backend cuda \
    --target sm_80 \
    --res 32 64 128 256
```

This runs the static checks, the 85-record fluid release gate, the 51-record swarm release gate and
compact publication-tier KNN suite,
the additional precise CUDA polar fields used by the raw cross-backend gate, and an archive
completeness check. Terminal output and stage status are stored in `qav/logs/run_all_cuda.json`.

Copy the complete project directory, including the ignored `qav/logs/` tree, to the second system.
A Git clone or pull alone does not copy generated evidence. Backend-specific output paths prevent
the second native run from overwriting the first. Then run the other backend with `--compare`; for
CUDA first and ROCm second, use

```bash
python3 qav/tool/run_all.py \
    --backend rocm \
    --target gfx942 \
    --res 32 64 128 256 \
    --compare
```

The reverse direction is equally valid. Start on ROCm with

```bash
python3 qav/tool/run_all.py \
    --backend rocm \
    --target gfx942 \
    --res 32 64 128 256
```

Then copy its archive to the CUDA system and run

```bash
python3 qav/tool/run_all.py \
    --backend cuda \
    --target sm_80 \
    --res 32 64 128 256 \
    --compare
```

Whichever backend runs second checks the copied counterpart archive, writes only its own native
paths, generates or uses the precise CUDA polar fields, and finally compares all fluid, swarm, and
KNN records. It rejects a missing or incomplete counterpart campaign and any source-tree
fingerprint mismatch before starting the native matrix. Comparison occurs only when `--compare` is
present; without it, the command never requires or reads the other backend archive.
Both campaign JSON files contain a SHA-256 fingerprint of the Makefile and every QAV-relevant
source file. The final comparator rejects archives made from different source snapshots instead of
silently attributing a code change to a GPU backend.
Raw-field comparison reports additionally fingerprint their JSON and binary-field inputs, so a
report remains valid after copying the archive to a different absolute path but is rejected if any
input record changes.
The final machine-readable reports are

```text
qav/logs/archive_check_cuda.json
qav/logs/archive_check_rocm.json
qav/logs/backend_field_comparison.json
qav/logs/backend_comparison.json
qav/logs/run_all_cuda.json
qav/logs/run_all_rocm.json
```

Use `--quick` on both systems for a two-resolution workflow check. Quick archives are deliberately
marked partial and are not evidence for the full convergence matrix.

## Native and focused commands

The backend-specific entry points remain available for focused work:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group diffusion --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group collision --res 32

python3 qav/rocm/fluid/test_common/run_suite.py --group diffusion --res 32 64 128 256 --target gfx942
python3 qav/rocm/swarm/test_common/run_suite.py --group collision --res 32 --target gfx942
```

`--group all` has exactly the same scientific meaning on both backends: the common analytical and
KNN matrix. ROCm-only robustness checks remain explicit:

```bash
python3 qav/rocm/fluid/test_common/run_suite.py --group failure --target gfx942
python3 qav/rocm/fluid/test_common/run_suite.py --group lds --target gfx942
python3 qav/rocm/swarm/test_common/run_suite.py --group restart --res 32 --target gfx942
python3 qav/rocm/swarm/test_common/run_suite.py --group failure --target gfx942
```

The complete `all` archive remains at the canonical paths used by the comparators. Focused groups
write their complete evidence below `groups/GROUP/`, and a model wrapper invoked directly writes
below `groups/manual/`. A focused or exploratory run therefore cannot delete any model record from
the completed `all` baseline. Fluid variants remain separate within each scope.
Sweep comparisons include resolution, frame count, and output time in their directory and result
names, preventing quick, full, fast-math, and precise-math runs from colliding.

## Output contract

```text
qav/logs/
├── fluid/
│   ├── cuda/SWEEP/[groups/SCOPE/]MODEL[/VARIANT]/
│   └── rocm/SWEEP/[groups/SCOPE/]MODEL[/VARIANT]/
├── swarm/
│   ├── cuda/[groups/SCOPE/]MODEL/
│   └── rocm/[groups/SCOPE/]MODEL/
├── bench/rocm/
├── archive_check_BACKEND.json
├── backend_field_comparison.json
└── backend_comparison.json
```

The analytical suites write `metrics_*.json`, raw `.dat` fields, per-model manifests, group-specific
suite manifests, and environment JSON. The ignored archive is self-contained evidence and may be
copied independently of object files and executables.

To validate a native archive without rerunning it:

```bash
python3 qav/tool/check_archive.py --backend cuda --component all
python3 qav/tool/check_archive.py --backend rocm --component all
```

To compare archives stored in the same project on either cluster or a CPU-only system, no path
options are required:

```bash
python3 qav/tool/compare_fluid_fields.py --expected-records 8
python3 qav/tool/compare_backends.py --component all
```

For separately stored QAV trees, pass `--cuda-root` and `--rocm-root` to both comparison scripts.
The legacy `--cuda-qav` and `--rocm-qav` spellings remain aliases. These arguments identify archive
contents, not the hardware on which the Python command is running.

```bash
python3 qav/tool/compare_fluid_fields.py \
    --cuda-root /path/to/cuda/qav \
    --rocm-root /path/to/rocm/qav \
    --expected-records 8 \
    --output qav/logs/backend_field_comparison.json

python3 qav/tool/compare_backends.py \
    --cuda-root /path/to/cuda/qav \
    --rocm-root /path/to/rocm/qav \
    --component all \
    --fluid-field-report qav/logs/backend_field_comparison.json \
    --output qav/logs/backend_comparison.json
```

Detailed equations, expected solutions, tolerances, and scientific interpretation are documented
in `doc/fluid_testset.md` and `doc/swarm_testset.md`.
