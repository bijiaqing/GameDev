# GameDev quality assurance workflow

The QAV tree separates backend-neutral test definitions in `comm/`, native drivers in `cuda/` and
`rocm/`, shared archive utilities in `tool/`, and ignored generated evidence in `logs/`.
`qav/tool/qav_config.py` is the authoritative common matrix used by both native runners and the
cross-backend comparator.

Each native build writes its transient `gamedev` executable beside the `flags.mk` selected for that
test model. Common cases therefore use `qav/comm/REPRESENTATION/MODEL/gamedev`, while a backend-only
qualification case uses its directory below `qav/BACKEND/`. These executables are ignored by Git;
the backend-separated result archive below `qav/logs/` is the persistent comparison interface.

The mathematical cases, evidence tiers, tolerances, latest archived results, and coverage limits are
documented in [`doc/fluid_testset.md`](../doc/fluid_testset.md) and
[`doc/swarm_testset.md`](../doc/swarm_testset.md). This file contains only the cross-backend archive
workflow and output contract.

## Transferable native campaign

An invocation without `--compare` creates and checks only the selected native archive. For example,
run CUDA first with

```bash
python3 qav/tool/run_all.py \
    --backend cuda \
    --target sm_80 \
    --res 32 64 128 256
```

Copy the complete project, including the ignored `qav/logs/` tree, to the ROCm system and run

```bash
python3 qav/tool/run_all.py \
    --backend rocm \
    --target gfx942 \
    --res 32 64 128 256 \
    --compare
```

The reverse order is equally valid: run ROCm without `--compare`, copy the complete archive, then
run CUDA with `--compare`. The second native run writes only its backend-owned paths, verifies the
copied archive, generates or reuses the precise CUDA polar fields, and executes both cross-backend
comparators. Use `--quick` on both systems only for a partial workflow check.

Each campaign records a SHA-256 fingerprint of the Makefile and every QAV-relevant source file.
Comparison rejects incomplete archives and different source fingerprints before interpreting a
backend difference. The comparison programs require only Python; raw-field comparison additionally
requires NumPy, so completed archives may also be compared on a CPU-only system.

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
├── backend_comparison.json
├── run_all_cuda.json
└── run_all_rocm.json
```

The analytical suites write metrics, metadata, environment records, manifests, and compact binary
fields. A focused group writes below `groups/GROUP/`, while a directly invoked model writes below
`groups/manual/`; neither can replace the canonical `all` archive. Fluid variants remain separated
by sweep and arithmetic mode. Generated evidence is ignored by Git and must be copied explicitly.

Validate completed native archives without rerunning them:

```bash
python3 qav/tool/check_archive.py --backend cuda --component all
python3 qav/tool/check_archive.py --backend rocm --component all
```

Compare archives already stored in the same project:

```bash
python3 qav/tool/compare_fluid_fields.py --expected-records 8
python3 qav/tool/compare_backends.py --component all
```

For separate archive trees, pass `--cuda-root` and `--rocm-root` to both scripts. The legacy
`--cuda-qav` and `--rocm-qav` spellings remain aliases. These paths identify archive contents and
do not depend on which GPU, if any, is present on the comparison host.
