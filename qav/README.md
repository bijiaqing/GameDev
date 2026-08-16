# GameDev quality assurance

The QAV tree keeps test definitions separate from generated evidence:

- `comm/fluid/` and `comm/swarm/` contain backend-neutral model flags, analytical helpers, criteria, and analyzers
- `cuda/` contains CUDA test drivers and CUDA-owned test source
- `rocm/` contains ROCm test drivers, ROCm-only failure/LDS cases, and AMD profiling tools
- `tool/` contains backend-neutral static and cross-backend analysis utilities
- `logs/` is generated at run time, separated by representation and backend, and ignored by Git

The source tree intentionally contains no saved JSON, text log, binary field, executable, object,
or Python-cache artifacts. Runners create every required output directory automatically.

## CUDA

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group all --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group all --res 32 64 128 256
```

Use `--quick` for a short workflow check. The fluid `sweep` group compares the thread-line and
block-line implementations. The swarm `knn` group validates both collision-neighborhood backends.

## ROCm

```bash
python3 qav/rocm/fluid/test_common/run_suite.py \
    --group all --res 32 64 128 256 --target gfx942

python3 qav/rocm/swarm/test_common/run_suite.py \
    --group all --res 32 64 128 256 --target gfx942
```

ROCm-only robustness groups are selected with `--group failure`; fluid LDS launch qualification is
selected with `--group lds`. AMD performance campaigns and profiling live in `qav/rocm/bench/`.

## Output layout

```text
qav/logs/
├── fluid/
│   ├── cuda/SWEEP/MODEL/
│   └── rocm/SWEEP/MODEL/
├── swarm/
│   ├── cuda/MODEL/
│   └── rocm/MODEL/
└── bench/rocm/
```

The two native archives can be compared after both suites finish:

```bash
python3 qav/tool/compare_backends.py --component all
```

Detailed equations, expected solutions, tolerances, and pass criteria are documented in
`doc/fluid_testset.md` and `doc/swarm_testset.md`
