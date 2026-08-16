# Fluid tests

This directory contains backend-neutral fluid test definitions:

- `test_*/`: model flags and analytical source overrides;
- `test_common/`: shared physical overrides and test documentation;
- `mock/`: one documented CPU regression program for numerical building blocks;

CUDA and ROCm drivers live under `qav/cuda/fluid/` and `qav/rocm/fluid/`, respectively.

From the repository root, run the complete CUDA verification matrix with:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

Use `FLUID_SWEEP=block python3 qav/cuda/fluid/test_common/run_suite.py ...` to run the same cases with
the one-block-per-line kernels instead of the default one-thread-per-line reference kernels.

Run the matched thread-versus-block branch with:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --quick
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --sweep-dim all
```

The quick command uses small grids and one short output interval. The full command runs the
transport-only `1024^2` and diffusion-enabled `128^3` comparisons, applies automatic field and mass
checks, reports accepted-step counts, and records timing ratios. Radiation is disabled in both pairs.
Because this branch is substantially more expensive, `--group all` continues to mean all analytical
convergence cases and does not implicitly run `--group sweep`.

Run the supplementary CPU algorithm checks with:

```bash
python3 qav/comm/fluid/mock/algorithm_checks.py
```

These checks are readable NumPy transcriptions and do not replace tests of the
actual CUDA kernels. See `mock/README.md` for their scope and interpretation.

CUDA test executables are generated below `bin/MODEL/cuda/` and are ignored by Git. Thread and block
results are generated under `qav/logs/fluid/cuda/thread/` and `qav/logs/fluid/cuda/block/`,
respectively. Raw outputs from parameter sweeps use subdirectories such as `cfl0.05/`,
`shift3.25/`, and `p-1/`, so neither sweep implementations nor parameter configurations overwrite
one another.
Production outputs remain under top-level `out/`.
