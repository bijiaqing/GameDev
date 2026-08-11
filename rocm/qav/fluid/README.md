# Fluid tests

This directory contains all test-only fluid material:

- `test_*/`: HIP/ROCm models with analytical reference solutions;
- `test_common/`: shared HIP/ROCm test driver, validation code, and suite runner;
- `mock/`: one documented CPU regression program for numerical building blocks;
- `out/`: generated verification metrics and environment records, grouped first by `thread` or
  `block`, then by model and swept parameter configuration.

From the repository root, run the complete HIP/ROCm verification matrix with:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
```

Use `FLUID_SWEEP=block python3 qav/fluid/test_common/run_suite.py ...` to run the same cases with
the one-block-per-line kernels instead of the default one-thread-per-line reference kernels.

Run the matched thread-versus-block branch with:

```bash
python3 qav/fluid/test_common/run_suite.py --group sweep --quick --target gfx942
python3 qav/fluid/test_common/run_suite.py --group sweep --sweep-dim all --target gfx942
```

The sweep runner prints a compact live simulation heartbeat every ten seconds while retaining the
complete executable stream and parsed accepted-step history in each model's `run.json`; use
`--progress-seconds N` to change the interval or `--progress-seconds 0` to suppress the heartbeat.
Compiler output is stored in `build.json`, while production parameters are converted to
`variables.json`, so generated QA records do not use ad hoc text files.
Every sweep configuration receives a tag such as `N32_save1_tout0p1` in its data-directory and
comparison filename, preventing quick, full, and custom-resolution runs from overwriting one another.

The quick command uses small grids and one short output interval. The full command runs the
transport-only `512^2` and diffusion-enabled `128^3` comparisons, applies automatic field and mass
checks, reports accepted-step counts, and records timing ratios. Radiation is disabled in both pairs.
Because this branch is substantially more expensive, `--group all` continues to mean all analytical
convergence cases and does not implicitly run `--group sweep`.

Run the supplementary CPU algorithm checks with:

```bash
python3 qav/fluid/mock/algorithm_checks.py
```

These checks are readable NumPy transcriptions and do not replace tests of the
actual HIP/ROCm kernels. See `mock/README.md` for their scope and interpretation.

HIP/ROCm test executables remain inside their respective `test_*` directories and are ignored by
Git. Thread and block results remain under `qav/fluid/out/thread/` and `qav/fluid/out/block/`,
respectively. Raw outputs from parameter sweeps use subdirectories such as `cfl0.05/`,
`shift3.25/`, and `p-1/`, so neither sweep implementations nor parameter configurations overwrite
one another.
Production outputs remain under top-level `out/`.
