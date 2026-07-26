# Fluid tests

This directory contains all test-only fluid material:

- `verify_*/`: CUDA models with analytical reference solutions;
- `verify_common/`: shared CUDA test driver, validation code, and suite runner;
- `mock/`: one documented CPU regression program for numerical building blocks;
- `out/`: generated verification metrics and environment records, grouped by model and swept
  parameter configuration.

From `new/`, run the complete CUDA verification matrix with:

```bash
python3 tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

Run the supplementary CPU algorithm checks with:

```bash
python3 tst/fluid/mock/algorithm_checks.py
```

These checks are readable NumPy transcriptions and do not replace tests of the
actual CUDA kernels. See `mock/README.md` for their scope and interpretation.

CUDA test executables remain inside their respective `verify_*` directories and are ignored by
Git. Test metrics remain under `tst/fluid/out/`; raw outputs from parameter sweeps use subdirectories
such as `cfl0.05/`, `shift3.25/`, and `p-1/` so configurations cannot overwrite one another.
Production outputs remain under top-level `out/`.
