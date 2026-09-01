# Fluid QAV source layout

This directory owns backend-neutral fluid model flags, analytical overrides, and validators.
Native wrappers and runtime drivers live under `qav/cuda/fluid/` and
`qav/rocm/fluid/`; the root Makefile continues to select production kernels unless a test model
provides an explicit same-named override.

Each model-local `const_defs.cuh` selects its analytical case before including the common constants.
Focused groups write below `qav/logs/fluid/BACKEND/SWEEP/groups/GROUP/`, direct wrappers write below
`groups/manual/`, and neither path can replace the canonical `all` archive. Executables, objects,
dependency files, and numerical results are all generated below `qav/logs/`.

The retained publication cases, expected solutions, acceptance criteria, and coverage limits are
maintained in [`doc/fluid_testset.md`](../../../doc/fluid_testset.md).
