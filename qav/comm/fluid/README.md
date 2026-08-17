# Fluid QAV source layout

This directory owns backend-neutral fluid model flags, analytical overrides, validators, and CPU
algorithm checks. Native wrappers and runtime drivers live under `qav/cuda/fluid/` and
`qav/rocm/fluid/`; the root Makefile continues to select production kernels unless a test model
provides an explicit same-named override.

Each model-local `const_defs.cuh` selects its analytical case before including the common constants.
Focused groups write below `qav/logs/fluid/BACKEND/SWEEP/groups/GROUP/`, direct wrappers write below
`groups/manual/`, and neither path can replace the canonical `all` archive. Each executable is
generated beside the model's `flags.mk` and is ignored by Git.

The complete test definitions, expected solutions, commands, acceptance criteria, evidence, and
remaining coverage are maintained in
[`doc/fluid_testset.md`](../../../doc/fluid_testset.md). The supplementary NumPy transcriptions are
documented in [`mock/README.md`](mock/README.md).
