# Swarm validation layout

- `mod/<model>/`: model flags, overrides, and individual runners.
- `src/`: shared drivers and analysis.
- `out/<model>/<backend>/`: numerical results and manifests.
- `obj/<model>/<backend>/`: executables, objects, dependencies, and build stamps.

Focused runs append `groups/<group>/`; direct runners default to `groups/manual/`.
Full suites omit that suffix. Suite records use `out/_suite/<backend>/`.
Select CUDA or ROCm with `GPU_BACKEND=cuda` or `GPU_BACKEND=rocm`.

See [test definitions](../../doc/swarm_testset.md) and [campaign commands](../README.md).
