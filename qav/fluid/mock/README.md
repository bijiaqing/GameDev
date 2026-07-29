# Fluid CPU algorithm checks

This directory contains one supplementary regression program:
`algorithm_checks.py`. It translates selected numerical building blocks into
NumPy so their intermediate values can be inspected without a CUDA compiler or
GPU.

From the repository root, run:

```bash
python3 qav/fluid/mock/algorithm_checks.py
```

NumPy is the only dependency. The complete run performs several refinement
sequences and dense reference solves, so it can take roughly a minute even
though it does not use a GPU. It exits with status zero only when every check
passes, making it suitable for a lightweight regression job.

## What the groups check

| Group | Purpose |
|---|---|
| T1 | periodic Crank-Nicolson diffusion and the Sherman-Morrison cyclic solve |
| T2 | cylindrical and spherical radial diffusion geometry |
| T3 | exact drag relaxation with a linearly changing external force |
| T4 | nonuniform PPM face weights and reconstruction order |
| T5 | periodic FARGO plus PPM transport behavior |
| T6 | radial optical-depth quadrature |
| T7 | the former radial Euler operator at small and production CFL numbers |
| T8 | the accuracy improvement from SSPRK2 to SSPRK(3,3) |
| T9 | the current HLL/PPM invariant-domain radial transport update |
| T10 | logarithmic-radial and polar mesh measures |

Each output line starts with `PASS` or `FAIL`, followed by the property being
tested and diagnostic numbers. The final line reports how many checks passed.

## Important limitation

This program is not an independent execution of the CUDA source. It is a
manually maintained Python transcription, so it can pass even if it has drifted
away from the production implementation. Use it for fast algorithm development
and algebraic regression checks, but use `test_common/run_suite.py` as the
authoritative end-to-end validation of the actual CUDA kernels.

When a production numerical algorithm changes, update its corresponding Python
section and explanatory comments together. Historical one-off diagnosis scripts
were removed after their relevant checks were consolidated here; Git history can
recover them if an old investigation ever needs to be reproduced.
