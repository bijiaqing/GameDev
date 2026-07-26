# Merged GameDev development tree

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

This tree is the staged merge of the Eulerian dust-fluid and Lagrangian dust-swarm solvers.
Each model selects one representation through `DUST_REPR` in its `flags.mk`.

At the present migration stage, the fluid solver, its fiducial model, and its CUDA verification
suite are available. The `swarm/` branches are structural placeholders and are not buildable yet.

Build a fluid model from this directory with, for example:

```bash
make MODEL=fluid_fiducial
```

All test models, test scripts, generated test results, and standalone algorithm mocks live below
`tst/`, separately from production models in `mod/`. Run the complete migrated fluid verification
suite with:

```bash
python3 tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

The source search order is model override, selected representation, then `share`. At this stage,
all fluid headers remain in `inc/fluid`; only the optical-depth CUDA kernels are in `src/share`.
