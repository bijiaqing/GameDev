# Merged GameDev development tree

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

This tree is the staged merge of the Eulerian dust-fluid and Lagrangian dust-swarm solvers.
Each model selects one representation through `DUST_REPR` in its `flags.mk`.

Both the fluid and swarm solvers are now available. During this migration stage, the swarm branch
keeps a private copy of every dependency, including cuKD and its optical-depth kernels; it does not
consume code from `inc/share` or `src/share` yet.

Build a fluid model from this directory with, for example:

```bash
make MODEL=fluid_fiducial
```

Build the migrated swarm model with:

```bash
make MODEL=swarm_fiducial
```

All test models, test scripts, generated test results, and standalone algorithm mocks live below
`tst/`, separately from production models in `mod/`. Run the complete migrated fluid verification
suite with:

```bash
python3 tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

For fluid models, the source search order is model override, fluid branch, then `share`. For swarm
models, it is model override followed by the swarm branch, with no `share` fallback. All fluid
headers remain in `inc/fluid`; the swarm headers and cuKD dependency remain in `inc/swarm`.
