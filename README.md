# GameDev

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

This project combines the Eulerian dust-fluid and Lagrangian dust-swarm solvers.
Each model selects one representation through `DUST_REPR` in its `flags.mk`.

Both the fluid and swarm solvers are active at the repository root. Each representation owns its
current headers, kernels, runtime, and optical-depth pipeline. `inc/share/` and `src/share/` are
reserved for future representation-independent interfaces and currently provide no production
implementation.

Build a fluid model from the repository root with, for example:

```bash
make MODEL=fluid_fiducial
```

Build the swarm model with:

```bash
make MODEL=swarm_fiducial
```

All test models, test scripts, generated test results, and standalone algorithm mocks live below
`qav/`, separately from production models in `mod/`. Run the complete fluid verification
suite with:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

For both representations, the source search order is model override followed by the selected
representation branch. Fluid compilation additionally places the reserved `inc/share/` after
`inc/fluid/` on the include path; swarm compilation uses `inc/swarm/` only. See
[`doc/README.md`](doc/README.md) for the canonical numerical, verification, architecture, and
naming references.
