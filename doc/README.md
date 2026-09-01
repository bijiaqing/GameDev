# GameDev documentation

This directory contains the canonical documentation for the published GameDev source. Historical
audit diaries, private naming notes, non-fiducial model descriptions, exploratory benchmarks, and
superseded test inventories are not part of the public documentation.

## Document map

| Document | Responsibility |
|---|---|
| [`fluid_numeric.md`](fluid_numeric.md) | Eulerian dust-fluid equations, initialization, geometry, operators, composition, and limitations |
| [`fluid_testset.md`](fluid_testset.md) | retained fluid publication cases, analytical references, measurements, and acceptance criteria |
| [`swarm_numeric.md`](swarm_numeric.md) | Lagrangian swarm equations, initialization, trajectories, diffusion, collisions, and limitations |
| [`swarm_testset.md`](swarm_testset.md) | retained swarm publication cases, statistical references, KNN validation, and collision-chain criteria |

Physical equations and production algorithms belong in the numerical guides. Test definitions and
evidence rules belong in the test guides. The paired fluid/swarm derivations remain independent even
where their gas-disk assumptions are physically identical.

## Source and QAV layout

- `inc/comm/fluid/`, `inc/comm/swarm/`, `src/comm/fluid/`, and `src/comm/swarm/` contain complete
  backend-neutral files.
- `inc/cuda/`, `inc/rocm/`, `src/cuda/`, and `src/rocm/` contain complete backend-owned files.
- `mod/` contains the published production model configuration.
- `qav/comm/` contains backend-neutral publication-test definitions and validators.
- `qav/cuda/` and `qav/rocm/` contain native drivers and backend-owned test source.
- `qav/tool/` contains archive and cross-backend utilities.
- all ignored QAV build products and numerical evidence are written below `qav/logs/`.

The build shares a file only when the complete file is backend-neutral. It does not split partially
portable translation units. Fluid and swarm files are also kept separate rather than creating a
cross-representation shared layer.

Production models select one GPU backend, one fluid sweep implementation where applicable, and one
collision-search backend. Their executable is written beside the model flags. QAV archives retain
explicit CUDA/ROCm paths because completed native results must coexist for comparison. Checkpoints
are not portable across backends.

## Publication validation state

The current source defines a deliberately compact publication suite:

- fluid: 19 models and 73 analytical metric records;
- swarm: 12 common models and 33 analytical/statistical records;
- one standalone KNN correctness matrix at $10^5$ particles;
- four production frozen-bath collision-chain models.

The former expanded archives and their micro-tests were retired. A fresh native CUDA and ROCm
campaign is required to populate the new publication manifests after this reduction.
The exact commands and archive contract are documented in [`qav/README.md`](../qav/README.md).

## Current cross-representation conventions

Both physical representations use spherical computational coordinates, consistent cylindrical
conversions, the same gas-profile convention, and the same dust surface-density normalization.
They intentionally differ in:

- Eulerian conserved fields versus Lagrangian representative particles;
- spherical fluid diffusion versus cylindrical swarm diffusion;
- conservative finite-volume transport versus semi-analytic particle trajectories;
- diffusive fluid momentum transport versus velocity-preserving stochastic displacement;
- pressureless Riemann evolution versus KNN-based stochastic collisions.

These are paired scientific conventions, not shared-code interfaces.

## Maintenance rule

When statements disagree, use this authority order:

1. current production source, flags, and constants;
2. freshly generated machine-readable evidence under `qav/logs/`;
3. the canonical documents in this directory;
4. historical results and development notes.

Add a new publication test only when it establishes a distinct physical or numerical claim not
already covered by the retained matrix. Local helper checks, failure injection, restart plumbing,
and performance experiments should not expand the canonical scientific suite unless a publication
claim explicitly depends on them.
