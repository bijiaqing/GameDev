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

The naming and commenting contract for production source is kept in the local, untracked
`doc/MEMORY.md`. The untracked `doc/extension/`, `doc/proposal/`, and `doc/publication/`
directories hold proposals, manuscript planning, and dated development records; they do not
describe current behavior unless a statement is confirmed against the source.

Physical equations and production algorithms belong in the numerical guides. Test definitions and
evidence rules belong in the test guides. The paired fluid/swarm derivations remain independent even
where their gas-disk assumptions are physically identical.

## Source and validation layout

- `inc/fluid/`, `inc/swarm/`, `src/fluid/`, and `src/swarm/` own the two numerical representations.
- `inc/gpu.cuh` maps GPU APIs and execution policies; CUDA and ROCm compile the same numerical sources.
- `mod/` contains production model configurations.
- `val/{fluid,swarm}/mod/` contains test definitions; `src/` contains shared drivers and validators.
- `val/*.py` contains native campaign, archive, and cross-backend utilities.
- `val/{fluid,swarm}/out/` holds backend-separated results; `obj/` holds disposable builds.
- `val/paper/` contains scientific campaigns with their own READMEs, model overrides, and outputs.

Fluid and swarm retain separate physical operators even though they use the same GPU API mappings.

Production models select one GPU backend, one fluid sweep implementation where applicable, and one
collision-search backend. Their executable is written beside the model flags. Validation archives retain
explicit CUDA/ROCm paths because completed native results must coexist for comparison. Checkpoints
are not portable across backends.

## Publication validation state

The current source defines a deliberately compact publication suite:

- fluid: 19 models, expanded to 22 cases by three fixed-grid limiter variants, with 76
  analytical or invariant-based metric records;
- swarm: 15 common entries, namely 14 analytical/statistical models with 42 records plus the
  standalone KNN correctness matrix at $10^5$ particles;
- four production frozen-bath collision-chain models run as the separate `chain` group.

The case lists and expected record counts are defined in `val/val_config.py`.

The active validation tree was reconstructed around current production operators. Previous native
passes, including the September Morton integration campaign, describe their recorded source and
configuration; they do not qualify the present shared-source implementation or rebuilt tests.
Use [`val/README.md`](../val/README.md) for current commands, archive paths, and evidence status.
A model run reported by the user is not a complete validation campaign.

Block-run source provenance and the separate native build checks remain unresolved without matching
evidence. The latter cover model-override selection, cached A → B → A relinking, and non-collision
production builds. Old comparator mismatch counts are historical diagnostics, not results for the
current comparator. Do not restart completed historical campaigns merely to regenerate reports.

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

Inspect live Git status, the relevant source, and `doc/MEMORY.md` before editing. Preserve existing
dirty changes, including separate coagulation campaigns; an old checkout snapshot is not authority
to reset files. If current native archives are missing locally, request the cluster outputs before
updating qualification claims.

When statements disagree, use this authority order:

1. current production source, flags, and constants;
2. matching-source machine-readable evidence under `val/{fluid,swarm}/out/` and `val/*.json`;
3. the canonical documents in this directory;
4. historical results and development notes.

Add a new publication test only when it establishes a distinct physical or numerical claim not
already covered by the retained matrix. Local helper checks, failure injection, restart plumbing,
and performance experiments should not expand the canonical scientific suite unless a publication
claim explicitly depends on them.
