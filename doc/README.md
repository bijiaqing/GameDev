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

## Source and validation layout

- `inc/fluid/`, `inc/swarm/`, `src/fluid/`, and `src/swarm/` contain complete
  backend-neutral files.
- `inc/cuda/`, `inc/rocm/`, `src/cuda/`, and `src/rocm/` contain complete backend-owned files.
- `mod/` contains the published production model configuration.
- `val/comm/` contains backend-neutral publication-test definitions and validators.
- `val/cuda/` and `val/rocm/` contain native drivers and backend-owned test source.
- `val/tool/` contains archive and cross-backend utilities.
- ignored numerical evidence is written below `val/logs/`, while disposable validation builds are
  written below `val/temp/`.

The build shares a file only when the complete file is backend-neutral. It does not split partially
portable translation units. Fluid and swarm files are also kept separate rather than creating a
cross-representation shared layer.

Production models select one GPU backend, one fluid sweep implementation where applicable, and one
collision-search backend. Their executable is written beside the model flags. Validation archives retain
explicit CUDA/ROCm paths because completed native results must coexist for comparison. Checkpoints
are not portable across backends.

## Publication validation state

The current source defines a deliberately compact publication suite:

- fluid: 19 models and 76 analytical or invariant-based metric records;
- swarm: 15 common entries, including 14 analytical/statistical models with 42 records and the KNN matrix;
- one standalone KNN correctness matrix at $10^5$ particles;
- four production frozen-bath collision-chain models.

The former expanded archives and their micro-tests were retired. The current retained matrices
have passing native CUDA and ROCm records; the dated assessments in the test guides distinguish
those numerical passes from unresolved comparison and source-provenance checks. Results apply to
their recorded source snapshot. Commands, archive rules, and separate production-build checks are
documented in [`val/README.md`](../val/README.md).

### Repair qualification, 2026-09-05

Commit `9cdb96a` contains the six source repairs below. Numerical methods and acceptance criteria
are maintained in the linked guides; native results apply to the source snapshots recorded there.

| Repaired defect | Current behavior and evidence |
|---|---|
| Unbounded old-donor diffusion transfer | All six fluid directional sweeps limit outgoing mass before transporting momentum; native limiter and smooth-diffusion cases pass on both backends and sweeps ([method](fluid_numeric.md#73-conservative-donor-momentum-closure), [tests](fluid_testset.md#native-archive-assessment-2026-09-05)) |
| Wrong collision-partner image orientation | Queries and caches retain the selected image for physical rates; native cache-rate, KNN, and collision-chain cases pass ([method](swarm_numeric.md#85-periodic-boundary-ghosts), [tests](swarm_testset.md#8-physical-collision-rates)) |
| Diffusion velocity projected at the wrapped wedge angle | Velocity uses the unwrapped stochastic endpoint; native planar and 3D seam cases pass ([method](swarm_numeric.md#72-boundaries-and-velocity-reprojection), [tests](swarm_testset.md#6-stochastic-diffusion)) |
| Model override bypassed by ROCm extension preference | Directory-first source selection and source-identity sidecars handle override changes; native override/amplitude evidence remains outstanding |
| Stale executable after returning to cached objects | Every configured build relinks the selected objects; native cached A → B → A evidence remains outstanding |
| Collision bookkeeping compiled without collisions | Controller declarations, reset, and output require `COLLISION && !BERNOULLI`; native non-collision production-runtime evidence remains outstanding |

The build contract is in the root [README](../README.md#source-and-header-overrides), and the
remaining native build gates are in the [validation guide](../val/README.md#production-build-qualification).
Full qualification also remains open on the swarm comparator's 60 diagnostic mismatches and the
missing block-run source fingerprints, as detailed in the dated testset assessments. Completed
numerical campaigns need not be rerun solely because older aggregate files report static failures.

The underlying source review examined fluid/swarm operators, helpers, runtimes, build routing,
model overrides, and validation references, with CUDA/ROCm differences inspected separately.
Vendored KD-tree support received dependency/difference inspection rather than an exhaustive
independent proof. Host algebra and mock-build reproductions established specific defects and
build-graph behavior; the later downloaded GPU archives provide the distinct native evidence.

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
2. freshly generated machine-readable evidence under `val/logs/`;
3. the canonical documents in this directory;
4. historical results and development notes.

Add a new publication test only when it establishes a distinct physical or numerical claim not
already covered by the retained matrix. Local helper checks, failure injection, restart plumbing,
and performance experiments should not expand the canonical scientific suite unless a publication
claim explicitly depends on them.
