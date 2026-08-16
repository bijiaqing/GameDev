# GameDev documentation

This directory is the canonical documentation for the current merged GameDev project. It describes the active source tree, the verified numerical methods, the evidence retained in the repository, and future work that has not yet been promoted into production.

Resolved audit diaries and rename histories are not canonical documents. Durable findings from the 2026-07-27 merged-code audit, the 2026-07-29 swarm-test audit, the 2026-07-30 KNN audit, and the 2026-08-01 static numerical audit have been incorporated into the documents below.

## Document map

| Document | Responsibility |
|---|---|
| [`fluid_numeric.md`](fluid_numeric.md) | Eulerian dust-fluid user guide covering model selection, initialization, equations, discretization, composition, state semantics, and limitations |
| [`fluid_testset.md`](fluid_testset.md) | Fluid analytical cases, measurement protocol, historical native results, commands, and missing verification |
| [`swarm_numeric.md`](swarm_numeric.md) | Lagrangian swarm user guide covering model selection, mass normalization, initialization, transport, diffusion, radiation, collisions, state semantics, and limitations |
| [`swarm_testset.md`](swarm_testset.md) | Swarm analytical and statistical cases, archived native results, commands, and missing verification |
| [`naming.md`](naming.md) | Parallel naming conventions and independent file-ownership rules for the fluid and swarm branches |

Numerical equations belong in the two `*_numeric.md` documents. Test definitions and evidence
belong in the two `*_testset.md` documents. Future designs must not be described as active
production behavior.

## Repository and build layout

The active merged project is at the repository root:

- `inc/comm/fluid/`, `inc/comm/swarm/`, `src/comm/fluid/`, and `src/comm/swarm/` contain complete backend-neutral files
- `inc/cuda/`, `inc/rocm/`, `src/cuda/`, and `src/rocm/` contain complete backend-owned files
- `mod/` contains production model configurations
- `qav/comm/fluid/` and `qav/comm/swarm/` contain shared verification definitions, criteria, and analyzers
- `qav/cuda/` and `qav/rocm/` contain backend test drivers and backend-owned test source
- `qav/tool/` contains backend-neutral QA utilities, while generated evidence is written below the
  ignored `qav/logs/` tree
- `qav/rocm/bench/` documents and runs the uninstrumented MI300A benchmark and profiler workflow

The root Makefile requires `MODEL`; `GPU_BACKEND=cuda|rocm` selects one compiler/backend and
`DUST_REPR := fluid|swarm` in the model flags selects one physical representation. Fluid builds
additionally select `FLUID_SWEEP := thread|block`. Collision-enabled swarm builds select
`COLLISION_SEARCH := kdtree|morton`.

`qav/tool/run_all.py` is the canonical transferable campaign entry point. It writes disjoint CUDA
and ROCm archives, checks each archive before transfer, and runs the cross-backend field and metric
comparisons after both archives are present. The exact command sequence is documented in
`qav/README.md`.

Model-local source files override same-named production translation units. Verification models use this mechanism only when an analytical setup cannot be expressed through the production interface.

The fluid and swarm representations do not share headers or translation units with one another
inside `comm/`. Related algorithms, including optical-depth construction, remain independently
implemented and tested. Backend-neutral complete files are shared between CUDA and ROCm.

The backend ownership rule is deliberately conservative: share a file only when the complete file
is backend-neutral, and otherwise keep complete CUDA and ROCm versions. The selected build may use
only `inc/comm`, `src/comm`, and one backend tree. Generated objects, executables, production
outputs, and QA records include the backend name, so CUDA and ROCm artifacts cannot be linked or
overwritten accidentally. Checkpoints are not supported across backends.

## Evidence status

| Area | Current repository evidence | Interpretation |
|---|---|---|
| fluid analytical suite | source definitions for the 85-case analytical matrix and thread/block comparisons | native CUDA and ROCm runs previously passed; generated records are intentionally not tracked and must be regenerated for current evidence |
| swarm analytical suite | source definitions for the 51-build analytical/statistical matrix, initialization, boundaries, collisions, and failure paths | native CUDA and ROCm runs previously passed; generated records are intentionally not tracked and must be regenerated for current evidence |
| adaptive-Morton KNN | ordinary, edge, periodic, wedge, and production-link harnesses for both backends | the validated block-parallel sorted top-$K$ merge remains implemented; regenerate backend manifests before making a current-machine claim |

The fluid and swarm source audits found no unresolved production correctness defect in their inspected scopes after the listed corrections were applied. That statement is a review result, not a substitute for the missing runtime cases documented in the verification files.

## Current cross-representation conventions

Both branches use spherical computational coordinates and the same cylindrical conversions, gas profiles, initial dust surface-density prescription, physical linear-velocity file convention, and model-selection mechanism. They intentionally differ in:

- Eulerian conserved fields versus Lagrangian representative states
- spherical fluid diffusion versus cylindrical swarm diffusion
- conservative fluid transport versus semi-analytic particle trajectories
- fluid density-diffusion momentum closure versus velocity-preserving stochastic particle displacement
- fluid pressureless Riemann evolution versus swarm KNN collision sampling

These are parallel scientific and naming conventions, not shared-code interfaces. The exact ownership rules are maintained in [`naming.md`](naming.md).

## Authority and maintenance

When statements disagree, use this order:

1. current production source, model constants, and model flags
2. freshly generated machine-readable native results under the ignored `qav/logs/` tree
3. the canonical documents in this directory
4. transcribed historical results, laboratory notes, and Git history

After a numerical change:

1. update the appropriate numerical-method document
2. add or update an analytical, statistical, or regression test
3. archive the metrics and environment needed to support the new claim
4. update the evidence status here

Do not create a new standalone audit diary for a resolved issue. Add the surviving invariant, limitation, or regression-test requirement to its canonical document
