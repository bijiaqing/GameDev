# GameDev documentation

This directory is the canonical documentation for the current merged GameDev project. It describes the active source tree, the verified numerical methods, the evidence retained in the repository, and future work that has not yet been promoted into production

Resolved audit diaries and rename histories are not canonical documents. Durable findings from the 2026-07-27 merged-code audit, the 2026-07-29 swarm-test audit, and the 2026-07-30 KNN audit have been incorporated into the documents below

## Document map

| Document | Responsibility |
|---|---|
| [`numerics_fluid.md`](numerics_fluid.md) | Eulerian dust-fluid equations, discretization, operator composition, state semantics, and unresolved numerical limitations |
| [`testset_fluid.md`](testset_fluid.md) | Fluid analytical cases, measurement protocol, historical native results, commands, and missing verification |
| [`numerics_swarm.md`](numerics_swarm.md) | Lagrangian representative-particle transport, diffusion, radiation, collisions, state semantics, and scientific limitations |
| [`testset_swarm.md`](testset_swarm.md) | Swarm analytical and statistical cases, archived native results, commands, and missing verification |
| [`format_variablename.md`](format_variablename.md) | Parallel naming conventions and independent file-ownership rules for the fluid and swarm branches |
| [`future_radial_model.md`](future_radial_model.md) | Planned vertically integrated radial-only swarm model, required corrections, isolation rules, and verification criteria |
| [`future_rocm_support.md`](future_rocm_support.md) | CUDA-to-ROCm portability assessment and implementation sequence |

Numerical equations belong in the two `numerics_*` documents. Test definitions and evidence belong in the two `testset_*` documents. Future designs must not be described as active production behavior

## Repository and build layout

The active merged project is at the repository root:

- `inc/fluid/` and `src/fluid/` contain the Eulerian fluid implementation
- `inc/swarm/` and `src/swarm/` contain the Lagrangian swarm implementation
- `mod/` contains production model configurations
- `qav/fluid/` and `qav/swarm/` contain verification models and validators

The root Makefile requires `MODEL` and reads that model's `flags.mk`. `DUST_REPR := fluid` or `DUST_REPR := swarm` selects exactly one source branch. Fluid builds additionally select `FLUID_SWEEP := thread` or `FLUID_SWEEP := block`. A model-local `const_defs.cuh` has include priority; models without one inherit the selected representation's defaults

Model-local source files override same-named production translation units. Verification models use this mechanism only when an analytical setup cannot be expressed through the production interface

The branches do not share headers or translation units. Related algorithms, including optical-depth construction, remain independently implemented and tested

## Evidence status

| Area | Current repository evidence | Interpretation |
|---|---|---|
| fluid analytical suite | `qav/fluid/out/` is currently empty | the 85-case A100 result table in `testset_fluid.md` is a retained historical baseline, not a presently archived machine-readable result set |
| fluid thread/block comparison | harness exists; no comparison JSON, build log, or profiler artifact is archived | implementation equivalence and timing claims should be regenerated on the target GPU before publication |
| swarm analytical suite | 25 metrics, 25 metadata files, and 10 environment records under `qav/swarm/out/` | all ten models passed natively on the recorded A100 environment |
| adaptive-Morton KNN | the clean promoted-source ordinary, wedge, and adversarial matrices are archived under `qav/swarm/test_knn/out/`, including exact topology at $N_P=10^6$; all current cases pass | the standalone archive records base revision `115aa791f9db149bf3d53838df2efc86134994b7` and its worktree state; copied collision-runtime comparisons remain an earlier development baseline and are not current production timings |

The fluid and swarm source audits found no unresolved production correctness defect in their inspected scopes after the listed corrections were applied. That statement is a review result, not a substitute for the missing runtime cases documented in the verification files

## Current cross-representation conventions

Both branches use spherical computational coordinates and the same cylindrical conversions, gas profiles, initial dust surface-density prescription, physical linear-velocity file convention, and model-selection mechanism. They intentionally differ in:

- Eulerian conserved fields versus Lagrangian representative states
- spherical fluid diffusion versus cylindrical swarm diffusion
- conservative fluid transport versus semi-analytic particle trajectories
- fluid density-diffusion momentum closure versus velocity-preserving stochastic particle displacement
- fluid pressureless Riemann evolution versus swarm KNN collision sampling

These are parallel scientific and naming conventions, not shared-code interfaces. The exact ownership rules are maintained in [`format_variablename.md`](format_variablename.md)

## Authority and maintenance

When statements disagree, use this order:

1. current production source, model constants, and model flags
2. current machine-readable native results retained under `qav/`
3. the canonical documents in this directory
4. transcribed historical results, laboratory notes, and Git history

After a numerical change:

1. update the appropriate numerical-method document
2. add or update an analytical, statistical, or regression test
3. archive the metrics and environment needed to support the new claim
4. update the evidence status here

Do not create a new standalone audit diary for a resolved issue. Add the surviving invariant, limitation, or regression-test requirement to its canonical document
