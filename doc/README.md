# GameDev documentation

This directory is the canonical documentation for the current merged GameDev project. It describes the active source tree, the verified numerical methods, the evidence retained in the repository, and future work that has not yet been promoted into production.

Resolved audit diaries and rename histories are not canonical documents. Durable findings from the 2026-07-27 merged-code audit, the 2026-07-29 swarm-test audit, the 2026-07-30 KNN audit, and the 2026-08-01 static numerical audit have been incorporated into the documents below.

## Document map

| Document | Responsibility |
|---|---|
| [`fluid_numeric.md`](fluid_numeric.md) | Eulerian dust-fluid user guide covering model selection, initialization, equations, discretization, composition, state semantics, and limitations |
| [`fluid_testset.md`](fluid_testset.md) | Fluid analytical cases, measurement protocol, latest archived native results, commands, and missing verification |
| [`swarm_numeric.md`](swarm_numeric.md) | Lagrangian swarm user guide covering model selection, mass normalization, initialization, transport, diffusion, radiation, collisions, state semantics, and limitations |
| [`swarm_testset.md`](swarm_testset.md) | Swarm analytical and statistical cases, latest archived native results, commands, and missing verification |

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
only `inc/comm`, `src/comm`, and one backend tree. Generated object directories remain
configuration-specific. Each model instead owns one executable beside its `flags.mk` and one
production output directory under `out/MODEL/`, because a production model selects only one
backend and algorithm configuration. QAV result archives retain explicit backend paths because
cross-backend comparison requires both records. Checkpoints are not supported across backends.

## Latest archived evidence

| Area | Current repository evidence | Interpretation |
|---|---|---|
| fluid analytical suite | archived 85-metric baseline on both CUDA and ROCm, plus eight direct polar-field comparisons | pre-expansion native archives and cross-backend comparison passed on 2026-08-17; the current 105-metric matrix awaits refresh |
| swarm analytical suite | archived 51-metric baseline on both CUDA and ROCm | pre-expansion native archives passed with no mismatches on 2026-08-17; the current 75-metric matrix awaits refresh |
| adaptive-Morton KNN | ordinary, edge, periodic, wedge, and production-link matrices on both backends | both compact native suites passed with equal coverage on 2026-08-17 |

The two complete campaigns used the same source SHA-256,
`efc181ba842c4dd6917db01d02afc3c7f6e501e7b63cc786983f6f2745243fd8`. CUDA used target `sm_80`,
CUDA 12.1 (`nvcc` 12.1.105), an NVIDIA
A100-SXM4-40GB, and driver 580.159.04. ROCm used target `gfx942`, ROCm 7.2.4, AMD clang
22.0.0git, an AMD Instinct MI300A, and amdgpu driver 6.16.13. Both used Python 3.13.5.

That fingerprint identifies the archived pre-expansion baseline and no longer matches the current
QAV source after addition of the four analytical sequences. Independent local archive checks and
both cross-backend comparators reproduced `PASS` for the archived source; new 105-fluid/75-swarm
native campaigns are required to qualify the expanded matrix. The remaining coverage
limits are listed in the two test-set documents.

## Current cross-representation conventions

Both branches use spherical computational coordinates and the same cylindrical conversions, gas profiles, initial dust surface-density prescription, physical linear-velocity file convention, and model-selection mechanism. They intentionally differ in:

- Eulerian conserved fields versus Lagrangian representative states
- spherical fluid diffusion versus cylindrical swarm diffusion
- conservative fluid transport versus semi-analytic particle trajectories
- fluid density-diffusion momentum closure versus velocity-preserving stochastic particle displacement
- fluid pressureless Riemann evolution versus swarm KNN collision sampling

These are parallel scientific conventions, not shared-code interfaces, and the two representations
remain independently implemented.

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
