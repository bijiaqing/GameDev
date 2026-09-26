# GameDev documentation

This directory holds the reference documentation for GameDev. The top-level
[`README.md`](../README.md) is the user guide for building, configuring, running, and reading
output; the guides here state the equations, algorithms, and validation evidence behind it.

## Document map

| Document | Contents |
|---|---|
| [`fluid_numeric.md`](fluid_numeric.md) | Eulerian dust fluid: coordinates and state, gas disk and initialization, continuum equations, transport, sources, diffusion, time stepping, boundaries, GPU implementation, limitations |
| [`swarm_numeric.md`](swarm_numeric.md) | Lagrangian swarm: particle state and mass weighting, initialization, trajectories, radiation, diffusion, collisions, nearest-neighbor search, time stepping, GPU implementation, limitations |
| [`fluid_testset.md`](fluid_testset.md) | fluid validation cases, analytical references, measurements, and acceptance criteria |
| [`swarm_testset.md`](swarm_testset.md) | swarm validation cases, statistical references, neighbor-search and collision-chain criteria |
| [`../val/README.md`](../val/README.md) | running, archiving, and comparing the validation suites |

Typical questions and where they are answered:

| Question | Where |
|---|---|
| Which flags and constants does a model use, and how do I build and run it? | [`README.md`](../README.md) |
| What equation does an operator solve, and with which discretization and accuracy? | the representation's `*_numeric.md` |
| Which claim does a validation case establish, and how is it judged? | the representation's `*_testset.md` |
| How do I run the validation campaign and compare CUDA with ROCm? | [`val/README.md`](../val/README.md) |
| How is a scientific campaign reproduced? | the campaign README under `val/paper/` |

Physical equations and production algorithms belong in the numerical guides, and test definitions
and evidence rules in the test-set guides. The paired fluid and swarm derivations are kept
independent even where their gas-disk assumptions are identical.

The code style, naming, and commenting rules are kept in the local, untracked
`doc/code_style.md` and checked by `val/tools/check_style.py`.

## Source layout

- `inc/fluid/`, `src/fluid/` and `inc/swarm/`, `src/swarm/` own the two representations; kernels
  launched by the runtimes mostly have one source file each, named after the kernel.
- `inc/gpu.cuh` maps the CUDA and HIP runtime, RNG, and Thrust APIs and defines the `GPU_CHECK` and
  `GPU_KERNEL_CHECK` error checks, so both backends compile the same numerical sources.
- `mod/` holds production model configurations.
- `val/fluid/` and `val/swarm/` hold the validation models (`mod/`), shared drivers and validators
  (`src/`), and generated results (`out/`, `obj/`); `val/*.py` are the campaign, archive, and
  comparison utilities; `val/paper/` holds the scientific campaigns.

A build selects one GPU backend, one fluid sweep where applicable, and one collision search.
Validation outputs keep backend-separated paths so that CUDA and ROCm results can coexist for
comparison; checkpoints are not portable between backends.

## Conventions shared by both representations

Both representations use spherical computational coordinates, the same cylindrical conversions,
the same gas-profile convention, and the same dust surface-density normalization. They
intentionally differ in:

- Eulerian conserved fields versus Lagrangian representative particles
- spherical fluid diffusion versus cylindrical swarm diffusion, which give different radial
  equilibria away from the midplane
- conservative finite-volume transport versus semi-analytic particle trajectories
- diffusive fluid momentum transport versus velocity-preserving stochastic displacement
- pressureless Riemann evolution versus neighbor-based stochastic collisions

These are paired scientific conventions, not shared-code interfaces.

## Authority order

When statements disagree, use this order:

1. current production source, flags, and constants
2. machine-readable validation results under `val/{fluid,swarm}/out/` and `val/*.json` produced
   from that source
3. the documents in this directory and the READMEs
4. historical results and development notes

A validation result qualifies only the source it was produced from: `val/run_all.py` records a
source fingerprint, and a model run reported on its own is not a validation campaign.

## Maintaining the documentation

- Check a statement against the source before changing it, and state current behavior in the
  present tense; development history belongs in version control.
- Update the numerical guide together with any change to an equation, algorithm, or default, and
  the test-set guide together with any change to a validation case or criterion.
- Add a validation case only when it establishes a physical or numerical claim that the retained
  matrix does not already cover; helper checks, failure injection, restart plumbing, and
  performance experiments stay outside the scientific suite unless a claim depends on them.
