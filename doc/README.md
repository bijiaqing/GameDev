# GameDev documentation

This directory holds the reference documentation for GameDev. The top-level
[`README.md`](../README.md) is the user guide for building, configuring, running, and reading
output; the guides here state the disk model, equations, algorithms, and validation evidence behind
it. This page maps the guides, says which one owns each fact, and defines the project's terms.

## Contents

- [Documentation map](#documentation-map)
- [Glossary](#glossary)
- [Shared conventions](#shared-conventions)
- [Authority order](#authority-order)
- [Maintaining the documentation](#maintaining-the-documentation)

## Documentation map

| Document | Contents |
|---|---|
| [`../README.md`](../README.md) | user guide: requirements, building, feature flags, constants, running, failure messages, restarts, output files, reproducibility, repository layout |
| [`guide_basis.md`](guide_basis.md) | shared disk model: units and notation, coordinates and grid, supported geometries, gas disk, stopping time, turbulent diffusivity, initial dust surface density, drift velocity, radiation pressure and optical depth |
| [`guide_swarm.md`](guide_swarm.md) | Lagrangian swarm: particle state and mass weighting, initialization, transport, forces and radiation, stochastic diffusion, representative-particle collisions (pair rates, outcomes, frozen-bath event chain, bath controller), nearest-neighbor search, time integration and boundaries, accuracy, GPU implementation |
| [`guide_fluid.md`](guide_fluid.md) | Eulerian dust fluid: state, initialization, finite-volume transport, sources and radiation, diffusion and momentum closure, time integration and boundaries, accuracy, GPU implementation |
| [`guide_tests.md`](guide_tests.md) | swarm and fluid validation cases, references, and acceptance criteria |
| [`../val/README.md`](../val/README.md) | running, checking, and comparing the validation suites; archive acceptance and qualification rules |
| `../val/paper/*/README.md` | scientific campaigns and how to reproduce them |

### Where to find an answer

| Question | Where |
|---|---|
| Which flags and constants does a model use, and how do I build, run, and restart it? | [`README.md`](../README.md) |
| How is the gas disk defined, and which coordinates, units, and geometries exist? | [`guide_basis.md`](guide_basis.md) |
| What equation does a swarm or fluid operator solve, with which discretization and accuracy? | [`guide_swarm.md`](guide_swarm.md) or [`guide_fluid.md`](guide_fluid.md) |
| How are collision rates, events, bath durations, and neighbors computed? | [swarm collisions](guide_swarm.md#8-collisions) and [neighbor search](guide_swarm.md#9-nearest-neighbor-search) |
| Which claim does a validation case establish, and how is it judged? | [`guide_tests.md`](guide_tests.md) |
| How do I run the validation campaign, check an archive, and compare CUDA with ROCm? | [`val/README.md`](../val/README.md) |
| How is a scientific campaign reproduced? | the campaign README under `val/paper/` |

### Where each fact lives

Every fact has one owner. Other files state it in a sentence at most and link to the owner.

| Topic | Owner |
|---|---|
| comparison of the two representations | [README: Dust representations](../README.md#dust-representations) |
| Make variables, search and sweep selection, build directories | [README: Make variables](../README.md#make-variables) |
| feature flags and their `#error` rules | [README: swarm flags](../README.md#swarm-feature-flags), [fluid flags](../README.md#fluid-feature-flags) |
| model directories and overrides | [README: Model directories](../README.md#model-directories), [overrides](../README.md#source-and-header-overrides) |
| commonly changed constants | [README: Constants](../README.md#constants) |
| every constant with its default | [swarm](guide_swarm.md#24-parameters), [collision](guide_swarm.md#24-parameters), and [fluid](guide_fluid.md#24-parameters) parameters |
| console output and failure messages | [README: Running](../README.md#console-output) |
| output times, checkpoints, and file contents | [README: Output times](../README.md#output-times-and-checkpoints) |
| restart rules | [README: Restarting](../README.md#restarting-a-simulation) |
| what saved fields mean numerically, restart numerics | [swarm](guide_swarm.md#128-output-and-restart-semantics) and [fluid](guide_fluid.md#106-output-and-restart-semantics) output semantics |
| where finite-state checks run | [swarm](guide_swarm.md#127-finite-state-and-error-checks) and [fluid](guide_fluid.md#105-finite-state-checks) finite-state checks |
| collision error codes | [collision error checks](guide_swarm.md#127-finite-state-and-error-checks) |
| repository layout | [README: Repository layout](../README.md#repository-layout) |
| units, notation, coordinates, grid, geometries, continuity equations | [disk: scope and notation](guide_basis.md#1-scope-and-notation), [coordinates and grid](guide_basis.md#2-coordinates-and-grid) |
| gas disk, viscosity, viscous flow, one-way coupling | [disk: gas disk](guide_basis.md#3-gas-disk) |
| Stokes number, diffusivities, initial dust surface density, drift velocity | [disk: stopping time](guide_basis.md#4-stopping-time-and-stokes-number), [diffusivity](guide_basis.md#5-turbulent-diffusivity), [dust surface density](guide_basis.md#6-dust-surface-density), [drift velocity](guide_basis.md#7-steady-drift-velocity) |
| radiation ratio, startup ramp, optical depth | [disk: radiation](guide_basis.md#8-radiation-pressure-and-optical-depth) |
| swarm state, initialization, transport, forces, diffusion, time stepping | [`guide_swarm.md`](guide_swarm.md) |
| collision rates, outcomes, bath controller, neighbor search | [swarm collisions](guide_swarm.md#8-collisions), [neighbor search](guide_swarm.md#9-nearest-neighbor-search) |
| fluid state, initialization, transport, sources, diffusion, time stepping | [`guide_fluid.md`](guide_fluid.md) |
| validation cases, per-suite record counts, file tables, and exclusions | [`guide_tests.md`](guide_tests.md) |
| archive acceptance, comparison rules, qualification, adding a case | [`val/README.md`](../val/README.md) |
| terms, shared conventions, authority order | this page |

## Glossary

| Term | Meaning | Defined in |
|---|---|---|
| representative particle (swarm particle) | computational particle standing for $N_p$ identical grains with represented mass $W_p$ | [state](guide_swarm.md#21-stored-state), [representatives](guide_swarm.md#81-representative-particles) |
| represented mass $W_p$, grain count $N_p$ | physical mass and number of grains a representative carries; collisions keep $W_p$ fixed | [weights](guide_swarm.md#34-grain-sizes-and-representative-weights) |
| mass bank | 128-entry table of the domain mass $I(s)$ on a logarithmic size axis | [domain mass](guide_swarm.md#33-dust-mass-in-the-domain) |
| equal-area proposal | grain-size sampling law used with radiation, corrected by importance weights | [weights](guide_swarm.md#34-grain-sizes-and-representative-weights) |
| absorbed representative (sentinel state) | particle that left through an absorbing boundary and is skipped afterwards | [boundaries](guide_swarm.md#103-boundary-conditions) |
| radial-only closure | `N_X == 1`, `N_Z == 1` swarm model with midplane dynamics and a retained $\ell_\phi$ | [closure](guide_swarm.md#25-radial-only-closure) |
| stored angular variables $(\ell_\phi,v_r,\ell_\theta)$ | internal velocity state of both representations | [swarm](guide_swarm.md#21-stored-state), [fluid](guide_fluid.md#21-stored-state) |
| evolved density $\varrho_d$ | surface density $\Sigma_d$ when `N_Z == 1`, volume density $\rho_d$ otherwise | [notation](guide_basis.md#14-notation) |
| geometry names | radial-only, radial–azimuthal, radial–polar, full 3D | [geometries](guide_basis.md#24-supported-geometries) |
| well-mixed closure | 2D assumption that dust shares the vertical profile of the gas | [optical depth](guide_basis.md#82-radial-optical-depth) |
| containment factor $\mathcal C_g$ | fraction of the gas column inside a finite spherical domain | [vertical structure](guide_basis.md#32-vertical-structure) |
| rotation-support parameter $\eta$ | effective sub-Keplerian parameter, including off-midplane gravity | [rotation](guide_basis.md#34-rotation-support) |
| edge taper | Gaussian convolution of the initial dust profile near the radial edges | [edge taper](guide_basis.md#62-edge-taper) |
| Stokes number, stopping time | drag coupling $\mathrm{St}=\Omega_Kt_s$ | [stopping time](guide_basis.md#4-stopping-time-and-stokes-number) |
| Schmidt number | ratio of gas viscosity to dust diffusivity before Stokes suppression | [diffusivities](guide_basis.md#51-directional-diffusivities) |
| density or concentration diffusion | whether the dust density or the dust-to-gas ratio is diffused (`DIFFUSE_CONCENTRATION`) | [diffusion](guide_basis.md#52-density-and-concentration-diffusion) |
| code units (`CODE_UNIT`) | collision-microphysics calibration without molecular gas constants | [collision parameters](guide_swarm.md#24-parameters) |
| owner | particle whose collision rate and size are being evolved | [representatives](guide_swarm.md#81-representative-particles) |
| partner | neighbor sampled for an event; never modified by it | [representatives](guide_swarm.md#81-representative-particles) |
| retained neighbors ($N_K$) | an owner's exact $N_K$ nearest neighbors inside the search cap | [search contract](guide_swarm.md#92-search-contract) |
| search cap (`H_SEARCH`) | largest neighbor distance, in local gas scale heights | [search contract](guide_swarm.md#92-search-contract) |
| KNN measure | boundary-corrected area or volume of the ball reaching the farthest retained neighbor | [KNN measure](guide_swarm.md#842-knn-measure) |
| image code | $c=3i+a$, packing a particle index $i$ and a periodic image $a$ | [periodic images](guide_swarm.md#95-periodic-images) |
| ghost record | Morton copy of a particle across a wedge seam | [periodic images](guide_swarm.md#95-periodic-images) |
| physical or synthetic kernel | `COAG_KERNEL = 3` versus the normalized kernels `0`–`2` | [kernels](guide_swarm.md#841-collision-kernels) |
| sticking packet | several tiny projectiles merged into one sticking event | [outcomes](guide_swarm.md#851-outcome-channels) |
| bath (frozen bath) | interval with fixed positions, neighbors, and published partner properties | [frozen reservoir](guide_swarm.md#861-frozen-reservoir) |
| reservoir publication | snapshot of partner sizes and numbers used during a bath | [frozen reservoir](guide_swarm.md#861-frozen-reservoir) |
| event cap, continuation launch | events allowed per kernel launch; unfinished owners continue in the next launch | [event cap](guide_swarm.md#863-event-cap-and-continuation) |
| no-event screen | quick draw that completes owners with no event in the interval | [screen](guide_swarm.md#864-cached-rates-and-the-no-event-screen) |
| controller group | spatial bin whose bath durations are scheduled together | [groups](guide_swarm.md#871-controller-groups-and-size-bins) |
| merged size bin | logarithmic size bin merged with neighbors to reach a target of `COL_BIN_MIN` owners | [groups](guide_swarm.md#871-controller-groups-and-size-bins) |
| operator horizon $h_{\rm op}$ | duration of one collision operator: half a dynamics step, or the whole output interval in collision-only runs | [bath duration](guide_swarm.md#872-requested-bath-duration) |
| bath level, tick lattice | power-of-two bath durations on a grid of $2^{52}$ ticks | [scheduler](guide_swarm.md#873-power-of-two-scheduler) |
| refresh wave | one advance of all currently due controller groups | [scheduler](guide_swarm.md#873-power-of-two-scheduler) |
| compensator | path-integrated hazard used by the audit | [audit](guide_swarm.md#874-post-bath-audit) |
| overshoot (activity, distribution, persistent) | audit exceedance; recorded, never rejected | [overshoots](guide_swarm.md#875-safety-factor-and-overshoots) |
| safety factor $\ell_c$ | per-group multiplier in $[0.25,1]$ on the bath tolerance | [overshoots](guide_swarm.md#875-safety-factor-and-overshoots) |
| geometry epoch | interval with unchanged positions, over which the search and neighbor cache are reused | [epochs](guide_swarm.md#125-geometry-epochs) |
| Strang (palindromic) composition | symmetric ordering of the operators within one step | [swarm](guide_swarm.md#101-operator-composition), [fluid](guide_fluid.md#81-operator-composition) |
| Itô process, Euler–Maruyama step | stochastic diffusion process and its one-step scheme | [Euler–Maruyama](guide_swarm.md#72-eulermaruyama-step) |
| velocity reprojection | keeping the Cartesian velocity through a diffusion displacement | [reprojection](guide_swarm.md#74-velocity-reprojection) |
| sweep (thread or block) | work decomposition of the fluid line kernels (`FLUID_SWEEP`) | [parallel mapping](guide_fluid.md#102-parallel-mapping) |
| directional operator ($A_x$, $D_y$, …) | one-direction advection or diffusion step | [composition](guide_fluid.md#81-operator-composition) |
| PPM, HLL | parabolic face reconstruction; two-wave flux | [PPM](guide_fluid.md#52-ppm-reconstruction), [HLL](guide_fluid.md#53-pressureless-hll-flux) |
| invariant-domain correction | face limiter keeping density nonnegative and primitives bounded | [correction](guide_fluid.md#54-invariant-domain-correction) |
| FARGO | integer orbital shift plus residual azimuthal transport | [FARGO](guide_fluid.md#55-fargo-azimuthal-transport) |
| SSPRK(3,3) | three-stage strong-stability-preserving Runge–Kutta scheme | [time integration](guide_fluid.md#56-radial-and-polar-time-integration) |
| vacuum state (`RHO_VAC`) | regularized velocity of near-empty cells | [state](guide_fluid.md#21-stored-state) |
| positivity subcycling (`POS_LIMIT`) | Crank–Nicolson substep count and donor-export bound | [Crank–Nicolson](guide_fluid.md#72-cranknicolson-solve) |
| donor-momentum closure | diffusive mass flux carries the donor cell's velocity | [closure](guide_fluid.md#74-donor-momentum-closure) |
| outer-face optical depth | cumulative $\tau$ stored at the outer radial cell faces | [optical depth](guide_basis.md#82-radial-optical-depth) |
| radiation ramp | smooth startup $f_\beta$ of radiation pressure over `T_BETA` | [ramp](guide_basis.md#81-radiation-ratio-and-startup-ramp) |
| frame, particle checkpoint, mesh-only frame | output index; frames with and without swarm particle files | [output times](../README.md#output-times-and-checkpoints) |
| RNG-state checkpoint | each representative's raw backend random-number state, saved with a particle checkpoint | [restarting](../README.md#restarting-a-simulation) |
| model directory, override, `MODEL_PARENT` | build configuration mechanism | [model directories](../README.md#model-directories), [overrides](../README.md#source-and-header-overrides) |
| metric record, model manifest, suite manifest | archive files, from one resolution up to the whole suite | [archive acceptance](../val/README.md#archive-acceptance) |
| tier | evidence label on records and manifests; only `publication` exists | [archive acceptance](../val/README.md#archive-acceptance) |
| source fingerprint | hash of the sources a campaign ran on | [qualification](../val/README.md#what-qualifies-a-result) |
| campaign record | `run_all_<backend>_<sweep>.json`, the record of one native campaign | [campaign](../val/README.md#running-a-complete-campaign) |
| activation check | test condition proving that the targeted branch was exercised | [swarm](guide_tests.md#12-what-passing-means), [fluid gates](guide_tests.md#25-fluid-acceptance-gates) |
| shared gate, validator-owned gate | the two acceptance paths of the fluid cases | [fluid gates](guide_tests.md#25-fluid-acceptance-gates) |

## Shared conventions

### Common to both representations

Both representations use the disk model of [`guide_basis.md`](guide_basis.md):

- the spherical computational coordinates and cylindrical conversions of
  [coordinates](guide_basis.md#21-computational-coordinates), with the mesh and geometries of
  [supported geometries](guide_basis.md#24-supported-geometries);
- the same prescribed gas disk, with one-way coupling ([gas disk](guide_basis.md#3-gas-disk));
- the same Stokes-number and diffusivity definitions
  ([stopping time](guide_basis.md#4-stopping-time-and-stokes-number),
  [diffusivities](guide_basis.md#51-directional-diffusivities));
- the same initial dust surface density with its edge taper
  ([edge taper](guide_basis.md#62-edge-taper));
- the same radiation ramp and outer-face optical depth
  ([radiation](guide_basis.md#81-radiation-ratio-and-startup-ramp)).

### Where the representations differ

The representations intentionally differ in:

- Lagrangian representative particles versus Eulerian conserved fields;
- cylindrical swarm diffusion versus spherical fluid diffusion, which give different radial
  equilibria away from the midplane ([why the bases
  differ](guide_basis.md#53-why-the-diffusion-bases-differ));
- semi-analytic particle trajectories versus conservative finite-volume transport;
- velocity-preserving stochastic displacement
  ([reprojection](guide_swarm.md#74-velocity-reprojection)) versus diffusive fluid momentum
  transport ([donor momentum](guide_fluid.md#74-donor-momentum-closure));
- the initial vertical velocity: terminal settling in the swarm
  ([swarm](guide_swarm.md#32-initial-velocity)) versus the diffusive balance in the fluid
  ([fluid](guide_fluid.md#32-initial-velocity));
- neighbor-based stochastic collisions versus pressureless Riemann evolution.

These are paired scientific conventions, not shared-code interfaces.

## Authority order

When statements disagree, use this order:

1. current production source, flags, and constants
2. machine-readable validation results under `val/{swarm,fluid}/out/` and `val/*.json` produced
   from that source
3. the documents in this directory and the READMEs
4. historical results and development notes

A validation result qualifies only the source it was produced from; [what qualifies a
result](../val/README.md#what-qualifies-a-result) defines the source fingerprint and the
qualification rules.

## Maintaining the documentation

- Check a statement against the source before changing it, and state current behavior in the
  present tense; development history belongs in version control.
- Change a fact in its owner ([Where each fact lives](#where-each-fact-lives)) and link to it from
  elsewhere rather than repeating it.
- Update the owning numerical guide together with any change to an equation, algorithm, or
  default, and the test-set guide together with any change to a validation case or criterion.
- Add a validation case only as described in [Adding a validation
  case](../val/README.md#adding-a-validation-case).
- Define each project term at its first use in a file and link it once to the
  [glossary](#glossary); add new terms here.

The code style, naming, and commenting rules, including the Markdown rules that keep these files
rendering correctly on GitHub, are kept in the local, untracked `doc/code_style.md` and checked by
`val/tools/check_style.py`.
