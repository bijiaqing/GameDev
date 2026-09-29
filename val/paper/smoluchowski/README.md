# Smoluchowski kernel benchmarks

These campaigns test whether the swarm model's collision code reproduces analytical solutions of the
Smoluchowski coagulation equation. They run the three normalized synthetic [collision
kernels](../../../doc/guide_swarm.md#841-collision-kernels) (constant, additive, and product)
through the production [frozen-bath event
chain](../../../doc/guide_swarm.md#86-frozen-bath-event-chain), [bath
controller](../../../doc/guide_swarm.md#87-bath-controller), and [KD-tree
search](../../../doc/guide_swarm.md#93-kd-tree), with the particles held in place, and sweep the
neighbor count and the refresh tolerance. The routine [collision-chain
tests](../../../doc/guide_tests.md#8-swarm-collision-chain) check that the same code conserves mass
and is reproducible; these campaigns measure its accuracy instead, as summarized in
[`guide_tests.md`](../../../doc/guide_tests.md#9-swarm-coagulation-campaigns). This README covers
all three kernels; what is specific to the additive kernel is in the companion
[`linear/README.md`](linear/README.md).

## Contents

- [At a glance](#at-a-glance)
- [Models](#models)
- [Setup](#setup)
- [Production code and overrides](#production-code-and-overrides)
- [Build and run](#build-and-run)
- [Outputs](#outputs)
- [Analysis](#analysis)
- [Limits](#limits)

## At a glance

| Item | Value |
|---|---|
| Representation | swarm, collisions only (neither `TRANSPORT` nor `DIFFUSION`) |
| Backend and search | CUDA with the KD tree, enforced by each kernel `Makefile` |
| Kernels | `const/` (constant), `linear/` (additive), `product/` (product) |
| Sweep | `N_K` = 16, 32, 64, 128, 256 by `COL_BATH_EPS` = 0.01, 0.02, 0.04, 0.08, 0.16 |
| Models | 25 per kernel, 75 in total |
| Seeds | `SEED` = 0 to 9 per model |
| Planned runs | 250 per kernel, 750 in total |
| Representatives | `N_P = 1000000` per run |
| Main flags | `COLLISION`, `MULTISIZE`, `CODE_UNIT`, `-DSEED=$(SEED)`; `LOGTIMING` in `const/` only |
| Analysis entry point | `<kernel>/src/score_model.py`, one call per run |
| Output location | raw output in `<kernel>/out/<model>/seed<N>/`, scores in `<kernel>/out/<model>/seed<N>.json` |

## Models

The three kernel directories share one layout. Each holds a `Makefile`, the shared sources in
`src/`, and 25 model directories under `mod/`, one for every combination of neighbor count and
refresh tolerance:

| Kernel directory | Kernel $\kappa(m_i,m_j)$ | `COAG_KERNEL` | Initial state |
|---|---|---|---|
| `const/` | $1$ | `0` | unit monomers |
| `linear/` | $m_i+m_j$ | `1` | exponential number distribution ([companion](linear/README.md#setup)) |
| `product/` | $m_im_j$ | `2` | unit monomers |

A model directory is named `k<N_K>_eps<COL_BATH_EPS>`, with the decimal point written as `p`; for
example, `k64_eps0p02` has `N_K = 64` and `COL_BATH_EPS = 0.02`. Its two files are:

- `const_defs.cuh`, an include fragment that defines `SWEEP_N_K` (the neighbor count `N_K`) and
  `SWEEP_COL_BATH_EPS` (the refresh tolerance `COL_BATH_EPS`) and then includes the kernel's shared
  `src/const_defs.cuh`;
- `flags.mk`, which selects `DUST_REPR := swarm`, `COLLISION_SEARCH := kdtree`, and
  `MODEL_PARENT := ../src` (the kernel's `src/`) and adds the flags listed above.

All 25 `flags.mk` files of a kernel are identical, and those of `linear/` and `product/` are
identical to each other. Particles do not move, because `TRANSPORT` and `DIFFUSION` are not
enabled.

## Setup

### Shared parameters

The shared `src/const_defs.cuh` of each kernel sets:

| Parameter | Value |
|---|---|
| Representatives | `N_P = 1000000`, equal represented mass |
| Initial grains | unit monomers of diameter 1 (`INIT_SMIN = INIT_SMAX = 1`); `linear/` replaces the sizes |
| Material density | `RHO_0` $=6/\pi$ |
| Total represented mass | `BENCHMARK_MASS = 1e30` |
| Positions | fixed jittered annular grid on $0.5\le R\le1.5$, full azimuth, midplane; mesh `(2, 100, 1)` |
| Pair measure | unit volume in place of the KNN measure |
| Search cap | `H_SEARCH = 128` gas scale heights |
| Fragmentation threshold | `V_FRAG = 1`; synthetic kernels have zero relative speed, so every event sticks |
| Threads per owner block | `COL_BATH_TPB = 64` |
| Event cap | `COL_EVENT_CAP = 32` |
| Size bins | `COL_BIN_S = 64` adaptive size bins, merged to at least `COL_BIN_MIN = 64` owners |
| Audit tail probability | `COL_BATH_ALPHA = 1e-3` |
| Longest bath | `COL_BATH_MAX = 1e100`, effectively disabled |

The positions form 399 radial rings of 2506 or 2507 particles each, spread evenly in azimuth; every
point is displaced by a uniform random fraction of up to a quarter of its cell in both directions.
The search cap, event cap, and merged size bins are defined in the
[glossary](../../../doc/README.md#glossary). Because `COL_BATH_MAX` never binds, the longest bath
is the operator horizon, which in these collision-only runs is the output interval.

### Per-kernel settings

| Setting | `const/` | `linear/` | `product/` |
|---|---|---|---|
| Output schedule | `LOGTIMING`, `DT_OUT = 0.1`, `LOG_BASE = 10`, `SAVE_MAX = 9` | linear, `DT_OUT = 1`, `SAVE_MAX = 4` | linear, `DT_OUT = 0.1`, `SAVE_MAX = 9` |
| Frame times | frame 0 is the monomer state; frames 1–9 are at $t=1,10,\ldots,10^8$ | $t=0,1,2,3,4$ | $t=0,0.1,\ldots,0.9$, strictly before gelation at $t=1$ |
| Controller groups | $8\times4$ (`COL_BIN_X`, `COL_BIN_Y`) | $8\times4$ | one |
| Partner mixing | none | none | before every refresh |

`COL_BIN_Z = 2` in `const/` and `linear/` collapses to one bin because the model has no polar
extent ([controller groups](../../../doc/guide_swarm.md#871-controller-groups-and-size-bins)).

### Seeds

`SEED` selects one of ten independent realizations; `src/const_defs.cuh` asserts that it lies in 0
to 9. Each random stream derives from it:

| Stream | Seed |
|---|---|
| Positions (host `std::mt19937`) | `SEED + 1` |
| Collisions (per-particle GPU stream) | `SEED + 1` |
| Initial masses, `linear/` only | `SEED` |
| Partner mixing, `product/` only (`COL_PARTNER_SEED = 2`) | `SEED + 2` |

The same seed gives the same initial state for every model of a kernel.

### Normalization and analytical solutions

The normalization makes each run a direct sample of the normalized Smoluchowski equation. Setting
the grain material density to $6/\pi$ makes the production grain-mass formula $m=\pi\rho s^3/6$
equal to $s^3$, preserving unit monomer mass without copying the production physics header.
Production uses $\lambda_0=N_P/(N_KM_\mathrm{tot})$, and with the unit-volume measure each
representative's summed constant-kernel propensity is initially one. The analytic normalized
initial number and mass densities are both one. For integer monomer mass $k$:

- Constant kernel: $g=1/(1+t/2)$, $n_k=g^2(1-g)^{k-1}$, and mass probability $kn_k$. The moments
  are $M_0=g$, $M_1=1$, and $M_2=1+t$.
- Product kernel: $n_k=k^{k-2}t^{k-1}e^{-kt}/k!$ and mass probability $kn_k$. The moments are
  $M_0=1-t/2$, $M_1=1$, and $M_2=1/(1-t)$ for $t\lt 1$.

The additive-kernel solution, a continuous distribution, is given in the
[companion](linear/README.md#analysis).

## Production code and overrides

The campaigns compile the root runtime, particle initializer, KD-tree builder and search,
collision controller, cached rates, sticking grouping, event updates, and moving size-bin bounds
directly; the bounds follow the production per-group policy of the
[bath controller](../../../doc/guide_swarm.md#871-controller-groups-and-size-bins). With neither
transport nor diffusion enabled, the
[geometry epoch](../../../doc/guide_swarm.md#125-geometry-epochs) never ends: tree build,
neighbor search, and neighbor-dependency graph construction occur once per fresh run, while grain
properties, rates, and collision refreshes continue to evolve.

Each kernel's `src/` directory contains only these files:

| File | What it replaces | Purpose |
|---|---|---|
| `const_defs.cuh` | the model constants header | the parameters above |
| `swarm_host.cuh` | wraps `inc/swarm/swarm_host.cuh` | total mass, seeded jittered positions, `[CAMPAIGN]` metadata; `linear/` also the gamma mass sampler, `product/` also partner mixing |
| `_collision.cuh` | wraps `inc/swarm/_collision.cuh` | unit-volume measure in place of the geometric one |
| `rngstate_init.cu` | `src/swarm/rngstate_init.cu` | collision RNG seeded with `SEED + 1` |
| `score_model.py` | none | the analytical scorer |

The two wrappers include the production header with the replaced functions renamed
(`get_total_dust_mass`, `rand_disk_mono`, and `save_variable`, plus `rand_powerlaw` in `linear/`;
`_get_ball_measure`) and then define their own versions. The unit-volume measure is one for any
positive neighbor distance and zero otherwise. No runtime, collision controller, physical velocity
model, or KD-tree implementation is copied. Production collision diagnostics (`COL_DIAGNOSTICS`)
are disabled to avoid large collision JSON and JSONL files.

### Product partner mixing

Partner mixing samples partners from the whole population in `product/`, instead of keeping
particles coupled to one fixed small partner population, which can amplify product-kernel growth
artificially. Before every collision refresh, including the first, a seeded global permutation
relabels all cached neighbor indices through the production `COL_PARTNER_REFRESH` hook. The tree
and geometric neighbor search are still performed only once. The partner RNG uses
`SEED + COL_PARTNER_SEED` (`SEED + 2`), separately from the collision RNG. Negative sentinel
entries and periodic image labels are preserved.

One spatial controller group, enforced by a static assertion, makes every refresh publish all
particles together; a refresh of fewer than `N_P` particles stops the run with an error. The 64
adaptive size bins still control the interval. Mixing precedes publication and full cached-rate
recalculation and never occurs during continuations or audits, so the single-group dependency graph
remains valid after relabeling. The idle continuation queue supplies GPU storage for the
permutation. Each refresh adds one CPU shuffle, one host-to-device copy of `N_P` integers, and one
relabeling kernel as a campaign-specific cost. Physical production models and the `const/` and
`linear/` campaigns do not enable the hook. Reshuffling changes partner sampling; it is not an
overflow guard.

## Build and run

Each kernel `Makefile` includes the root `Makefile` and redirects models, objects, and output into
the kernel directory. It forces `GPU_BACKEND=cuda` and stops with an error for any other backend or
for a `COLLISION_SEARCH` other than `kdtree`; `GPU_TARGET` defaults to `sm_80` and `SEED` to 0. Run
from the repository root. Build one model and seed:

```bash
make -C val/paper/smoluchowski/const -j8 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
```

Run it:

```bash
val/paper/smoluchowski/const/obj/seed0/k64_eps0p02/gamedev
```

Score it:

```bash
python3 -B val/paper/smoluchowski/const/src/score_model.py \
  val/paper/smoluchowski/const/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/const/out/k64_eps0p02/seed0.json
```

Use `linear` or `product` in place of `const` for the other kernels, and `SEED=<N>` with the
matching `seed<N>` paths for the other seeds.

The output path is compiled into the executable, so every seed has its own build. Separate build
and output paths for each seed permit independent jobs:

| Item | Path under the kernel directory |
|---|---|
| Executable | `obj/seed<N>/<model>/gamedev` |
| Objects | `obj/seed<N>/<model>/swarm/cuda/kdtree/<GPU_TARGET>/` |
| Raw output | `out/<model>/seed<N>/` |
| Score JSON | `out/<model>/seed<N>.json` |

`make -C <kernel directory> clean MODEL=<model> SEED=<N>` removes only that executable and its
objects. No job scripts or ensemble aggregation are included. Python entry points disable bytecode
caches.

## Outputs

Each run writes `variables.txt` and, at every frame, `particle_<frame>.dat` and
`rngstate_<frame>.dat`. A particle record holds eight doubles, 64 bytes: position, velocity, grain
diameter `par_size`, and represented grain count `par_numr`. One particle file is 64 MB and one RNG
checkpoint 48 MB with CUDA's `curandState`:

| Kernel | Frames | Size per run |
|---|---|---|
| `const/` | 10 | about 1.1 GB |
| `linear/` | 5 | about 0.56 GB |
| `product/` | 10 | about 1.1 GB |

See [Output files](../../../README.md#output-files) for the formats.

Besides the production metadata, which includes `COL_SIZE_BIN_POLICY` and
`COL_SIZE_RANGE_FACTORS`, `variables.txt` gains a `[CAMPAIGN]` section with `SEED`,
`POSITION_SEED`, `COLLISION_SEED`, `UNIT_VOLUME`, `GEOMETRY_REUSE`, `SIZE_BIN_POLICY`,
`SIZE_MIN_FACTOR`, and `SIZE_MAX_FACTOR`. `product/` adds `PARTNER_SEED`, `PARTNER_RESHUFFLE`, and
`PARTNER_GROUPS`, and `linear/` adds its [initial-mass keys](linear/README.md#outputs).

All raw output stays under `out/`, which `make clean` never touches, and nothing deletes it
automatically. Keep the seed-0 snapshots; reduce each nonzero-seed run to its score JSON before
removing its raw files: ten seeds of all 25 models hold about 0.28 TB for `const/` and `product/`
and 0.14 TB for `linear/`.

## Analysis

`score_model.py` reduces one run to a JSON file (the `--json` argument is required). It reads the
final particle frame only and scores the physical-mass probability, weighting each representative
by $N_i m_i/M_\mathrm{tot}$ with $m_i=s_i^3$. For `const/` and `product/` it histograms
$\log_{10}(m/m_0)$ on 200 coarse and 4096 fine bins between $-0.5$ and $9.5$, with explicit
underflow and overflow bins; the `linear/` bins and reference differ
([companion](linear/README.md#analysis)). The JSON file records:

- the total-variation distance on the coarse bins (the primary shape metric), the Jensen–Shannon
  distance, the CDF supremum distance on the fine bins, and the log-mass Wasserstein distance $W_1$
  in dex, integrated over the fine bins;
- mass, number, and $M_2/M_1$ ratios with their relative errors;
- the minimum and maximum mass and, for `const/` and `product/`, the maximum deviation from integer
  mass;
- simulated and analytical underflow and overflow mass;
- the coarse and fine histograms, the seeds, `N_P`, `N_K`, `COL_BATH_EPS`, and the reference
  definitions.

The constant-kernel reference is the closed-form CDF of the mass probability $kn_k$. The product
reference is the Borel mass distribution, tabulated to $k=10^5$. The outputs contain no formal pass
threshold. Look at how the distances change with `N_K` and `COL_BATH_EPS` against the scatter of
the ten seeds.

## Limits

These campaigns measure how closely the production chain, with its fixed neighbor sets and finite
bath durations, reproduces the analytical solutions at the sampled `N_K` and `COL_BATH_EPS`.

- Assess neighbor-count and refresh-tolerance dependence separately from seed scatter; a small
  controller tolerance is not a histogram-error bound.
- Distribution distances measure the actual numerical approximation, including production sticking
  grouping ([sticking packets](../../../doc/guide_swarm.md#851-outcome-channels)). Stored
  diameters carry roundoff, so cluster masses are integers only approximately; the scorer reports
  the largest deviation.
- Product results hold only before gelation and with partner mixing; they do not establish that a
  small fixed neighbor set suffices for the physical collision problem.
- The campaigns do not test convergence with `N_P`, Morton search, ROCm, transport, restarts,
  physical kernels, or performance scaling.
- The analytical scorers can be exercised on synthetic samples drawn from their reference
  distributions; neither such checks nor build dry runs are evidence of native CUDA compilation or
  GPU accuracy. Fresh CUDA runs are required.

The rules for qualifying a result are in
[What qualifies a result](../../README.md#what-qualifies-a-result).
