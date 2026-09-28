# Smoluchowski kernel benchmarks

## Purpose

These campaigns compare the production frozen-bath collision chain with analytical solutions of the
Smoluchowski coagulation equation for the three normalized synthetic kernels. This README covers the
constant-kernel (`const/`) and product-kernel (`product/`) campaigns and the layout shared by all
three. The additive-kernel campaign is documented in [linear/README.md](linear/README.md); it uses
the same parameter grid and seed layout, an exponential physical initial mass distribution, and
fixed partners without reshuffling.

## Models

Each of `const/` and `product/` contains 25 models, `k{16,32,64,128,256}_eps0p{01,02,04,08,16}`.
Each `mod/<model>/const_defs.cuh` sets `SWEEP_N_K` (the neighbor count `N_K`) and
`SWEEP_COL_BATH_EPS` (the collision-refresh tolerance `COL_BATH_EPS`) and includes the shared
`src/const_defs.cuh`. `SEED=0` through `9` selects ten independent realizations; the source asserts
this range. Constant and product together define 500 runs; with the 250 linear-kernel runs, the
three families define 750 runs.

Every `flags.mk` selects `DUST_REPR := swarm`, `COLLISION_SEARCH := kdtree`,
`MODEL_PARENT := ../src`, `COLLISION`, `MULTISIZE`, `CODE_UNIT`, and `-DSEED=$(SEED)`; constant
models add `LOGTIMING`. `TRANSPORT` and `DIFFUSION` are not enabled, so particles do not move.

| Parameter | Value |
|---|---|
| Representatives | `N_P = 1000000`, equal represented mass |
| Initial grains | unit monomers, diameter 1 (`INIT_SMIN = INIT_SMAX = 1`) |
| Material density | `RHO_0` $=6/\pi$ |
| Total represented mass | `BENCHMARK_MASS = 1e30` |
| Positions | fixed jittered annular grid on $0.5\le R\le1.5$, full azimuth, mesh `(2, 100, 1)` |
| Kernel | `COAG_KERNEL = 0` (constant) or `2` (product), in a unit volume |
| Neighbor search | KD-tree, `H_SEARCH = 128` gas scale heights |
| Chain width and cap | `COL_BATH_TPB = 64`, `COL_EVENT_CAP = 32` |
| Size bins | `COL_BIN_S = 64` adaptive size bins, `COL_BIN_MIN = 64` |
| Spatial groups | constant: $8\times4$ (`COL_BIN_X`, `COL_BIN_Y`); product: one group |
| Audit tail probability | `COL_BATH_ALPHA = 1e-3` |
| Duration cap | `COL_BATH_MAX = 1e100`, effectively disabled |
| Seeds | positions and collisions use `SEED + 1`; product partner mixing uses `SEED + 2` |

The same seed gives matched initialization across parameter combinations.

| Kernel | Output schedule | Frames |
|---|---|---|
| Constant | `LOGTIMING`, `DT_OUT = 0.1`, `LOG_BASE = 10` | frame 0 is the monomer state; frames 1–9 are at $t=1,10,\ldots,10^8$ |
| Product | linear, `DT_OUT = 0.1` | $t=0,0.1,\ldots,0.9$, strictly before gelation at $t=1$ |

## Analytical normalization

Setting the grain material density to $6/\pi$ makes the production grain-mass formula
$m=\pi\rho s^3/6$ equal to $s^3$, preserving unit monomer mass without copying the production
physics header. Production uses $\lambda_0=N_P/(N_KM_\mathrm{tot})$, so initially each
representative's summed constant-kernel propensity is one. The analytic normalized initial number
and mass densities are both one. For integer monomer mass $k$:

- Constant kernel: $g=1/(1+t/2)$, $n_k=g^2(1-g)^{k-1}$, and mass probability $kn_k$. The moments
  are $M_0=g$, $M_1=1$, and $M_2=1+t$.
- Product kernel: $n_k=k^{k-2}t^{k-1}e^{-kt}/k!$ and mass probability $kn_k$. The moments are
  $M_0=1-t/2$, $M_1=1$, and $M_2=1/(1-t)$ for $t\lt 1$.

## Production code and overrides

The root runtime, particle initializer, KD-tree builder and search, collision controller, cached
rates, sticking grouping, event updates, and moving size-bin bounds are compiled directly; the
bounds follow the production per-group policy described in [the frozen-bath collision
chain](../../../doc/numerics_swarm.md#86-frozen-bath-event-chain). With neither
dynamics nor diffusion enabled, geometry is never invalidated: tree build, neighbor search, and
neighbor-dependency graph construction occur once per fresh run, while grain properties, rates, and
collision refreshes continue to evolve.

Each campaign `src/` directory contains only:

- `const_defs.cuh`: the constants above.
- `swarm_host.cuh`: substitutes the total mass, the seeded jittered positions, and appended campaign
  metadata; for product only, it also supplies partner reshuffling.
- `_collision.cuh`: substitutes a unit-volume neighbor measure for the geometric one.
- `rngstate_init.cu`: seeds the collision RNG with `SEED + 1`.
- `score_model.py`: the analytical scorer.

No runtime, collision controller, physical velocity model, or KD-tree implementation is copied.
Production collision diagnostics (`COL_DIAGNOSTICS`) are disabled to avoid large collision JSON and
JSONL files.

### Product-only partner mixing

Before every collision refresh, including the first, a seeded global permutation relabels all cached
neighbor indices through the production `COL_PARTNER_REFRESH` hook. The tree and geometric neighbor
search are still performed only once. The partner RNG uses `SEED + COL_PARTNER_SEED` (`SEED + 2`),
separately from the collision RNG. Negative sentinel entries and periodic image labels are
preserved. This gives well-mixed partner sampling instead of keeping particles coupled to one fixed
small partner population, which can amplify product-kernel growth artificially.

One spatial controller group, enforced by a static assertion, makes every refresh publish all
particles together; the 64 adaptive size bins still control the interval. Mixing precedes
publication and full cached-rate recalculation and never occurs during continuations or audits, so
the single-group dependency graph remains valid after relabeling. The idle continuation queue
supplies GPU storage for the permutation. Each refresh adds one CPU shuffle, one host-to-device copy
of `N_P` integers, and one relabeling kernel as a campaign-specific cost. Physical production models
and the constant campaign do not enable the hook. Reshuffling changes partner sampling; it is not an
overflow guard.

## Build and run

The campaign `Makefile` of each kernel forces `GPU_BACKEND=cuda` and rejects any other backend or a
`COLLISION_SEARCH` other than `kdtree`; `GPU_TARGET` defaults to `sm_80` and `SEED` to 0. From the
repository root:

```sh
make -C val/paper/smoluchowski/const -j8 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
val/paper/smoluchowski/const/obj/seed0/k64_eps0p02/gamedev
python3 -B val/paper/smoluchowski/const/src/score_model.py \
  val/paper/smoluchowski/const/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/const/out/k64_eps0p02/seed0.json
```

Use `product` instead of `const` for that kernel. Separate build and output paths for each seed
permit independent jobs:

| Item | Path under the kernel directory |
|---|---|
| Executable and objects | `obj/seed<N>/<model>/` |
| Raw output, seed 0 | `out/<model>/seed0/` |
| Raw output, seeds 1–9 | `obj/seed<N>/<model>/out/` |
| Score JSON | `out/<model>/seed<N>.json` |

No job scripts or ensemble aggregation are included. Python entry points disable bytecode caches.

## Outputs

Each run writes `variables.txt` and, at every frame, `particle_<frame>.dat` and
`rngstate_<frame>.dat`. A particle record holds eight doubles, 64 bytes: position, velocity, grain
diameter `par_size`, and represented grain count `par_numr`. One particle file is 64 MB and one RNG
checkpoint 48 MB with CUDA's `curandState`, so a run's ten frames occupy about 1.1 GB. See
[Output files](../../../README.md#output-files) for the formats.

Besides the production metadata, which includes `COL_SIZE_BIN_POLICY` and
`COL_SIZE_RANGE_FACTORS`, `variables.txt` gains a `[CAMPAIGN]` section with `SEED`,
`POSITION_SEED`, `COLLISION_SEED`, `UNIT_VOLUME`, `GEOMETRY_REUSE`, `SIZE_BIN_POLICY`,
`SIZE_MIN_FACTOR`, and `SIZE_MAX_FACTOR`; product adds `PARTNER_SEED`, `PARTNER_RESHUFFLE`, and
`PARTNER_GROUPS`.

Seed-0 snapshots are kept under `out/`. Nothing deletes raw output automatically; reduce each
nonzero-seed run to its score JSON before removing its raw files.

## Analysis

`score_model.py` reads the final particle frame only and scores the physical-mass probability,
weighting each representative by $N_i m_i/M_\mathrm{tot}$ with $m_i=s_i^3$. It histograms
$\log_{10}(m/m_0)$ on 200 coarse and 4096 fine bins between $-0.5$ and $9.5$, with explicit
underflow and overflow bins. The JSON file (the `--json` argument is required) records:

- the total-variation distance on the coarse bins (the primary shape metric), the Jensen–Shannon
  distance, the CDF supremum distance, and the log-mass Wasserstein distance $W_1$ in dex;
- mass, number, and $M_2/M_1$ ratios with their relative errors;
- the minimum and maximum mass and the maximum deviation from integer mass;
- simulated and analytical underflow and overflow mass;
- the coarse and fine histograms, the seeds, `N_P`, `N_K`, `COL_BATH_EPS`, and the reference
  definitions.

The product reference is the Borel mass distribution, tabulated to $k=10^5$. The outputs contain no
formal pass threshold. Distribution distances measure the actual numerical approximation, including
production sticking grouping; grouped events need not preserve integer cluster masses exactly.

## Evidence boundary

The analytical scorers can be exercised on synthetic samples drawn from their reference
distributions; neither such checks nor build dry runs are evidence of native CUDA compilation or GPU
accuracy. Fresh CUDA runs are required. Assess neighbor-count and refresh-tolerance dependence
separately from seed scatter; a small controller tolerance is not a histogram-error bound. Product
results hold only before gelation and with partner mixing; they do not establish that a small fixed
neighbor set suffices for the physical collision problem. The campaigns do not test Morton search,
ROCm, transport, or restarts.
