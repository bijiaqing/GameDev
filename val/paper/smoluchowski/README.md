# Smoluchowski kernel benchmarks

The reproduced additive-kernel campaign is documented in [linear/README.md](linear/README.md).
It uses the same parameter grid and seed layout, an exponential physical initial
mass distribution, and fixed partners without reshuffling.

CUDA and KD-tree only. Each of `const/` and `product/` contains 25 models:
`k{16,32,64,128,256}_eps0p{01,02,04,08,16}`. `SEED=0` through `9`
selects the ten independent realizations; position and collision seeds are `SEED+1`.
The same seed gives matched initialization across parameter combinations.
There are 500 planned runs. No Slurm or campaign orchestration scripts are included.

## Layout and individual use

Within each kernel directory:

- `mod/<model>/`: neighbor count and collision-refresh tolerance.
- `src/`: common constants, thin normalization/initialization overrides, RNG seed selection, and analytical scorer.
- `out/<model>/seed0/`: retained seed-0 snapshots and metadata.
- `out/<model>/seed<N>.json`: intended compact per-seed distributions and scores.
- `obj/seed<N>/<model>/`: executable and build products; nonzero-seed raw outputs are staged in `out/` here.

Separate seed build/output paths permit independent jobs later. No raw outputs
are deleted automatically. Reduce nonzero-seed snapshots to JSON before removing
those temporary files. Ensemble aggregation and job scripts will be planned separately.

Example from the repository root:

```sh
make -C val/paper/smoluchowski/const -j8 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
val/paper/smoluchowski/const/obj/seed0/k64_eps0p02/gamedev
python3 -B val/paper/smoluchowski/const/src/score_model.py \
  val/paper/smoluchowski/const/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/const/out/k64_eps0p02/seed0.json
```

For seed 1, the executable is `obj/seed1/<model>/gamedev` and its raw results
are `obj/seed1/<model>/out/`. Use `product` instead of `const` for that kernel.

## Model and analytical normalization

Both use 1,000,000 equal represented-mass monomers, diameter 1, a fixed jittered
annular grid, and the normalized synthetic kernel in a unit volume. The total
represented mass is 1e30. Setting grain material density to 6/pi makes the
production grain-mass formula m=pi*rho*s^3/6 equal to s^3, preserving unit monomer
mass without copying the production physics header.

Production uses lambda0=N_P/(N_K*M_total), so initially each representative's
summed constant-kernel propensity is one. The analytic normalized initial number
and mass densities are both one. For integer monomer mass k:

- Constant kernel: g=1/(1+t/2), n_k=g^2*(1-g)^(k-1), mass probability k*n_k.
  Number moment M0=g and mass moment M1=1. Frames 1--9 are at 1,10,...,1e8;
  frame 0 is the monomer state.
- Product kernel: n_k=k^(k-2)*t^(k-1)*exp(-k*t)/k!, mass probability k*n_k.
  M0=1-t/2, M1=1, M2=1/(1-t). Frames 1--9 are at 0.1,...,0.9,
  strictly before gelation at t=1.

The scorer retains final mass-weighted histograms, CDF and distribution distances,
number/mass errors, seed identifiers and model parameters. Its outputs contain no
formal pass threshold. Distribution distances measure the actual numerical
approximation, including production sticking grouping; integer cluster masses
need not be preserved exactly by grouped events.

## What is reused and what differs

The current root runtime, particle initializer, KD-tree builder/search, collision
controller, cached rates, grouping and event updates are compiled directly. With
neither dynamics nor diffusion enabled, geometry is not invalidated: tree build,
neighbor search and neighbor-dependency graph construction occur once per fresh
run. Grain properties, rates and collision refreshes continue to evolve.

The local host header substitutes total mass, seeded jittered positions
and appended campaign metadata, and for product only supplies partner reshuffling.
The collision header substitutes unit-volume neighbor measures.
The RNG kernel uses the campaign seed. No copied old runtime,
collision controller, physical velocity model or KD-tree implementation is retained.

Controller settings use 64 collision threads, 32 events per continuation,
8x4 spatial bins for constant and one spatial group for product, 64 size bins,
minimum merged population 64 and the selected eps.
The absolute duration cap is effectively disabled (1e100), as in the old campaign.
The campaign uses moving logarithmic size-bin bounds independently in each spatial
group: [0.5*minimum_current_size, 8*maximum_current_size]. They update immediately
before the group's size-bin mapping is rebuilt and remain fixed through collision
execution and its audit. Groups not being refreshed retain their bounds. Extrema
are reduced on the GPU; no new device-to-host copies are needed. This adds three
kernel launches per rate/bin refresh. Tree and neighbor caching remain unchanged.
Moving size bins now come directly from the production collision header; no campaign
collision-controller override is needed. `COL_SIZE_BIN_POLICY = moving_per_group`
and `COL_SIZE_RANGE_FACTORS = 0.5 8` record the policy in production output metadata.

Full production diagnostics are disabled to avoid large
collision JSON/JSONL files.

### Product-only partner mixing

Before every collision refresh (including the first), a seeded global permutation
relabels all cached neighbor indices. The tree and geometric neighbor search are
still performed only once. The partner RNG uses `SEED + COL_PARTNER_SEED` (currently
`SEED+2`), separately from the collision RNG. Negative sentinel entries and periodic
image labels are preserved. This restores the old campaign's well-mixed sampling
instead of keeping particles coupled to one fixed small partner population, which
can amplify product-kernel growth artificially.

One spatial controller group makes every refresh publish all particles together;
64 adaptive size bins still control the interval. Mixing precedes publication and
full cached-rate recalculation. It never occurs during continuations or audits.
Thus the single-group dependency graph remains valid after relabeling. The idle
continuation queue supplies GPU permutation storage. Each refresh adds one CPU
shuffle, one N_P-integer host-to-device copy and one relabeling kernel, as a
campaign-specific cost. Physical production models and the constant campaign do
not enable the hook. `PARTNER_RESHUFFLE`, `PARTNER_SEED`, and `PARTNER_GROUPS` are
recorded in `variables.txt`. A fresh CUDA run is required to confirm that the
previous nonfinite-rate failure is resolved; reshuffling is not an overflow guard.

## Verification status

Local build dry runs and source routing checks cover all 50 models at seeds 0 and 9.
The analytical scorers were exercised on synthetic samples drawn from their
reference distributions. Neither is evidence of native CUDA compilation or GPU
accuracy. Fresh CUDA runs remain required. No stale outputs were copied, and
Python entry points disable bytecode caches.
