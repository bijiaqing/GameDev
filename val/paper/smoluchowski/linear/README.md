# Linear-kernel campaign

This campaign tests the production frozen-bath collision chain against the analytical solution of
the Smoluchowski equation for the normalized additive kernel $K(m_i,m_j)=m_i+m_j$, without a factor
$1/2$, starting from an exponential physical number distribution instead of the monomers of the
other two kernels. It is the `linear/` family of the
[Smoluchowski kernel benchmarks](../README.md), which own the shared sweep, parameters, seeds,
production-code reuse, build layout, output format, scorer structure, and limits; this page gives
only what differs. The additive kernel is described with the other synthetic kernels in
[collision kernels](../../../../doc/guide_swarm.md#841-collision-kernels).

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
| Representation | swarm, collisions only, CUDA with the KD tree ([shared setup](../README.md#at-a-glance)) |
| Kernel | additive, `COAG_KERNEL = 1` |
| Initial state | $n(m,0)=e^{-m}$, sampled as equal-mass representatives |
| Models | the 25 shared `N_K` by `COL_BATH_EPS` combinations |
| Planned runs | 250 (`SEED` = 0 to 9 per model) |
| Output times | $t=0,1,2,3,4$ |
| Controller groups and partners | $8\times4$ groups, fixed partners without mixing |
| Analysis entry point | `linear/src/score_model.py` |
| Output location | as for the other kernels ([paths](../README.md#build-and-run)) |

## Models

The 25 models `k{16,32,64,128,256}_eps0p{01,02,04,08,16}` are the
[shared sweep](../README.md#models) of the constant and product campaigns. Their `flags.mk` files
are identical to those of `product/`: `COLLISION`, `MULTISIZE`, `CODE_UNIT`, and `-DSEED=$(SEED)`,
without `LOGTIMING`.

## Setup

### Initial mass distribution

The physical initial number distribution is $n(m,0)=e^{-m}$. Equal represented masses therefore
sample $m$ from a gamma distribution with shape 2 and scale 1, whose density $`m\,e^{-m}`$ is the
initial mass probability; sizes are $s=m^{1/3}$, with `RHO_0` $=6/\pi$ so that grain mass is $s^3$.
The mass initializer uses its own `std::mt19937` seeded with `SEED`; the position and collision
streams use `SEED + 1`, as in the other kernels. The normalized initial moments are $M_0=1$,
$M_1=1$, and $M_2=2$.

### Output schedule and controller

Outputs 0–4 are at $t=0,1,2,3,4$ (`DT_OUT = 1`, `SAVE_MAX = 4`, linear output). Geometry and
neighbor lists are built once and reused. There is no partner mixing. The controller uses the
constant campaign's $8\times4$ spatial grouping; all other parameters are the
[shared ones](../README.md#shared-parameters).

## Production code and overrides

Production collision evolution, cached rates, local refreshes, and moving size bins are used
directly, as described in the [parent README](../README.md#production-code-and-overrides). The
local `_collision.cuh` and `rngstate_init.cu` match those of `const/`. The differences are:

- `src/const_defs.cuh` sets `COAG_KERNEL = 1`, `DT_OUT = 1`, and `SAVE_MAX = 4`;
- `src/swarm_host.cuh` additionally replaces `rand_powerlaw` with the gamma sampler and appends the
  initial-mass keys to `[CAMPAIGN]`.

No runtime or collision controller is copied.

## Build and run

Run from the repository root. Build one model and seed:

```bash
make -C val/paper/smoluchowski/linear -j8 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
```

Run it:

```bash
val/paper/smoluchowski/linear/obj/seed0/k64_eps0p02/gamedev
```

Score it:

```bash
python3 -B val/paper/smoluchowski/linear/src/score_model.py \
  val/paper/smoluchowski/linear/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/linear/out/k64_eps0p02/seed0.json
```

For other seeds, the executable is `obj/seed<N>/<model>/gamedev`, the raw output goes to
`out/<model>/seed<N>/`, and the score JSON belongs at `out/<model>/seed<N>.json`. The campaign
`Makefile` forces CUDA and the KD-tree search, as in the
[other kernel campaigns](../README.md#build-and-run).

## Outputs

Each run writes `variables.txt` and five frames of `particle_<frame>.dat` (64 MB) and
`rngstate_<frame>.dat` (48 MB with CUDA's `curandState`), about 0.56 GB in total. Besides the
[shared keys](../README.md#outputs), the `[CAMPAIGN]` section of `variables.txt` records
`INITIAL_MASS_SEED`, `INITIAL_MASS_DISTRIBUTION = gamma_shape2_scale1`, and
`PARTNER_RESHUFFLE = none`. The scorer does not delete raw files. Keep seed-0 snapshots, and remove
other seeds' raw output only after successful scoring and JSON retention
([parent Outputs](../README.md#outputs)).

## Analysis

For $g=e^{-t}$, the continuous mass-probability density is

```math
p(m,t)=\frac{g\,e^{-(2-g)m}\,I_1\left(2m\sqrt{1-g}\right)}{\sqrt{1-g}},
```

with $`p(m,0)=m\,e^{-m}`$. The normalized physical moments are $M_0=e^{-t}$, $M_1=1$, and
$M_2=2e^{2t}$.

The scorer evaluates this density with a polynomial approximation of the exponentially scaled
Bessel function $I_1$. It integrates the density into a CDF on a wider work grid, $10^{-12}$ to
$10^{12}$ in mass on 65537 logarithmic nodes, stops with an error unless the integral is one to a
relative tolerance of $10^{-5}$, rescales it to exactly one, and interpolates it onto the scoring
bins. It saves the
final-time ($t=4$) total-variation, Jensen–Shannon, and CDF supremum distances, $W_1$ in dex, and
the mass, number, and $M_2/M_1$ errors, in the same JSON layout as the
[other scorers](../README.md#analysis). Histogram and CDF support is $10^{-8}$ to $10^{8}$ in mass,
on 320 coarse and 8192 fine logarithmic bins with explicit underflow and overflow bins; $W_1$ is
integrated over this support. Histograms describe physical mass, weighting each representative by
`par_numr*par_size**3`. The JSON records the initial-mass seed with the other seeds. Integer-mass
errors do not apply to this continuous initial distribution, so the scorer omits them.

## Limits

Fixed partners can cause finite-`N_K` bias even when `COL_BATH_EPS` is small; ten-seed scatter and
the `N_K` sweep must be assessed separately from refresh convergence. The scorer can be exercised
on a synthetic sample drawn from the analytical final distribution, and the host gamma initializer
is deterministic for a given seed. These checks do not establish GPU accuracy; native CUDA
compilation and simulation are required. The shared limits are listed in the
[parent README](../README.md#limits).

The rules for qualifying a result are in
[What qualifies a result](../../../README.md#what-qualifies-a-result).
