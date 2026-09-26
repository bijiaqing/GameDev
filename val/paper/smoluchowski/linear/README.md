# Linear-kernel campaign

## Purpose

This campaign compares the production frozen-bath collision chain with the analytical solution of
the Smoluchowski equation for the normalized additive kernel $K(m_i,m_j)=m_i+m_j$, without a factor
$1/2$, starting from an exponential physical number distribution. It is the third family of the
[Smoluchowski kernel benchmarks](../README.md), which document the shared layout, controller
settings, and production code reuse.

## Models

The 25 models `k{16,32,64,128,256}_eps0p{01,02,04,08,16}` set `N_K` = 16, 32, 64, 128, 256 and
`COL_BATH_EPS` = 0.01, 0.02, 0.04, 0.08, 0.16, matching the constant and product campaigns. Each
model runs for `SEED=0` through `9`, giving 250 runs. The build is CUDA with the KD-tree search,
1,000,000 representatives, `COAG_KERNEL = 1`, and the flags `COLLISION`, `MULTISIZE`, `CODE_UNIT`,
and `-DSEED=$(SEED)`.

The physical initial number distribution is $n(m,0)=e^{-m}$. Equal represented masses therefore
sample $m$ from a gamma distribution with shape 2 and scale 1; sizes are $s=m^{1/3}$, with
`RHO_0` $=6/\pi$ so that grain mass is $s^3$. The mass initializer uses seed `SEED`; the position
and collision streams use `SEED + 1`.

Outputs 0–4 are at $t=0,1,2,3,4$ (`DT_OUT = 1`, `SAVE_MAX = 4`). Geometry and neighbor lists are
built once and reused. There is no partner mixing. Production collision evolution, cached rates,
local refreshes, and moving size bins are used directly, with the constant campaign's $8\times4$
spatial grouping. The only extra initialization override is the gamma
sampler, which replaces `rand_powerlaw` in `src/swarm_host.cuh`; no runtime or collision controller
is copied.

## Build and run

From the repository root:

```bash
make -C val/paper/smoluchowski/linear -j8 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
val/paper/smoluchowski/linear/obj/seed0/k64_eps0p02/gamedev
python3 -B val/paper/smoluchowski/linear/src/score_model.py \
  val/paper/smoluchowski/linear/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/linear/out/k64_eps0p02/seed0.json
```

For nonzero seeds, the executable is `obj/seed<N>/<model>/gamedev`, raw output is staged at
`obj/seed<N>/<model>/out/`, and the score JSON belongs at `out/<model>/seed<N>.json`. The campaign
`Makefile` forces CUDA and the KD-tree search, as in the other kernel campaigns.

## Outputs

Each run writes `variables.txt` and five frames of `particle_<frame>.dat` (64 MB) and
`rngstate_<frame>.dat` (48 MB with CUDA's `curandState`), about 0.56 GB in total. The `[CAMPAIGN]`
section of `variables.txt` records `SEED`, `POSITION_SEED`, `COLLISION_SEED`, `INITIAL_MASS_SEED`,
`INITIAL_MASS_DISTRIBUTION = gamma_shape2_scale1`, `PARTNER_RESHUFFLE = none`, and the unit-volume,
geometry-reuse, and size-bin settings. The scorer does not delete raw files. Keep seed-0 snapshots;
remove other seeds' raw output only after successful scoring and JSON retention.

## Analysis

For $g=e^{-t}$, the continuous mass-probability density is

$$
p(m,t)=\frac{g\,e^{-(2-g)m}\,I_1\left(2m\sqrt{1-g}\right)}{\sqrt{1-g}},
$$

with $p(m,0)=m\,e^{-m}$. The normalized physical moments are $M_0=e^{-t}$, $M_1=1$, and
$M_2=2e^{2t}$. The scorer uses an exponentially scaled Bessel approximation, checks the
normalization of the reference before integrating its CDF, and saves the final-time total-variation,
Jensen–Shannon, and CDF supremum distances, $W_1$ in dex, and the mass, number, and $M_2/M_1$
errors. Histogram and CDF support is $10^{-8}$ to $10^{8}$ in mass, on 320 coarse and 8192 fine
logarithmic bins with explicit underflow and overflow bins; $W_1$ is integrated over this support.
Histograms describe physical mass, weighting each representative by `par_numr*par_size**3`.
Integer-mass errors do not apply to this continuous initial distribution.

## Evidence boundary

Fixed partners can cause finite-`N_K` bias even when `COL_BATH_EPS` is small; ten-seed scatter and
the `N_K` sweep must be assessed separately from refresh convergence. The scorer can be exercised on
a synthetic sample drawn from the analytical final distribution, and the host gamma initializer is
deterministic for a given seed. These checks do not establish GPU accuracy; native CUDA compilation
and simulation are required.
