# Linear-kernel campaign

CUDA / KD-tree, 1,000,000 representative particles, seeds 0--9.
The 25 models use N_K = 16, 32, 64, 128, 256 and eps = 0.01, 0.02,
0.04, 0.08, 0.16, matching the constant/product campaigns.

The normalized additive kernel is K(m_i,m_j)=m_i+m_j, without a factor 1/2.
The physical initial number distribution is exp(-m). Equal represented masses
therefore sample m from Gamma(shape=2, scale=1); sizes are cbrt(m), with RHO_0=6/pi
so grain mass is size^3. The mass initializer uses seed SEED; position and collision
streams use SEED+1. Seeds are recorded in variables.txt.

Outputs 0--4 are at t=0,1,2,3,4. Geometry and neighbor lists are built once and
reused. There is no partner mixing. Production collision evolution, cached rates,
local refreshes and moving size bins are used directly, with the constant
campaign's spatial grouping. The only extra initialization override is the Gamma
sampler; no copied runtime or collision controller is needed.

## Run one model

From the repository root:

```bash
make -C val/paper/smoluchowski/linear -j18 MODEL=k64_eps0p02 SEED=0 GPU_TARGET=sm_80
val/paper/smoluchowski/linear/obj/seed0/k64_eps0p02/gamedev
python3 -B val/paper/smoluchowski/linear/src/score_model.py \
  val/paper/smoluchowski/linear/out/k64_eps0p02/seed0 \
  --json val/paper/smoluchowski/linear/out/k64_eps0p02/seed0.json
```

For nonzero seeds, raw output is staged at obj/seed<N>/<model>/out/;
score JSON belongs at out/<model>/seed<N>.json. The scorer does not delete raw
files. Keep seed-0 snapshots; remove other seeds' raw output only after successful
scoring and JSON retention, as in the other campaigns.

## Analytical checks

For g=exp(-t), the continuous mass-probability density is

    p(m,t) = g exp(-(2-g)m) I1(2m sqrt(1-g)) / sqrt(1-g).

At t=0, p(m,0)=m exp(-m). The normalized physical moments are
M0=exp(-t), M1=1, M2=2 exp(2t). The scorer uses an exponentially scaled
Bessel approximation, checks normalization before integrating the CDF, and saves
final-time W1 (dex), histogram distances, mass, number and second-moment errors.
Histogram/CDF support is 10^-8 to 10^8, with explicit underflow/overflow bins;
W1 is integrated over this stated support. Histograms describe physical mass,
weighted by par_numr*par_size^3. Integer-mass errors do not apply here.

Fixed partners can cause finite-N_K bias even when eps is small; ten-seed scatter
and the N_K sweep must be assessed separately from refresh convergence.

## Verification

The scorer can be exercised on a synthetic sample drawn from the analytical final
distribution, and the host Gamma initializer is deterministic for a given seed.
These checks do not establish GPU accuracy; native CUDA compilation and simulation
are required. The campaign's build file is named `makefile`, which GNU Make finds
with `make -C val/paper/smoluchowski/linear`.
