# Product-kernel accuracy--cost balance

## Recommendation

For the `N_P = 1e6` shuffled product-kernel campaign, the meaningful measured
accuracy--cost knee is

```text
COL_BATH_EPS = 0.02
```

The campaign does not identify a meaningful `N_K` knee. `N_K = 10`, the
smallest tested value, is already sufficient for this specialized algorithm;
`N_K = 20` costs little more and is a conservative alternative, but its lower
mean W1 is not statistically resolved across seeds. This weak `N_K` dependence
must not be generalized to a fixed local-neighbor or physical-kernel model.

## Data and scoring

This updated assessment uses
`val/paper/coagulation/test_product/multiseed/summary.json`, containing 10
independent seeds for each of the 25 models: 250 completed model runs in total.
All models used `N_P = 1e6`. Reported scores and wall times are across-seed
means with sample standard deviations. Process wall times exclude compilation.

Only the final `particle_00009.dat` snapshot at `t = 0.9` was scored. This is
the final requested pre-gelation snapshot. Grain mass was calculated as
`m = par_size^3`, and the normalized mass probability represented by each
particle was

```text
par_numr * m / TOTAL_DUST_MASS.
```

For the monodisperse product kernel, the exact pre-gelation mass probability
at integer mass `k = m/m0` is the Borel distribution

```text
p_k = exp(-k*t) * (k*t)^(k - 1) / k!.
```

At `t = 0.9`, this distribution has mean representative-particle mass
`1/(1-t) = 10`. The numerical and analytical probabilities were integrated
over identical fixed bins in `log10(m/m0)`, using the same 0.05-dex resolution
as the constant-kernel assessment. The campaign's declared primary score is
mean final-snapshot total-variation distance. The one-dimensional Wasserstein
distance in log mass provides a scale-aware shape measure. Total
particle-number error, mass conservation, and the second-moment error were
evaluated separately.

## Why `N_K` has little effect

Every representative swarm retains the same represented mass,

```text
M_rep = N_j*m_j.
```

For the product kernel, the rate contribution from partner `j` is therefore

```text
lambda_ij = lambda_0*N_j*m_i*m_j
          = lambda_0*M_rep*m_i,
```

which is independent of partner mass. Summing `N_K` valid slots and applying
the `N_P/N_K` subsampling normalization consequently gives the same total
owner collision rate for every `N_K`. Larger `N_K` can improve only the
sampled distribution of partner masses and hence the coagulation jump sizes;
it does not improve the product-kernel rate estimate.

The product test also applies a new global random permutation to the cached
neighbor labels before every frozen bath. The spatial cache remains fixed, but
its `N_K` entries become well-mixed global partner-sampling slots. A particle
therefore samples new partner masses across baths instead of remaining coupled
to one local reservoir. At `COL_BATH_EPS = 0.02`, all tested `N_K` values use
approximately 2010--2020 baths. Even `N_K = 10` is consequently exposed to
about 20,000 partner-slot samples per owner over the calculation. These are not
20,000 independent collision events, but the repeated relabelling strongly
reduces the persistent finite-`N_K` error.

This same relabelling prevents the product-kernel avalanche caused by a rapidly
growing owner repeatedly drawing from one unusually favorable frozen partner
reservoir. It also deliberately weakens the diagnostic power of this campaign
as a general `N_K`-convergence test.

## Multiseed `N_K` comparison

At the selected `COL_BATH_EPS = 0.02`:

| `N_K` | Mean wall time | Mean TV distance | Mean W1 in log mass | Mean `M_2` error |
| ---: | ---: | ---: | ---: | ---: |
| 10  | 62.5 +/- 0.8 s  | 0.001779 +/- 0.000282 | 0.001252 +/- 0.000706 dex | 1.320 +/- 0.718% |
| 20  | 64.0 +/- 1.0 s  | 0.001866 +/- 0.000390 | 0.001247 +/- 0.000822 dex | 1.341 +/- 0.899% |
| 50  | 70.0 +/- 1.4 s  | 0.002028 +/- 0.000435 | 0.001540 +/- 0.000949 dex | 1.673 +/- 1.200% |
| 100 | 81.7 +/- 1.3 s  | 0.002000 +/- 0.000307 | 0.001377 +/- 0.001007 dex | 1.354 +/- 1.045% |
| 200 | 111.9 +/- 1.0 s | 0.002000 +/- 0.000540 | 0.001380 +/- 0.000984 dex | 1.367 +/- 1.036% |

The mean TV scores for `N_K = 10` and 20 differ by `8.62e-5`, or 4.8%,
far less than their seed scatter. `N_K = 20` costs 2.5% more and has a 0.4%
lower mean W1, but the paired difference is not statistically resolved.
Increasing `N_K` to 100 or 200 adds cost without a systematic improvement in
TV, W1, or the second moment.

## Multiseed tolerance comparison

Using `N_K = 20` as a representative low-cost row:

| `COL_BATH_EPS` | Mean wall time | Mean TV distance | Mean W1 in log mass | Mean `M_2` error |
| ---: | ---: | ---: | ---: | ---: |
| 0.005 | 258.2 +/- 3.6 s | 0.001862 +/- 0.000417 | 0.000792 +/- 0.000422 dex | 0.537 +/- 0.418% |
| 0.01  | 129.3 +/- 2.0 s | 0.001820 +/- 0.000271 | 0.000836 +/- 0.000624 dex | 0.782 +/- 0.674% |
| **0.02** | **64.0 +/- 1.0 s** | **0.001866 +/- 0.000390** | **0.001247 +/- 0.000822 dex** | **1.341 +/- 0.899%** |
| 0.04  | 31.9 +/- 0.6 s  | 0.002392 +/- 0.000493 | 0.002346 +/- 0.000984 dex | 2.839 +/- 1.117% |
| 0.08  | 15.8 +/- 0.2 s  | 0.004533 +/- 0.000430 | 0.006035 +/- 0.000891 dex | 6.520 +/- 0.835% |

Tightening `COL_BATH_EPS` from 0.04 to 0.02 doubles the runtime while improving
mean TV by 22%, mean W1 by 47%, and mean second-moment error by 53%. Tightening
from 0.02 to 0.01 doubles the runtime again, but improves the primary mean TV
score by only 2.5%; its 33% W1 and 42% second-moment improvements make 0.01 a
defensible tail-sensitive setting, but not the primary accuracy--cost knee.
Tightening from 0.01 to 0.005 doubles the runtime once more, improves mean W1
by only 5%, and slightly worsens mean TV; neither difference is resolved.

## Sampling floor and limits

A multinomial calculation for `N_P = 1e6` independent samples from the exact
`t = 0.9` distribution gave an approximate TV sampling-floor median of
0.00173 and a 5th--95th percentile range of 0.00132--0.00225. The expected
one-sigma relative sampling uncertainty of the second moment is approximately
0.30%.

The `COL_BATH_EPS = 0.01` and 0.02 TV results are already near this
finite-sample floor. The multiseed scatter confirms that small differences
between `N_K` values cannot be ranked reliably. The sampling-floor calculation
assumes independent samples; correlations introduced by the
representative-particle algorithm can modify it.

Across all 250 runs, the maximum reported relative mass error was
`5.33e-15`. Two runs recorded one recoverable activity overshoot and 55 runs
recorded one recoverable distribution overshoot. No run recorded a persistent
controller overshoot.

The resulting campaign conclusion is:

- use `COL_BATH_EPS = 0.02` as the product-kernel accuracy--cost knee;
- treat `N_K = 10` as sufficient for this shuffled product test, with
  `N_K = 20` as a conservative but statistically equivalent alternative;
- use `COL_BATH_EPS = 0.01` only when W1 or second-moment accuracy matters more
  than the campaign's primary TV score;
- do not infer a general `N_K` recommendation from this globally relabelled
  product-kernel test;
- keep this conclusion specific to the pre-gelation product-kernel test.
