# Product-kernel accuracy--cost balance

## Recommendation

For the `N_P = 1e6` product-kernel campaign, the best measured balance is

```text
N_K          = 100
COL_BATH_EPS = 0.02
```

This corresponds to model `1e+6n_1e+2k_2e-2b`.

A cheaper approximate setting is `N_K = 100` and `COL_BATH_EPS = 0.04`.
The campaign provides no demonstrated benefit from using
`COL_BATH_EPS < 0.02` or increasing `N_K` beyond 100.

## Data and scoring

This assessment used the 25 completed models downloaded to
`/Users/jiaqingbi/Scratch/gamedev/test_product`. All models used `N_P = 1e6`.
Process wall times came from `wall_time_summary.json` and exclude compilation.

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
as the constant-kernel assessment. Total-variation distance and the
one-dimensional Wasserstein distance in log mass measure the distribution
shape. Total particle-number error, mass conservation, and the second-moment
error were evaluated separately.

## Relevant comparison

| `N_K` | `COL_BATH_EPS` | Wall time | TV distance | W1 in log mass | Number error | `M_2` error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50  | 0.02  | 72.5 s  | 0.00161 | 0.00094 dex | 0.0081% | 1.21% |
| **100** | **0.02** | **82.8 s** | **0.00150** | **0.00020 dex** | **0.0083%** | **0.21%** |
| 200 | 0.02  | 113.1 s | 0.00103 | 0.00031 dex | 0.0193% | 0.30% |
| 100 | 0.01  | 164.8 s | 0.00183 | 0.00055 dex | 0.0191% | 0.41% |
| 100 | 0.005 | 331.0 s | 0.00152 | 0.00032 dex | 0.0096% | 0.22% |
| 100 | 0.04  | 41.2 s  | 0.00202 | 0.00175 dex | 0.0279% | 2.39% |

At `COL_BATH_EPS = 0.02`, increasing `N_K` from 50 to 100 costs 14% more
wall time, improves the log-mass Wasserstein distance by 79%, and reduces the
second-moment error by 83%. This is a favorable exchange for an
accuracy-validation model.

Increasing `N_K` from 100 to 200 costs another 37%, but does not produce a
consistent accuracy improvement. The TV distance improves, while the
Wasserstein, particle-number, and second-moment errors become slightly worse.
These differences are small enough to be explained by the finite-sample noise.

Tightening `COL_BATH_EPS` from 0.02 to 0.01 at `N_K = 100` doubles the runtime
without improving the measured distribution. Tightening to 0.005 doubles it
again. Relaxing the tolerance from 0.02 to 0.04 halves the runtime, but raises
the tail-sensitive second-moment error from 0.21% to 2.39%.

## Sampling floor and limits

A multinomial calculation for `N_P = 1e6` independent samples from the exact
`t = 0.9` distribution gave an approximate TV sampling-floor median of
0.00173 and a 5th--95th percentile range of 0.00132--0.00225. The expected
one-sigma relative sampling uncertainty of the second moment is approximately
0.30%.

The `N_K = 100` and `N_K = 200`, `COL_BATH_EPS = 0.02` results are therefore
already at the finite-sample floor. Their small differences cannot be ranked
reliably from one random realization. The sampling-floor calculation assumes
independent samples; correlations introduced by the representative-particle
algorithm can modify it.

Unlike the constant-kernel campaign, the product campaign randomly permutes
the globally sampled bath partners between frozen baths. The remaining
finite-`N_K` bias is consequently small, and the campaign does not support the
additional cost of `N_K = 200`.

All 25 models conserved mass to approximately `7e-15` relative error. No
activity, distribution, or persistent controller overshoots were recorded.

The resulting campaign conclusion is:

- use `N_K = 100`, `COL_BATH_EPS = 0.02` for product-kernel accuracy tests;
- use `N_K = 100`, `COL_BATH_EPS = 0.04` when lower cost is more important;
- do not use `COL_BATH_EPS < 0.02` by default based on this single-seed result;
- do not increase `N_K` beyond 100 based on this campaign;
- keep this conclusion specific to the pre-gelation product-kernel test.
