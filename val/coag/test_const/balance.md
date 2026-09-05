# Constant-kernel accuracy--cost balance

## Recommendation

For the `N_P = 1e6` constant-kernel campaign, the best measured balance is

```text
N_K          = 200
COL_BATH_EPS = 0.02
```

This corresponds to model `1e+6n_2e+2k_2e-2b`.

The completed 15-seed campaign confirms this recommendation. The selected
model is on the wall-time--Wasserstein Pareto frontier and is the practical
knee before the cost of tighter bath tolerances rises much faster than the
accuracy improves. This is a judgment based on the measured trade-off, not a
unique knee independent of the chosen accuracy metric or axis scaling.

For a cheaper production-oriented setting, `N_K = 100` and
`COL_BATH_EPS = 0.02` is a reasonable alternative. The constant-kernel result
alone is not sufficient to establish defaults for the linear or physical
kernels.

## Data and scoring

The updated assessment uses
`val/coag/test_const/multiseed/summary.json`, containing 15
independent seeds for each of the 25 models: 375 completed model runs in total.
All models used `N_P = 1e6`. Reported scores and wall times are across-seed
means with sample standard deviations. Process wall times exclude compilation.

Only the final `particle_00009.dat` snapshot at dimensionless coagulation time
`tau_coag = 1e8` was scored. Grain mass was calculated as
`m = par_size^3`, and the normalized mass probability
represented by each particle was

```text
par_numr * m / TOTAL_DUST_MASS.
```

The exact monodisperse constant-kernel mass probability is

```text
p_k = k * g^2 * (1 - g)^(k - 1),
g   = 1 / (1 + tau_coag/2).
```

The numerical and analytical probabilities were integrated over identical
fixed bins in `log10(m/m0)`, with 200 ordinary bins spanning `[-0.5, 9.5]`
and explicit underflow and overflow bins. The campaign's declared primary
score was mean final-snapshot total-variation distance. The one-dimensional
Wasserstein distance in `log10(m/m0)` was retained as the scale-aware shape
measure used for the trade-off plot. Total particle-number error and mass
conservation were evaluated separately.

## Log-mass Wasserstein derivation

Let

```text
k = m/m0,
g = 1/(1 + tau_coag/2).
```

For monodisperse initial conditions and the constant kernel, the number of
physical grains in integer mass class `k` is

```text
n_k = N0 * g^2 * (1 - g)^(k - 1).
```

The Wasserstein comparison uses physical-mass probability, rather than number
probability. Because `M0 = N0*m0`, the normalized mass probability in class
`k` is

```text
p_k = (k*m0*n_k)/M0
    = k * g^2 * (1 - g)^(k - 1).
```

Its exact cumulative distribution through integer class `K` is

```text
F_ana(K) = sum(p_k, k=1..K)
         = 1 - (1 - g)^K * (1 + K*g).
```

For an edge `x` in log mass, where `x = log10(m/m0)`, the analytical CDF
strictly below that edge uses

```text
K(x) = ceil(10^x) - 1.
```

The implementation evaluates the same CDF stably as

```text
log_tail = K*log1p(-g) + log1p(K*g)
F_ana    = -expm1(log_tail).
```

For simulated representative particle `i`, the empirical mass-probability
weight is

```text
w_i = par_numr_i * m_i / sum(par_numr_j * m_j).
```

Placing these weights at `x_i = log10(m_i/m0)` and cumulatively summing them
gives `F_sim(x)`. The one-dimensional log-mass Wasserstein distance is then

```text
W1_logm = integral |F_sim(x) - F_ana(x)| dx,
```

with units of dex because the integration coordinate is `log10(m/m0)`. The
score evaluates this integral by the trapezoidal rule on 4097 fixed points
over `-0.5 <= x <= 9.5`.

The plotting function `const_kernel` returns the differential quantity
`m^2*f(m)/(M0*m0)`. It is therefore not called by the Wasserstein calculation:
one-dimensional Wasserstein distance is defined through cumulative
probabilities. Using the exact analytical CDF also prevents the score from
depending on the plotting histogram width or on numerical quadrature of the
displayed analytical curve.

## Multiseed comparison

| `N_K` | `COL_BATH_EPS` | Mean wall time | Mean TV distance | Mean W1 in log mass | Mean number error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50  | 0.02  | 43.6 +/- 0.1 s  | 0.003431 +/- 0.000536 | 0.002149 +/- 0.000229 dex | 3.305 +/- 0.119% |
| 100 | 0.02  | 49.1 +/- 0.1 s  | 0.002932 +/- 0.000275 | 0.001522 +/- 0.000129 dex | 2.271 +/- 0.166% |
| **200** | **0.02** | **72.4 +/- 0.1 s** | **0.002570 +/- 0.000209** | **0.001119 +/- 0.000134 dex** | **1.478 +/- 0.108%** |
| 200 | 0.01  | 139.1 +/- 0.3 s | 0.002426 +/- 0.000286 | 0.000913 +/- 0.000166 dex | 1.387 +/- 0.168% |
| 200 | 0.005 | 272.0 +/- 0.4 s | 0.002398 +/- 0.000356 | 0.000715 +/- 0.000197 dex | 1.255 +/- 0.180% |
| 200 | 0.04  | 38.6 +/- 0.1 s  | 0.003283 +/- 0.000339 | 0.001738 +/- 0.000258 dex | 1.740 +/- 0.173% |
| 200 | 0.08  | 21.4 +/- 0.1 s  | 0.004857 +/- 0.000425 | 0.002977 +/- 0.000214 dex | 2.268 +/- 0.123% |

At `COL_BATH_EPS = 0.02`, increasing `N_K` from 100 to 200 costs 47% more
wall time, improves the mean log-mass Wasserstein distance by 27%, improves
mean TV distance by 12%, and reduces the mean particle-number error by 35%.
This remains a useful exchange for an accuracy-validation model.

Tightening `COL_BATH_EPS` from 0.02 to 0.01 at `N_K = 200` increases wall time
by 92%, while improving mean Wasserstein distance by 18%, mean TV distance by
6%, and mean particle-number error by 6%. Tightening from 0.02 to 0.005 costs
276% more for a 36% improvement in mean Wasserstein distance but only a 7%
improvement in mean TV distance. The tighter tolerances therefore give a
reproducible W1 improvement, but not enough to justify their cost as the
default. In the other direction, relaxing from 0.02 to 0.04 saves 47% of the
runtime while worsening mean Wasserstein distance by 55%, mean TV distance by
28%, and mean particle-number error by 18%.

The systematic particle-number error decreases with neighbor count:

| `N_K` | Range of mean final number error across tolerances |
| ---: | ---: |
| 10  | 8.68--9.10% |
| 20  | 5.72--6.26% |
| 50  | 3.16--3.95% |
| 100 | 2.02--2.94% |
| 200 | 1.25--2.27% |

The signed error is negative: the models retain fewer physical grains than the
analytical solution and therefore coagulate slightly too quickly.

## Sampling floor and limits

The former single-seed assessment estimated an approximate TV sampling-floor
median of 0.00230 and a 5th--95th percentile range of 0.00185--0.00282 using a
10,000-draw multinomial calculation for `N_P = 1e6`. The 15-seed campaign now
measures the combined seed scatter directly. It confirms that the smallest TV
differences among the strictest cases are modest compared with their
realization scatter, whereas the systematic W1 and particle-number trends with
`N_K` remain clear. The multinomial floor still assumes independent samples;
correlations in the KNN representative-particle system can change it.

Across all 375 runs, the maximum reported relative mass error was
`5.44e-15`, and no model reported a persistent controller overshoot.

The resulting campaign conclusion is:

- use `N_K = 200`, `COL_BATH_EPS = 0.02` for constant-kernel accuracy tests;
- use `N_K = 100`, `COL_BATH_EPS = 0.02` when lower cost is more important;
- use `COL_BATH_EPS < 0.02` only when its smaller W1 is worth the
  disproportionate runtime increase;
- reassess the balance independently for the linear and physical kernels.
