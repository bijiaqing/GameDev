# Constant-kernel accuracy--cost balance

## Recommendation

For the `N_P = 1e6` constant-kernel campaign, the best measured balance is

```text
N_K          = 200
COL_BATH_EPS = 0.02
```

This corresponds to model `1e+6n_2e+2k_2e-2b`.

For a cheaper production-oriented setting, `N_K = 100` and
`COL_BATH_EPS = 0.02` is a reasonable alternative. The constant-kernel result
alone is not sufficient to establish defaults for the linear or physical
kernels.

## Data and scoring

This assessment used the 25 completed models downloaded to
`/Users/jiaqingbi/Scratch/gamedev/test_const`. All models used `N_P = 1e6`.
Process wall times came from `wall_time_summary.json` and exclude compilation.

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
and explicit underflow and overflow bins. The primary shape measures were
total-variation distance and the one-dimensional Wasserstein distance in
`log10(m/m0)`. Total particle-number error and mass conservation were evaluated
separately.

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

## Relevant comparison

| `N_K` | `COL_BATH_EPS` | Wall time | TV distance | W1 in log mass | Number error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50  | 0.02  | 43.6 s  | 0.00365 | 0.00224 dex | 3.55% |
| 100 | 0.02  | 49.2 s  | 0.00285 | 0.00142 dex | 2.58% |
| **200** | **0.02** | **72.3 s** | **0.00216** | **0.00098 dex** | **1.45%** |
| 200 | 0.01  | 139.2 s | 0.00267 | 0.00092 dex | 1.20% |
| 200 | 0.005 | 272.0 s | 0.00249 | 0.00066 dex | 1.28% |
| 200 | 0.04  | 38.9 s  | 0.00334 | 0.00188 dex | 1.77% |

At `COL_BATH_EPS = 0.02`, increasing `N_K` from 100 to 200 costs 47% more
wall time, improves the log-mass Wasserstein distance by 31%, and reduces the
particle-number error by 44%. This remains a useful exchange for an
accuracy-validation model.

Tightening `COL_BATH_EPS` from 0.02 to 0.01 at `N_K = 200` increases wall time
by 92%, while improving the Wasserstein distance by only 6%. The TV distance
does not improve, and the particle-number error changes by only 0.25 percentage
points. Tightening to 0.005 doubles the cost again without a reproducible
overall improvement. In the other direction, relaxing from 0.02 to 0.04 nearly
halves the runtime but approximately doubles the Wasserstein error.

The systematic particle-number error decreases with neighbor count:

| `N_K` | Final number-error range across tolerances |
| ---: | ---: |
| 10  | 8.6--9.0% |
| 20  | 5.6--6.3% |
| 50  | 3.2--3.8% |
| 100 | 2.2--2.9% |
| 200 | 1.2--2.2% |

The signed error is negative: the models retain fewer physical grains than the
analytical solution and therefore coagulate slightly too quickly.

## Sampling floor and limits

The lowest measured TV score occurred at `N_K = 100` and
`COL_BATH_EPS = 0.005`, but this does not establish that model as the most
accurate. A 10,000-draw multinomial calculation for `N_P = 1e6` samples from
the exact final distribution gave an approximate TV sampling-floor median of
0.00230 and a 5th--95th percentile range of 0.00185--0.00282. The best measured
TV scores therefore cannot be reliably ranked using one random realization.
This sampling-floor estimate assumes independent samples; correlations in the
KNN representative-particle system can change it.

All 25 models conserved mass to approximately `7e-15` relative error. Across
the 225 controller records there were no persistent controller overshoots.

The resulting campaign conclusion is:

- use `N_K = 200`, `COL_BATH_EPS = 0.02` for constant-kernel accuracy tests;
- use `N_K = 100`, `COL_BATH_EPS = 0.02` when lower cost is more important;
- do not use `COL_BATH_EPS < 0.02` by default based on this single-seed result;
- reassess the balance independently for the linear and physical kernels.
