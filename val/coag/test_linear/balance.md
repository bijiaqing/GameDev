# Linear-kernel accuracy--cost balance

## Recommendation

For the `N_P = 1e6` linear-kernel campaign, the measured accuracy--cost balance is

```text
N_K          = 200
COL_BATH_EPS = 0.04
```

`N_K = 200` is not an observed convergence knee. Accuracy continues to improve
strongly through the largest tested neighbor count, so it is the best tested
boundary; the campaign cannot exclude further improvement above 200.

At fixed `N_K = 200`, the ten-seed shape errors at `COL_BATH_EPS = 0.005`
through 0.04 are statistically indistinguishable. The 0.04 case is therefore
the linear-only cost knee. The more conservative cross-kernel production
default remains `N_K = 200`, `COL_BATH_EPS = 0.02`, supported by the constant-
and product-kernel campaigns rather than by a resolved linear-only improvement
from 0.04 to 0.02.

## Data and scoring

This assessment uses `multiseed/summary.json`, containing 10 independent seeds
for each of 25 models: 250 completed runs in total. All models used
`N_P = 1e6`. Values below are across-seed means with sample standard deviations.
Process times exclude compilation; the mean complete-grid cost is 25.87
GPU-hours per seed.

Only the final `particle_00004.dat` snapshot at dimensionless coagulation time
`tau = 4` was scored. Grain mass was calculated as

```text
m = par_size^3,
```

and the normalized physical-mass probability represented by particle `i` was

```text
w_i = par_numr_i*m_i / sum(par_numr_j*m_j).
```

For the additive kernel and the initial exponential number distribution, write

```text
mu = m/m0,
g  = exp(-tau).
```

The continuous analytical number distribution used by the campaign notebook
is

```text
f(m,t) = (N0/m) * (g/sqrt(1-g))
         * I1(2*mu*sqrt(1-g)) * exp(-mu*(2-g)).
```

The analytical physical-mass probability was integrated in
`x = log10(m/m0)`. Its density per dex is

```text
p_x(x) = ln(10)*m^2*f(m,t)/M0.
```

The numerical and analytical cumulative distributions were evaluated on 4097
points over `-0.5 <= x <= 9.5`. The reported scale-aware shape score is

```text
W1_logm = integral |F_sim(x) - F_ana(x)| dx,
```

with units of dex. The Bessel expression was evaluated with the exponentially
scaled function `i1e`. Total-variation distance used common 0.05-dex bins with
explicit underflow and overflow mass.

## Neighbor-count comparison

At `COL_BATH_EPS = 0.02`:

| `N_K` | Mean wall time | Mean W1 in log mass | Mean TV distance |
| ---: | ---: | ---: | ---: |
| 10  | 0.792 +/- 0.274 h | 0.11123 +/- 0.00166 dex | 0.08184 +/- 0.00096 |
| 20  | 0.604 +/- 0.104 h | 0.05860 +/- 0.00128 dex | 0.04539 +/- 0.00085 |
| 50  | 0.523 +/- 0.080 h | 0.02500 +/- 0.00082 dex | 0.02016 +/- 0.00047 |
| 100 | 0.548 +/- 0.060 h | 0.01263 +/- 0.00152 dex | 0.01096 +/- 0.00087 |
| **200** | **0.755 +/- 0.063 h** | **0.00682 +/- 0.00148 dex** | **0.00636 +/- 0.00052** |

Increasing `N_K` from 100 to 200 costs 38% more wall time while improving mean
W1 by 46% and mean TV distance by 42%. This is a favorable exchange for an
accuracy validation. The non-monotonic wall times below `N_K = 100` arise because the
frozen-bath count and continuation work also change with the sampled local
partner reservoir; cost is not simply proportional to `N_K`.

The strong accuracy dependence follows from the equal-mass representative
swarm relation

```text
M_rep = N_j*m_j.
```

For the additive kernel, an individual sampled-partner contribution is

```text
lambda_ij = lambda_0*N_j*(m_i + m_j)
          = lambda_0*M_rep*(m_i/m_j + 1).
```

It therefore depends strongly on partner mass. Larger `N_K` improves both the
owner collision-rate estimate and the sampled distribution of coagulation jump
sizes. Unlike the specialized product-kernel test, the linear campaign does
not globally relabel the cached partners between baths. Each owner remains
coupled to its fixed local reservoir, so temporal bath repetition does not
average away the finite-`N_K` network error.

The final W1 falls by nearly a factor of two from `N_K = 100` to 200. The tested
range therefore does not demonstrate `N_K` convergence.

## Bath-tolerance comparison

At `N_K = 200`:

| `COL_BATH_EPS` | Mean wall time | Mean W1 in log mass | Mean TV distance |
| ---: | ---: | ---: | ---: |
| 0.005 | 3.045 +/- 0.328 h | 0.00641 +/- 0.00112 dex | 0.00633 +/- 0.00061 |
| 0.01  | 1.462 +/- 0.093 h | 0.00649 +/- 0.00159 dex | 0.00644 +/- 0.00064 |
| 0.02  | 0.755 +/- 0.063 h | 0.00682 +/- 0.00148 dex | 0.00636 +/- 0.00052 |
| **0.04** | **0.443 +/- 0.157 h** | **0.00647 +/- 0.00129 dex** | **0.00638 +/- 0.00069** |
| 0.08  | 0.199 +/- 0.025 h | 0.00765 +/- 0.00139 dex | 0.00707 +/- 0.00052 |

Tightening `COL_BATH_EPS` from 0.08 to 0.04 costs 122% more while improving
mean W1 by 15% and mean TV by 9.8%. Tightening again from 0.04 to 0.02 costs
71% more, changes mean TV by only 0.34%, and slightly worsens mean W1. Even the
0.005 case improves both mean shape scores by less than 1% relative to 0.04
while costing nearly seven times as much. These differences are smaller than
the seed scatter. Paired 95% t-intervals for both TV and W1 differences between
0.04 and either 0.02 or 0.005 include zero. This makes
`COL_BATH_EPS = 0.04` the measured linear-only knee.

Across all 250 runs, the maximum relative represented-mass error is
`6.67e-15`, and no run records a persistent controller overshoot.

The resulting conclusions are:

- use `N_K = 200` as the best tested boundary, not as a demonstrated
  neighbor-count convergence knee;
- use `COL_BATH_EPS = 0.04` for the measured linear-only accuracy--cost balance;
- retain `COL_BATH_EPS = 0.02` as the conservative cross-kernel production
  default;
- do not infer convergence with representative-particle count because only
  `N_P = 1e6` was tested.
