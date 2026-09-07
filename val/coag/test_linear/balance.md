# Linear-kernel accuracy--cost balance

## Provisional recommendation

For the `N_P = 1e6` linear-kernel campaign, the best measured setting within
the tested domain is

```text
N_K          = 200
COL_BATH_EPS = 0.02
```

Unlike the constant-kernel result, `N_K = 200` is not an observed convergence
knee. Accuracy continues to improve strongly through the largest tested
neighbor count, so this is the best tested boundary. The campaign cannot
exclude further improvement above `N_K = 200`.

This recommendation is provisional because the quantitative tables below use
only seed 0. They should be replaced by the 10-seed statistics after the
completed seed records are available locally.

## Data and scoring

This assessment used the 25 completed models downloaded to
`/Users/jiaqingbi/Scratch/gamedev/test_linear`. All models used `N_P = 1e6`.
All 25 runs reported `passed`. Their process times exclude compilation and sum
to 29.46 GPU-hours for the complete seed-0 grid.

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

| `N_K` | Wall time | W1 in log mass | TV distance |
| ---: | ---: | ---: | ---: |
| 10  | 0.640 h | 0.11051 dex | 0.08172 |
| 20  | 0.591 h | 0.05820 dex | 0.04609 |
| 50  | 0.483 h | 0.02512 dex | 0.02063 |
| 100 | 0.552 h | 0.01156 dex | 0.01079 |
| **200** | **0.684 h** | **0.00632 dex** | **0.00603** |

Increasing `N_K` from 100 to 200 costs 24% more wall time while improving W1
by 45% and TV distance by 44%. This is a favorable exchange for an accuracy
validation. The non-monotonic wall times below `N_K = 100` arise because the
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

The initial particle realization has `W1 = 0.00121 dex` relative to its exact
mass distribution. The best final score, `0.00632 dex` at `N_K = 200`, remains
well above that initial sampling discrepancy. The tested range therefore does
not demonstrate `N_K` convergence.

## Bath-tolerance comparison

At `N_K = 200`:

| `COL_BATH_EPS` | Wall time | W1 in log mass | TV distance |
| ---: | ---: | ---: | ---: |
| 0.005 | 3.155 h | 0.00587 dex | 0.00591 |
| 0.01  | 1.332 h | 0.00643 dex | 0.00624 |
| **0.02** | **0.684 h** | **0.00632 dex** | **0.00603** |
| 0.04  | 0.423 h | 0.00744 dex | 0.00643 |
| 0.08  | 0.177 h | 0.00748 dex | 0.00697 |

Tightening `COL_BATH_EPS` from 0.04 to 0.02 costs 62% more and improves W1 by
15%. Tightening from 0.02 to 0.01 nearly doubles the runtime without improving
either single-seed shape score. Tightening from 0.02 to 0.005 costs 361% more
for a 7% W1 improvement and a 2% TV improvement. Within this realization,
`COL_BATH_EPS = 0.02` is the practical tolerance knee.

## Ten-seed campaign

The maintained runner executes replicate labels 0--9 sequentially. Because the
linear test also has a random initial mass distribution, each replicate records
three streams:

```text
initialization seed = replicate
position seed       = 2*replicate + 1
collision seed      = 2*replicate + 1
```

The stream triples are `(0, 1, 1)` through `(9, 19, 19)`. The purpose is to
determine whether the strong `N_K = 100` to 200 improvement and the weaker
`COL_BATH_EPS = 0.02` knee persist across realizations.

The current conclusion is:

- use `N_K = 200`, `COL_BATH_EPS = 0.02` as the provisional production
  default supported jointly by the constant- and linear-kernel campaigns;
- treat `N_K = 200` as a tested upper boundary rather than a demonstrated
  convergence knee for the linear kernel;
- reassess the ranking from the 10-seed summary before treating it as
  statistically established.
