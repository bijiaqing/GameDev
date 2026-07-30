# Lagrangian swarm verification

## Purpose

The CUDA models under `qav/swarm/` compare production swarm kernels and device helpers with
analytical or statistically exact reference problems. They are verification models, not
production disk setups. Each model supplies a test-local `swarm_runtime.cu` and
`const_defs.cuh`; unless a case explicitly defines a small test kernel, the numerical operation
being measured still comes from `src/swarm/` or `inc/swarm/`.

The initial suite covers particle-to-grid projection, optical-depth construction, semi-analytic
transport, stiff gas drag, cylindrical stochastic diffusion, radiation pressure, Poynting–
Robertson drag, collision-neighborhood measures, the three dimensionless coagulation kernels, and
exact KD-tree/Morton neighbor search. It does not yet validate initialization sampling, imported
gas, complete stochastic collision events, boundaries, restart reproducibility, or long-term
coupled evolution.

This document defines what each test proves, what it does not prove, and what result is
expected. Machine-readable results are written under `qav/swarm/out/`. The shorter implementation
index in `qav/swarm/test_common/TEST_CASES.md` should remain consistent with this document.

## Measurement protocol

### Deterministic errors

For a deterministic scalar sample error $e_i$, the current validator reports unweighted norms

$$
L_1=\frac{1}{n}\sum_{i=1}^n|e_i|,
\qquad
L_2=\left(\frac{1}{n}\sum_{i=1}^n e_i^2\right)^{1/2},
\qquad
L_\infty=\max_i|e_i|.
$$

When several state components are tested together, their error arrays are concatenated before the
norms are calculated. These are particle-sample or grid-value norms, not the volume-weighted field
norms used by the Eulerian fluid suite. The grid tests separately reconstruct the finite-volume
mass

$$
M_d=\sum_i\rho_{d,i}\Delta V_i
$$

and require $M_d=1$ to roundoff because their test particles carry unit total dust mass.

For the orbit refinement, the runner can formally calculate

$$
p=\log_2\left(\frac{E_N}{E_{2N}}\right),
$$

where $N$ is the number of timesteps in one orbit and therefore $\Delta t\propto N^{-1}$. However,
the circular-orbit case is an exactly preserved equilibrium of the staggered update. Its error
is consequently dominated by floating-point roundoff rather than temporal truncation, so the
formal values of $p$ are not convergence orders and no fitted-order threshold is imposed. A future
non-equilibrium trajectory test is needed to measure the transport scheme's temporal order.

### Stochastic errors

A diffusion trajectory is not expected to converge pointwise when the ensemble size changes.
Instead, the validator compares the sample mean and variance with exact Itô moments. For an exact
variance $\sigma^2$ and $N_P$ independent representatives, the acceptance limits are

$$
|\overline X-\mu|\le 6\sqrt{\frac{\sigma^2}{N_P}},
$$

$$
|s_X^2-\sigma^2|
\le 6\sigma^2\sqrt{\frac{2}{N_P-1}}.
$$

These six-standard-error bounds make a false rejection very unlikely while still tightening as
$N_P^{-1/2}$. Every diffusion build initializes cuRAND with seed 17, so a given compiler, GPU, and
resolution should be reproducible, but correctness is assessed through moments rather than the
particular random sequence.

### Meaning of `--res`

The suite intentionally reuses one command-line resolution parameter for three different kinds of
refinement:

| Test family | Meaning of $N$ | Derived size |
|---|---|---|
| 2D grid | mesh cells in both active directions | $N_X=N_Y=N$, $N_P=N^2$ |
| 3D grid | radial and polar mesh cells | $N_X=4$, $N_Y=N_Z=N$, $N_P=4N^2$ |
| circular orbit | timesteps in one orbit | $\Delta t=2\pi/N$, $N_P=64$ |
| 2D/3D diffusion | ensemble-control parameter | $N_P=16N^2$ on a fixed mesh |
| drag, radiation, P-R, collision algebra | no physical dependence on $N$ | only the first requested value is built |
| KNN benchmark | no dependence on `--res` | $10^5$ particles by default; add `--knn-full` for $10^6$ |

Consequently, an order derived from the orbit is a temporal order, while decreasing diffusion
errors demonstrate Monte Carlo sampling convergence. The exactly constructed grid cases are
geometry regressions rather than truncation-error convergence tests.

The CUDA drivers write six structure-of-arrays fields in `state_N*.dat`:

$$
(x,y,z,\ell_x,v_y,\ell_z),
$$

and multisize tests additionally write grain size and represented grain number. The Python
validator constructs all reference values independently from these raw outputs and test metadata.

## Analytical and statistical cases

| Model | Production calculation exercised | Reference result |
|---|---|---|
| `test_grid_2d` | 2D mass deposition, density conversion, opacity deposition, radial prefix sum, ring mean | exact cell density and cumulative optical depth from one equal-mass particle at every interpolation centroid |
| `test_grid_3d` | full-3D spherical projection and optical-depth geometry | the same construction using exact $r^2\,dr\,d\Omega$ measures |
| `test_orbit_2d` | complete non-radiative `ssa_transport` kernel | one pressure-free circular Kepler orbit |
| `test_drag_2d` | frozen-coefficient gas-drag response in `ssa_transport` | exact exponential angular relaxation and its induced radial response |
| `test_diffusion_2d` | azimuthal cylindrical diffusion SDE and velocity reprojection | exact Gaussian angular moments and invariant Cartesian velocity |
| `test_diffusion_3d` | cylindrical radial and vertical SDE mapped to spherical storage | exact radial/vertical moments including cylindrical Itô drift and invariant Cartesian velocity |
| `test_radiation_2d` | midpoint radiation split without P-R damping | exact radiation-modified frozen response at zero optical depth |
| `test_prdrag_2d` | combined gas and P-R exponential response | exact component-dependent damping at zero optical depth |
| `test_collision_2d` | 2D accessible-neighborhood measure and coagulation-kernel numerators | disk and circular-cap areas plus constant, additive, and product rates |
| `test_collision_3d` | 3D accessible-neighborhood measure and coagulation-kernel numerators | interior, radial-cap, and polar-cap ball volumes plus the same three rates |
| `test_knn` | exact KD-tree and adaptive-Morton construction, cooperative top-$K$, periodic query images, and production boundary ghosts | independent brute-force neighbors, adversarial topology, overflow-free traversal, and timing/memory records |

## Grid projection and optical depth

### Particle placement

The 2D and 3D grid cases put exactly one representative in every mesh cell and assign each
representative mass

$$
m_p=\frac{1}{N_P}.
$$

Particles are placed at the locations where the production trilinear deposition stencil assigns
unit weight to the intended cell. Azimuthal and polar coordinates use the ordinary half-cell
offset. On the logarithmic radial grid with ratio

$$
q=\left(\frac{Y_{\max}}{Y_{\min}}\right)^{1/N_Y}
$$

and radial measure power $d=2$ in the vertically integrated disk or $d=3$ in full 3D, the
fractional centroid offset is

$$
f_y=
\log_q\left[
\frac{d}{d+1}
\frac{q^{d+1}-1}{q^d-1}
\right].
$$

The representative in radial cell $i_y$ is placed at

$$
y_p=Y_{\min}q^{i_y+f_y}.
$$

This setup makes interpolation leakage a direct error rather than hiding it inside a sampled
density profile.

### Expected density

The exact spherical cell measure used by the production grid is

$$
\Delta V_{i_xi_yi_z}
=\Delta x
\frac{y_o^d-y_i^d}{d}
\left(\cos z_i-\cos z_o\right),
$$

where the polar factor is one when `N_Z == 1`. Therefore the expected deposited density is

$$
\rho_{d,i}=\frac{m_p}{\Delta V_i}.
$$

This checks x-fastest indexing, the 2D versus 3D radial Jacobian, the polar sine Jacobian, the
deposition weights, and the conversion from accumulated cell mass to density. The validator
requires

$$
L_\infty(\rho_d)<5\times10^{-12},
\qquad
|M_d-1|<5\times10^{-12}
$$

up to the same density tolerance used in the implementation.

### Expected optical depth

In 3D the extinction weight of each monodisperse representative is

$$
w_i=\kappa_0m_p.
$$

In the vertically integrated 2D model, the well-mixed closure converts surface mass to midplane
volume density,

$$
w_i=\frac{\kappa_0m_p}{\sqrt{2\pi}H_g(R_i)},
\qquad
H_g=h_gR.
$$

After deposition, one cell contributes

$$
\Delta\tau_{i_y}=\frac{w_i}{\Delta V_i}\Delta r_{i_y},
\qquad
\Delta r_{i_y}=y_i(q-1).
$$

The reference outer-face optical depth is the inclusive radial prefix sum

$$
\tau_{i_y}=\sum_{j=0}^{i_y}\Delta\tau_j.
$$

All azimuthal cells are identical, so the production ring-mean kernel should leave the result
unchanged. The required maximum optical-depth error is $5\times10^{-11}$. These cases verify the
opacity reconstruction itself, but not the subsequent factor $e^{-\tau}$ in a particle force.

## Circular-orbit transport

Every representative begins at

$$
y=R=1,
\qquad
z=\frac{\pi}{2},
\qquad
(\ell_x,v_y,\ell_z)=(1,0,0),
$$

with initial azimuths uniformly distributed around the complete ring. The test parameters
$p=2$ and $q=-1$ make the midplane pressure-support parameter vanish, so both gas and dust have
unit Keplerian angular speed. Drag therefore has no effect on the exact orbit.

The analytical trajectory is

$$
x(t)=x_0+t\pmod{2\pi},
\qquad
y(t)=1,
\qquad
(\ell_x,v_y,\ell_z)=(1,0,0).
$$

The CUDA kernel advances to $T=2\pi$ using $N$ equal steps. The validator wraps the azimuthal error
to $[-\pi,\pi)$ and combines errors in $x$, $y$, $\ell_x$, and $v_y$. It rejects nonfinite states
or $L_\infty\ge0.5$. Because this circular equilibrium is preserved to roundoff, refinement does
not expose the nominal temporal order: ratios of errors near machine precision fluctuate and may
produce negative formal orders. This case verifies equilibrium preservation and long-orbit phase
consistency, not general second-order convergence.

## Frozen drag and radiation responses

The three one-step response tests use eight grain sizes

$$
s_i=0.05\,2^i,
\qquad i=0,\ldots,7,
$$

at $R=y=1$, with

$$
\mathrm{St}_i=0.2s_i,
\qquad
t_{s,i}=\mathrm{St}_i,
\qquad
\Delta t=0.1.
$$

All representatives start from

$$
(\ell_{x,0},v_{y,0},\ell_{z,0})=(1.2,0,0),
\qquad
\ell_{x,g}=1,
\qquad
v_{y,g}=0.
$$

Because $v_{y,0}=0$, the staggered midpoint radius remains exactly one. All coefficients used by
the semi-analytic response are therefore known constants for the complete step.

Define

$$
\gamma_{\rm PR}=\frac{\beta GM_\star}{cR^2},
\qquad
k_x=t_s^{-1}+\gamma_{\rm PR},
\qquad
k_y=t_s^{-1}+2\gamma_{\rm PR}.
$$

Without P-R drag, $\gamma_{\rm PR}=0$. The midpoint angular momentum is

$$
\ell_{x,1}
=e^{-k_x\Delta t/2}\ell_{x,0}
+\frac{1-e^{-k_x\Delta t/2}}{k_x}
\frac{\ell_{x,g}}{t_s}.
$$

At $R=y=1$ the midpoint radial force is

$$
F_{y,1}=-(1-\beta)+\ell_{x,1}^2.
$$

The completed exact frozen response is

$$
\ell_{x,j}
=e^{-k_x\Delta t}\ell_{x,0}
+\frac{1-e^{-k_x\Delta t}}{k_x}
\frac{\ell_{x,g}}{t_s},
$$

$$
v_{y,j}=\frac{1-e^{-k_y\Delta t}}{k_y}F_{y,1},
\qquad
y_j=1+\frac{1}{2}v_{y,j}\Delta t.
$$

The individual cases select:

- `test_drag_2d`: $\beta=0$ and $\gamma_{\rm PR}=0$
- `test_radiation_2d`: $\beta=\beta_0/(s/S_0)$ and $\gamma_{\rm PR}=0$
- `test_prdrag_2d`: the same size-dependent $\beta$ with
  $\gamma_{\rm PR}=\beta/c$, using $c=25$ in test units

The radiation tests supply a zero optical-depth array and unit startup taper. They therefore
isolate force and damping algebra from opacity reconstruction. The validator concatenates errors
in $\ell_x$, $v_y$, and $y$ and requires $L_\infty<2\times10^{-13}$. This checks stiff responses
from $\Delta t/t_s=0.078125$ through 10 without requiring explicit drag timesteps.

## Cylindrical stochastic diffusion

The diffusion tests use constant

$$
D=\nu=2\times10^{-2},
\qquad
\Delta t=2\times10^{-2},
$$

so the active-direction variance is

$$
2D\Delta t=8\times10^{-4}.
$$

Every representative starts at $(x,R,Z)=(0,1,0)$ with cylindrical velocity

$$
(v_\phi,v_R,v_Z)=(0.7,0.2,0).
$$

The diffusion operation is a positional redistribution rather than an impulse. The production
kernel reconstructs the Cartesian velocity before displacement and projects that unchanged vector
back into the new local spherical basis afterward. Both cases therefore require the reconstructed
physical velocity to agree with the initial Cartesian vector to $2\times10^{-12}$.

### Two-dimensional azimuthal diffusion

`test_diffusion_2d` activates only $D_x$; radial diffusion is suppressed with an enormous Schmidt
number. At $R=1$,

$$
\Delta x=\sqrt{2D\Delta t}\,\xi,
\qquad
E[\Delta x]=0,
\qquad
\operatorname{Var}(\Delta x)=2D\Delta t,
$$

where $\xi\sim\mathcal N(0,1)$. Angular displacements are wrapped to $[-\pi,\pi)$ before their
moments are measured.

### Three-dimensional radial and vertical diffusion

`test_diffusion_3d` suppresses azimuthal diffusion and activates $D_R=D_Z=D$. For constant
diffusivity, cylindrical geometry supplies the Itô drift $D/R$:

$$
\Delta R=\frac{D}{R}\Delta t+\sqrt{2D\Delta t}\,\xi_R,
$$

$$
\Delta Z=\sqrt{2D\Delta t}\,\xi_Z.
$$

At $R=1$ the exact moments are

$$
E[\Delta R]=D\Delta t=4\times10^{-4},
\qquad
E[\Delta Z]=0,
$$

$$
\operatorname{Var}(\Delta R)
=\operatorname{Var}(\Delta Z)
=8\times10^{-4}.
$$

The selected timestep and domain keep the ensemble far from reflecting boundaries, so these are
unbounded one-step moments. The test does not yet validate reflection statistics or a spatially
varying diffusion coefficient.

The radial drift signal is only $4\times10^{-4}$. With $N_P=16N^2$, its six-standard-error mean
band is approximately $1.33\times10^{-3}$ at $N=32$, $6.63\times10^{-4}$ at $N=64$,
$3.31\times10^{-4}$ at $N=128$, and $1.66\times10^{-4}$ at $N=256$. The $N=32$ and $N=64$
ensembles therefore check gross diffusion behavior but cannot independently resolve the Itô drift;
the $N\ge128$ results provide that evidence.

## Collision helper mathematics

The collision cases execute a test-local CUDA kernel that directly calls the production
`_get_ball_measure` and `_get_col_rate_ij` device helpers. They do not build or query the KD tree.

For neighborhood radius $a=0.2$, an interior neighborhood has measure

$$
V_2=\pi a^2
$$

in the 2D radial–azimuthal model and

$$
V_3=\frac{4\pi a^3}{3}
$$

in full 3D. A second point is placed a distance $d=a/4$ from the inner radial boundary. The 3D
case also places a third point the same physical distance $y(z-Z_{\min})=a/4$ from the lower polar
boundary, directly testing the locally planar polar-distance conversion. Each accessible measure
must subtract one boundary cap. In 2D,

$$
A_{\rm cap}
=a^2\arccos\left(\frac{d}{a}\right)
-d\sqrt{a^2-d^2},
$$

while in 3D,

$$
V_{\rm cap}
=\frac{\pi(a-d)^2(2a+d)}{3}.
$$

The expected radial and 3D polar boundary-corrected measures are $V_n-V_{\rm cap}$. The 2D case
has no active polar boundary, so its third stored measure repeats the interior disk area.

The two species have sizes $s_i=1$, $s_j=2$, represented target number $N_j=7$, compact density
$\rho_0=1$, and normalization $\lambda_0=0.3$. Their physical grain masses are

$$
m_i=\frac{\pi}{6},
\qquad
m_j=\frac{8\pi}{6}.
$$

The three exact pair-propensity numerators are

$$
\lambda_{ij}^{\rm const}=\lambda_0N_j,
$$

$$
\lambda_{ij}^{\rm add}=\lambda_0N_j(m_i+m_j),
$$

$$
\lambda_{ij}^{\rm prod}=\lambda_0N_jm_im_j.
$$

The validator requires the maximum error across the three measures and three kernel values to
remain below $2\times10^{-13}$. These cases catch incorrect dimensions, missing cap corrections, and an
incorrect additive-kernel definition. They do not test neighbor identity, periodic images,
normalization by an actual KNN radius, partner selection, coagulation events, or fragmentation.

## Exact KNN and periodic-ghost tests

`qav/swarm/test_knn/` is a standalone CUDA test family registered as the `knn` common-suite group.
It compiles four drivers:

- ordinary smooth, ring, and clump benchmarks in 2D and 3D
- adversarial edge cases for ties, coincident points, cutoff equality, sparse neighborhoods, and
  Morton split planes
- periodic query-image cases covering both wedge faces, full-$2\pi$ geometry, image overlap, and
  deduplication
- periodic wedge benchmarks comparing the three-copy KD-tree, query-image Morton, and the
  production compact boundary-ghost Morton owner
- a deliberately narrow seam-clump wedge that forces physical-identifier deduplication and the
  production $3N_K$ fallback rather than the ordinary disjoint-image shortcut

For every configured brute-force query, the CPU reference enumerates every physical particle,
applies the exact minimum-image wedge geometry, sorts by $(d^2,\mathrm{id})$, and retains the first
$N_K$. If the two GPU methods disagree outside the initially configured brute-force subset, the
validator also checks every disagreement against exhaustive search. A case fails on an incorrect
identifier, a distance outside the single-precision tolerance, a missing valid neighbor, or a
traversal-stack overflow.

The clean A100 validation baseline covered:

- 10 of 10 ordinary adversarial cases
- 10 of 10 periodic adversarial cases
- 12 smooth, ring, and clump cases at $N_P=10^5$ and $10^6$
- 16 periodic-wedge cases at $N_P=10^5$ and $10^6$
- copied collision-runtime comparisons in full-disk 2D, wedge 2D, full-disk 3D, and full-disk 3D
  at $N_P=10^6$

All recorded topology checks passed, including the million-particle cases. At $N_P=10^6$, ordinary
Morton query time ranged from 0.991 to 1.136 times the KD-tree speed while using 0.706-0.752 of its
persistent search memory. For periodic wedges, boundary ghosts gave KD-tree/ghost query speed
ratios of 1.007-1.133 and used 0.249-0.455 of the three-image KD-tree persistent memory. Ratios above
one favor Morton. These are hardware-specific A100 measurements, not universal performance claims.

Those standalone memory ratios compare the owned search hierarchies while treating query points as
common inputs. The copied-runtime experiments counted backend-specific support arrays as well:
Morton/KD-tree storage was 1.379-1.446 for the tested full disks and 0.468 for the tested wedge.
The promoted production owner has changed since those measurements, so current total VRAM must be
remeasured rather than inferred from the hierarchy-only ratios.

The archived JSON, manifests, environment records, and terminal summaries were moved from the
development laboratory to `qav/swarm/test_knn/out/`. That directory records the clean pre-promotion
baseline. Because the QA driver now calls the production ghost owner directly, it must be rerun on
native CUDA before the current source revision is treated as a publication artifact.

## Recorded native CUDA results

The analytical matrix was run natively on the Vera CUDA cluster on 2026-07-29 with the then-current
`all` group:

```bash
python3 qav/swarm/test_common/run_suite.py \
    --group all \
    --res 32 64 128 256
```

At that date the KNN group had not yet been registered, so this historical command comprised only
the ten analytical models. The completed archive contains all 25 analytical builds, and all ten
models passed their validators.
Resolved development compilation errors are not part of the current verification claim.

| Model | Requested $N$ | Recorded $L_2$ errors | Result |
|---|---:|---|---|
| `test_grid_2d` | 32, 64, 128, 256 | density: $8.16\times10^{-16}$ to $8.58\times10^{-15}$; optical depth: $1.08\times10^{-15}$ to $7.02\times10^{-15}$ | PASS |
| `test_grid_3d` | 32, 64, 128, 256 | density: $9.16\times10^{-16}$ to $1.30\times10^{-14}$; optical depth: $1.19\times10^{-16}$ to $3.17\times10^{-15}$ | PASS |
| `test_orbit_2d` | 32, 64, 128, 256 | state: $2.22\times10^{-15}$ to $8.44\times10^{-15}$ | PASS |
| `test_drag_2d` | 32 | response: $3.25\times10^{-18}$ | PASS |
| `test_diffusion_2d` | 32, 64, 128, 256 | velocity: $5.64\times10^{-17}$ to $5.68\times10^{-17}$ | PASS |
| `test_diffusion_3d` | 32, 64, 128, 256 | velocity: $2.03\times10^{-17}$ to $2.04\times10^{-17}$ | PASS |
| `test_radiation_2d` | 32 | response: $3.17\times10^{-18}$ | PASS |
| `test_prdrag_2d` | 32 | response: $6.95\times10^{-17}$ | PASS |
| `test_collision_2d` | 32 | collision helpers: $1.13\times10^{-17}$ | PASS |
| `test_collision_3d` | 32 | collision helpers: $2.83\times10^{-18}$ | PASS |

The downloaded archive contains all 25 `metrics_N*.json` files, all 25 matching `meta_N*.txt`
files, and one `environment.txt` for each of the ten models. Every JSON record has
`"passed": true`. All environment records agree on:

- NVIDIA CUDA compiler 12.1, build 12.1.105
- NVIDIA A100-SXM4-40GB GPU
- NVIDIA driver 580.159.04

### Deterministic accuracy and conservation

The largest recorded $L_\infty$ error and mass drift over every tested resolution were

| Calculation | Maximum $L_\infty$ | Maximum $|\Delta M_d/M_d|$ |
|---|---:|---:|
| 2D deposited density | $5.46\times10^{-14}$ | $6.66\times10^{-15}$ |
| 2D optical depth | $1.38\times10^{-14}$ | — |
| 3D deposited density | $1.45\times10^{-13}$ | $2.55\times10^{-15}$ |
| 3D optical depth | $1.53\times10^{-14}$ | — |
| circular-orbit state | $1.73\times10^{-14}$ | — |
| frozen drag response | $1.39\times10^{-17}$ | — |
| frozen radiation response | $6.94\times10^{-18}$ | — |
| frozen P-R response | $2.22\times10^{-16}$ | — |
| 2D diffusion velocity reprojection | $2.22\times10^{-16}$ | — |
| 3D diffusion velocity reprojection | $2.22\times10^{-16}$ | — |
| 2D collision helpers | $2.78\times10^{-17}$ | — |
| 3D collision helpers | $6.94\times10^{-18}$ | — |

These errors are consistent with floating-point evaluation and accumulation roundoff and remain
well inside their validator thresholds. The orbit's printed formal orders, $0.1096$, $-1.5315$,
and $-0.3951$, only compare roundoff-level errors and therefore carry no convergence meaning.

### Diffusion moment statistics

For each diffusion ensemble, define the normalized discrepancy as the absolute measured moment
error divided by its six-standard-error acceptance limit. A value below one passes. For the 3D
case, the table reports the larger ratio between the active $R$ and $Z$ directions.

| Model | $N$ | $N_P$ | mean-error fraction | variance-error fraction |
|---|---:|---:|---:|---:|
| `test_diffusion_2d` | 32 | 16,384 | 0.207 | 0.012 |
| `test_diffusion_2d` | 64 | 65,536 | 0.162 | 0.230 |
| `test_diffusion_2d` | 128 | 262,144 | 0.043 | 0.255 |
| `test_diffusion_2d` | 256 | 1,048,576 | 0.084 | 0.213 |
| `test_diffusion_3d` | 32 | 16,384 | 0.139 | 0.132 |
| `test_diffusion_3d` | 64 | 65,536 | 0.163 | 0.064 |
| `test_diffusion_3d` | 128 | 262,144 | 0.374 | 0.257 |
| `test_diffusion_3d` | 256 | 1,048,576 | 0.276 | 0.366 |

All measured means and variances lie within 0.374 of their six-standard-error limits. Their
non-monotonic individual errors are expected for nested samples from one stochastic realization;
the tightening statistical limits, rather than monotonic sample error, define the convergence
criterion. The independently checked velocity-reprojection invariant remains at roundoff for
every ensemble.

The archived JSON files are the authoritative records for the complete $L_1$, $L_2$,
$L_\infty$, mass, and stochastic-moment values. The metadata and environment records document the
compiled problem sizes and native CUDA platform, so the downloaded `qav/swarm/out/` directory now
forms a complete machine-readable record of this verification run.

## Running the suite

From the repository root, run a short workflow check with

```bash
python3 qav/swarm/test_common/run_suite.py --group all --quick
```

This performs the 15 analytical-suite builds, compiles the four KNN drivers, links the production
collision sources once per backend, and then runs the $10^5$-particle KNN matrix.

Run the complete default matrix with

```bash
python3 qav/swarm/test_common/run_suite.py \
    --group all \
    --res 32 64 128 256
```

The complete command performs the 25 analytical-suite builds plus the four KNN driver builds and
the two production-backend links.
Individual groups can be selected with

```bash
python3 qav/swarm/test_common/run_suite.py --group grid      --res 32 64 128 256
python3 qav/swarm/test_common/run_suite.py --group transport --res 32 64 128 256
python3 qav/swarm/test_common/run_suite.py --group diffusion --res 32 64 128 256
python3 qav/swarm/test_common/run_suite.py --group radiation --res 32
python3 qav/swarm/test_common/run_suite.py --group collision --res 32
python3 qav/swarm/test_common/run_suite.py --group knn       --res 32
```

The KNN group can also be run directly:

```bash
python3 qav/swarm/test_knn/run.py
```

Add the million-particle matrix with either

```bash
python3 qav/swarm/test_knn/run.py --full
python3 qav/swarm/test_common/run_suite.py --group knn --knn-full
```

Use `--build-only` to compile every selected configuration without executing or validating it.
Each resolution is cleaned and rebuilt because the mesh sizes, particle count, or timestep count
are compile-time constants. Output is stored as

```text
qav/swarm/out/MODEL/
```

with raw binary arrays, `meta_N*.txt`, `metrics_N*.json`, and `environment.txt`. Keep the output
directory and full terminal log together when archiving a native run.

KNN benchmark JSON and manifests are stored separately under

```text
qav/swarm/test_knn/out/
```

## Verification still required

- Rerun the promoted KNN suite against the production boundary-ghost owner; the test group now
  compiles and links both production collision backends before running the standalone topology matrix
- Add direct end-to-end rate/event comparisons for axisymmetry, partial wedges, and full 3D
- Test complete frozen collision batches, the exact Bernoulli probability
  $1-e^{-\lambda\Delta t}$, partner sampling, representative-mass conservation, and convergence
  with `CFL_COL`, $N_K$, and $N_P$
- Recover analytical constant, additive, and product Smoluchowski moment evolution rather than
  checking only the pair-kernel numerators
- Validate the monodisperse and conditional multisize initialization CDFs, the convolved radial
  profile, finite-sample mass normalization, and initial drift velocities
- Validate imported-gas spatial and temporal interpolation, the analytical `STOKES_0` anchor, and
  response to a depleted imported midplane density
- Add deterministic tests of periodic azimuth, radial absorption, full-disk polar absorption,
  half-disk reflection, and reflecting diffusion boundaries
- Add a direct `dyn_rate_calc` test for every active rate and a regression proving accepted
  timesteps do not exceed the configured crossing or diffusion limits
- Test finite attenuation by coupling the reconstructed optical depth to
  $\beta e^{-\tau}$; the current grid and radiation cases validate the two pieces separately
- Recover long-term reduced-gravity circular motion and secular P-R inspiral, including the
  factor-of-two radial P-R damping over many steps
- Validate vertical settling–diffusion equilibrium and spatially varying diffusivity-gradient
  terms
- Compare uninterrupted and restarted stochastic runs bitwise, including restored cuRAND state
- Add end-to-end operator-combination tests for transport plus diffusion, transport plus radiation,
  transport plus collision, and all enabled swarm physics
- Repeat important publication runs without `--use_fast_math`, or document and measure its effect

The existing 2D and 3D helper tests do not establish correctness of complete 3D disk evolution.
The KNN suite establishes exact search topology and isolated performance, but the collision algebra
tests still do not establish the statistical convergence of the complete event pipeline.

## References

- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Ormel & Cuzzi (2007), [turbulent relative velocities](https://arxiv.org/abs/astro-ph/0702303)
- Zsom & Dullemond (2008), [representative-particle coagulation](https://arxiv.org/abs/0807.5052)
- Gillespie (1977), [stochastic reaction simulation](https://doi.org/10.1021/j100540a008)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Burns, Lamy & Soter (1979), [radiation pressure and P-R drag](<https://doi.org/10.1016/0019-1035(79)90050-2>)
