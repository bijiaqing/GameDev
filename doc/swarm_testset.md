# Swarm publication test suite

## 1. Purpose

The swarm validation suite verifies the scientific behavior of the Lagrangian dust model:
deterministic trajectories, stochastic diffusion, continuous finite-domain initialization, physical
collision rates, nearest-neighbor geometry, the production frozen-bath collision integrator, and
idealized coagulation against analytical Smoluchowski solutions. The retained tests compare complete
algorithms with closed-form, statistical, or brute-force references.

The canonical matrix is `SWARM_GROUPS` in `val/val_config.py`. It contains 15 named entries: 14
analytical or statistical models and the standalone KNN matrix. With the standard resolutions
$N=32,64,128,256$ these write 42 metric records (`EXPECTED_SWARM_METRICS`): four each for the nine
resolution-swept models, two for the endpoint-resolution initialization model, and one each for
the four fixed-resolution models. The KNN matrix writes a suite manifest instead of metric records.
The collision-chain group `SWARM_CHAIN_MODELS` adds four production-runtime models, each with one
manifest. CUDA and ROCm use the same backend-neutral definitions under `val/swarm/mod/` and shared
drivers and validators under `val/swarm/src/`. The analytical coagulation campaigns under
`val/paper/smoluchowski/` are CUDA-only scientific campaigns and deliberately remain outside this
routine matrix.

The production equations and numerical methods are documented in
[swarm_numeric.md](swarm_numeric.md). Commands, output paths, and job layout are in
[`val/README.md`](../val/README.md).

## 2. Evidence criterion

A swarm case is retained when it establishes at least one of these claims:

1. a complete deterministic trajectory follows a known physical orbit or relaxation path;
2. a stochastic operator reproduces the probability law implied by its stochastic differential
   equation;
3. the initialized ensemble samples the intended continuous mass distribution;
4. the collision prescription reproduces published physical-rate formulae across all regimes;
5. the selected neighbors are exact under ordinary, periodic, wedge, boundary, and clumped
   geometries;
6. the production collision integrator conserves represented mass and is pathwise reproducible;
7. an evolving representative-particle mass distribution agrees with an analytical Smoluchowski
   solution across independent stochastic realizations.

The suite intentionally does not retain elementary grid/index checks, flag-guard checks, deliberate
failure injection, restart plumbing, or performance measurements as separate scientific cases.

## 3. Common measurements

### 3.1 Error norms

For deterministic particle values $q_p$, the validators report

$$
L_1=\frac{1}{N_P}\sum_p|e_p|,\qquad
L_2=\left(\frac{1}{N_P}\sum_pe_p^2\right)^{1/2},\qquad
L_\infty=\max_p|e_p|,
$$

where $e_p=q_p^{\rm num}-q_p^{\rm ref}$. Angular differences are reduced to the principal interval,

$$
\Delta\phi=\bigl[(\phi_{\rm num}-\phi_{\rm ref}+\pi)\bmod 2\pi\bigr]-\pi.
$$

For trajectory tests the resolution $N$ is the number of timesteps over a fixed interval, and the
observed order is computed from the $L_1$ errors of successive factor-of-two refinements as in the
fluid suite.

### 3.2 Statistical and distributional measurements

Stochastic diffusion is not required to match the same realization on CUDA and ROCm. Instead, each
backend must independently satisfy sampling limits derived from the expected mean and variance.
Continuous initialization uses probability-integral transforms and Kolmogorov–Smirnov statistics.

For represented particle number $N_p$ and grain size $s_p$, the mass proxy used by collision tests
is

$$
M_{\rm rep}\propto\sum_p N_p s_p^3.
$$

The constant material-density factor cancels in relative conservation errors.

### 3.3 Acceptance gate

Each analytical validator returns one metric record with a `passed` flag and raises an error when
that flag is false, so a failing case stops the suite. `val/swarm/src/run_model.py` then requires
every record of the model to pass and, for models that declare a convergence field, the observed
order between the two finest resolutions to reach the declared minimum. The collision-chain and
KNN drivers apply their own criteria and write their own manifests (Sections 9 and 10).

## 4. Retained matrix

| Group | Models | Records | Meaning of $N$ | Main claim |
|---|---|---|---|---|
| transport | `test_orbit_ecc_2d`, `test_orbit_beta_2d`, `test_orbit_inc_3d`, `test_drag_path_1d` | 4 each | timesteps | deterministic gravity, radiation, and drag trajectories |
| transport | `test_prdrag_2d` | 1 | fixed build | exact Poynting–Robertson relaxation over one step |
| diffusion | `test_diffusion_1d`, `test_diffusion_2d`, `test_diffusion_3d`, `test_diffusion_wedge_2d`, `test_diffusion_wedge_3d` | 4 each | sample size $N_P=16N^2$ | drift-diffusion statistics, inactive velocity constraints, and wedge-vector identification |
| initialization | `test_initial_3d` | 2 | polar cells, $N=32$ and 256 | exact continuous finite-domain sampling and polydisperse mass weights |
| collision physics | `test_colphys_code`, `test_colphys_cgs`, `test_colphys_3d` | 1 each | fixed build | turbulent, Brownian, drift, vertically integrated, and periodic-image collision rates |
| neighbor search | `test_knn` | suite manifest | none | exact K-nearest-neighbor results for KD-tree and Morton backends |
| collision chain | `test_colchain_2d`, `test_colchain_frag_2d`, `test_colchain_wedge_2d`, `test_colchain_3d` | chain manifest | none | production frozen-bath evolution and controller behavior |
| analytical coagulation | `paper/smoluchowski/const`, `paper/smoluchowski/linear`, `paper/smoluchowski/product` | separate campaign | none | constant-, additive-, and product-kernel mass distributions against Smoluchowski solutions |

The Poynting–Robertson and physical-collision cases use one fixed build because their test
coordinate is a parameter index rather than a spatial resolution. `test_initial_3d` uses only the
endpoint builds $N=32$ and $256$: its continuous sampler is intentionally independent of polar cell
count, and the two builds must produce byte-identical initialized samples, mass banks, and mass
summaries without two redundant intermediate records.

Unless a case states otherwise, the models use $0.5\le R\le1.5$, the full azimuthal domain
$-\pi\le\phi\lt\pi$, the midplane or the polar domain $0.35\le\theta\le\pi-0.35$, gas aspect ratio
$0.05$, and midplane Stokes number $0.2$ at unit radius and size. Each model's `const_defs.cuh`
selects its `TEST_*` branch of `val/swarm/src/const_defs.cuh`, which replaces the production
physical setup with test constants.

## 5. Deterministic trajectories

The orbit and drag-path tests call the production `_ssa_advance` transport stages; only the drag
coefficient is specialized (zero drag for the orbits, prescribed stopping time and gas velocity for
the drag path).

### 5.1 Eccentric Kepler orbit

`test_orbit_ecc_2d` evolves particles on eccentric planar orbits. The validator advances the mean
anomaly

$$
M(t)=M_0+nt,\qquad n=\sqrt{\frac{GM}{a^3}},
$$

solves Kepler's equation

$$
M=E-e\sin E,
$$

and reconstructs

$$
R=a(1-e\cos E),\qquad
\phi=\operatorname{atan2}\!\left(\sqrt{1-e^2}\sin E,\cos E-e\right).
$$

It compares the full stored phase-space state and also records the specific-energy error,

$$
\mathcal{E}=\frac{v_R^2+v_\phi^2}{2}-\frac{GM}{R}=-\frac{GM}{2a},
$$

and the angular-momentum error, with $\ell=\sqrt{GMa(1-e^2)}$. Because the stored state contains
both position and velocity, compensating errors that a radius-only comparison would miss fail the
state comparison.

Acceptance: zero-drag specialization active, finite state, maximum state error below
$2\times10^{-2}$, and a final observed order of the state $L_1$ error of at least 1.8.

### 5.2 Radiation-pressure orbit

`test_orbit_beta_2d` repeats the exact Kepler comparison with reduced gravity

$$
GM_{\rm eff}=GM(1-\beta)
$$

for two grain sizes with different $\beta$, at zero optical depth and unit radiation taper. It
demonstrates that the radiation flag changes the central acceleration consistently in both the
orbit frequency and conserved quantities. Acceptance is as for the eccentric orbit, with the
additional activation requirement of two distinct values $0\lt\beta\lt1$.

### 5.3 Inclined three-dimensional orbit

`test_orbit_inc_3d` rotates an eccentric Kepler ellipse by known periapsis, inclination, and node
angles, then converts the exact Cartesian orbit into the code's spherical position and velocity
variables. It tests radial, azimuthal, and polar transport simultaneously and compares the full
state; the orbital energy error is recorded. The build enables the full-3D contract with zero
diffusivity. Acceptance: maximum state error below $3\times10^{-2}$ and a final state order of at
least 1.8.

### 5.4 Drag path

`test_drag_path_1d` uses three particles with distinct constant stopping times, a constant gas
velocity $v_g$, and a constant force $F$. With the terminal speed $v_\infty=v_g+Ft_s$, the exact
response is

$$
v(t)=v_\infty+(v_0-v_\infty)e^{-t/t_s},
$$

and the integrated radial path is

$$
R(t)=R_0+v_\infty t+(v_0-v_\infty)t_s\left(1-e^{-t/t_s}\right).
$$

The case compares both position and velocity after multiple production transport steps. Inactive
azimuthal and polar variables must remain at their radial-model constraints.

Acceptance: maximum position error below $10^{-2}$, velocity error below $2\times10^{-12}$,
inactive-variable error below $2\times10^{-14}$, and a final position order of at least 1.8.

### 5.5 Poynting–Robertson drag

`test_prdrag_2d` verifies the coupled radiation/drag response for eight grain sizes over one step
$\Delta t=0.1$. To first order in $v/c$, the Poynting–Robertson acceleration is

$$
\boldsymbol{a}_{\rm PR}
=-\beta\frac{GM}{cR^2}
\left(2v_R\,\hat{\boldsymbol{R}}+\boldsymbol{v}_{\perp}\right),
$$

where $\boldsymbol{v}_\perp$ is the velocity perpendicular to $\hat{\boldsymbol{R}}$, in addition
to the radial radiation-pressure term. It therefore adds the damping rate $\gamma=\beta GM/(cR^2)$
to the tangential components and $2\gamma$ to the radial component. The fixed-step reference
evaluates the exact linear relaxation factors for the radial and tangential components and compares
angular momentum, radial velocity, and the time-centered position update. Acceptance: maximum
response error below $2\times10^{-13}$.

## 6. Stochastic diffusion

The swarm diffusion update represents an Itô process of the form

$$
dX=A(X)\,dt+\sqrt{2D(X)}\,dW,
$$

where $A$ contains the required diffusivity-gradient and coordinate drift. The retained cases
apply one production `diffusion_pos` Euler–Maruyama step from prescribed initial positions. With
$D=\nu/(1+{\rm St}^2)$ evaluated at that initial position,

$$
\operatorname{Var}(\Delta X)=2D\Delta t.
$$

`test_diffusion_1d`, `test_diffusion_2d`, and `test_diffusion_3d` start every particle at $R=1$
on the midplane and activate the radial, azimuthal, and cylindrical radial-plus-vertical forms
respectively, with $\nu=2\times10^{-2}$ (`CONST_NU`) and $\Delta t=0.02$. The validator checks
each sampled component (radius, azimuth, or radius and height) with

$$
|\bar X-\mu|\le 6\sqrt{\frac{2D\Delta t}{N_P}},
$$

and

$$
|s_X^2-2D\Delta t|
\le 6(2D\Delta t)\sqrt{\frac{2}{N_P-1}}.
$$

It separately requires the particle velocities to remain unchanged to $2\times10^{-12}$. In the
radial cases, the expected mean includes both the cylindrical $D/R$ term and the Stokes-dependent
diffusivity gradient. These one-step density-mode tests do not establish concentration-mode
equilibria or long-time convergence in arbitrary imported gas fields.

These are distributional tests: differing random-number streams across GPU backends are acceptable
when both ensembles satisfy the same law.

`test_diffusion_wedge_2d` and `test_diffusion_wedge_3d` place matched populations $0.005$ inside
both faces of the periodic wedge $-0.1\le\phi\lt0.1$, advanced with $\Delta t=0.002$; the 3D
population starts one gas scale height off the midplane with nonzero polar motion. For each
realized displacement, the validator reconstructs the unique unwrapped endpoint $x_u$ satisfying

$$
|x_u-x_0|\lt\frac{\Delta\phi_w}{2}
$$

and independently projects the unchanged initial Cartesian velocity at $x_u$. It requires both
crossing directions to occur and compares every stored velocity component with that deterministic
reference to $2\times10^{-12}$. The azimuthal displacements must also satisfy the mean and variance
limits above, with the variance averaged over the initial Stokes numbers. The displacement bound is
an activation requirement: if any particle moves far enough to make the wrap count ambiguous, the
test fails rather than guessing an image. The 3D specialization exercises the full spherical basis
transformation.

## 7. Continuous polydisperse initialization

`test_initial_3d` validates the exact-containment initialization used when a vertically settled dust
distribution is truncated by spherical and polar domain boundaries. It calls the production host
initializer for 65 536 particles in the thin polar domain $\vert\theta-\pi/2\vert\le0.01$.

For size $s$, the radial marginal is

$$
\frac{dI}{dR}
=\Delta\phi\,R\Sigma_d(R)
\sum_k\left[
\Phi\!\left(\frac{Z_{k,+}}{H_d(R,s)}\right)
-\Phi\!\left(\frac{Z_{k,-}}{H_d(R,s)}\right)
\right],
$$

where the allowed vertical domain may split into two intervals and $\Phi$ is the standard normal
CDF. The size-dependent domain mass is

$$
I(s)=\int \frac{dI}{dR}\,dR.
$$

The production sampler first draws $R$ from the normalized radial CDF and then draws $Z$ from the
corresponding truncated Gaussian. The representative number assigned to a sampled particle is
normalized so that

$$
\sum_p N_p\,m(s_p)=M_{d,\Omega},
$$

with the finite-domain mass $M_{d,\Omega}$ integrated over the chosen size distribution.

The test uses three populations of about one third of the particles each, at $s_{\min}$,
$\sqrt{s_{\min}s_{\max}}$, and $s_{\max}$. It independently reconstructs the 128-entry mass bank,
uses the logarithmic interpolation of that bank for the middle size, checks represented mass, and
applies radial and vertical probability-integral transforms. For a correct sampler those transformed
values are uniform.

Acceptance: finite output, the expected grain size in each population, every particle inside the
domain, mass-bank entries within $2\times10^{-12}$ relative error, mass normalization and
represented mass within $5\times10^{-12}$, and radial and vertical KS distances below
$6/\sqrt{N_{\rm population}}$ for each population. The endpoint resolution builds must produce the
same continuous initialization result because the algorithm does not integrate over polar cells.

## 8. Physical collision rates

### 8.1 Relative speed and pair rates

`test_colphys_code`, `test_colphys_cgs`, and `test_colphys_3d` evaluate the production physical
collision helpers (`COAG_KERNEL = 3`) for two fixed particles with sizes $0.5$ and $1.75$ and
represented numbers $3$ and $7$. The relative speed is assembled as

$$
\Delta v_{ij}=
\sqrt{\Delta v_{\rm drift}^2+\Delta v_{\rm turb}^2+\Delta v_{\rm Brown}^2},
$$

where $\Delta v_{\rm drift}$ combines the query-local differential radial, azimuthal, and capped
settling speeds evaluated at the owner's position for both grain sizes.

For the vertically integrated planar model, the pair numerator is

$$
q_{ij}^{2D}
=N_j\,\sigma_{ij}\,\Delta v_{ij}
\left[2\pi(H_{g,i}^2+H_{g,j}^2)\right]^{-1/2},
\qquad
\sigma_{ij}=\frac{\pi}{4}(s_i+s_j)^2,
$$

where $H_{g,i}$ and $H_{g,j}$ are the gas scale heights at the two particles' cylindrical radii.
For the volumetric 3D model, the Gaussian overlap-depth factor is absent,

$$
q_{ij}^{3D}=N_j\,\sigma_{ij}\,\Delta v_{ij}.
$$

The total propensity of owner $i$ divides the retained pair numerators by its KNN measure,

$$
\Gamma_i^{2D}=\frac{1}{A_i}\sum_{j\in\mathcal N_i}q_{ij}^{2D},
\qquad A_i=\pi h_i^2,
$$

$$
\Gamma_i^{3D}=\frac{1}{V_i}\sum_{j\in\mathcal N_i}q_{ij}^{3D},
\qquad
V_i=\frac{4\pi}{3}h_i^3,
$$

where $h_i$ is the distance to the owner's farthest retained neighbor.

### 8.2 Regime coverage

An independent Python implementation evaluates all six Ormel–Cuzzi turbulent-velocity regimes
with one point inside each and straddles all five branch boundaries with points $10^{-6}$ below and
above each. `test_colphys_code` and `test_colphys_3d` use code units with a prescribed Reynolds
number and no Brownian motion. `test_colphys_cgs` omits `CODE_UNIT`, so the molecular Reynolds
closure and Brownian motion are compiled, and it additionally checks the Brownian speed and its
sound-speed cap. All three check the query-local drift speed and the assembled custom-kernel rate.
The 3D reference evaluates gas stratification, Stokes numbers, turbulent velocity, and the
Brownian cap at the actual off-midplane cylindrical position; it does not reuse the midplane
specialization.

### 8.3 Periodic images

The planar and 3D paths additionally represent the same local pair once in the interior of the
wedge $-0.5\le\phi\lt0.5$ and once across its rotational seam. The partner image code

$$
c=3j+a
$$

is decoded back to physical index $j$ and image $a$. Because the physical closure is query-local,
the seam pair, the same pair evaluated with a deliberately wrong image, and the interior pair must
all give the same relative speed to $2\times10^{-11}$. The image code and its decoded index and
image must still be returned exactly, because they determine the selected neighbor and its search
distance.

### 8.4 Search, cache, and bath-rate handoff

The cache check does not insert image codes manually. It places the two particles across the seam,
launches the production site initialization `col_site_init`, builds the selected KD-tree or
Morton/ghost index, and launches the production KNN cache query `col_cache_get` with $K=2$. It then
launches the production frozen-bath rate kernel `col_bath_rate` on both owners as one bath, whose
partner reservoir is the cache just built. In `test_colphys_cgs`, the only model that uses the
per-owner gas-environment cache, `col_env_cache` fills that cache first. At the probe grain sizes,
sticking packets contain one projectile, so each bath-start rate (result entries 30 and 31) equals
the physical propensity $\Gamma_i$ over the owner's two cached neighbors: itself and the periodic
image of its partner.

Each of the three physical-collision models is compiled and validated with both searches while
retaining one aggregate metric record, which passes only when both searches pass. The validator
requires the expected physical-index/image sets for both owners, compares the cached bath-start
rates with independent pair-rate numerators divided by the returned measure to $2\times10^{-11}$,
and checks the returned KNN area or volume against an independent float-geometry reconstruction to
relative tolerance $2\times10^{-4}$. The latter tolerance reflects that search coordinates and tree
nodes are single precision; it is not applied to the physical collision formulas.

Acceptance: all activation checks above (six regimes, five straddled boundaries, Brownian cap in
the physical-unit model, image invariance and interior equivalence, exact image codes, expected
cache images, positive measures), finite results, maximum relative error below $2\times10^{-11}$
over every physical value, and relative measure error below $2\times10^{-4}$, for both searches.

These fixed-input cases are retained because the piecewise published prescription is scientifically
material and difficult to infer from an end-to-end stochastic size distribution.

## 9. Exact nearest neighbors

### 9.1 Publication matrix

`test_knn` validates both collision-search backends with $K=200$ and $10^5$ particles. It is a
correctness matrix, not a speed benchmark.

The reference sorts candidates by the same physical squared distance used by the collision model,

$$
d_{ij}^2=|\boldsymbol{x}_i-\boldsymbol{x}_j|^2,
$$

after applying periodic minimum-image or wedge-ghost geometry as appropriate. Neighbor identity and
the $K$th radius are compared with brute force. In a restricted wedge, each backend is compared
with the exact candidate set admitted by its own representation: the KD tree stores both adjacent
periodic images, whereas Morton creates a ghost only when the source particle's search ball reaches
that seam. The 48 retained checks are

- 7 ordinary cases: smooth, ring, and clumped distributions in two and three dimensions, and a
  one-dimensional radial line;
- 12 wedge cases: smooth, ring, interior-clump, and seam-clump distributions, plus narrow
  ($0.2$ rad) and nearly full-period ($2\pi-0.1$ rad) seam-centered clumps, each in two and three
  dimensions;
- 14 periodic checks: lower and upper seams, interior queries, a narrow wedge, a full $2\pi$ disk,
  and two period-limit configurations, each in two and three dimensions;
- 15 edge checks: distance ties, duplicate points, queries on the search-radius boundary, sparse
  leaves, split-plane configurations, and inactive-particle filtering in two and three dimensions,
  a radial line, and KD-tree-specific inactive-filter checks.

For every selected wedge neighbor, the test also retains the packed image code returned by the
actual search. It independently rotates the decoded physical particle by that image and requires
the resulting squared distance to reproduce the stored search distance. The nearly-full-period
case must activate at least one legitimate KD-tree/Morton difference while each result still agrees
with its backend-specific oracle. The benchmark evaluates the fixed brute-force prefix and every
additional query on which the two searches disagree; a disagreement beyond the prefix therefore
cannot evade its two representation-specific references. Its first query is a deterministic,
radially isolated version of the wide-wedge eligibility counterexample, so activation does not
depend on the random sample. The physical-collision cases above then verify the complete
query $\rightarrow$ cache $\rightarrow$ physical-rate handoff.

Acceptance: every labeled adversarial check and every matrix case must pass. Timing is recorded
only as a diagnostic and is not an acceptance criterion.

### 9.2 Optional topology regression

The optional `test_knn` Make target `topology` compares the production Morton hierarchy with an
independent serial oracle in 36 cases: 2D/3D, leaf targets 1/128, depths 1/5/20, mixed random,
coincident and boundary points, all-coincident inputs, and singleton roots, with the index rebuilt
for every case. It checks stable point ordering, every node range and bound, child links, and leaf
statistics while allowing different node numbering. This structural regression is not part of the
48-check publication matrix; it does not replace the neighbor or collision tests.

## 10. Production frozen-bath collision chain

### 10.1 Setup and variants

The four `test_colchain_*` models compile `src/swarm/swarm_runtime.cu` and the production
frozen-bath continuous-time chain with validation constants and `COL_DIAGNOSTICS`. They run
collisions only, without transport, for one output interval, and exercise complete event histories
of 2048 representatives with $K=200$ neighbors and 256 threads per owner chain.

| Model | Geometry | Kernel | Output interval | Variants |
|---|---|---|---|---|
| `test_colchain_2d` | planar disk, $32\times32$ | normalized synthetic (`COAG_KERNEL = 0`) | $10^{-3}$ | KD tree and Morton, event caps 1 and 32 |
| `test_colchain_frag_2d` | planar disk, $32\times32$ | physical (`COAG_KERNEL = 3`), `V_FRAG = 0` | $1$ | KD tree and Morton, event cap 32 |
| `test_colchain_wedge_2d` | periodic wedge $-0.1\le\phi\lt0.1$ | physical (`COAG_KERNEL = 3`) | $1$ | KD tree and Morton, event cap 32 |
| `test_colchain_3d` | spherical, $8\times16\times8$ | normalized synthetic (`COAG_KERNEL = 0`) | $10^{-2}$ | KD tree and Morton, event cap 32 |

The zero fragmentation threshold forces fragmentation in `test_colchain_frag_2d`, the narrow wedge
forces periodic-image deduplication, and the 3D model exercises the volume-density closure. Event
cap 1 in the planar coagulation case forces the continuation path. Every variant is built and run
twice from a clean output directory.

### 10.2 Acceptance criteria

For every realization, `val/swarm/src/run_chain.py` requires:

- 2048 particle records, finite particle state, and positive size and number;
- at least one changed particle and, for the fragmentation case, at least one shrunken particle;
- relative represented-mass error no larger than $2\times10^{-12}$;
- run provenance in `variables.txt` matching the selected search, event cap, and expected
  `COAG_KERNEL`, with the path-integrated controller audit;
- a complete adaptive-bath archive: positive bath, operator, and wave counts, a bath count equal to
  the number of bath records, finite controller extrema, and, for every bath, positive duration,
  nonnegative finite diagnostics, and limit scales in $[0.25,1]$;
- no persistent activity/distribution overshoot;
- at least one chain launch for event cap 32, where a refresh wave may launch no chain when the
  no-event screen completes all of its owners, and more chain launches than refresh waves for
  event cap 1.

Across realizations and variants it requires:

- identical final particle and RNG hashes for the two repeated runs;
- equivalent controller histories, with every numeric entry equal to relative and absolute
  tolerance $10^{-12}$;
- byte-identical initial particle states for every variant of a model;
- pathwise identity (equal particle and RNG hashes) between cap-1 and cap-32 results for the same
  search backend.

The controller divides a collision operator interval into frozen baths. Within each bath the
positions and KNN geometry are fixed, while the event chain updates particle properties. The
post-interval audit flags activity or distribution overshoots and adjusts later refreshes; it does
not reject and replay the stochastic path. The validation criteria require no persistent overshoot.

### 10.3 Scope

These tests establish conservation, deterministic continuation, search-backend coverage, and
production-path integrity. They do not constitute a convergence proof for arbitrary physical
coagulation histories. When a publication depends on such a history, its model-specific evidence
must add bath-tolerance refinement and independent-seed comparisons at equal physical time; those
campaign outputs are scientific results, not additional permanent validation records.

## 11. Analytical coagulation distributions

The separate CUDA/KD-tree campaigns live under
[`val/paper/smoluchowski/`](../val/paper/smoluchowski/README.md), with `const`, `linear`,
and `product` kernels. Each uses $10^6$ representatives, 25 combinations of
`N_K = 16, 32, 64, 128, 256` and `COL_BATH_EPS = 0.01, 0.02, 0.04, 0.08, 0.16`,
and ten seeds. These define 750 planned runs, not 750 completed or validated runs.

The normalized kernels are constant, additive, and multiplicative in grain mass. Constant and
product start from unit monomers; linear samples equal-mass representatives from a gamma
mass distribution of shape two and scale one, corresponding to an exponential physical number
distribution. Grain material density $6/\pi$ makes the production mass formula $m=s^3$.
Constant output times extend to $10^8$; linear uses $t=0,1,2,3,4$; product stops at
$t=0.9$, before gelation at $t=1$.

These campaigns compile the production runtime, collision controller, cached rates, and event
updates. Local overrides provide normalization, seeded initialization, unit-volume neighbor
measures, and scoring. Fixed positions permit geometry reuse. Product additionally reshuffles
cached partner identities at each collision refresh and uses one spatial controller group; its
well-mixed evolution does not establish that a small fixed neighbor set suffices for the physical
collision problem.

The scorers record mass-weighted distributions, CDF and histogram distances, log-mass
Wasserstein distance, and moment errors. They impose no formal accuracy pass threshold.
Assess neighbor-count and refresh-tolerance dependence separately from seed scatter; a small
controller tolerance is not a histogram-error bound. Campaign definitions, commands, and output
layout are in the campaign READMEs.

These campaigns do not establish convergence with representative count, Morton or ROCm
accuracy, post-gelation product behavior, physical-kernel accuracy, restart behavior, or
performance scaling. They are not registered with `val/run_all.py` and do not enter the archive
counts below.

## 12. Interpreting results and the archive contract

Commands, output directories, and job layout are documented in
[`val/README.md`](../val/README.md). This section explains what the archived files mean.

### 12.1 Records and manifests

- A metric record (`metrics_N<res>.json`) is the validator output for one analytical model at one
  resolution: error norms, statistical ratios, activation values, the evidence tier, and `passed`.
  For a physical-collision model it aggregates the KD-tree and Morton results.
- A model manifest (`manifest.json`) covers one analytical model across its resolutions: its
  metric files, environment record, resolution tiers, convergence assessment, and `passed`.
- The KNN suite manifest (`suite_manifest.json`) summarizes the ordinary, wedge, periodic, and
  edge components with their case counts and `passed`.
- A collision-chain manifest (`manifest.json` of the chain group, schema 4) records, for every
  search and event-cap variant, both realizations with their hashes, mass error, provenance, and
  controller summary, together with the initial-state and cap-pathwise equalities, the mass
  tolerance, and `passed`.
- The suite manifest (`manifest_all.json`, with `environment_all.json`) lists the canonical
  models, their resolutions and tiers, the metric-tier counts, and the overall status.

### 12.2 What a pass means

`val/check_archive.py` accepts the swarm archive only when the suite manifest passes for the
complete canonical model list; every component manifest exists, passes, and agrees with the suite
manifest in identity and tiers, with its referenced metrics and environment record present; all 42
metric records exist and pass; the tier counts partition the records; the KNN suite manifest
passes as publication evidence; and all four collision-chain manifests pass with every variant
passing. A complete standard campaign therefore reports 42 passing analytical/statistical records,
a passing KNN suite manifest, and four passing collision-chain manifests.

The strongest evidence is the combination of:

- closed-form phase-space trajectories rather than single-step force checks;
- statistical acceptance derived from the SDE rather than matching RNG bytes;
- continuous CDF and mass-integral validation of initialization;
- independent coverage of every physical collision-rate regime;
- brute-force KNN identity in geometrically difficult domains;
- conserved, deterministic end-to-end collision histories;
- analytical constant-, additive-, and product-kernel mass distributions across
  independent seeds;
- consistent CUDA and ROCm publication manifests from the same source fingerprint.

### 12.3 Cross-backend comparison

`val/compare_backends.py` compares corresponding deterministic CUDA and ROCm swarm metric records
field by field with a default relative tolerance of $10^{-5}$ and absolute tolerance of $10^{-11}$.
The collision-physics KNN-measure diagnostics (`maximum_measure_relative_error` and
`errors.knn_measure`) are excluded, because they carry the single-precision rounding of the search
and differ between backends at about $10^{-5}$ relative; each backend judges them natively against
the $2\times10^{-4}$ bound of Section 8.4. The stochastic diffusion records need not agree; both
backends must pass their own statistical validation for the same case and resolution. The KNN suite
manifests must both pass with equal case counts, and each collision-chain model must pass on both
backends with the same variant coverage; chain hashes are compared only within one backend. Both
suite manifests must pass with equal tier counts and, unless explicitly disabled, the two campaign
records must carry the same source fingerprint.

### 12.4 Evidence boundary

These are source-defined tests and acceptance criteria. Only a fresh native campaign whose source
fingerprint matches the cited source qualifies the CUDA or ROCm swarm implementation. Source
inspection, backend build routing, Makefile dry runs, and static checks establish code structure
and routing but do not constitute native numerical qualification. Cross-backend parity requires
matching campaign source fingerprints unless the comparison is explicitly exploratory.

## 13. Deliberate limits

The suite does not provide an exhaustive flag matrix, a restart guarantee, or a
hardware-performance claim. The one-step diffusion cases do not test concentration-mode equilibria
or imported gas fields, and the collision-chain cases do not establish population-distribution
accuracy. The analytical coagulation campaigns have the additional limits stated in
[Section 11](#11-analytical-coagulation-distributions). The suite does not retain one-off tests for
indexing, guard clauses, or helper return values. A new test belongs in the publication matrix
only when it validates a distinct scientific algorithm or physically relevant configuration not
already covered here.
