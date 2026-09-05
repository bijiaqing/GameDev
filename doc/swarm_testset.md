# Swarm publication test suite

## 1. Purpose

The swarm validation suite verifies the scientific behavior of the Lagrangian dust model: deterministic
trajectories, stochastic diffusion, continuous finite-domain initialization, physical collision
rates, nearest-neighbor geometry, and the production frozen-bath collision integrator. The retained
tests compare complete algorithms with closed-form, statistical, or brute-force references.

The common matrix is defined in `val/tool/val_config.py`. For the standard resolutions
$N=32,64,128,256$, it contains 15 named entries: 14 analytical or statistical models producing
42 metric records, plus the standalone KNN matrix. The all-in-one campaign also runs four native
collision-chain models. CUDA and ROCm use the same
backend-neutral definitions and validators wherever their runtime APIs permit.

The production equations and numerical methods are documented in
[swarm_numeric.md](swarm_numeric.md). Archive layout and commands are in
[`val/README.md`](../val/README.md).

## 2. Publication criterion

A swarm case is retained when it establishes at least one of these claims:

1. a complete deterministic trajectory follows a known physical orbit or relaxation path;
2. a stochastic operator reproduces the probability law implied by its stochastic differential
   equation;
3. the initialized ensemble samples the intended continuous mass distribution;
4. the collision prescription reproduces published physical-rate formulae across all regimes;
5. the selected neighbors are exact under ordinary, periodic, wedge, boundary, and clumped
   geometries;
6. the production collision integrator conserves represented mass and is pathwise reproducible.

The suite intentionally does not retain elementary grid/index checks, flag-guard checks, deliberate
failure injection, restart plumbing, or performance measurements as separate scientific cases.

## 3. Error and statistical measurements

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

Stochastic diffusion is not required to match the same realization on CUDA and ROCm. Instead, each
backend must independently satisfy sampling limits derived from the expected mean and variance.
Continuous initialization uses probability-integral transforms and Kolmogorov--Smirnov statistics.

For represented particle number $N_p$ and grain size $s_p$, the mass proxy used by collision tests is

$$
M_{\rm rep}\propto\sum_p N_p s_p^3.
$$

The constant material-density factor cancels in relative conservation errors.

## 4. Retained matrix

| Group | Models | Main claim |
|---|---|---|
| transport | `test_orbit_ecc_2d`, `test_orbit_beta_2d`, `test_orbit_inc_3d`, `test_drag_path_1d`, `test_prdrag_2d` | deterministic gravity, radiation, drag, and Poynting--Robertson trajectories |
| diffusion | `test_diffusion_1d`, `test_diffusion_2d`, `test_diffusion_3d`, `test_diffusion_wedge_2d`, `test_diffusion_wedge_3d` | drift-diffusion statistics, inactive velocity constraints, and wedge-vector identification |
| initialization | `test_initial_3d` | exact continuous finite-domain sampling and polydisperse mass weights |
| collision physics | `test_colphys_code`, `test_colphys_cgs`, `test_colphys_3d` | turbulent, Brownian, resolved, vertically integrated, and periodic-image collision rates |
| neighbor search | `test_knn` | exact K-nearest-neighbor results for KD-tree and Morton backends |
| collision chain | `test_colchain_2d`, `test_colchain_frag_2d`, `test_colchain_wedge_2d`, `test_colchain_3d` | production frozen-bath evolution and controller behavior |

The PR-drag and physical-collision cases use one fixed build because their test coordinate is a
parameter index rather than a spatial resolution. `test_initial_3d` uses only the endpoint builds
$N=32$ and $256$: its continuous sampler is intentionally independent of polar cell count, and the
two builds verify byte-identical initialized samples, mass banks, and mass summaries without two
redundant intermediate records.

## 5. Deterministic trajectories

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

It compares the full stored phase-space state as well as specific energy

$$
\mathcal{E}=\frac{v_R^2+v_\phi^2}{2}-\frac{GM}{R}=-\frac{GM}{2a}
$$

and angular momentum $\ell=\sqrt{GMa(1-e^2)}$. This catches compensating position and velocity
errors that a radius-only comparison would miss.

### 5.2 Radiation-pressure orbit

`test_orbit_beta_2d` repeats the exact Kepler comparison with reduced gravity

$$
GM_{\rm eff}=GM(1-\beta).
$$

It demonstrates that the radiation flag changes the central acceleration consistently in both the
orbit frequency and conserved quantities.

### 5.3 Inclined three-dimensional orbit

`test_orbit_inc_3d` rotates an eccentric Kepler ellipse by a known inclination and node angle, then
converts the exact Cartesian orbit into the code's spherical position and velocity variables. It
tests radial, azimuthal, and polar transport simultaneously and compares the full state and orbital
energy.

### 5.4 Drag path

`test_drag_path_1d` uses constant stopping time and gas velocity. The exact response is

$$
v(t)=v_g+(v_0-v_g)e^{-t/t_s},
$$

and the integrated radial path is

$$
R(t)=R_0+v_g t+(v_0-v_g)t_s\left(1-e^{-t/t_s}\right).
$$

The case compares both position and velocity after multiple production transport steps. Inactive
azimuthal and polar variables must remain at their radial-model constraints.

### 5.5 Poynting--Robertson drag

`test_prdrag_2d` verifies the coupled radiation/drag response. To first order in $v/c$, the
Poynting--Robertson acceleration is

$$
\boldsymbol{a}_{\rm PR}
=-\beta\frac{GM}{R^2}
\left(\frac{v_R}{c}\,\hat{\boldsymbol{R}}
+\frac{\boldsymbol{v}_{\perp}}{c}\right),
$$

in addition to the radial radiation-pressure term. The fixed-step reference evaluates the exact
linear relaxation factors for radial and tangential components and compares angular momentum,
radial velocity, and the time-centered position update.

## 6. Stochastic diffusion

The swarm diffusion update represents an Itô process of the form

$$
dX=A(X)\,dt+\sqrt{2D(X)}\,dW,
$$

where $A$ contains the required diffusivity-gradient and coordinate drift. For the constant-$D$
test problems,

$$
\operatorname{Var}(\Delta X)=2D\Delta t.
$$

`test_diffusion_1d`, `test_diffusion_2d`, and `test_diffusion_3d` activate the radial, planar, and
cylindrical-vertical forms respectively. The validator checks each sampled component with

$$
|\bar X-\mu|\le 6\sqrt{\frac{2D\Delta t}{N_P}},
$$

and

$$
|s_X^2-2D\Delta t|
\le 6(2D\Delta t)\sqrt{\frac{2}{N_P-1}}.
$$

It separately requires particle velocities to remain unchanged to roundoff. In the cylindrical
cases, the expected geometric mean drift is included rather than assuming a Cartesian zero mean.

These are distributional tests: differing random-number streams across GPU backends are acceptable
when both ensembles satisfy the same law.

`test_diffusion_wedge_2d` and `test_diffusion_wedge_3d` place matched populations next to both
faces of a narrow periodic wedge. For each realized displacement, the validator reconstructs the
unique unwrapped endpoint $x_u$ satisfying

$$
|x_u-x_0|<\frac{\Delta\phi_w}{2}
$$

and independently projects the unchanged initial Cartesian velocity at $x_u$. It requires both
crossing directions to occur and compares every stored velocity component with that deterministic
reference to $2\times10^{-12}$. The displacement bound is an activation requirement: if any
particle moves far enough to make the wrap count ambiguous, the test setup fails rather than
guessing an image. The 3D specialization uses nonzero polar motion and therefore exercises the full
spherical basis transformation.

## 7. Continuous polydisperse initialization

`test_initial_3d` validates the exact-containment initialization used when a vertically settled dust
distribution is truncated by spherical and polar domain boundaries.

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

The test uses populations at $s_{\min}$, $\sqrt{s_{\min}s_{\max}}$, and $s_{\max}$. It independently
reconstructs the 128-entry mass bank, tests the logarithmic interpolation at the middle size, checks
represented mass, and applies radial and vertical probability-integral transforms. For a correct
sampler those transformed values are uniform; each population must satisfy a KS limit
$6/\sqrt{N_{\rm population}}$. The endpoint resolution builds must produce the same continuous
initialization result because the algorithm no longer integrates over polar cells.

## 8. Physical collision rates

`test_colphys_code`, `test_colphys_cgs`, and `test_colphys_3d` evaluate the production physical
collision helpers for fixed particle pairs. The relative speed is assembled as

$$
\Delta v_{ij}=
\sqrt{\Delta v_{\rm resolved}^2+\Delta v_{\rm turb}^2+\Delta v_{\rm Brown}^2},
$$

For the vertically integrated planar model, the pair numerator is

$$
q_{ij}^{2D}
=N_j\,\sigma_{ij}\,\Delta v_{ij}
\left[2\pi(H_i^2+H_j^2)\right]^{-1/2},
\qquad
\sigma_{ij}=\frac{\pi}{4}(s_i+s_j)^2.
$$

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
V_i=\frac{4\pi}{3}h_i^3.
$$

The 3D reference evaluates gas stratification, Stokes numbers, turbulent velocity, and the
Brownian cap at the actual off-midplane cylindrical position; it does not reuse the midplane
specialization.

An independent Python implementation evaluates all six Ormel--Cuzzi turbulent-velocity regimes and
straddles all five branch boundaries. The physical-unit case additionally checks Brownian motion and
its sound-speed cap. Both branches check the resolved Cartesian relative speed and the assembled
custom-kernel rate. The maximum relative discrepancy must remain below $2\times10^{-11}$.

The code-unit planar and 3D paths additionally represent the same local pair once in the wedge
interior and once across its rotational seam. The partner image code

$$
c=3j+a
$$

is decoded back to physical index $j$ and image $a$. The resolved speed and complete physical rate
must equal the interior reference, while deliberately ignoring the image must produce a relative
speed at least ten times larger. The 3D case uses nonzero radial and polar velocity so this is not a
planar identity compiled on a three-dimensional grid.

The cache regression does not insert this code manually. It launches the production site
initialization, builds the selected KD-tree or Morton/ghost index, launches the production KNN cache
query, and then launches the cached production rate kernel. Each of the three physical-collision
models is compiled and validated with both searches while retaining one aggregate metric record per
resolution. The validator requires the expected
physical-index/image sets for both owners, compares the cached physical rates with independent
pair-rate numerators to $2\times10^{-11}$, and checks the returned KNN area or volume against an
independent float-geometry reconstruction to relative tolerance $2\times10^{-4}$. The latter
tolerance reflects that search coordinates and tree nodes are single precision; it is not applied
to the physical collision formulas.

These fixed-input cases are retained because the piecewise published prescription is scientifically
material and difficult to infer from an end-to-end stochastic size distribution.

## 9. Exact nearest neighbors

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
that seam. The retained cases cover:

- smooth, ring, and clumped distributions in two and three dimensions;
- duplicate distances, inactive particles, sparse leaves, and domain-edge queries;
- full $2\pi$ periodic seams;
- restricted periodic wedges, including narrow and nearly full-period seam-centered clumps;
- KD-tree and adaptive Morton/ghost representations.

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

All labeled adversarial checks and every matrix record must pass. Timing is recorded only as a
diagnostic and is not an acceptance criterion.

## 10. Production frozen-bath collision chain

The four `test_colchain_*` cases run the production collision runtime rather than an isolated event
formula. They cover coagulation in a planar disk, fragmentation, a periodic wedge, and full 3D.
Each case runs both KD-tree and Morton searches. The planar coagulation case also compares event caps
1 and 32, forcing the continuation path.

For every realization, the validator requires:

- finite particle state and positive size/number;
- at least one changed particle and, for the fragmentation case, at least one shrunken particle;
- relative represented-mass error no larger than $2\times10^{-12}$;
- a complete adaptive-bath audit with finite nonnegative controller quantities;
- no persistent activity/distribution overshoot;
- repeated runs with identical final particle and RNG hashes;
- equivalent controller histories to $10^{-12}$;
- pathwise identity between cap-1 and cap-32 results for the same search backend.

The controller divides a collision operator interval into frozen baths. Within each bath the
positions and KNN geometry are fixed, while the event chain updates particle properties. A bath is
accepted only when its measured activity and distribution changes remain within the controller
limits. The wedge chain must report `COAG_KERNEL=3`, ensuring that it exercises the physical
collision kernel rather than a synthetic rate. These tests establish conservation, deterministic
continuation, search-backend coverage, and production-path integrity. They do not constitute a convergence proof for arbitrary physical
coagulation histories. When a publication depends on such a history, its model-specific evidence
must add bath-tolerance refinement and independent-seed comparisons at equal physical time; those
campaign outputs are scientific results, not additional permanent validation records.

## 11. Running and interpreting the suite

The canonical common archive is below `val/logs/swarm/BACKEND/`; disposable executables, objects,
KNN binaries, and compiler stamps are isolated below `val/temp/`. The collision-chain publication
records are below `groups/chain/`. A complete standard campaign reports 42 analytical/statistical metrics, a
passing KNN suite manifest, and four passing collision-chain manifests.

Source inspection, normalized CUDA/ROCm diffs, Makefile dry runs, and static checks establish code
structure and routing but do not constitute native numerical qualification. Publication evidence
requires a completed GPU campaign for the cited source snapshot. Cross-backend parity additionally
requires matching campaign source fingerprints unless the comparison is explicitly exploratory.

The strongest evidence is the combination of:

- closed-form phase-space trajectories rather than single-step force checks;
- statistical acceptance derived from the SDE rather than matching RNG bytes;
- continuous CDF and mass-integral validation of initialization;
- independent coverage of every physical collision-rate regime;
- brute-force KNN identity in geometrically difficult domains;
- conserved, deterministic end-to-end collision histories;
- matching CUDA and ROCm publication manifests from the same source fingerprint.

## 12. Deliberate limits

The suite does not provide an analytical coagulation-size-distribution solution, an exhaustive flag
matrix, a restart guarantee, or a hardware-performance claim. It does not retain one-off tests for
indexing, guard clauses, or helper return values. A new test belongs in the publication matrix only
when it validates a distinct scientific algorithm or physically relevant configuration not already
covered here.
