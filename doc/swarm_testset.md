# Lagrangian swarm verification

## Overview and scope

The shared definitions under `qav/comm/swarm/`, together with the drivers under `qav/cuda/swarm/` and
`qav/rocm/swarm/`, compare production swarm kernels and device helpers with
analytical or statistically exact reference problems. They are verification models, not
production disk setups. Each model supplies a backend-local `swarm_runtime.cu` or
`swarm_runtime.hip` and a
`const_defs.cuh`; unless a case explicitly defines a small test kernel, the numerical operation
being measured comes from either the backend-neutral `comm/swarm/` tree or the selected backend's
`swarm/` tree.

The suite covers particle-to-grid projection, optical-depth construction, semi-analytic
transport, stiff gas drag, cylindrical stochastic diffusion, radiation pressure, Poynting–
Robertson drag, collision-neighborhood measures, the three dimensionless coagulation kernels, and
exact KD-tree/Morton neighbor search. Dedicated radial-only cases additionally test exact inactive
coordinates, annular collision normalization, and imported surface-density Stokes/Reynolds scaling.
It does not yet validate monodisperse initialization and initial drift together, complete stochastic
collision events against an analytical or converged reference, boundary-event convergence, restart
reproducibility, or long-term coupled evolution. It now includes non-equilibrium eccentric-orbit
and multi-step constant-drag trajectories in addition to its equilibrium and one-step tests.

This document defines what each test proves, what it does not prove, and what result is
expected. Machine-readable results are generated below `qav/logs/swarm/BACKEND/` and are ignored by
Git; the latest local archive is summarized below but is not part of the tracked source tree. The
QAV implementation points back to this document rather than maintaining a second copy of the case
definitions.

## Verification architecture and claim levels

The swarm tests provide four complementary kinds of evidence:

1. **Analytical device and kernel tests** compare deterministic grid, transport, drag, radiation,
   boundary, collision-helper, and imported-gas calculations with independent CPU formulas
2. **Statistical ensemble tests** compare diffusion and initialization samples with exact moments,
   probability transforms, represented-mass closure, or distributional thresholds
3. **Backend differential tests** compare KD-tree and Morton neighbor topology with independent
   brute force adjudicating search disagreements; retained development records additionally compare
   collision-rate probes and final collision distributions but are not current common-suite cases
4. **Diagnostic performance records** report KNN construction time, query time, persistent memory,
   record multiplicity, and traversal counters without making speed part of numerical `PASS`

Most models invoke production kernels from a small test-local backend runtime. Helper tests use
small GPU kernels only to call production device functions and return their values. The KNN family
compiles standalone controlled drivers and also links the actual 1D, 2D, and 3D production collision
translation units with each backend. The other GPU backend is never treated as analytical truth;
backend disagreement is resolved by exhaustive CPU search where the test design promises it.

### Evidence tiers

The common archive retains narrow regressions but distinguishes them from code-paper evidence:

- **Publication:** one representative resolution of each exact grid-geometry case and the
  stationary radial orbit; the full 2D-orbit and stochastic-diffusion sequences; the endpoint
  polar-resolution comparison for continuous initialization; drag, viscous-flow, radiation,
  P-R drag, imported-gas, collision-measure, collision-kernel, and compact KNN checks
- **Release:** extra resolutions of exact grid and stationary-orbit cases, the intermediate
  initialization repetitions, and the four deterministic boundary-policy helper models
- **Qualification:** restart, deliberate KNN/collision failure injection, extended million- and
  ten-million-particle KNN runs, and performance/profiling campaigns

With the standard four requested resolutions, the expanded 75-metric analytical swarm matrix
partitions into 57 publication and 18 release-only records. The source-matched 2026-08-23/24
CUDA/ROCm campaign completed all six new four-resolution sequences on both backends. The compact
KNN suite is publication evidence for exact
neighbor search; `--knn-full` adds qualification evidence rather than strengthening a physical
convergence claim. These labels are stored in model and aggregate manifests and do not remove any
test from the release gate.

Unlike the fluid analytical runner, the swarm runner implements numerical pass/fail criteria. Each
analytical or statistical resolution writes `passed`, each model writes a manifest, each KNN
component writes a component manifest, and the common dispatcher accepts a model only if its
manifest reports `passed: true`. Therefore the final line

```text
SWARM TEST SUITE: PASS
```

means that every selected model completed and every implemented criterion passed. It does not
extend the claim beyond the coverage limits stated in each test definition.

## Measurement and acceptance protocol

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
formal values of $p$ are not convergence orders and no fitted-order threshold is imposed. The eccentric-orbit case supplies the missing non-equilibrium temporal sequence and requires the
final state-error order to remain at least 1.8.

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

### Reference independence and pass propagation

Deterministic expected states are reconstructed in Python from test constants and raw GPU output;
the validator does not read a GPU result back as its own reference. Grid measures, optical-depth
prefix sums, drag exponentials, viscous targets, boundary folds, collision measures, and imported
gas scalings are evaluated independently. Initialization uses independent containment integrals and
probability-integral transforms. KNN edge cases exhaustively enumerate every candidate, while large
matrices brute-force a configured query subset and every backend disagreement.

Every record must remain finite in addition to satisfying its case-specific tolerance. The model
manifest passes only when all requested resolutions pass. The aggregate dispatcher then verifies
the component manifest before advancing to the next model, so a failed numerical comparison cannot
be hidden by later successful runs.

### Meaning of `--res`

The suite intentionally reuses one command-line resolution parameter for three different kinds of
refinement:

| Test family | Meaning of $N$ | Derived size |
|---|---|---|
| 1D grid | radial mesh cells | $N_X=N_Z=1$, $N_Y=N$, $N_P=N$ |
| 2D grid | mesh cells in both active directions | $N_X=N_Y=N$, $N_P=N^2$ |
| 3D grid | radial and polar mesh cells | $N_X=4$, $N_Y=N_Z=N$, $N_P=4N^2$ |
| circular orbit | timesteps in one orbit | $\Delta t=2\pi/N$, $N_P=64$ |
| eccentric orbit | timesteps to a non-integer orbital phase | $\Delta t=1.3/N$, $N_P=4$ |
| reduced-gravity orbit | timesteps to a non-integer orbital phase | $\Delta t=1.3/N$, $N_P=4$ |
| inclined 3D orbit | timesteps to a non-integer orbital phase | $\Delta t=1.3/N$, $N_P=4$ |
| constant drag path | timesteps over one time unit | $\Delta t=1/N$, $N_P=3$ |
| deterministic radial absorption | accepted transport steps over one time unit | $\Delta t=1/N$, $N_P=4$ |
| 1D/2D/3D diffusion | ensemble-control parameter | $N_P=16N^2$ on a fixed mesh |
| drag, radiation, P-R, collision algebra | no physical dependence on $N$ | only the first requested value is built |
| deterministic boundary helpers | no physical dependence on $N$ | only the first requested value is built |
| KNN benchmark | no dependence on `--res` | $10^5$ particles by default; add `--knn-full` for $10^6$ and $10^7$ |

Consequently, an order derived from the orbit is a temporal order, while decreasing diffusion
errors demonstrate Monte Carlo sampling convergence. The exactly constructed grid cases are
geometry regressions rather than truncation-error convergence tests.

The CUDA drivers write six structure-of-arrays fields in `state_N*.dat`:

$$
(x,y,z,\ell_x,v_y,\ell_z),
$$

and multisize tests additionally write grain size and represented grain number. The Python
validator constructs all reference values independently from these raw outputs and test metadata.

### Output artifact contract

The active QA runners use file formats according to the role of each artifact:

| Artifact | Format | Purpose |
|---|---|---|
| raw GPU fields | `*.dat` | headerless binary `real` arrays read by NumPy |
| per-build metadata | `meta_N*.json` | structured grid, time, and case parameters written directly by the CUDA driver |
| numerical assessment | `metrics_N*.json` | machine-readable errors, statistics, thresholds, and pass state |
| KNN benchmark record | `<distribution>_<dimension>d_N<particles>.json` | search configuration, correctness counters, timing, memory, and pass state |
| model or matrix index | `manifest.json` | authoritative list of current result JSON files and component pass state |
| KNN aggregate | `suite_manifest.json` | ordinary, edge, periodic, wedge, and production-link pass states |
| full swarm aggregate | `qav/logs/swarm/BACKEND/manifest_all.json` | complete common-matrix live state and final pass state |
| focused swarm aggregate | `qav/logs/swarm/BACKEND/groups/GROUP/manifest.json` | isolated focused-group state and final pass state |
| compiler and GPU record | `environment.json` | structured backend compiler, GPU, driver, and test-specific build settings |
| optional captured transcript | `*.json` | command, return state, and escaped human-readable output from a build or run |

The analytical and KNN harnesses originally evolved independently and used a mixture of text and
JSON records. There was no physical or validation reason for the distinction. Active runners now
use JSON for metadata, environments, metrics, manifests, and persistent build or run records, and
both current QA trees contain no persistent text results.

The remaining format differences are intentional. Raw numerical arrays remain compact binary
files, while small descriptive and diagnostic records use JSON. KNN tests do not need separate
`meta_N*.json` files because
their result JSON already contains the particle count, physical and search dimensions, cutoff,
$K$, tree parameters, and quality-query counts. An active manifest is authoritative; obsolete
development JSON files elsewhere in an output directory are not included in validation.
The KNN runner leaves the ordinary and wedge manifests scoped to the files they index and writes
the cross-component state separately to `suite_manifest.json`. A skipped periodic or wedge
component is stored as JSON `null`, distinguishing “not selected” from either pass or failure.

The complete `all` dispatcher writes the canonical model directories and `manifest_all.json`.
Focused groups write their model records and manifest below `groups/GROUP/`, and direct model
wrappers use `groups/manual/`. Consequently, no focused or exploratory run can alter the completed
common-matrix archive used for transfer and cross-backend comparison. `--rebuild-manifest`
reconstructs one selected group from copied component manifests without rerunning native
executables.

The complete `qav/tool/run_all.py` campaign also records a SHA-256 fingerprint of the Makefile and
every compiled or interpreted QAV source. Cross-backend comparison rejects two complete campaign
archives when those fingerprints differ, so source drift cannot masquerade as a CUDA/ROCm result.

## Coverage matrix and case inventory

The current CUDA matrix covers the following combinations:

| Capability | 1D radial | 2D radial–azimuthal | full 3D |
|---|---:|---:|---:|
| grid deposition and optical depth | yes | yes | yes |
| orbital transport, gas drag, and radial absorption | equilibrium, multi-step drag path, and exact radial crossings | circular, eccentric, and reduced-gravity trajectories | inclined eccentric trajectory |
| stochastic diffusion | cylindrical radial | azimuthal | cylindrical radial and vertical mapped to spherical storage |
| finite-domain multisize initialization | no | no | yes |
| radiation pressure and P-R drag | yes | yes | no |
| deterministic boundary helpers | yes | narrow wedge | full disk and half disk |
| collision measures and kernel numerators | annular | disk/circular cap | ball/radial and polar caps |
| exact KD-tree/Morton search | collinear radial embedding | ordinary and periodic wedge | ordinary and periodic wedge |
| end-to-end coupled evolution | no | guarded CUDA/Morton collision-only runtime | no current registered case |

The 1D model is vertically integrated and azimuthally symmetric but retains midplane dynamical
closure. The 2D model evolves surface density in $R$ and azimuth. The 3D model stores spherical
coordinates while diffusion is prescribed in cylindrical $R$ and $Z$; each test states explicitly
which representation its analytical reference uses. The older collision-runtime differential
records described below are development evidence, not directories registered in the current common
suite, and therefore do not change the “no current registered case” entries.

| Model | Production calculation exercised | Reference result |
|---|---|---|
| `test_grid_1d` | radial annular deposition, density conversion, opacity deposition, and radial prefix sum | exact annular surface density, cumulative midplane optical depth, and exact inactive coordinates |
| `test_grid_2d` | 2D mass deposition, density conversion, opacity deposition, radial prefix sum, ring mean | exact cell density and cumulative optical depth from one equal-mass particle at every interpolation centroid |
| `test_grid_3d` | full-3D spherical projection and optical-depth geometry | the same construction using exact $r^2\,dr\,d\Omega$ measures |
| `test_orbit_1d` | radial-only non-radiative `ssa_transport` | stationary radius and angular momentum with exact inactive coordinates during one pressure-free circular orbit |
| `test_orbit_2d` | complete non-radiative `ssa_transport` kernel | one pressure-free circular Kepler orbit |
| `test_orbit_ecc_2d` | production staggered drift and geometric force update with QAV-only zero drag | eccentric Kepler state, energy, angular momentum, and second-order temporal convergence |
| `test_orbit_beta_2d` | production split radiation transport with zero optical depth, unit taper, and QAV-only zero drag | reduced-gravity Kepler state for $\mu_{\rm eff}=(1-\beta)GM_S$, invariants, and second-order temporal convergence |
| `test_orbit_inc_3d` | complete spherical transport with QAV-only zero drag and an unscheduled zero-diffusivity closure | independently rotated inclined Kepler state, energy, and second-order temporal convergence |
| `test_drag_1d` | radial-only frozen gas-drag response | exact exponential angular relaxation, radial response, and inactive-state invariants |
| `test_drag_path_1d` | production staggered radial drift with QAV-only constant drag coefficients | exact velocity and displacement for three stopping times plus second-order position convergence |
| `test_absorb_path_1d` | production staggered drift and radial transport boundary with QAV-only constant paths | exact crossings and sentinel plus density, optical-depth, dynamical-rate, collision-mask, and collision-rate exclusion |
| `test_viscflow_1d` | radial initialization with vertically integrated `VISC_FLOW` | exact viscous gas target and steady dust drift across several radii |
| `test_drag_2d` | frozen-coefficient gas-drag response in `ssa_transport` | exact exponential angular relaxation and its induced radial response |
| `test_diffusion_1d` | direct cylindrical radial SDE | exact Itô mean and variance, invariant physical velocity, and exact inactive coordinates |
| `test_diffusion_2d` | azimuthal cylindrical diffusion SDE and velocity reprojection | exact Gaussian angular moments and invariant Cartesian velocity |
| `test_diffusion_3d` | cylindrical radial and vertical SDE mapped to spherical storage | exact radial/vertical moments including cylindrical Itô drift and invariant Cartesian velocity |
| `test_settle_diffuse_3d` | production vertical diffusion combined with a QAV-only local linear settling map | exact finite-step Ornstein--Uhlenbeck mean, variance, Gaussian CDF, and invariant physical velocity |
| `test_initial_3d` | continuous finite-domain multisize initialization with geometrically thin size-dependent settling | independent truncated-Gaussian containment, radial and vertical probability transforms, represented-mass closure, exact domain bounds, and invariance under changes to simulation $N_Z$ |
| `test_radiation_1d` | radial-only midpoint radiation split | exact radiation-modified frozen response and inactive-state invariants |
| `test_radiation_2d` | midpoint radiation split without P-R damping | exact radiation-modified frozen response at zero optical depth |
| `test_prdrag_1d` | radial-only gas plus P-R exponential response | exact component-dependent damping and inactive-state invariants |
| `test_prdrag_2d` | combined gas and P-R exponential response | exact component-dependent damping at zero optical depth |
| `test_collision_1d` | exact radial annular KNN measure and coagulation-kernel numerators | interior, inner-edge, outer-edge, and both-edge annular areas plus constant, additive, and product rates |
| `test_collision_2d` | 2D accessible-neighborhood measure and coagulation-kernel numerators | disk and circular-cap areas plus constant, additive, and product rates |
| `test_collision_3d` | 3D accessible-neighborhood measure and coagulation-kernel numerators | interior, radial-cap, and polar-cap ball volumes plus the same three rates |
| `test_import_1d` | imported vertically integrated gas coupling | exact $\mathrm{St}=\mathrm{St}_0\Sigma_0/\Sigma_g$ and external-$\Sigma_g$ turbulent Reynolds scaling |
| `test_colchain_2d` | guarded CUDA/Morton production runtime with the continuous-time frozen-bath collision chain | finite positive species, nontrivial evolution, represented-mass conservation, provenance, RNG checkpointing, and exact repeat determinism |
| `test_boundary_1d` | radial-only transport absorption and diffusion reflection | exact inactive-coordinate locking, repeated radial folding, and absorbing radial endpoint states |
| `test_boundary_2d` | radial–azimuthal boundary helpers on a narrow periodic wedge | exact multi-wrap azimuth, radial diffusion reflection, radial absorption, and inactive polar state |
| `test_boundary_3d` | full-disk 3D boundary helpers | exact periodic azimuth, radial/polar diffusion reflection, and radial/polar transport absorption |
| `test_boundary_half` | half-disk 3D boundary helpers | exact periodic azimuth, diffusion reflection, lower-polar absorption, and upper-midplane transport reflection |
| `test_knn` | exact KD-tree and adaptive-Morton construction, block-parallel sorted Morton top-$K$, radial-line embedding, periodic-image correctness, and production boundary ghosts | independent brute-force neighbors, adversarial topology, overflow-free traversal, and production-relevant timing/memory records |

### Radial-only validation contract

The `radial` group is a deliberately mixed validation matrix. Deterministic cases compare every
returned scalar with an independently evaluated CPU formula. The diffusion case compares ensemble
moments with exact stochastic moments. The KNN case combines differential comparison over many
queries with exhaustive CPU enumeration over a configured subset and every backend disagreement.
Thus `PASS` does not mean merely that a GPU kernel completed without an error.

| Model | GPU quantities compared | Independent ground truth | Required for `PASS` |
|---|---|---|---|
| `test_grid_1d` | annular dust density, cumulative optical depth, total mass, and inactive $x,z,\ell_z$ | one equal-mass representative at each exact logarithmic annular centroid; analytical annular area and radial optical-depth prefix sum | density $L_\infty<5\times10^{-12}$, optical-depth $L_\infty<5\times10^{-11}$, relative mass error $<5\times10^{-12}$, and inactive fields exact |
| `test_orbit_1d` | all six stored state components after one orbit | stationary $R=1$, $\ell_x=1$, $v_y=0$ equilibrium with inactive $x=0$, $z=\pi/2$, $\ell_z=0$ | finite state and combined $L_\infty<5\times10^{-13}$ |
| `test_drag_1d` | final $\ell_x$, $v_y$, $R$, and inactive fields for eight sizes | closed-form frozen-coefficient gas-drag exponential and midpoint radial force | combined $L_\infty<2\times10^{-13}$ |
| `test_viscflow_1d` | initialized $R$, $\ell_x$, $v_y$, and inactive fields at eight radii | analytical Kanagawa viscous gas velocity, radial Stokes scaling, and steady no-backreaction dust drift | combined $L_\infty<2\times10^{-13}$ |
| `test_diffusion_1d` | radial displacement mean and variance, reconstructed $v_\phi,v_R$, and inactive fields | exact one-step cylindrical SDE, $E[\Delta R]=D\Delta t$ and $\mathrm{Var}(\Delta R)=2D\Delta t$ | mean and variance errors each within six standard errors, with velocity and inactive-field errors jointly satisfying $L_\infty<2\times10^{-12}$ |
| `test_radiation_1d` | the same response fields as drag with size-dependent $\beta$ | closed-form frozen drag plus radiation-pressure response at zero optical depth | combined $L_\infty<2\times10^{-13}$ |
| `test_orbit_beta_2d` | complete stored state, reduced-gravity energy, and angular momentum | eccentric Kepler solution with $\mu_{\rm eff}=(1-\beta)GM_S$ | state $L_\infty<2\times10^{-2}$ and final temporal order $\ge1.8$ |
| `test_orbit_inc_3d` | complete spherical stored state and orbital energy | independently rotated inclined eccentric Kepler orbit | state $L_\infty<3\times10^{-2}$ and final temporal order $\ge1.8$ |
| `test_prdrag_1d` | component-dependent angular and radial damping plus updated radius | closed-form gas plus Poynting–Robertson exponential with independently calculated $k_x$ and $k_y$ | combined $L_\infty<2\times10^{-13}$ |
| `test_absorb_path_1d` | first inactive endpoint, sentinel, downstream grids, collision masks/rates, and active mass | exact crossings, two surviving representatives, and inactive-mask contract | crossing delay within one step, exact inactive state/mask/rates, active rates positive, no Morton overflow, survivor error $<2\times10^{-12}$, and mass error $<2\times10^{-12}$ |
| `test_boundary_1d` | seven returned $(x,R,z,\ell_x,v_R,\ell_z)$ states | independent repeated radial reflection for diffusion, endpoint absorption for transport, and inactive-coordinate locking | all 42 scalars finite and combined $L_\infty<2\times10^{-13}$ |
| `test_collision_1d` | four accessible radial KNN measures and three kernel numerators | exact boundary-clipped annular areas and constant, additive, and product propensity formulas | maximum absolute error across all seven scalars $<2\times10^{-13}$ |
| `test_import_1d` | local Stokes number and $\mathrm{Re}^{-1/2}$ at eight imported-density samples | prescribed $\Sigma_g$, $\mathrm{St}=\mathrm{St}_0\Sigma_0/\Sigma_g$, and analytical Reynolds scaling | maximum absolute error across all 16 scalars $<2\times10^{-13}$ |
| radial KNN matrix | KD-tree and Morton neighbor identifiers and squared distances for 4096 queries, plus overflow and stored-record diagnostics | exhaustive CPU search over all $10^5$ collinear points for the first 32 queries and for every backend disagreement | zero KD-tree/Morton disagreements over all 4096 queries, zero brute-force failures, and zero Morton traversal overflows |
| radial KNN edge checks | inner, middle, and outer line queries plus rejection of nearer inactive particles | exhaustive enumeration of each small point set | exact valid identifier set, squared-distance error within $2\times10^{-6}\max(1,|d_{\rm ref}^2|)$, correct unused Morton slots, and zero Morton overflows; active-mask rejection must pass independently for both backends |

For deterministic cases, the other GPU implementation is never treated as ground truth. The
Python reference uses formulas derived independently from the GPU output. For the radial KNN
matrix, backend agreement over all 4096 queries is supplemented rather than replaced by brute
force: the first 32 queries are always exhaustively checked, and any later disagreement would
also be exhaustively adjudicated.

## Detailed test definitions

### Grid projection and optical depth

#### Particle placement

The 1D, 2D, and 3D grid cases put exactly one representative in every mesh cell and assign each
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

For `test_grid_1d`, the inactive coordinates must additionally be assigned exactly as

$$
x=0,
\qquad
z=\frac{\pi}{2},
\qquad
\ell_\theta=0.
$$

The expected cell measure is the annular area $\pi(R_o^2-R_i^2)$ rather than a Cartesian line
length. The same $5\times10^{-12}$ mass/density and $5\times10^{-11}$ optical-depth tolerances are
used.

#### Expected density

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

#### Expected optical depth

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

### Circular-orbit transport

Every representative begins at

$$
y=R=1,
\qquad
z=\frac{\pi}{2},
\qquad
(\ell_x,v_y,\ell_z)=(1,0,0),
$$

The 2D case distributes initial azimuths uniformly around the complete ring. The radial-only case
sets $x=0$ exactly and requires it to remain zero while $\ell_x$ still supplies centrifugal support. The test parameters
$p=2$ and $q=-1$ make the midplane pressure-support parameter vanish, so both gas and dust have
unit Keplerian angular speed. Drag therefore has no effect on the exact orbit.

The 2D analytical trajectory is

$$
x(t)=x_0+t\pmod{2\pi},
\qquad
y(t)=1,
\qquad
(\ell_x,v_y,\ell_z)=(1,0,0).
$$

For `test_orbit_1d`, the corresponding spatial trajectory is $x(t)=0$, with the same constant
$y$, $\ell_x$, and $v_y$, and with $z=\pi/2$, $\ell_z=0$ exactly

The selected GPU kernel advances to $T=2\pi$ using $N$ equal steps. The 2D validator wraps the azimuthal error
to $[-\pi,\pi)$ and combines errors in $x$, $y$, $\ell_x$, and $v_y$. It rejects nonfinite states
or $L_\infty\ge0.5$. The radial validator additionally checks exact inactive $z$ and $\ell_z$ and
requires $L_\infty<5\times10^{-13}$. Because this circular equilibrium is preserved to roundoff, refinement does
not expose the nominal temporal order: ratios of errors near machine precision fluctuate and may
produce negative formal orders. This case verifies equilibrium preservation and long-orbit phase
consistency, not general second-order convergence.

### Non-equilibrium eccentric orbit

`test_orbit_ecc_2d` complements the circular equilibrium with four eccentric trajectories at
initial mean anomalies $0.1$, $0.7$, $1.4$, and $2.2$. The QAV model activates a zero-drag
specialization while retaining the production staggered drift, spherical geometric force
calculation, boundary handling, and `ssa_transport` kernel. For $GM_S=a=1$ and $e=0.2$, the
independent reference solves

$$
E-e\sin E=M_0+t
$$

and evaluates

$$
y=1-e\cos E,
\qquad
x=\operatorname{atan2}\left(\sqrt{1-e^2}\sin E,\cos E-e\right),
$$

$$
\ell_x=\sqrt{1-e^2},
\qquad
v_y=\frac{e\sin E}{1-e\cos E},
\qquad
z=\frac{\pi}{2},
\qquad
\ell_z=0.
$$

The final time $T=1.3$ is not an integer orbital period, preventing phase errors from being hidden
by a return-time comparison. The validator reports the wrapped azimuthal error, complete stored
state, specific orbital energy

$$
\mathcal E=\frac12\left[\left(\frac{\ell_x}{y}\right)^2+v_y^2\right]-\frac1y,
$$

and angular momentum. Every state error must remain below $2\times10^{-2}$, the activation flag
must be true, and the final state $L_1$ order over $N=32,64,128,256$ must exceed 1.8. An independent
host replica predicts second-order errors, but native CUDA and ROCm records are still required
before quoting measured values.

### Reduced-gravity radiation orbit

`test_orbit_beta_2d` passes a zero optical-depth grid and unit taper through the production
`ssa_substep_1`/`ssa_substep_2` radiation split while retaining the QAV-only zero-drag force
update. Two monodisperse-in-trajectory size groups use $\beta=0.2$ and $0.1$, so both the global
radiation coefficient and its production $S_0/s$ scaling are exercised. For constant
$0<\beta<1$, each exact problem is Keplerian with

$$
\mu_{\rm eff}=(1-\beta)GM_S,
\qquad
n=\sqrt{\frac{\mu_{\rm eff}}{a^3}}.
$$

The validator advances Kepler's equation with this $n$ and replaces $GM_S$ by $\mu_{\rm eff}$ in
the exact radius, radial velocity, angular momentum, and energy. It requires the zero-drag,
zero-optical-depth, and unit-taper activation records, finite output, a maximum state error below
$2\times10^{-2}$, and final $L_1$ temporal order at least 1.8. A missing radiation factor would
therefore appear as a persistent phase and energy error rather than a one-step force discrepancy.

### Inclined three-dimensional orbit

`test_orbit_inc_3d` rotates the perifocal conic by fixed longitude of ascending node $\Omega$,
inclination $i$, and argument of periapsis $\omega$. The independent reference constructs

$$
\mathbf r'=
\begin{pmatrix}a(\cos E-e)\\a\sqrt{1-e^2}\sin E\\0\end{pmatrix},
\qquad
\mathbf v'=\frac{an}{1-e\cos E}
\begin{pmatrix}-\sin E\\\sqrt{1-e^2}\cos E\\0\end{pmatrix},
$$

applies $R_z(\Omega)R_x(i)R_z(\omega)$, and converts the Cartesian vectors independently to
$(x,y,z,\ell_x,v_y,\ell_z)$. This exercises polar motion and both stored angular momenta. The
full-3D build enables the required `DIFFUSION` configuration with $\nu=0$, but deliberately
schedules no diffusion kernel. The validator requires those activation records, finite states,
maximum state error below $3\times10^{-2}$, and final $L_1$ temporal order at least 1.8.

### Multi-step constant-coefficient drag path

`test_drag_path_1d` tests displacement as well as the already verified one-step velocity response.
A QAV-only coefficient specialization retains the production first/second half drifts and
`ssa_transport` launch while prescribing

$$
v_g=0.15,
\qquad
a=-0.08,
\qquad
t_s\in\{0.02,0.2,2.0\}.
$$

For $y_0=0.8$ and $v_0=0.25$, define $v_{\rm eq}=v_g+a t_s$. The exact solution at $T=1$ is

$$
v(T)=v_{\rm eq}+(v_0-v_{\rm eq})e^{-T/t_s},
$$

$$
y(T)=y_0+v_{\rm eq}T+t_s(v_0-v_{\rm eq})(1-e^{-T/t_s}).
$$

The velocity relaxation is exact for every numerical step, while the staggered two-half-drift
position is a trapezoidal composition and should converge at second order. The validator requires
velocity error below $2\times10^{-12}$, position error below $10^{-2}$, inactive coordinates and
momenta below $2\times10^{-14}$, an active-specialization flag, and final position $L_1$ order at
least 1.8. The three stopping times cover rapid, intermediate, and weak relaxation relative to the
one-unit trajectory; the existing frozen-response cases retain the more extreme one-step stiffness
ratios.

### Frozen drag and radiation responses

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

- `test_drag_1d`: the radial-only version of the zero-$\beta$ response
- `test_drag_2d`: $\beta=0$ and $\gamma_{\rm PR}=0$
- `test_radiation_1d`: the radial-only size-dependent radiation response
- `test_radiation_2d`: $\beta=\beta_0/(s/S_0)$ and $\gamma_{\rm PR}=0$
- `test_prdrag_1d`: the radial-only component-dependent P-R response
- `test_prdrag_2d`: the same size-dependent $\beta$ with
  $\gamma_{\rm PR}=\beta/c$, using $c=25$ in test units

The radiation tests supply a zero optical-depth array and unit startup taper. They therefore
isolate force and damping algebra from opacity reconstruction. The validator concatenates errors
in $\ell_x$, $v_y$, and $y$ and requires $L_\infty<2\times10^{-13}$. This checks stiff responses
from $\Delta t/t_s=0.078125$ through 10 without requiring explicit drag timesteps.

### Vertically integrated viscous flow

`test_viscflow_1d` calls the production particle initializer at eight radii with constant kinematic
viscosity. For the test parameters $p=2$, $q=-1$, and $\nu=0.02$, pressure support vanishes and the
vertically integrated Kanagawa target is

$$
v_{R,g}=-\frac{3\nu}{R}\left(p+\frac12\right)=-\frac{0.15}{R}.
$$

For a reference-size grain, the independent analytic Stokes number and steady no-backreaction
dust velocities are

$$
\mathrm{St}=\frac{\mathrm{STOKES}_0}{R^2},
\qquad
v_R=\frac{v_{R,g}}{1+\mathrm{St}^2},
$$

$$
v_\phi=R^{-1/2}-\frac12\mathrm{St}\,v_R,
\qquad
\ell_\phi=Rv_\phi.
$$

The case requires the initialized $R$, $v_R$, and $\ell_\phi$ to match these values to
$2\times10^{-13}$ and requires the inactive $x$, $z$, and $\ell_\theta$ fields to be exact. This
checks the radial `VISC_FLOW` closure and its use by production initialization; it does not yet
measure a long-time viscous mass flux.

### Imported radial gas scaling

`test_import_1d` stores a prescribed surface-density sequence

$$
\Sigma_{g,i}=\Sigma_0(1+0.1i)
$$

on the radial mesh and evaluates production helpers at selected cell centers. For a reference-size
grain, the independent Stokes reference is

$$
\mathrm{St}_i=\mathrm{STOKES}_0\frac{\Sigma_0}{\Sigma_{g,i}}.
$$

With the test's constant-$\alpha$ code-unit closure, the turbulent Reynolds reference is

$$
\mathrm{Re}_i=\mathrm{REYNOLDS}_0\frac{\Sigma_{g,i}}{\Sigma_0},
\qquad
\mathrm{Re}_i^{-1/2}=\left(\mathrm{Re}_i\right)^{-1/2}.
$$

The case passes only when every GPU Stokes and inverse-Reynolds value agrees with these independent
references to absolute error below $2\times10^{-13}$. It specifically detects treating imported
$\Sigma_g$ as a volume density or retaining the analytical gas profile inside collision turbulence

### Cylindrical stochastic diffusion

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

#### Two-dimensional azimuthal diffusion

`test_diffusion_2d` activates only $D_x$; radial diffusion is suppressed with an enormous Schmidt
number. At $R=1$,

$$
\Delta x=\sqrt{2D\Delta t}\,\xi,
\qquad
E[\Delta x]=0,
\qquad
\mathrm{Var}(\Delta x)=2D\Delta t,
$$

where $\xi\sim\mathcal N(0,1)$. Angular displacements are wrapped to $[-\pi,\pi)$ before their
moments are measured.

#### One-dimensional radial diffusion

`test_diffusion_1d` activates only $D_R=D$. At $R=1$, the exact one-step process is

$$
\Delta R=D\Delta t+\sqrt{2D\Delta t}\,\xi_R.
$$

The validator applies the same six-standard-error limits to its mean and variance as the 3D radial
component. It also reconstructs $v_\phi=\ell_x/R$ and requires $v_\phi=0.7$, $v_R=0.2$ while
$x=0$, $z=\pi/2$, and $\ell_z=0$ remain exact

The production radial diffusion policy is identical in 1D, 2D, and 3D: repeatedly reflect the
spherical radius into $[Y_{\min},Y_{\max}]$. For vertically integrated models $y=R$, and radial
reflection does not change azimuth. The configured diffusion ensembles are far enough from the
boundaries that reflection is not the source of their analytical moments

#### Three-dimensional radial and vertical diffusion

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
\mathrm{Var}(\Delta R)
=\mathrm{Var}(\Delta Z)
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

#### Three-dimensional settling plus diffusion

`test_settle_diffuse_3d` isolates the cylindrical vertical process

$$
dZ=-\gamma Z\,dt+\sqrt{2D}\,dW
$$

with $\gamma=0.7$, $D=0.02$, $Z_0=0.25$, and final time $T=1$. A test-only kernel applies the
documented Euler settling map, while the unmodified production `diffusion_pos` kernel supplies the
vertical Gaussian increment and spherical-coordinate reprojection. The azimuthal and cylindrical
radial diffusivities are suppressed through large Schmidt numbers.

For $n=N$ steps of size $h=T/N$, define

$$
a=1-\gamma h.
$$

The implemented recurrence is exactly

$$
Z_{k+1}=aZ_k+\sqrt{2Dh}\,\xi_k,
\qquad \xi_k\sim\mathcal N(0,1),
$$

so the independent discrete reference is

$$
E[Z_n]=a^nZ_0,
$$

$$
\operatorname{Var}(Z_n)
=2Dh\frac{1-a^{2n}}{1-a^2}.
$$

Each resolution uses 65,536 representatives. The validator applies conservative six-standard-error
limits to the sample mean and variance and a two-sided Kolmogorov--Smirnov test against the complete
Gaussian law. The CDF gate receives one third of a familywise significance level of 0.01; the much
stricter moment gates keep the combined false-rejection budget below that level. It also reconstructs
$(v_R,v_x,v_Z)$ after every accumulated basis transformation and
requires the final physical velocity to remain unchanged to $2\times10^{-11}$; $R$ and $x$ must
remain unchanged to $2\times10^{-12}$. The configured radial and polar boundaries lie sufficiently
far into the Gaussian tails that the reference is the unbounded recurrence rather than a reflected
process. This is a coefficient-specialized statistical test of the production diffusion kernel,
not a claim that the unrestricted disk settling force is globally linear.

### Continuous finite-domain initialization

`test_initial_3d` exercises the host initialization path with `MULTISIZE` and `DIFFUSION`. It uses
$N_Y=96$, $N_P=65\,536$, the broad size interval

$$
s\in[0.05,6.4],
$$

and a finite polar domain $|z-\pi/2|\le0.01$. Approximately one third of the representatives use
each endpoint size and the remaining third use

$$
s_{\rm mid}=\sqrt{s_{\min}s_{\max}}.
$$

The midpoint lies halfway between the two central entries of the logarithmic 128-knot size axis,
so it directly exercises interpolation of both the conditional radial CDF and the contained-mass
table. All three populations retain enough samples for separate distribution tests.

At each cylindrical radius, the Python validator independently intersects the radial shell and
polar wedge to recover the allowed vertical intervals. For every interval it evaluates

$$
F_Z(R,s)=
\Phi\left(\frac{Z_{\rm hi}}{H_d(R,s)}\right)
-\Phi\left(\frac{Z_{\rm lo}}{H_d(R,s)}\right),
$$

reconstructs the convolved surface-density profile, and integrates

$$
I(s)=2\pi\int R\Sigma_d(R)F_Z(R,s)\,dR
$$

on the production size axis. The 128 returned containment masses must agree with this independent
reference to relative $L_\infty<2\times10^{-12}$. The validator independently interpolates the two
central positive mass entries linearly in $\log I$ to reconstruct the expected normalization of
the midpoint population. Its
relative error and the finite representative normalization error must both remain below
$5\times10^{-12}$, with the latter written as

$$
\left|\frac{\sum_i m_g(s_i)N_i-M_{\rm dust}}{M_{\rm dust}}\right|<5\times10^{-12}.
$$

For spatial sampling, the validator maps every sampled radius through its independently rebuilt
radial CDF and every sampled height through the appropriate truncated-Gaussian conditional CDF.
For the midpoint population, the radial reference is independently formed by interpolating the
two adjacent normalized CDFs exactly as specified by the production size discretization. Each of
the three transformed populations must be uniform by a two-sided KS statistic below
$6/\sqrt{N_j}$ for its own population count $N_j$, and every converted spherical position must
remain inside the configured domain. The validator also reports, without using them as pass/fail
criteria, the midpoint CDF and contained-mass interpolation errors relative to a CDF integrated
directly at $s_{\rm mid}$.

The requested test resolution changes `N_Z` while leaving all physical parameters, auxiliary
radial resolution, size table, and random seed fixed. The model runner compares `initial`,
`mass_bank`, and `mass_summary` byte for byte across those builds. This directly detects any
accidental return of polar-cell-center quadrature to the initialization path. A single-resolution
invocation still validates the analytical and statistical criteria but cannot establish the
cross-`N_Z` byte comparison.

### Deterministic radial absorption path

`test_absorb_path_1d` exercises the two production radiation-split transport kernels rather than
invoking the boundary helper directly. A model-local transport specialization removes gravity and
drag but retains the production staggered drift, midpoint and endpoint boundary calls,
active-state test, and absorbed-particle sentinel. Two representatives move toward the inner and
outer faces, and two remain inside the domain as an active collision pair. For either crossing,

$$
t_{\rm hit}=\frac{y_{\rm face}-y_0}{v_y}.
$$

After every accepted step the driver records the first endpoint at which the stored radius becomes
the production inactive value $y=0$. Since an endpoint is checked no later than the first step that
contains the exact crossing, the deterministic acceptance condition is

$$
0\le t_{\rm record}-t_{\rm hit}\le\Delta t.
$$

This bounded delay tends to zero with the refined step even though its ratio to $\Delta t$ need not
be monotone when a fixed crossing time moves through different step phases. The test therefore
checks the analytical bound at every resolution instead of fitting a misleading order to the
ceiling operation.

At $T=1$, both crossed representatives must have

$$
y=0,
\qquad z=\frac{\pi}{2},
\qquad \ell_x=v_y=\ell_z=0,
$$

while each surviving path remains the exact translation $y(T)=y_0+v_yT$. The final state is then
passed through the production dynamical-rate, density-deposition, complete optical-depth, and
collision-site/rate paths. The required collision mask is $(0,0,1,1)$; inactive collision rates
and radii must be exactly zero, active rates must be finite and positive, and Morton traversal must
not overflow. The finite-volume density integral must equal the two units of active represented
mass. The separate KNN edge suite places nearer inactive records around a query and requires both
KD-tree and Morton searches to return only active records. Together, these cases connect the
production $y=0$ sentinel to its collision mask and then verify both search consumers.

### Deterministic boundary policies

The four fixed-resolution boundary models call the production `_apply_transport_boundary` and
`_apply_diffusion_boundary` device helpers directly. No time integrator or test-local
reimplementation sits between the configured endpoint and the helper being checked. Each model
stores the same seven endpoint states

$$
(x,y,z,\ell_x,v_y,\ell_z),
$$

for a total of 42 checked scalars. The seven inputs and their intended roles are:

| Endpoint | Helper | Constructed excursion | Required map |
|---:|---|---|---|
| 0 | diffusion | above the azimuthal upper face and below the radial and, in 3D, lower polar faces | periodic azimuth plus lower-face reflection |
| 1 | diffusion | below the azimuthal lower face and above the radial and, in 3D, upper polar faces | periodic azimuth plus upper-face reflection |
| 2 | diffusion | $2.25$ radial-domain widths below $Y_{\min}$ | repeated reflection until the radius lies inside the domain |
| 3 | transport | above the azimuthal upper face and below $Y_{\min}$ | wrap azimuth, then absorb at the inner radial boundary |
| 4 | transport | below the azimuthal lower face and above $Y_{\max}$ | wrap azimuth, then absorb at the outer radial boundary |
| 5 | transport | above the azimuthal upper face and below the lower polar face | lower-polar absorption in 3D; inactive-$z$ locking in 1D/2D |
| 6 | transport | below the azimuthal lower face and above the upper polar face | upper-polar absorption in a full disk, upper-midplane reflection in a half disk, or inactive-$z$ locking in 1D/2D |

The Python reference does not call either CUDA helper. It independently applies

$$
x_{\rm wrap}
=X_{\min}+\bigl[(x-X_{\min})\bmod (X_{\max}-X_{\min})\bigr]
$$

when azimuth is active and repeatedly folds a diffusion coordinate about its lower or upper face
until it lies in the closed interval. For a radial or absorbing polar transport exit, the expected
sentinel state is

$$
y=0,\qquad z=\frac{\pi}{2},\qquad
\ell_x=v_y=\ell_z=0.
$$

The one-dimensional case requires $x$ to remain at its inactive reference, $z=\pi/2$, and
$\ell_z=0$. The two-dimensional case uses the narrow wedge $[-0.1,0.1)$ so a wrap cannot be mistaken
for a full-$2\pi$ assumption. The full 3D case absorbs transport endpoints beyond either polar face.
The half-disk case sets $Z_{\max}=\pi/2$, absorbs through the lower polar face, and reflects an upper
endpoint as

$$
z\leftarrow\pi-z,
\qquad
\ell_z\leftarrow-\ell_z.
$$

For an inactive polar coordinate, the helper must restore $z=\pi/2$ and $\ell_z=0$ without
changing the active radial state. Diffusion reflection changes position only; the test therefore
also detects an unintended momentum sign change.

A model passes only if all 42 returned scalars are finite and their combined absolute
$L_\infty$ error is below $2\times10^{-13}$. There is no resolution-order requirement because
these are direct deterministic maps with fixed inputs. They prove the configured endpoint
policies, including repeated diffusion folding, but do not prove that a finite transport or
stochastic step detects every continuous-time boundary crossing. That stronger claim requires
the boundary-event convergence tests described below.

### Collision helper mathematics

The collision cases execute a test-local GPU kernel that directly calls the production
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

For `test_collision_1d`, the reference is instead the exact accessible annular area

$$
A_K(R,a)=\pi\left[
\min(Y_{\max},R+a)^2-
\max(Y_{\min},R-a)^2
\right].
$$

The test evaluates an interior ring, inner- and outer-boundary clipping, and a radius large enough
to reach both radial boundaries

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

The validator requires the maximum error across the stored measures and three kernel values to
remain below $2\times10^{-13}$. These cases catch incorrect dimensions, missing cap corrections, and an
incorrect additive-kernel definition. They do not test neighbor identity, periodic images,
normalization by an actual KNN radius, partner selection, coagulation events, or fragmentation.

Thus a collision-helper case passes only when all seven GPU values agree with the
independently evaluated CPU formulas above to the stated absolute tolerance. The other backend is
not used as the reference, and resolution convergence is neither required nor implied because this
is a direct device-helper test with fixed inputs.

### Guarded production collision-chain runtime

`test_colchain_2d` is a CUDA-only qualification case that builds the actual production
`src/cuda/swarm/swarm_runtime.cu` with `COL_CHAIN`, `COLLISION_MORTON`, $N_P=2048$, $N_K=200$, and
a $32\times32\times1$ radial–azimuthal grid. Unlike the collision-helper cases, it constructs the
Morton hierarchy, caches physical neighbors, selects controller baths, advances exact local event
chains with continuation support, writes production checkpoints, and runs through the production
output path.

The wrapper starts the executable twice from the same deterministic initialization and validates
both the particle and RNG checkpoints. With particle size $s_i$ and represented grain number $N_i$,
the conserved mass proxy is

$$
\mathcal M_3=\sum_iN_is_i^3,
$$

because the constant factor $\pi\rho_0/6$ cancels in the relative comparison. A run passes only if

- every stored particle scalar is finite and every final $s_i,N_i$ is positive
- at least one particle size changes, proving the event path was exercised
- $|\mathcal M_3^{\rm final}-\mathcal M_3^{\rm initial}|/\mathcal M_3^{\rm initial}le2\times10^{-12}$
- `variables.txt` records the chain integrator and Morton search
- a nonempty RNG checkpoint is written
- the two fresh runs have identical final-particle and RNG SHA-256 digests

This is an integration and reproducibility gate, not an analytical collision-distribution test. It
does not by itself establish bath-size convergence, force an event-cap continuation, validate the
custom physical kernel or fragmentation, exercise imported gas or operator coupling, or determine
production-scale memory and performance.

## KNN implementation and periodic-ghost comparison

`qav/cuda/swarm/test_knn/` and `qav/rocm/swarm/test_knn/` provide the backend KNN drivers;
backend-neutral result analyzers remain under `qav/comm/swarm/test_knn/`. The group compiles four drivers:

- ordinary smooth, ring, and clump benchmarks in 2D and 3D, plus a collinear radial distribution
- adversarial edge cases for ties, coincident points, cutoff equality, sparse neighborhoods, Morton
  split planes, and active-mask rejection of nearer absorbed records in both search backends
- correctness-only periodic query-image cases covering both wedge faces, full-$2\pi$ geometry,
  image overlap, deduplication, and both sides of the shared $10^{-6}$ near-full-domain cutoff
- periodic wedge benchmarks comparing the production three-copy KD-tree and compact
  boundary-ghost Morton owner, including a deliberately narrow seam clump that forces
  physical-identifier deduplication and the production $3N_K$ fallback rather than the ordinary
  disjoint-image shortcut

The radial distribution writes every point as $(R,0,0)$ and compares both backends with exhaustive
one-dimensional brute force. The Morton search internally uses its two-coordinate container for
these collinear points, so the JSON retains `dimension: 2` as the search embedding and adds
`physical_dimension: 1`; the case and file are named `radial_1d_*`. The adversarial driver also
includes inner, middle, and outer radial queries on a collinear point set. Production translation
units for `test_collision_1d` are compiled with both search backends during the KNN suite

For every configured brute-force query, the CPU reference enumerates every physical particle,
applies the exact minimum-image wedge geometry, sorts by $(d^2,\mathrm{id})$, and retains the first
$N_K$. If the two GPU methods disagree outside the initially configured brute-force subset, the
validator also checks every disagreement against exhaustive search. A case fails on an incorrect
identifier, a distance outside the single-precision tolerance, a missing valid neighbor, or a
traversal-stack overflow. Backend-to-backend list differences remain diagnostic but do not fail a
case when each list is independently equivalent to the brute-force result, because separate
single-precision image rotations can exchange distance-equivalent boundary neighbors.
When the GPU and brute-force lists contain the same particle indices, the validator compares each
distance by its matching index rather than by independently rounded distance rank; genuinely
different index sets retain the stricter boundary-tie adjudication.

### KNN ground truth and pass criteria

The KNN family contains four distinct levels of evidence; the word `PASS` has a different precise
meaning at each level.

| Test family | Ground truth | Required for `PASS` |
|---|---|---|
| ordinary adversarial cases | exhaustive CPU enumeration of every candidate | exact selected identifiers and valid count, squared-distance agreement, and zero traversal overflows |
| periodic adversarial cases | exhaustive CPU minimum-image enumeration over the physical particle set | the same neighbor conditions, no repeated physical identifier, the expected number of query images, and zero overflows |
| ordinary smooth/ring/clump/radial matrix | exhaustive CPU search for a configured subset and every KD-tree/Morton disagreement; the remaining queries have differential agreement but no independent ground truth | zero KD-tree/Morton query disagreements, zero brute-force failures, zero record/geometry failures, and zero Morton traversal overflows |
| periodic wedge matrix | exhaustive CPU minimum-image search for a configured subset and every KD-tree/Morton disagreement | every brute-force-checked KD-tree and compact-ghost Morton result is valid, with zero Morton overflows; backend differences are allowed only when each differing result is independently equivalent to brute force |

For the non-periodic CPU reference, every point inside the inclusive cutoff
$d^2\le q^2$ is collected, sorted lexicographically by $(d^2,\mathrm{id})$, and truncated to
$N_K$. The ordinary adversarial and matrix comparisons accept a squared-distance error no larger
than

$$
2\times10^{-6}\max(1,|d^2_{\rm ref}|).
$$

The adversarial cases also require unused output slots to be exactly represented by identifier
`-1` and infinite distance. They exercise equal-distance identifiers, coincident points, a point
exactly on the cutoff, fewer than $N_K$ valid points, and points on Morton split planes in both 2D
and 3D. The validator independently sorts valid returned pairs on the host before comparison.

For a periodic wedge, the CPU reference evaluates the original query and its two adjacent images,
retains the minimum squared distance to each physical identifier, applies the inclusive cutoff,
sorts by $(d^2,\mathrm{id})$, and truncates to $N_K$. The periodic adversarial driver requires an
exact identifier match and the same scaled $2\times10^{-6}$ distance tolerance. The large wedge
matrix uses an absolute $2\times10^{-6}$ squared-distance tolerance. If separately rounded CPU and
GPU rotations exchange identifiers at the cutoff, the matrix accepts the substitution only when
the corresponding ranked squared distances differ by at most $10^{-7}$; it records such neighbors
as tie-equivalent rather than silently ignoring them.

The ordinary matrix samples four deliberately different geometries:

- `smooth` fills an annulus with uniform surface number density and, in 3D, a Gaussian vertical
  spread
- `ring` concentrates particles around $R=1$ with a narrower vertical spread
- `clump` places 80 percent of the particles in a compact Cartesian Gaussian clump and leaves a
  diffuse background
- `radial` places all particles on the collinear set $(R,0,0)$ with
  $R\in[0.5,1.5]$, thereby testing the physical 1D use of the 2D search container

The periodic matrix uses a wedge of width $\pi/2$ for smooth, ring, interior-clump, and
seam-clump populations. It adds a width-$0.2$ seam clump whose periodic images overlap within the
cutoff; this forces stable-identifier deduplication in the KD-tree and the $3N_K$ Morton fallback.
The compact Morton owner is also checked record by record: every stored Cartesian record must be
the correct physical point or one-wedge rotation of its source identifier.

The periodic adversarial driver now contains 14 cases. In both 2D and 3D it tests one domain whose
width lies within $10^{-6}$ of $2\pi$ and must use the full-period path, and one just outside that
tolerance that must construct periodic images. The independently evaluated reference uses the same
numerical threshold, while expected image counts verify that the intended branch was actually
selected.

The ordinary and wedge matrices use 4096 quality queries and brute-force the first 32 by default.
Every additional backend-disagreement query is also brute-forced, so KD-tree agreement is not used
to adjudicate a disagreement. This is strong differential coverage but is not exhaustive CPU
validation of all 4096 queries when all GPU backends make the same choice. The adversarial drivers
provide the complementary fully exhaustive small-set checks.

### Latest archived source-matched KNN evidence

The KNN branches of the 2026-08-23/24 CUDA and ROCm campaigns used $N_K=200$, a search cutoff of
approximately $0.1$, 4096 quality queries, 32 unconditional exhaustive CPU queries, and
$N_P=10^5$. Each backend compiled all four standalone KNN drivers and linked the real
`test_collision_1d`, `test_collision_2d`, and `test_collision_3d` production translation units
once with KD-tree search and once with Morton search. Every component and aggregate manifest
reports `"passed": true`, and the cross-backend report confirms equal coverage.

| Component | CUDA | ROCm |
|---|---:|---:|
| edge adversarial driver | 15/15 PASS | 15/15 PASS |
| periodic adversarial driver | 14/14 PASS | 14/14 PASS |
| ordinary smooth/ring/clump/radial matrix | 7/7 PASS | 7/7 PASS |
| compact-ghost periodic-wedge matrix | 10/10 PASS | 10/10 PASS |
| production collision links | 6/6 PASS | 6/6 PASS |

Across the seven ordinary cases on each backend there were no KD-tree/Morton query disagreements,
brute-force mismatches, Morton-record mismatches, record-geometry mismatches, or traversal-stack
overflows. The largest squared-distance discrepancy was $9.31\times10^{-10}$ on CUDA and
$1.86\times10^{-9}$ on ROCm.

The periodic-wedge matrices contained six CUDA and ten ROCm topology disagreements. Every one was
independently brute-forced, and both searches agreed with an admissible minimum-image reference.
There were no brute-force, record-geometry, or overflow failures. The largest squared-distance
discrepancy was $2.65\times10^{-8}$ on CUDA and $3.73\times10^{-9}$ on ROCm, both within the
single-precision periodic tolerance.

Timing is diagnostic and is not part of `passed`. Define the query-time ratio

$$
S_{\rm query}=\frac{t_{\rm KD}}{t_{\rm Morton}}
$$

where values above one favor Morton. The current measurements are

| Backend | Ordinary $S_{\rm query}$ | Wedge $S_{\rm query}$ | Ordinary Morton/KD memory | Wedge Morton/KD memory |
|---|---:|---:|---:|---:|
| CUDA A100 | 0.258--0.793 | 0.669--1.274 | 0.718--0.756 | 0.247--0.491 |
| ROCm MI300A | 0.386--0.865 | 0.668--1.871 | 0.718--0.756 | 0.247--0.491 |

The compact ghost-record count ranged from $1.0248N_P$ for an interior clump to
$2.00196N_P$ for a narrow seam clump, compared with the KD-tree's fixed $3N_P$ wedge records.
These are search-batch and owned-hierarchy measurements, not end-to-end collision or simulation
speedups.

The memory ratios above compare owned search hierarchies while treating query coordinates as common
inputs. Production also owns backend-specific support arrays, so total simulation VRAM must be
measured in the production executable rather than reconstructed from these ratios.

The `--full` path is configured to add $N_P=10^6$ and $10^7$, producing 21 ordinary and 30 wedge
cases. The archived source-matched manifests document the compact $N_P=10^5$ campaign only, so no
large-particle timing is promoted as archived evidence.

The standalone `*_query_ms` fields measure one complete batch of $N_P$ search queries using native
CUDA or HIP events after an untimed warm-up and report the mean of the requested repeats. They include tree
traversal, top-$K$ maintenance, and the checksum write, but exclude index construction, allocation,
host-device transfers, brute-force validation, collision-rate physics, and collision events. The
corresponding `*_build_ms` fields report index construction separately. These search-only timings
must not be presented as end-to-end collision-operator or simulation timings.

The current elapsed-time tables contain only production search structures: the periodic KD-tree
with particle images and compact-ghost Morton with block-parallel sorted top-$K$. Query-image
Morton remains a correctness-only adversarial driver and contributes no timing, memory, or wedge
benchmark JSON fields.

The promoted QA benchmark uses the production `index_old` heap, and its periodic-wedge path
deduplicates physical particles represented by multiple KD-tree images. The optimized heap stores
`index_old` directly and activates its duplicate scan only
for geometrically overlapping image neighborhoods; the JSON field `kd_deduplicate` records which
path each wedge case exercises.

The validated source also uses level-aware Morton AABB padding and block-parallel pair-propensity
evaluation with serial ordered accumulation. The standalone QA and production KD paths now include
the same generic `index_old_heap` from each backend's
`inc/<backend>/swarm/kdtree/index_heap.cuh`, eliminating the
former test transcription. The clean topology archive validates the shared search components;
the production pair-rate optimization is compiled by the suite but still needs an end-to-end
production timing comparison.

## Latest archived native and cross-backend evidence

### Complete source-matched archives from 2026-08-23/24

The complete CUDA and ROCm campaigns each passed all 31 analytical/statistical models and all 75
requested resolution records. Every model manifest reports `passed: true`; all numeric JSON values
are finite. On each backend, the compact KNN suite also passed all four standalone builds, 7/7
ordinary cases, 15/15 edge cases, 14/14 periodic cases, 10/10 wedge cases, and 6/6 production
collision links.

The six added four-resolution sequences produced the following strongest summary diagnostics:

| Model | Convergence or statistical result | Finest or worst retained diagnostic |
|---|---|---|
| eccentric 2D orbit | state orders 2.0004, 2.0001, 2.0000 | state $L_\infty=1.01\times10^{-6}$; energy error $2.32\times10^{-7}$ |
| radiation-modified 2D orbit | state orders 2.0003, 2.0001, 2.0000 | state $L_\infty=8.18\times10^{-7}$; energy error $1.56\times10^{-7}$ |
| inclined 3D orbit | state orders 2.0005, 2.0001, 2.0000 | state $L_\infty=2.35\times10^{-6}$; energy error $3.54\times10^{-7}$ |
| multi-step 1D drag | position orders 1.9626, 1.9903, 1.9976 | position $L_\infty=6.46\times10^{-6}$ |
| deterministic 1D absorption | all state, sentinel, collision-mask, collision-rate, and optical-depth gates pass | finest absorption-time error $4.69\times10^{-4}$; active-mass error $8.88\times10^{-16}$ |
| 3D settling--diffusion equilibrium | all four ensemble distributions pass | KS $=0.00330$--$0.00546$ below the $0.00699$ limit |

The continuous 3D initialization regression also remained independent of the nominal polar
resolution: the four initialized arrays and mass-bank checks passed, with domain-mass relative
$L_\infty=3.71\times10^{-15}$, radial PIT KS $=0.00522$, vertical PIT KS $=0.00675$, and
middle-size contained-mass interpolation error $6.61\times10^{-9}$.

In the compact CUDA KNN archive, ordinary KD-tree/Morton queries had no topology mismatches and a
largest squared-distance discrepancy of $9.31\times10^{-10}$. The wedge matrix recorded at most
four topology disagreements in a case; every disagreement passed independent brute-force
adjudication, with no record-geometry mismatch or traversal-stack overflow. The largest wedge
squared-distance discrepancy was $2.65\times10^{-8}$, within the documented single-precision
periodic tolerance.

The two campaigns use the same source fingerprint recorded in [`README.md`](README.md). The
cross-backend comparator found all 75 records on both sides, equal tier metadata, no missing
records, and zero acceptance mismatches. The analytical metrics partition into 57 publication and
18 release records.

| Group | Builds or cases | CUDA | ROCm |
|---|---:|---:|---:|
| grid deposition and optical depth in 1D, 2D, and 3D | 12 builds | PASS | PASS |
| orbit, drag, viscous flow, absorption, and transport | 31 builds | PASS | PASS |
| diffusion and settling--diffusion in 1D, 2D, and 3D | 16 builds | PASS | PASS |
| continuous finite-domain initialization | 4 builds | PASS | PASS |
| radiation and P-R drag in 1D and 2D | 4 builds | PASS | PASS |
| full-disk, wedge, and half-domain boundary maps | 4 builds | PASS | PASS |
| collision and imported-gas helpers in 1D, 2D, and 3D | 4 builds | PASS | PASS |

The detailed KNN component counts, correctness criteria, memory ratios, and timing ranges are
reported once in “Latest archived native KNN evidence” above.

Sixteen stochastic records are compared by pass state and acceptance metadata rather than
requiring equal vendor RNG realizations. All other deterministic metric records use $10^{-5}$
relative and $10^{-11}$ absolute tolerances. The KNN comparison found equal component coverage
and passing aggregate manifests on both backends. Independent local archive checks reproduced
`PASS` for both downloaded archives without invoking either GPU runtime.

The common `all` group is the current publication-and-release baseline. Same-backend restart,
deliberate nonfinite injection, large-particle KNN, and performance measurements are qualification
branches and are not included in these counts. Refresh them separately when those claims are
needed. Vendor RNG-state files must only be tested by same-backend restart cases.

### Qualification branches

The complete campaign does not refresh same-backend restart or deliberate failure injection.
Run those qualification-only branches independently when their claims are needed:

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group chain --target sm_80
python3 qav/rocm/swarm/test_common/run_suite.py --group restart --res 32 --target gfx942
python3 qav/rocm/swarm/test_common/run_suite.py --group failure --target gfx942
```

Focused evidence is stored below `groups/GROUP/`, so it cannot replace the common `all` archive.
The complete backend-neutral transfer and comparison workflow is specified in `qav/README.md`.

The source-matched native `sm_80` chain qualification completed on 2026-08-26. Its two fresh
production-runtime executions each changed 516 of 2048 representatives, retained finite positive
states, and reported zero relative drift in $\mathcal M_3$. Their final particle checkpoints were
byte-identical with SHA-256
`bbdb178cf729b42767c27542159319aa4ba79b6193485220a8ac6727df051973`; their RNG checkpoints were
also byte-identical with SHA-256
`2dca612c5d133cc44f5eb95fb14d71c9f5ec9cbcc21097c8b8288737c5c95e2c`. The aggregate and component
manifests both report `passed: true`.

The accompanying CUDA legacy-collision regression also passed all four $N=32$ cases after the RNG
commit correction. Maximum $L_\infty$ errors were zero in 1D, $2.78\times10^{-17}$ for imported-gas
scaling, $2.78\times10^{-17}$ in 2D, and $6.94\times10^{-18}$ in 3D. This confirms that the guarded
integration and RNG housekeeping did not change the existing collision-helper results.

### Recorded initialization results

The complete CUDA and ROCm archives include `test_initial_3d` at

$$
N_Z=32,64,128,256.
$$

All four builds passed, and the model manifest reports
`"polar_resolution_independent": true`. The initialized particle arrays, 128-entry domain-mass
bank, and mass summary were byte-identical across all four values of `N_Z`. Every sampled point
remained in the spherical radial-polar domain. The worst metrics, identical at every resolution,
were

| Quantity | Recorded value | Acceptance limit |
|---|---:|---:|
| domain-mass relative $L_\infty$ | $3.71\times10^{-15}$ | $2\times10^{-12}$ |
| mass-normalization relative error | $2.16\times10^{-16}$ | $5\times10^{-12}$ |
| represented-mass relative error | $4.76\times10^{-16}$ | $5\times10^{-12}$ |
| radial PIT KS | $5.22\times10^{-3}$ | $4.06\times10^{-2}$ |
| vertical PIT KS | $6.75\times10^{-3}$ | $4.06\times10^{-2}$ |

For the geometric-mean grain size, interpolation between the two central size knots differed from
a directly integrated midpoint reference by $4.07\times10^{-9}$ in CDF $L_\infty$ and
$6.61\times10^{-9}$ in contained mass. These are fidelity diagnostics rather than acceptance
thresholds. Both backends accepted the same metrics; the per-resolution JSON, metadata,
environment, and manifest files are the machine-readable record.

### Deterministic accuracy and conservation

The largest recorded $L_\infty$ error and mass drift over every tested resolution were

| Calculation | Maximum $L_\infty$ | Maximum $|\Delta M_d/M_d|$ |
|---|---:|---:|
| 1D deposited density | $3.93\times10^{-14}$ | $6.66\times10^{-15}$ |
| 1D optical depth | $8.66\times10^{-15}$ | — |
| 2D deposited density | $5.46\times10^{-14}$ | $6.66\times10^{-15}$ |
| 2D optical depth | $1.38\times10^{-14}$ | — |
| 3D deposited density | $1.45\times10^{-13}$ | $2.55\times10^{-15}$ |
| 3D optical depth | $1.53\times10^{-14}$ | — |
| circular-orbit state | $1.73\times10^{-14}$ | — |
| frozen drag response | $1.39\times10^{-17}$ | — |
| viscous-flow initialization | $2.22\times10^{-16}$ | — |
| frozen radiation response | $6.94\times10^{-18}$ | — |
| frozen P-R response | $2.22\times10^{-16}$ | — |
| 1D diffusion velocity reprojection | $1.11\times10^{-16}$ | — |
| 2D diffusion velocity reprojection | $2.22\times10^{-16}$ | — |
| 3D diffusion velocity reprojection | $2.22\times10^{-16}$ | — |
| imported-gas scaling | $2.78\times10^{-17}$ | — |
| boundary policies | $2.78\times10^{-17}$ | — |
| 1D collision helpers | $0$ | — |
| 2D collision helpers | $2.78\times10^{-17}$ | — |
| 3D collision helpers | $6.94\times10^{-18}$ | — |

These errors are consistent with floating-point evaluation and accumulation roundoff and remain
well inside their validator thresholds. The orbit's printed formal orders, $0.1096$, $-1.5315$,
and $-0.3951$, only compare roundoff-level errors and therefore carry no convergence meaning.
Likewise, the exact grid-deposition errors can grow slowly with resolution because more
floating-point contributions are accumulated; these cases test exact finite-volume geometry,
indexing, and conservation at several grid sizes rather than a truncation-error convergence order.

### Diffusion moment statistics

For each diffusion ensemble, define the normalized discrepancy as the absolute measured moment
error divided by its six-standard-error acceptance limit. A value below one passes. For the 3D
case, the table reports the larger ratio between the active $R$ and $Z$ directions.

| Model | $N$ | $N_P$ | mean-error fraction | variance-error fraction |
|---|---:|---:|---:|---:|
| `test_diffusion_1d` | 32 | 16,384 | 0.207 | 0.012 |
| `test_diffusion_1d` | 64 | 65,536 | 0.162 | 0.230 |
| `test_diffusion_1d` | 128 | 262,144 | 0.043 | 0.255 |
| `test_diffusion_1d` | 256 | 1,048,576 | 0.084 | 0.213 |
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
compiled problem sizes and each native platform. The CUDA and ROCm component directories preserve
the machine-readable evidence for their respective campaigns.

## Running and archiving the suite

Each native suite requires its backend toolchain, Python 3.9 or newer, and NumPy. Optional
GPU diagnostic commands are recorded when available but do not replace numerical validation.

From the repository root, run a short workflow check with

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group all --quick
```

This performs the 45 analytical-suite builds, compiles the four KNN drivers, links the 1D, 2D,
and 3D production collision sources with both backends, and then runs the
$10^5$-particle KNN matrix.

Run the complete default matrix with

```bash
python3 qav/cuda/swarm/test_common/run_suite.py \
    --group all \
    --res 32 64 128 256
```

The current complete command performs 75 analytical-suite builds plus four KNN driver builds and
six production backend/geometry links.
Individual groups can be selected with

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group grid      --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group transport --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group diffusion --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group initialization --res 32 64 128 256
python3 qav/cuda/swarm/test_common/run_suite.py --group radiation --res 32
python3 qav/cuda/swarm/test_common/run_suite.py --group boundary  --res 32
python3 qav/cuda/swarm/test_common/run_suite.py --group collision --res 32
python3 qav/cuda/swarm/test_common/run_suite.py --group knn       --res 32
python3 qav/cuda/swarm/test_common/run_suite.py --group chain     --target sm_80
```

The `chain` group is a CUDA-only qualification branch and is intentionally excluded from `all`
until the same production method exists and passes natively on ROCm.

The radial implementation can be isolated before the full regression with

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group radial --res 32 64 128 256
```

In this command, the KNN entry is deliberately radial-only. It builds the ordinary and edge
drivers, links only `test_collision_1d` with KD-tree and Morton backends, runs the collinear radial
matrix plus radial-line and inactive-particle rejection checks, and skips the unrelated generic
2D/3D, periodic, and wedge matrices. Its default result is
`qav/logs/swarm/cuda/groups/radial/test_knn/radial_1d_N100000.json`; `--knn-full` adds the $10^6$- and
$10^7$-particle radial cases in the same directory.

The KNN group can also be run directly:

```bash
python3 qav/cuda/swarm/test_knn/run.py
```

Add the million- and ten-million-particle matrices with either

```bash
python3 qav/cuda/swarm/test_knn/run.py --full
python3 qav/cuda/swarm/test_common/run_suite.py --group knn --knn-full
```

These commands build and test the sole Morton path, the block-parallel sorted top-$K$ merge, against
the KD-tree and independent brute-force references. The `--full` matrix contains 21 ordinary cases
and 30 periodic-wedge cases across $N_P=10^5$, $10^6$, and $10^7$; the $10^7$ cases are
substantially more expensive and require correspondingly more GPU memory. The runners print the KD-tree and
Morton query batch times and their ratio. Standalone JSON, manifests, and analysis summaries are
written directly under `qav/logs/swarm/cuda/test_knn/`, with periodic-wedge results in its `wedge/`
subdirectory.

Use `--build-only` to compile every selected configuration without executing or validating it.
Each resolution is cleaned and rebuilt because the mesh sizes, particle count, or timestep count
are compile-time constants. Output is stored as

```text
qav/logs/swarm/cuda/groups/manual/MODEL/
```

with raw binary arrays, `meta_N*.json`, `metrics_N*.json`, `environment.json`, and a per-model
`manifest.json`. Each selected model removes its previous owned artifacts before execution, so the
manifest cannot silently mix the new run with older resolutions.

KNN benchmark JSON and component manifests are collected under

```text
qav/logs/swarm/cuda/test_knn/
```

The focused radial-group KNN JSON, environment record, and manifest are stored under

```text
qav/logs/swarm/cuda/groups/radial/test_knn/
```

The complete dispatcher maintains `qav/logs/swarm/cuda/manifest_all.json`; a focused dispatcher
maintains `qav/logs/swarm/cuda/groups/GROUP/manifest.json`. It records every expected model as
pending, running, passed, failed, or interrupted and prints a final `SWARM TEST SUITE: PASS` only
after every selected model has returned successfully.

## Coverage limits and verification still required

### Deferred analytical extensions

The secular Poynting--Robertson reference for a nearly circular, optically thin orbit is

$$
\frac{dR}{dt}=-\frac{2\beta GM_S}{cR},
\qquad
R^2(t)=R_0^2-\frac{4\beta GM_S}{c}t.
$$

It is deferred because it is an asymptotic, many-orbit test rather than an exact pointwise
trajectory; the exact reduced-gravity orbit is already covered by `test_orbit_beta_2d`.

For Brownian motion starting a distance $d$ from an absorbing boundary, the analytical crossing
probability is

$$
P(T_{\rm hit}\le t)
=
\operatorname{erfc}\left(\frac{d}{\sqrt{4Dt}}\right).
$$

This first-passage test is deferred because production stochastic diffusion currently reflects
radial and polar crossings. It requires a genuine continuous-time absorbing boundary-event
algorithm, not an endpoint-only test specialization. Extension of the existing local
settling--diffusion regression to unrestricted disk settling and spatially varying diffusivity
gradients is also deferred.

### Other missing verification

- Extend the current-source production collision-chain runtime gate beyond its CUDA/Morton 2D
  qualification case to axisymmetry, partial wedges, full 3D, imported gas, coupled operators, and
  $N_P=10^6$; statistically compare complete histories and bath-size convergence
- Test complete legacy frozen collision batches, the exact Bernoulli probability
  $1-e^{-\lambda\Delta t}$, and partner sampling; for both integrators, add convergence with
  `CFL_COL` or bath tolerance, $N_K$, $N_P$, and `H_SEARCH`, including a uniform-density field with
  a known rate and forced fragmentation that exercises the size and represented-number update
- Validate the physical `CUSTOM_KERNEL` directly across all Ormel–Cuzzi turbulent regimes and
  their boundaries, the physical-unit Brownian term, and the vertically integrated Gaussian
  overlap factor $[2\pi(H_{g,i}^2+H_{g,j}^2)]^{-1/2}$
- Recover analytical constant, additive, and product Smoluchowski moment evolution rather than
  checking only the pair-kernel numerators
- Add an axisymmetric radial–polar collision-helper case for the revolved
  $2\pi R$ neighborhood measure
- Add independent nonzero-$(p,q)$ tests of the midplane pressure-support target and complete 3D
  gas rotation law, followed by the corresponding dust drift at several heights
- Add a resolved-vertical `VISC_FLOW` test at several heights against the complete Kanagawa target
- Test monodisperse initialization and initial drift velocities in the complete production path;
  the native `test_initial_3d` archive now covers conditional multisize initialization, finite-domain
  containment, intermediate-size interpolation, mass normalization, and independence from `N_Z`
- Validate imported-gas spatial and temporal interpolation, the analytical `STOKES_0` anchor, and
  response to a depleted imported midplane density
- Add boundary-event convergence tests that can detect within-step exits and returns; the current
  deterministic cases establish only the endpoint helper maps
- Add a direct `dyn_rate_calc` test for every active rate and a regression proving accepted
  timesteps do not exceed the configured crossing or diffusion limits
- Test finite attenuation by coupling the reconstructed optical depth to
  $\beta e^{-\tau}$; the current grid and radiation cases validate the two pieces separately
- Compare uninterrupted and restarted stochastic runs with byte-identical restored RNG files but
  tolerance-based physical state, conservation, and ensemble criteria because velocity-file
  conversion need not preserve internal angular momentum bitwise
- Compare the radial-only model with an azimuthally uniform radial–azimuthal model using matched
  surface density, gas targets, and enabled physics
- Add end-to-end operator-combination tests for transport plus diffusion, transport plus radiation,
  transport plus collision, and all enabled swarm physics
- Add CUDA nonfinite-injection coverage matching the existing ROCm particle, collision-rate, and
  KNN-radius failure cases
- Reject an all-zero imported $\rho_g\epsilon$ mass before CDF normalization and test that failure
  path, and directly test the documented outermost-half-cell optical-depth clamp
- Validate the multisize radiation proposal and importance weights statistically, including their
  recovery of total represented dust mass and effective sample size
- Compile and run a small legal flag matrix covering `HALF_DISK`, `DIFFUSION`, `RADIATION`,
  `PR_EFFECT`, `COLLISION`, `MULTISIZE`, `IMPORTGAS`, `CONST_ST`, `VISC_FLOW`, and both KNN
  backends where compatible, while also requiring illegal combinations to fail at compile time
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
- Smoluchowski (1916), coagulation-equation moment evolution
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Kanagawa et al. (2017), [viscous disk gas flow](https://arxiv.org/abs/1706.08975)
- Burns, Lamy & Soter (1979), [radiation pressure and P-R drag](<https://doi.org/10.1016/0019-1035(79)90050-2>)
- Bentley (1975), [multidimensional binary search trees](https://doi.org/10.1145/361002.361007)
- Morton (1966), [geodetic database and file sequencing](https://dominoweb.draco.res.ibm.com/0dabf9473b9c86d48525779800566a39.html)
- Acklam (2000), *An algorithm for computing the inverse normal cumulative distribution function*
