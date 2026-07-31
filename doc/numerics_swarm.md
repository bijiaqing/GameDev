# Lagrangian swarm numerics

## Scope and status

The active representative-particle implementation is under `inc/swarm/` and `src/swarm/`.
It supports analytic or imported gas, semi-analytic particle transport, stochastic turbulent
diffusion, radiation pressure, particle-to-grid diagnostics, and representative-particle
coagulation/fragmentation.

The active tree includes the corrected axisymmetric measure, well-mixed vertically integrated
closures, imported-gas Stokes calibration, frozen collision snapshots, random-state restart
semantics, and an explicit radial-only path with `N_X == 1 && N_Z == 1`. The complete native CUDA
suite, including the radial analytical and KNN cases, is archived and summarized in
[`testset_swarm.md`](testset_swarm.md)

## Coordinates and particle state

The spherical computational coordinates are

$$
(x,y,z)=(\phi,r,\theta),
\qquad
R=y\sin z,
\qquad
Z=y\cos z.
$$

Each active representative stores

$$
\boldsymbol q=(\phi,r,\theta),
\qquad
\boldsymbol u=(\ell_\phi,v_r,\ell_\theta)
=(Rv_\phi,v_r,rv_\theta).
$$

When `N_Z == 1`, the code evaluates $R=y$ and $Z=0$ directly and keeps every particle at
$z=\pi/2$ with $\ell_\theta=0$. When `N_X == 1`, one stored azimuthal cell represents the complete
axisymmetric ring and particles are centered exactly in the inactive coordinate. The radial-only
model retains $v_R$ and $\ell_\phi$ as dynamical variables even though azimuth is spatially inactive

The mesh measure uses $d=2$ for a vertically integrated disk and $d=3$ when the polar dimension is
active:

$$
\Delta V_y=\frac{y_o^d-y_i^d}{d},
\qquad
\Delta V_z=\cos z_i-\cos z_o.
$$

For an axisymmetric cell, deposition and collision normalization include the missing complete
azimuthal circumference or extent rather than treating the stored meridional section as a physical
volume.

### Radial-only closure

The radial model is selected intrinsically by

$$
N_X=1,\qquad N_Y>1,\qquad N_Z=1,
$$

without a separate preprocessor flag. It represents an azimuthally symmetric, vertically
integrated disk: grid dust and gas densities are surface densities $\Sigma_d$ and $\Sigma_g$,
while gas drag, gravity, radiation, and particle dynamics use a midplane closure at $Z=0$.
Representative particles sample the radial mass distribution rather than acting as rigid rings.

The active particle state is

$$
R,\qquad v_R,\qquad \ell_\phi=Rv_\phi,
$$

with exact inactive assignments

$$
x=\frac{X_{\min}+X_{\max}}{2},\qquad
z=\frac{\pi}{2},\qquad
\ell_\theta=v_\theta=0.
$$

Azimuth is spatially inactive, but $\ell_\phi$ remains dynamical because it supplies centrifugal
support, azimuthal drag, collision velocity, and Poynting–Robertson damping. The radial surface
density obeys

$$
\frac{\partial\Sigma_d}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma_dv_R)
=\frac{1}{R}\frac{\partial}{\partial R}
\left(RD_R\frac{\partial\Sigma_d}{\partial R}\right),
$$

where transport follows particle characteristics and stochastic diffusion represents the
right-hand side. The midplane characteristic variables satisfy

$$
\frac{dR}{dt}=v_R,
$$

$$
\frac{d\ell_\phi}{dt}
=-\frac{\ell_\phi-\ell_{\phi,g}}{t_s}-\gamma_{\rm PR}\ell_\phi,
$$

$$
\frac{dv_R}{dt}
=\frac{\ell_\phi^2}{R^3}
-(1-\beta)\frac{GM_\star}{R^2}
-\frac{v_R-v_{R,g}}{t_s}
-2\gamma_{\rm PR}v_R.
$$

The binary particle layout remains the common three-coordinate layout. Science files store
physical linear velocities $(v_\phi,v_R,v_\theta)$, with $v_\theta=0$. Grid density files contain
$N_Y$ values of $\Sigma_d$, optical-depth files contain the radial cumulative outer-face field,
and runtime metadata labels the geometry, density kinds, inactive coordinates, and midplane
dynamical closure.

The radial timestep retains the orbital rate because centrifugal and epicyclic dynamics remain
active. It also limits radial particle and gas-target crossing, force-driven displacement,
diffusion RMS displacement, deterministic diffusion drift, `DT_MAX`, and the remaining output
interval; azimuthal and polar crossing rates are inactive.

## Dust normalization and initialization

The physical dust profile starts from

$$
\Sigma_d(R)=Z_{\rm metal}\Sigma_g(R)
$$

before the Gaussian edge convolution. `get_total_dust_mass()` integrates that initialized
profile over the configured domain. Representative grain counts, opacity, diagnostics, and
dimensionless collision normalizations use this runtime mass; there is no independent arbitrary
dust-mass parameter.

Monodisperse initialization samples a joint $(y,z)$ CDF proportional to density times the exact
cell measure. For a multisize 3D disk, grain size is sampled first, then the conditional spatial
CDF uses the size-dependent dust thickness

$$
H_d(R,s)=H_g(R)
\sqrt{\frac{\alpha_z}{\mathrm{St}_{\rm mid}(R,s)}}.
$$

The joint spatial draw is necessary because cylindrical $R=y\sin z$ couples the radial surface
profile and the vertical thickness on a spherical grid. Within a selected cell, the sampler is
uniform in $y^d$ and $\cos z$, so it does not apply a geometry Jacobian twice.

The initial velocity is the same no-backreaction steady drift used by the fluid branch:

$$
v_{R,d}
=\frac{v_{R,g}+2\mathrm{St}(v_{\phi,g}-v_K)}
{1+\mathrm{St}^2},
\qquad
v_{\phi,d}=v_{\phi,g}-\frac{\mathrm{St}}{2}v_{R,d}.
$$

An active vertical dimension also receives terminal settling
$v_Z=-\mathrm{St}\,\Omega_K Z$. `VISC_FLOW` replaces the zero gas radial velocity with the
analytic viscous prescription, requires `DIFFUSION`, and cannot be combined with `IMPORTGAS` because
imported gas velocities already prescribe the gas flow.

Imported gas initialization samples the imported gas density times $\epsilon$ using the same exact
cell-measure construction. Active azimuthal cells are sampled with `_get_dx()`; an inactive
azimuth is centered. Here $\epsilon$ controls the spatial shape of the dust sampling distribution;
the total represented dust mass is still normalized from `SIGMA_0`, `METAL_Z`, and the configured
domain rather than taken as an independent absolute mass from the imported field. When
`N_Z == 1`, the imported field is $\Sigma_g$ and

$$
\mathrm{St}(R,s)
=\mathrm{St}_0\frac{s}{S_0}\frac{\Sigma_0}{\Sigma_g(R)}.
$$

When `N_Z > 1`, the imported field is $\rho_g$ and `STOKES_0` is anchored to the analytical
reference midplane through

$$
\mathrm{St}(R,Z,s)
=\mathrm{St}_0\frac{s}{S_0}
\frac{\rho_{g,0}H_{g,0}}{\rho_g(R,Z)H_g(R)},
\qquad
\rho_{g,0}=\frac{\Sigma_0}{\sqrt{2\pi}H_{g,0}}.
$$

The imported density is assumed to have been calibrated from that $\Sigma_0,H_{g,0}$ reference
disk. A gap or other subsequent density depletion therefore increases the local Stokes number
above `STOKES_0` rather than resetting the depleted midplane to the reference value.
`CONST_ST` is therefore an analytic-gas-only option and is rejected when `IMPORTGAS` is enabled.

## Semi-analytic transport

Particle trajectories use a staggered semi-analytic drag integrator. Coordinate drifts account for
the angular-momentum storage convention and the changing spherical radius. At the force stage,
frozen-coefficient drag is integrated exponentially, so a short stopping time is not an explicit
stability restriction.

The analytic force model includes:

- central gravity, optionally reduced by attenuated radiation pressure
- optional Poynting-Robertson damping when `PR_EFFECT` is enabled
- pressure-supported gas rotation
- optional viscous gas radial motion
- radial centrifugal acceleration
- polar geometric torque

When imported gas is enabled, the current and next snapshots are interpolated to each dynamics
step midpoint. This avoids holding the gas piecewise constant over a complete output interval.

The runtime timestep is the inverse of the maximum active rate, capped by `DT_MAX`. The rate
calculation bounds orbital motion, azimuthal/radial/polar cell crossing, net-force displacement,
diffusion RMS displacement, deterministic diffusion drift, and radiation acceleration for the
smallest allowed grain where applicable.

## Radiation and optical depth

The swarm deposits extinction weight to the grid, converts it to a local radial optical-depth
increment, and performs a radial outer-face prefix sum. Particle forces interpolate that
cumulative field continuously at the particle radius.

In a vertically integrated 2D radial–azimuthal disk, every dust species is assumed to share the
gas vertical profile. The midplane density closure is therefore

$$
\rho_{d,0}=\frac{\Sigma_d}{\sqrt{2\pi}H_g},
$$

independent of grain size and Stokes number. This well-mixed closure is used only when `N_Z == 1`;
radial–polar and full-3D models retain explicit vertical structure and settling.

For geometric opacity $\kappa\propto s^{-1}$, the deposited extinction weight has the correct area
dimension. Radiation for each particle is

$$
\beta(s,t,\tau)
=\frac{\beta_0}{s/S_0}
f_\beta(t)e^{-\tau},
\qquad
f_\beta=s_t^2(3-2s_t),
\qquad
s_t=\operatorname{clip}(t/T_\beta,0,1).
$$

With radiation enabled, transport first drifts particles to midpoint positions, reconstructs
optical depth there, and then completes the force/drag step.

When `PR_EFFECT` is enabled, the same attenuated and size-dependent $\beta$ supplies the
first-order Poynting-Robertson rate

$$
\gamma_{\rm PR}=\frac{\beta GM_\star}{C_{\rm LIGHT}y^2}.
$$

The implemented first-order radiation acceleration is

$$
\boldsymbol a_{\rm grav+rad}
=-\frac{GM_\star}{y^2}\boldsymbol e_y
+\beta\frac{GM_\star}{y^2}
\left[
\left(1-\frac{v_y}{C_{\rm LIGHT}}\right)\boldsymbol e_y
-\frac{\boldsymbol v}{C_{\rm LIGHT}}
\right].
$$

Equivalently,

$$
\boldsymbol a_{\rm grav+rad}
=-(1-\beta)\frac{GM_\star}{y^2}\boldsymbol e_y
-\gamma_{\rm PR}
\left(
v_x\boldsymbol e_x+2v_y\boldsymbol e_y+v_z\boldsymbol e_z
\right).
$$

The stored tangential variables $\ell_\phi$ and $\ell_\theta$ are damped at
$\gamma_{\rm PR}$, while the spherical radial velocity is damped at
$2\gamma_{\rm PR}$. Gas and P-R drag are integrated together in the frozen-midpoint
exponential response:

$$
k_x=\frac{1}{t_s}+\gamma_{\rm PR},
\qquad
k_y=\frac{1}{t_s}+2\gamma_{\rm PR},
\qquad
k_z=\frac{1}{t_s}+\gamma_{\rm PR}.
$$

For any frozen component equation

$$
\frac{du}{dt}=-ku+\frac{u_g}{t_s}+F,
$$

the code evaluates

$$
u(h)
=e^{-kh}u(0)
+\frac{1-e^{-kh}}{k}
\left(
\frac{u_g}{t_s}+F
\right).
$$

The half response uses $h=\Delta t/2$ and forces evaluated from the initial angular momenta.
Those angular momenta are then used to reevaluate the centrifugal and polar terms, and the full
response uses $h=\Delta t$ from the original state with the updated midpoint forces. The
`expm1` form evaluates $1-e^{-kh}$ without small-argument cancellation.

The gas targets enter only through $u_g/t_s$, so P-R drag damps toward the stellar rest frame
rather than toward the gas. This integration is stable even when either damping rate is large,
although the trajectory timestep must still resolve spatial motion and coefficient variation.
No separate P-R stability term is added to `dyn_rate_calc` because the damping is analytic and
the P-R contribution is dissipative.

The current dynamics always uses orbital code units, including builds without `CODE_UNIT`;
that flag changes collision microphysics rather than $G$, $M_\star$, $R_0$, position, or
velocity units. Therefore `C_LIGHT = 10^4` is a code-unit velocity in the present implementation.
A future fully cgs dynamics branch must instead use
$c=2.99792458\times10^{10}\ {\rm cm\,s^{-1}}$.

Models enable this term with `NVCC += -DPR_EFFECT`. Compile-time validation rejects `PR_EFFECT`
without `RADIATION`.

## Stochastic density diffusion

The selected swarm diffusion equation represents diffusion of dust density rather than
dust-to-gas concentration. Its diffusion tensor is diagonal in cylindrical coordinates:

$$
(D_\phi,D_R,D_Z)
=\left(\frac{\nu}{\mathrm{Sc}_x},
\frac{\nu}{\mathrm{Sc}_R},
\frac{\nu}{\mathrm{Sc}_Z}\right).
$$

The Itô increments are

$$
d\phi
=\frac{\partial_\phi D_\phi}{R^2}\,dt
+\frac{\sqrt{2D_\phi}}{R}\,dW_\phi,
$$

$$
dR
=\left(\partial_R D_R+\frac{D_R}{R}\right)dt
+\sqrt{2D_R}\,dW_R,
$$

$$
dZ
=\partial_ZD_Z\,dt
+\sqrt{2D_Z}\,dW_Z.
$$

The current analytic viscosity depends only on $R$, so the azimuthal and vertical coefficient
gradients are zero; explicit placeholders mark where more general dependencies would enter. The
radial drift includes both the variable-diffusivity derivative and the cylindrical Itô term
$D_R/R$.

All supported geometries impose the same zero-flux radial diffusion boundary by repeatedly
reflecting the spherical radius into $[Y_{\min},Y_{\max}]$. In a vertically integrated model,
$y=R$, so this is also the cylindrical-radial boundary and does not change azimuth. In 3D, a
negative intermediate cylindrical $R$ is first mapped as
$(R,\phi)\mapsto(-R,\phi+\pi)$; this is a coordinate continuation across the cylindrical axis,
not a radial-boundary condition

A random displacement is a spatial redistribution, not an impulse. The kernel reconstructs the
particle's Cartesian velocity before moving it and projects that unchanged velocity into the new
local spherical basis afterward. This is internally consistent, but it is not the same momentum
closure as the fluid's donor-angular-momentum diffusion; density-only fluid–swarm diffusion
comparisons are valid until a common momentum equation is derived.

## Representative-particle collisions

Collision neighborhoods use one exact backend selected in `flags.mk`:

```make
COLLISION_SEARCH := kdtree
```

or

```make
COLLISION_SEARCH := morton
```

The same selection can be supplied on the build command line, for example
`make MODEL=my_model COLLISION_SEARCH=morton`. Backend-specific objects are stored under separate
search directories, so switching between KD-tree and Morton cannot silently reuse objects compiled
for the other backend.

The KD-tree is the default reference. The Morton backend uses an adaptive pointer-free hierarchy,
cooperative top-$K$ selection, and boundary-only periodic ghosts. Locally planar accessible-volume
corrections handle physical boundaries in 2D and 3D. In the radial-only model, both backends search
the collinear points $(R,0,0)$ and the exact accessible annular area is

$$
A_K=\pi\left[
\min(Y_{\max},R+d_K)^2-
\max(Y_{\min},R-d_K)^2
\right].
$$

For an axisymmetric radial-polar model, the reduced-dimensional neighborhood retains the existing
$2\pi R$ revolution factor

For a pair $i,j$, the physical rate contains

$$
\lambda_{ij}
\propto
\frac{N_j\,\sigma_{ij}\,\Delta v_{ij}}{V_{\rm local}},
\qquad
\sigma_{ij}=\frac{\pi}{4}(s_i+s_j)^2.
$$

In a vertically integrated 2D disk, both species use their local gas scale heights in the Gaussian
vertical overlap

$$
\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)}.
$$

The resolved relative speed is reconstructed from the complete Cartesian velocity difference.
Brownian and Ormel–Cuzzi turbulent speeds are added in quadrature as unresolved contributions.
Stokes numbers remain local inputs to drag and unresolved relative velocities, but not to the 2D
vertical overlap closure. In an imported vertically integrated model, the turbulent Reynolds
number uses the same external $\Sigma_g$ that determines the local Stokes number

For an imported 3D model, the local Stokes number uses the imported gas volume density, whereas
the turbulent Reynolds closure retains the analytic $\Sigma_g(R)$ profile because the imported
interface provides no vertically integrated gas column. This is a deliberate limitation of the
current 3D imported-gas interface rather than a reconstruction of $\Sigma_g$ from the imported
volume-density field

The GPU event update is a controlled parallel Bernoulli leap, not an exact serial Gillespie
trajectory. During each batch:

1. freeze particle size and represented grain number
2. evaluate each representative's total propensity and cumulative partner weights
3. choose
   $\Delta t_{\rm col}\le\mathrm{CFL}_{\rm col}/\max_i\lambda_i$
4. give each representative probability
   $1-\exp(-\lambda_i\Delta t_{\rm col})$
   of at most one event
5. sample its partner from the frozen pair propensities

The frozen species snapshot prevents concurrent partner reads from observing partially updated
sizes or grain counts. Accuracy still depends on convergence with decreasing `CFL_COL`.

For the Morton backend, the block evaluates the individual pair propensities cooperatively after
the neighbor search. Thread zero then accumulates the stored values and samples the partner in the
original Morton neighbor order. This preserves the serial floating-point summation and stochastic
selection rule while avoiding one-thread evaluation of all pair physics.

Coagulation uses

$$
s_{\rm new}=(s_i^3+s_j^3)^{1/3}
$$

and adjusts the represented grain number to conserve the representative mass. The fragmentation
law is a configured modeling choice rather than a universal fragment distribution. Dimensionless
constant, additive, and product kernels are available for analytical tests; the additive kernel is
$K=m_i+m_j$.

### Exact neighbor-search contract

The search backend is part of the estimator because the normalization contains

$$
\frac{N_P}{(N_K-1)M_{\mathrm{dust}}}.
$$

For particle $i$, the location-dependent search cap is

$$
q_i=H_{\mathrm{SEARCH}}H_g(R_i)R_i.
$$

Both backends must return the same original particle indices, squared distances, valid count,
and farthest valid distance under the lexicographic $(d^2,\mathrm{id})$ selection rule. They retain
the same $N_K$, $q_i$, accessible-ball normalization, pair physics, maximum-rate reduction, event
probability, partner sampling, and coagulation or fragmentation update. Approximate neighbors or a
fixed-radius population estimator would be a different numerical model and require a new
derivation.

The contract fixes the selected set, not the physical order in which a backend stores it. Morton
returns a sorted list, whereas the KD-tree retains the same set in heap order. Ordered rate sums are
reproducible within each backend, but an identical random target can therefore select a different
partner after switching backends. Backend trajectories and RNG states need not remain byte-equal;
mass conservation, rates, topology, and ensemble distributions are the cross-backend invariants.

The KD-tree candidate heap uses `index_old` as its equal-distance tie breaker
and expands its culling radius by one floating-point unit. Each heap slot stores only the encoded
pair $(d^2,\mathrm{id})$; the shuffled tree slot is not retained because collision physics consumes
`index_old` directly. This prevents mutable tree slots from changing which member of an
exact-distance tie is retained while minimizing thread-local storage. For partial wedges, the heap
checks for repeated physical identifiers only when the minimum separation between adjacent
periodic images is no larger than twice the current query radius. Wider wedges use the ordinary
$O(\log N_K)$ heap insertion without an $O(N_K)$ duplicate scan.

At the beginning of each fixed-position collision interval, the code records one active byte per
physical representative. Absorbed representatives may remain in the immutable search hierarchy,
but the KD heap rejects their stable identifiers before insertion. They therefore cannot occupy one
of the retained $N_K$ slots or reduce the active-neighbor radius. This avoids rebuilding a compacted
tree while making absorbed particles invisible to collision selection.

### Adaptive Morton hierarchy

The Morton builder maps Cartesian coordinates to integer cells, interleaves their bits into 64-bit
keys, stable-sorts the records, and recursively refines cells above `MORTON_LEAF_TARGET`. Compact
record and node arrays contain no pointers. The current hierarchy topology is assembled on the host
after the GPU key sort; GPU-native construction remains a performance improvement rather than a
correctness requirement.

For node bounds $[\boldsymbol b_{\min},\boldsymbol b_{\max}]$, traversal uses the conservative
lower bound

$$
d_{\min}^2=\sum_\alpha
\left[\max(b_{\min,\alpha}-x_\alpha,0,x_\alpha-b_{\max,\alpha})\right]^2.
$$

A level-aware single-precision padding

$$
p=2(L_{\max}+2)\epsilon_{\rm float}
\max\!\left(1,|\boldsymbol b_{\min}|_\infty,|\boldsymbol b_{\max}|_\infty\right)
$$

prevents accumulated recursive-subdivision roundoff from making the pruning bound too large. One
CUDA block evaluates candidate tiles cooperatively and retains the nearest $N_K$ pairs in shared
memory. The traversal reports an overflow rather than silently accepting an incomplete result.

The validated repeated block-parallel bitonic merge retains the nearest pairs in lexicographic
$(d^2,\mathrm{id})$ order. Distances remain available because traversal pruning, closest-ghost
deduplication, and the collision-volume radius require them even though collision physics consumes
the original particle identifiers. When periodic images overlap, the query temporarily retains
$3N_K$ records, deduplicates physical identifiers, and performs a final ordered selection.

All threads in a query block consume a shared traversal-node index. A block barrier precedes every
replacement of that index, while the existing following barrier publishes the replacement. Both
halves are required: a following barrier alone allows a fast warp to begin the next iteration and
overwrite the index while a slower warp is still reading the preceding value. CUDA Racecheck
identified this hazard during the top-$K$ validation; the corrected traversal reports zero
Racecheck hazards on the 100,000-particle three-dimensional ring case.

The block-parallel sorted merge is the only Morton top-$K$ implementation and requires no secondary
selection flag.

The Morton candidate tiles use the same active-byte array as the KD heap. An absorbed record is
replaced by the empty sentinel before each shared-memory merge, so it cannot enter the retained
top-$K$ list. The hierarchy may still contain its spatial record, which costs traversal work but
does not affect neighbor identity or collision normalization.

### Periodic boundary ghosts

For a wedge, the Morton owner copies only source records whose Cartesian distance to either
azimuthal face is no larger than

$$
q_{\max}=\max_i q_i.
$$

Lower-face sources are rotated by one positive wedge width and upper-face sources by one negative
wedge width. The original and compact ghost records are indexed together, while every record keeps
the original index of its physical representative. Ghost construction uses GPU `sincosf`, so
the periodic isometry follows the same single-precision arithmetic as the search records.

The global maximum is conservative for position-dependent cutoffs: a query still uses its own
$q_i$, while the larger construction halo guarantees that no eligible source was omitted. If a
wedge is narrow enough for multiple images of one representative to enter the same query ball, the
query temporarily retains up to $3N_K$ records, deduplicates by original particle index, and then selects
the exact nearest $N_K$ physical particles. Ordinary disk wedges use the cheaper disjoint-image
path.

When $H_gR$ varies strongly, a later memory optimization may replace $q_{\max}$ by certified radial
halo bins. For source and query radial bins $s$ and $b$, define

$$
D_{sb}=\max(0,R_s-R_{b+1},R_b-R_{s+1}),
$$

and use for source bin $s$

$$
h_s=\max_{\{b:D_{sb}\le c_b\}}c_b,
\qquad
c_b=\max_{i:R_i\in b}q_i.
$$

A source is ghosted when its seam distance is at most $h_s$ plus roundoff padding. This optimization
must reproduce the global-halo neighbor sets exactly before it can replace the current policy.

### Backend characteristics

The Morton representation offers contiguous pointer-free storage, bounded cooperative scratch,
boundary-only periodic records, explicit workload diagnostics, a more direct ROCm port, and natural
Morton-range ownership for multiple GPUs. Its hierarchy alone is compact, but the current production
owner also retains unsorted query coordinates, azimuths, per-particle cutoffs, and overflow flags.
Consequently, full-disk total search storage can exceed the single-array KD-tree even when the
Morton hierarchy is smaller; partial wedges benefit more strongly because the KD-tree stores three
complete copies. The latest isolated query matrix found that Morton used less persistent search
memory in every tested case through $N_P=10^7$, while KD-tree was faster in most cases and in every
$N_P=10^7$ case. The KD-tree therefore remains both an independent mature reference and a strong
single-GPU performance option. Backend choice is a measured model configuration, not a change in
collision physics; detailed timing and pass criteria belong in `testset_swarm.md`.

For distributed Morton search, each GPU can own contiguous coarse cells or key ranges and import
read-only halo records. A local result is globally certified when every intersecting remote cell has
been examined and either $d_{K,i}$ or $q_i$ is below the conservative distance to all remaining
remote cells. Particle migration, halo exchange, and collision-property synchronization are not yet
implemented.

### Future removal of the global collision timestep

The present batch step

$$
\Delta t_{\mathrm{col}}=
\min\left(\frac{\mathrm{CFL}_{\mathrm{COL}}}{\max_i\lambda_i},
\Delta t_{\mathrm{remaining}}\right)
$$

is controlled by the fastest representative. Raising `CFL_COL` does not solve this bottleneck: if
$\lambda_i\Delta t$ is large, collapsing several expected physical events into one Bernoulli trial
changes the stochastic process.

The preferred future integrator is a frozen-bath local continuous-time event chain. Over a larger
bath interval $\tau_{\mathrm{bath}}$, each representative keeps a local clock, draws exact waiting
times

$$
\delta t_i=-\frac{\ln U}{\lambda_i},
$$

and processes zero, one, or many events against immutable partner properties. After each event its
own rate and partner weights are recomputed. This removes the fastest-particle global microstep and
is exact conditional on the frozen bath; convergence under halving $\tau_{\mathrm{bath}}$ controls
the bath-freezing error. Rate-binned work queues and continuation launches can mitigate divergent
chain lengths without truncating a stochastic path. The current Bernoulli method must remain
selectable until waiting-time statistics, Smoluchowski moments, mass conservation, and bath-interval
convergence pass.

## Operator composition and boundaries

For one dynamics interval, the combined sequence is

$$
C^{1/2}D^{1/2}T D^{1/2}C^{1/2},
$$

where $C$ is collision, $D$ positional diffusion, and $T$ transport. This removes first-order Lie
splitting between enabled modules, but it does not make the stochastic diffusion or Bernoulli
collision leap deterministically second order.

Boundary policies are operator-specific:

- transport is periodic in azimuth
- transport absorbs radial exits and full-disk polar exits
- `HALFDISK` reflects transport at the midplane and absorbs at the other polar edge
- diffusion is periodic in azimuth and reflecting at finite radial and polar boundaries in every
  supported geometry

An active `HALFDISK` configuration is accepted only when $Z_{\max}=\pi/2$; the swarm runtime checks
this before allocating or evolving particle state.

An absorbed representative is parked at `y=0`, assigned zero stored velocity, and skipped by
subsequent transport, diffusion, deposition, opacity, timestep, and collision calculations.

The production helpers currently apply boundary conditions to the completed operator endpoint.
Repeated folding makes diffusion positions lie inside the reflecting interval and is consistent with
the zero-flux process in the small-step limit, but it is not an exact finite-step transition for
spatially varying drift or diffusivity. Likewise, endpoint absorption cannot detect a trajectory
that crosses an absorbing face and returns inside during one transport step. A rigorous future
boundary-event implementation should do the following:

1. construct a dense trajectory consistent with the staggered transport step
2. solve for the earliest radial or polar face crossing within the step
3. terminate an absorbing trajectory at that event, or reflect the normal velocity and reintegrate
   the unused part of the step for a reflecting face
4. treat stochastic crossings with a Brownian-bridge or reflected-transition construction and
   verify weak convergence under diffusion-step refinement

The existing CFL controls make missed transport crossings unlikely in ordinary runs, but do not
constitute a mathematical event detector. The deterministic boundary tests exercise the endpoint
policy itself; they do not yet establish boundary-event convergence.

The 2D and 3D collision-volume corrections also remain locally planar approximations. The exact
quantity is the measure of the Cartesian KNN ball intersected with the spherical disk domain. A
robust replacement should retain the exact 1D annular formula and evaluate boundary-overlapping 2D
and 3D measures by deterministic quadrature of the exact intersection: solve the ball/domain radial
limits along each angular ray, integrate $R\,dR\,d\phi$ in a vertically integrated disk or
$r^2\sin z\,dr\,dz\,d\phi$ in 3D, and handle periodic wedge faces as identified copies rather than
physical cuts. Interior queries keep the analytic disk/ball measure. A tabulated correction in
dimensionless boundary distances can amortize the quadrature, but it must be checked against direct
high-order quadrature before replacing the present cap formula.

## File and restart semantics

Science files store physical linear velocity $(v_\phi,v_r,v_\theta)$. Loading converts it back to
the internal angular variables before device evolution. Grain size and represented grain number
are included only for multisize builds, matching the declared binary dtype.
Loading also reasserts every inactive coordinate, so a radial restart has exact
$x=(X_{\min}+X_{\max})/2$, $z=\pi/2$, and $\ell_\theta=0$ before evolution resumes.

Collision or diffusion builds save `rngstate_<frame>.dat` beside every particle checkpoint and
restore it on resume. The companion file contains one raw `curandState` per representative, so a
resumed stochastic run continues the same random streams as an uninterrupted run built with the
same CUDA state layout.

## Current limitations

- The passed CUDA suite covers circular orbits, frozen stiff drag, one-step diffusion moments,
  optical-depth reconstruction, radiation and P-R response algebra, accessible collision measures,
  and constant, additive, and product kernel numerators. It does not establish long-time coupled
  production evolution; see [`testset_swarm.md`](testset_swarm.md)
- P-R validation still needs secular optically thin circular-orbit decay with a fixed orbital-plane
  direction beyond the prepared constant-coefficient one-step response
- Collision validation still needs complete event statistics and convergence with `CFL_COL`,
  neighbor count, and particle number; exact KD-tree/Morton/brute-force neighbor tests are now in
  `qav/swarm/test_knn/`
- Settling equilibrium, initialization CDFs, imported-gas trajectory coupling, boundary-event
  convergence, restart reproducibility, timestep rates, and complete flag/operator combinations
  remain untested; deterministic endpoint boundary helpers now have dedicated CUDA cases
- The current fluid and swarm diffusion-momentum closures differ and should not be compared as the
  same velocity equation
- A zero integrated dust mass, `N_K == 1`, or a polar domain reaching a coordinate singularity is
  not a scientifically meaningful configuration and is not fully guarded
- The locally planar KNN boundary-cap correction is asymptotically consistent, not an exact
  curved-boundary intersection
- Multi-GPU Morton ownership, radial halo bins, GPU-native hierarchy construction, and local
  continuous-time collision chains remain future work

## References

- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Ormel & Cuzzi (2007), [turbulent relative velocities](https://arxiv.org/abs/astro-ph/0702303)
- Zsom & Dullemond (2008), [representative-particle coagulation](https://arxiv.org/abs/0807.5052)
- Gillespie (1977), [stochastic reaction simulation](https://doi.org/10.1021/j100540a008)
- Johnson, Douze & Jégou (2017), [GPU similarity search](https://arxiv.org/abs/1702.08734)
- García et al. (2012), [multi-GPU spatial decomposition and halos](https://arxiv.org/abs/1210.1017)
- Cao, Gillespie & Petzold (2005), [explicit Poisson tau-leaping](https://people.cs.vt.edu/~ycao/publication/JChemPhys_123_054104.pdf)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](https://doi.org/10.1016/0019-1035(79)90050-2)
