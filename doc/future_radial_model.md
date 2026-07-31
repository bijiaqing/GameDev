# Future radial-only swarm model

## Implementation status

The first production implementation described in this document was added on 2026-07-30

- exact vertically integrated cylindrical geometry and inactive-coordinate assignments
- direct radial SSA and diffusion paths
- imported-surface-density Stokes scaling
- imported-$\Sigma_g$ collision Reynolds scaling
- exact annular radial KNN normalization
- canonical collinear KD-tree and Morton search coordinates
- radial metadata and restart locking
- radial grid, orbit, drag, viscous-flow, diffusion, radiation, P-R, collision, imported-gas, and KNN test cases

Local static checks and build dry-runs pass, but native CUDA results have not yet been generated for
the new cases. Boundary-event convergence, restart continuation, long-time coupled evolution, and
the radial versus azimuthally uniform 2D comparison also remain outstanding. The title and
completion criteria therefore remain future-facing until that evidence is available

## Decision and scope

The radial swarm model will use

$$
N_X=1,
\qquad
N_Y>1,
\qquad
N_Z=1,
$$

and will represent an azimuthally symmetric, vertically integrated dust disk with midplane particle dynamics

The physical closure is

- mesh dust and gas densities are surface densities $\Sigma_d$ and $\Sigma_g$
- representative particles sample the radial distribution of an axisymmetric population
- particle dynamics retain cylindrical radial velocity $v_R$ and specific azimuthal angular momentum $\ell_\phi=Rv_\phi$
- the gas, gravity, radiation, and drag coefficients are evaluated at the midplane $Z=0$
- vertical density is reconstructed analytically only for opacity and physical collision rates
- all grain sizes use the gas scale height $H_g$ in the vertically integrated closure

This is a midplane dynamical closure, not a strict vertical average of every force

The radial model must coexist with the existing geometries

| Geometry | Dimensions | Density meaning |
|---|---|---|
| radial only | `N_X == 1 && N_Z == 1` | surface density |
| radial–azimuthal | `N_X > 1 && N_Z == 1` | surface density |
| axisymmetric radial–polar | `N_X == 1 && N_Z > 1` | volume density |
| full 3D | `N_X > 1 && N_Z > 1` | volume density |

Radial behavior must therefore be inferred from `N_X == 1 && N_Z == 1`; no independent `RADIAL_1D` flag should be introduced

## Meaning of a representative particle

A representative particle does not act as one rigid physical ring. Its radius is one Monte Carlo sample from the axisymmetric mass distribution, while its represented grain number carries the physical statistical weight of that sample

The meaningful radial state is

$$
R,
\qquad
v_R,
\qquad
\ell_\phi=Rv_\phi,
$$

with the inactive stored coordinates constrained to

$$
x=x_{\rm ref}=\frac{X_{\min}+X_{\max}}{2},
\qquad
z=\frac{\pi}{2},
\qquad
\ell_\theta=0.
$$

Azimuth is spatially inactive, but $v_\phi$ and $\ell_\phi$ remain dynamical because they determine centrifugal support, gas drag, collision velocities, radiation forces, and Poynting–Robertson drag

The existing `swarm` structure and particle-file layout should remain unchanged. Science files should continue storing the physical linear velocities $(v_\phi,v_R,v_\theta)$, with $v_\theta=0$ in the radial model

## Governing equations

### Surface-density evolution

Ignoring collisions for the moment, the radial density equation is

$$
\frac{\partial\Sigma_d}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}
\left(R\Sigma_d v_R\right)
=
\frac{1}{R}\frac{\partial}{\partial R}
\left(RD_R\frac{\partial\Sigma_d}{\partial R}\right).
$$

Transport follows particle characteristics, while stochastic diffusion represents the right-hand side

### Midplane dynamics

At $Z=0$, the characteristic equations are

$$
\frac{dR}{dt}=v_R,
$$

$$
\frac{d\ell_\phi}{dt}
=-\frac{\ell_\phi-\ell_{\phi,g}}{t_s}
-\gamma_{\rm PR}\ell_\phi,
$$

$$
\frac{dv_R}{dt}
=\frac{\ell_\phi^2}{R^3}
-(1-\beta)\frac{GM_S}{R^2}
-\frac{v_R-v_{R,g}}{t_s}
-2\gamma_{\rm PR}v_R,
$$

where

$$
\ell_{\phi,g}=Rv_{\phi,g},
\qquad
t_s=\frac{\mathrm{St}}{\Omega_K},
\qquad
\gamma_{\rm PR}=\frac{\beta GM_S}{C_{\rm LIGHT}R^2}.
$$

The current staggered semi-analytic method remains appropriate: it evaluates coefficients at the midpoint, integrates frozen drag and optional P-R damping exponentially, reevaluates centrifugal acceleration with midpoint angular momentum, and completes the drift

### Vertically reconstructed density

The adopted well-mixed closure is

$$
\rho_d(R,Z)
=\frac{\Sigma_d(R)}{\sqrt{2\pi}H_g(R)}
\exp\left[-\frac{Z^2}{2H_g^2(R)}\right].
$$

The midplane value used by opacity is

$$
\rho_{d,0}(R)=\frac{\Sigma_d(R)}{\sqrt{2\pi}H_g(R)}.
$$

For a pair of well-mixed species, vertical integration of the product of their Gaussian profiles gives the physical collision overlap thickness

$$
H_{ij,\rm eff}
=\sqrt{2\pi\left(H_{g,i}^2+H_{g,j}^2\right)}.
$$

## Current implementation assessment

The production prescription below is now implemented. Native CUDA evidence is still required before declaring radial support production-validated

### Calculations that already reduce correctly

#### Mesh measure, deposition, and interpolation

`_get_mesh_dim()` returns $2$ when `N_Z == 1`, `_get_vol_x()` returns the complete azimuthal extent $2\pi$ when `N_X == 1`, and `_get_vol_z()` returns the neutral factor one when `N_Z == 1`

The exact radial cell area is therefore

$$
A_i
=2\pi\frac{R_{i+1}^2-R_i^2}{2}
=\pi\left(R_{i+1}^2-R_i^2\right).
$$

The trilinear grid routines degenerate algebraically to radial interpolation because the inactive-direction fractions and offsets are zero. Consequently

- `dustdens_depo` plus `dustdens_calc` produces cell-centred $\Sigma_d$
- imported `gasdens` is sampled with the correct annular measure
- optical depth is interpolated from radial outer faces
- x-fastest indexing remains valid because the one-cell inactive dimensions contribute unit strides

No separate one-dimensional grid container is required

#### Analytic initialization mass and radial sampling

For `N_Z == 1`, the initialized density helper returns $\Sigma_d$ directly. The total-mass integral and the radial position CDF use $2\pi R\,dR$, and the within-cell draw is uniform in $R^2$

The present initializer is a piecewise-cell approximation to the convolved profile rather than an exact continuous inverse CDF. This is a normal spatial discretization and should be verified by radial-resolution convergence; it is not a radial-only defect

When `MULTISIZE` is enabled with `N_Z == 1`, all sizes intentionally share the same radial CDF because the adopted vertically integrated model contains no differential settling in its spatial initialization

#### Analytic midplane gas model

At $Z=0$, the existing analytic prescriptions reduce to

$$
H_g=h_gR,
\qquad
\eta=-\frac{h_g^2}{2}
\left(p+\frac{q}{2}-\frac{3}{2}\right),
$$

and

$$
\mathrm{St}(R,s)
=\mathrm{STOKES}_0\frac{s}{S_0}
\left(\frac{R}{R_0}\right)^{-p}
$$

unless `CONST_ST` is selected

The vertically integrated viscous gas velocity already uses

$$
v_{R,g}
=-\frac{3\nu}{R}
\left(
\frac{d\ln\nu}{d\ln R}+p+\frac12
\right),
$$

which is the required surface-density expression

#### Radial stochastic diffusion

For `N_X == 1 && N_Z == 1`, the current diffusion kernel draws only the cylindrical radial increment

$$
dR
=\left(\frac{dD_R}{dR}+\frac{D_R}{R}\right)dt
+\sqrt{2D_R}\,dW.
$$

This is exactly the Itô process corresponding to diffusion of $\Sigma_d$ rather than diffusion of $\Sigma_d/\Sigma_g$

The current operator-specific boundaries are physically distinct and should remain so

- transport absorbs radial exits
- stochastic diffusion reflects radial crossings, imposing zero diffusive flux

#### Optical depth and radiation

For `N_Z == 1`, opacity deposition divides each extinction weight by $\sqrt{2\pi}H_g$. Division by annular cell area and multiplication by the radial cell width then give

$$
\Delta\tau_i
\simeq
\kappa_i\frac{\Sigma_{d,i}}{\sqrt{2\pi}H_{g,i}}\Delta R_i.
$$

`optdepth_csum` produces the radial midplane prefix integral with zero incident optical depth at the inner boundary. With `N_X == 1`, `optdepth_mean` is an identity and need not be called, but it is harmless

The midpoint opacity reconstruction used by radiation is still required in the radial model

#### Physical collision overlap and velocity

The physical custom kernel already divides the surface-neighbor rate by

$$
\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)},
$$

which is the correct well-mixed vertical overlap. At a common inactive azimuth, reconstructed radial and azimuthal velocity differences are also the correct local resolved collision velocity

#### Operator composition

The existing sequence

$$
C^{1/2}D^{1/2}TD^{1/2}C^{1/2}
$$

remains applicable. A radial implementation does not require a different main loop or a separate particle structure

## Implemented production corrections

### R1 — make the midplane and inactive-coordinate invariants exact

Before this correction, the code frequently obtained $R$ and $Z$ from `y*sin(z)` and `y*cos(z)`. Although $z$ was assigned `0.5*M_PI`, floating-point $\cos(\pi/2)$ is not identically zero, permitting small but avoidable violations

- `particle_init` can initialize a nonzero $\ell_\theta$ from $v_R\cos z$
- `diffusion_pos` can reconstruct and save a nonzero $\ell_\theta$ after an otherwise radial displacement
- general SSA coordinate drifts perform unnecessary $x$ and $z$ arithmetic before boundaries reset them
- collision and I/O conversions repeatedly reconstruct nearly, rather than exactly, midplane geometry

The shared host/device geometry helpers now enforce the exact contract

$$
R=
\begin{cases}
y, & N_Z=1,\\
y\sin z, & N_Z>1,
\end{cases}
\qquad
Z=
\begin{cases}
0, & N_Z=1,\\
y\cos z, & N_Z>1.
\end{cases}
$$

The helper names should follow the existing cylindrical convention, for example `_get_cyl_R(y,z)` and `_get_cyl_Z(y,z)`

Use them consistently in

- `particle_init.cu`
- `_transport.cuh`
- `diffusion_pos.cu`
- `dyn_rate_calc.cu`
- `_collision.cuh`
- `col_rate_calc.cu` and `col_event_run.cu`
- `col_site_init.cu`
- `optdepth_depo.cu`
- host initialization, velocity-file conversion, and restart paths in `swarm_host.cuh`

For `N_Z == 1`, initialize and save directly

$$
v_y=v_R,
\qquad
\ell_\theta=0,
$$

rather than projecting through trigonometric functions

For `N_X == 1 && N_Z == 1`, add an explicit compile-time radial path in the SSA drifts

$$
x_1=x_j=x_{\rm ref},
\qquad
z_1=z_j=\frac{\pi}{2},
\qquad
\ell_{\theta,1}=\ell_{\theta,j}=0,
$$

while still evolving $R$, $v_R$, and $\ell_\phi$

The diffusion radial path should preserve the selected unchanged-physical-velocity closure without spherical round trips

1. retain $v_\phi=\ell_\phi/R$ and $v_R$
2. draw and reflect the new radius
3. set $x=x_{\rm ref}$ and $z=\pi/2$
4. write $\ell_\phi=R_{\rm new}v_\phi$, $v_y=v_R$, and $\ell_\theta=0$

On restart, `load_particle_data` now locks both inactive coordinates, setting $x=x_{\rm ref}$ when `N_X == 1` and resetting $z$ and $\ell_\theta$ when `N_Z == 1`

The model-local constants should satisfy

$$
Z_{\min}=Z_{\max}=\frac{\pi}{2}.
$$

Host coordinate helpers should nevertheless return the exact midplane for `N_Z == 1` so density integration cannot silently depend on inactive polar bounds

### R2 — correct imported-gas Stokes dimensions in every vertically integrated model

For `N_Z == 1`, imported `gasdens` is $\Sigma_g$, not $\rho_g$. The corrected imported branch in `_get_stokes` therefore uses the vertically integrated expression without an additional $H_g$

The required vertically integrated branch is

$$
\mathrm{St}(R,s)
=\mathrm{STOKES}_0
\frac{s}{S_0}
\frac{\Sigma_0}{\Sigma_g(R)}.
$$

For `N_Z > 1`, retain the volume-density formula

$$
\mathrm{St}(R,Z,s)
=\mathrm{STOKES}_0
\frac{s}{S_0}
\frac{\rho_{g,0}H_{g,0}}
{\rho_g(R,Z)H_g(R)}.
$$

The implementation should use dimensionally descriptive locals: `sigma_g` in the vertically integrated branch and `rhog` in the vertically resolved branch

`CONST_ST` remains a valid analytic-gas experiment in the radial geometry, in which case the prescribed Stokes number is spatially constant. It must remain incompatible with `IMPORTGAS`: importing a spatially varying gas density while requesting constant Stokes coupling would give two contradictory closures, and the existing compile-time guard should be retained

This correction affects

- initialization velocities
- every transport drag solve
- collision turbulent relative velocities
- any future imported-gas timestep estimate that depends on stopping time

It also fixes the existing radial–azimuthal imported-gas model, not only the new radial model

### R3 — use the exact annular KNN measure in radial-only collisions

The previous one-dimensional branch started from a clipped line length and multiplied by the circumference at the query radius. The implemented early radial branch instead uses the exact annulus, including neighborhoods that reach both boundaries

For radial-only geometry, let $a=d_K$ be the farthest valid KNN distance and define

$$
R_{\rm lo}=\max(Y_{\min},R_i-a),
\qquad
R_{\rm hi}=\min(Y_{\max},R_i+a).
$$

The accessible surface area must be

$$
A_{K,i}
=\pi\left(R_{\rm hi}^2-R_{\rm lo}^2\right).
$$

This is implemented as an early exact branch in `_get_ball_measure` when `N_X == 1 && N_Z == 1`, before the planar cap multipliers and local-circumference factor

The existing locally planar 2D and 3D boundary corrections should remain unchanged because replacing them is a separate geometric project

### R4 — use imported $\Sigma_g$ in collision turbulence

The physical collision kernel evaluates turbulent relative speeds through the Reynolds number. The corrected path supplies imported $\Sigma_g$ to `_get_re_inv_sqrt` whenever a vertically integrated `IMPORTGAS` model is active

For an imported radial model, this is inconsistent with the imported Stokes number: a gap would change drag but would not change the Reynolds-number contribution to the turbulent collision speed

For `N_Z == 1`, evaluate the local or pair-centred imported surface density and use

$$
\mathrm{Re}(R)
=\mathrm{REYNOLDS}_0
\frac{\alpha(R)}{\alpha_0}
\frac{\Sigma_g(R)}{\Sigma_0}
$$

in code units, or

$$
\mathrm{Re}(R)
=\frac{1}{2}
\frac{\alpha(R)\Sigma_g(R)X_{\rm SEC}}{M_{\rm MOL}}
$$

in physical units

The required local `sigma_g` is passed through `_get_vrel_t` and `_get_re_inv_sqrt`, with analytic `_get_sigma_g(R)` retained when imported surface density is unavailable

The pair location should be defined once and used consistently. For the radial model, a suitable choice is the arithmetic midpoint $R=(R_i+R_j)/2$ followed by radial grid interpolation of $\Sigma_g$

The imported temperature and scale height remain analytic because the file interface currently imports density and velocity, not temperature

### R5 — make radial KNN support explicit and test both backends

The current search coordinates already give exact radial distances if all particles lie on one ray

$$
|\boldsymbol r_i-\boldsymbol r_j|=|R_i-R_j|.
$$

For the radial branch, `col_site_init` now writes the canonical search point

$$
\boldsymbol r_i=(R_i,0,0),
$$

independent of the arbitrary inactive azimuth

The KD-tree can search this collinear point set without an algorithm change. The Morton implementation currently supports dimensions two and three; the radial points can remain a collinear two-dimensional embedding. A dedicated one-dimensional Morton encoding is an optional performance optimization, not a correctness prerequisite

Required validation is nevertheless broader than compilation

- exact KD-tree versus brute-force neighbor identifiers and distances on radial lines
- exact Morton versus brute-force results using the production two-dimensional embedding
- coincident radii and equal-distance ties on both sides of a query
- sparse searches returning fewer than `N_K` valid records
- query radii at the inner and outer boundaries
- search radii reaching both domain boundaries
- inactive or absorbed particles excluded from physical rates
- zero traversal overflows

The radial tests must verify that the farthest valid distance used by the area normalization is the same distance returned by the selected neighbor set

### R6 — preserve and document the imported-data normalization contract

For radial `IMPORTGAS` models, files must mean

- `gasdens`: $\Sigma_g(R)$
- `epsilon`: radial sampling weight proportional to the desired dust-to-gas surface-density ratio
- `gasvelx`: physical $v_{\phi,g}(R)$
- `gasvely`: physical $v_{R,g}(R)$
- `gasvelz`: retained for file-layout compatibility but dynamically ignored

The current code uses `gasdens*epsilon` to determine the spatial CDF, then normalizes represented particle masses to the configured `SIGMA_0`, `METAL_Z`, and convolved-domain mass. Thus imported `epsilon` controls the radial shape, not an independent absolute total dust mass

This behavior is retained for the first radial implementation because it matches the current mass-allocation decision, and `variables.txt` records the imported epsilon normalization as shape-only

If imported `epsilon` should later determine the absolute total mass, that must be a separate, explicit normalization mode with new tests

### R7 — expose density and geometry semantics in metadata

Configuration output now records

- geometry: radial, radial–azimuthal, radial–polar, or full 3D
- dust density kind: surface or volume
- imported gas density kind when enabled
- midplane dynamical closure for the radial model
- inactive-coordinate values

The binary layout does not need to change. In a radial model

- `dustdens_<frame>.dat` contains `N_Y` cell-centred values of $\Sigma_d$
- `optdepth_<frame>.dat` contains `N_Y` radial outer-face cumulative optical depths
- particle files still contain all three positions and linear velocities

### R8 — extend verification infrastructure without weakening existing tests

`qav/swarm/test_common/const_defs.cuh` now permits `N_X == 1` for dedicated radial cases while retaining geometry requirements local to each test case

Dedicated radial cases were added without changing the expected values of existing 2D and 3D tests

## Changes that are not required

The following redesigns are unnecessary for accurate first-stage radial support

- no new particle structure
- no separate radial runtime
- no `RADIAL_1D` preprocessor flag
- no removal of azimuthal angular momentum
- no downgrade from `real3`
- no new diffusion equation
- no new optical-depth algorithm
- no strict vertical averaging of drag or radiation
- no dedicated one-dimensional KD-tree or Morton hierarchy
- no change to the collision Bernoulli-leap algorithm solely because the geometry is radial

## Timestep requirements

The radial model must retain the orbital rate

$$
\Delta t\lesssim\frac{\mathrm{CFL}_{\rm DYN}}{\Omega_K}
$$

even though azimuth is not a spatial coordinate. Radial epicyclic motion and centrifugal acceleration still occur on the orbital timescale

The required active limits are

- orbital rate
- particle radial cell crossing $|v_R|/\Delta R$
- imported or analytic gas radial target crossing $|v_{R,g}|/\Delta R$
- net radial acceleration displacement
- radial diffusion RMS displacement
- deterministic radial diffusion drift
- `DT_MAX` and the remaining output interval

Azimuthal and polar crossing limits must remain inactive

### Conservative centrifugal hardening

The acceleration estimate now bounds centrifugal acceleration using the present angular momentum, zero angular momentum, and the largest gas-target angular momentum from the two imported temporal endpoints

Specifically, form

$$
\ell_{\max}=\max(|\ell_\phi|,|\ell_{\phi,g}|)
$$

where $|\ell_{\phi,g}|$ is maximized over both imported temporal endpoints, and include $\ell_{\max}^2/R^3$ in the acceleration-displacement estimate. This is cross-cutting timestep hardening rather than a dimensional correction; dedicated timestep stress coverage remains desirable before production use

## Boundary behavior

The first radial implementation should retain the established operator policies

### Transport

Particles with

$$
R<Y_{\min}
\quad\text{or}\quad
R\ge Y_{\max}
$$

are absorbed and parked at the inactive state. This represents open advective boundaries

The crossing time is resolved only to the dynamics timestep. Boundary-loss convergence must therefore be checked under timestep refinement; exact event localization can remain a later enhancement

### Diffusion

Stochastic crossings reflect repeatedly until they lie inside the radial interval, with the outer endpoint shifted infinitesimally inward. This implements zero diffusive flux

### Inactive coordinates

Initialization, every operator, output conversion, and restart must preserve

$$
x=x_{\rm ref},
\qquad
z=\frac{\pi}{2},
\qquad
\ell_\theta=v_\theta=0
$$

to machine-exact assignment, not merely within a trigonometric tolerance

## File-by-file implementation map

| File | Required radial work |
|---|---|
| `inc/swarm/param_grid.cuh` | add exact cylindrical $R,Z$ helpers or equivalent compile-time geometry primitives; make inactive polar host/device locations exact |
| `inc/swarm/param_phys.cuh` | branch imported Stokes scaling by density kind; provide imported $\Sigma_g$ to collision Reynolds calculations |
| `inc/swarm/_transport.cuh` | add exact radial SSA drift and exact inactive-coordinate assignments |
| `inc/swarm/_diffusion.cuh` | retain the present cylindrical radial drift formula; no equation change |
| `inc/swarm/_collision.cuh` | add exact annular area and imported-$\Sigma_g$ turbulent Reynolds path |
| `inc/swarm/swarm_host.cuh` | lock inactive coordinates on load, use exact midplane geometry, document file semantics, and write geometry metadata |
| `src/swarm/particle_init.cu` | initialize $R=y$, $Z=0$, $v_y=v_R$, and $\ell_\theta=0$ exactly |
| `src/swarm/diffusion_pos.cu` | add direct radial redistribution preserving $v_R$ and $v_\phi$ without spherical round trips |
| `src/swarm/dyn_rate_calc.cu` | retain only active spatial rates and add the conservative gas-target centrifugal bound |
| `src/swarm/col_site_init.cu` | emit canonical collinear radial search points |
| `src/swarm/col_rate_calc.cu` | use the exact radial area returned by `_get_ball_measure` and validate radial neighbor counts |
| `src/swarm/col_event_run.cu` | use the identical area and neighbor contract used during rate construction |
| `src/swarm/optdepth_depo.cu` | use exact $R=y$ and the well-mixed midplane closure |
| `src/swarm/swarm_runtime.cu` | keep Morton radial search as a 2D collinear embedding; no separate runtime |
| `qav/swarm/test_common/` | permit radial geometry and add independent analytical references |
| `qav/swarm/test_knn/` | add radial-line KD-tree/Morton/brute-force cases and annular-area checks |

## Required verification suite

### 1. Geometry, deposition, and mass

Place known represented masses in radial cells and verify

$$
\Sigma_{d,i}
=\frac{M_i}{\pi(R_{i+1}^2-R_i^2)}.
$$

Verify

- exact annular cell areas
- radial interpolation at interior cells and both boundaries
- total deposited mass
- convergence of the initialized convolved profile and its integrated mass
- exact inactive coordinate values in the initial particle file

Suggested model: `test_grid_1d`

### 2. Circular orbit and temporal convergence

Without radiation, initialize

$$
v_R=0,
\qquad
\ell_\phi=\sqrt{GM_SR}
$$

and verify constant radius and angular momentum over one or more orbits

Unlike the current equilibrium-only orbit test, add at least one non-equilibrium epicycle so temporal order is measurable rather than hidden by an exact fixed point

Suggested model: `test_orbit_1d`

### 3. Frozen drag

Use constant gas targets and stopping time. Verify the exact exponential angular response

$$
\ell_\phi(t)
=\ell_{\phi,g}
+\left(\ell_{\phi,0}-\ell_{\phi,g}\right)e^{-t/t_s}
$$

and a force configuration with an independently evaluable radial response

Test both analytic and imported gas, including stiff $t_s\ll\Delta t$

Suggested models: `test_drag_1d` and `test_import_1d`

### 4. Imported surface-density Stokes scaling

Provide a known radial $\Sigma_g$ profile and verify

$$
\frac{\mathrm{St}(R_2)}{\mathrm{St}(R_1)}
=\frac{\Sigma_g(R_1)}{\Sigma_g(R_2)}.
$$

A gap depleted by a factor $f$ must increase Stokes number by exactly $1/f$. The same test should verify the corresponding imported-$\Sigma_g$ Reynolds-number scaling used by turbulent collision velocities

### 5. Steady radial drift and viscous flow

Compare initialization and evolved motion with the no-backreaction steady-drift solution, first with $v_{R,g}=0$ and then with `VISC_FLOW`

The comparison should include $v_R$, $v_\phi$, $\ell_\phi$, and radial mass flux

### 6. Radial diffusion

For constant $D_R$ and a narrow ring away from boundaries, verify

$$
\left\langle R^2(t)\right\rangle
=R_0^2+4D_Rt.
$$

The unbounded axisymmetric Green function is

$$
\Sigma(R,t)
=\frac{M}{4\pi D_Rt}
\exp\left[-\frac{R^2+R_0^2}{4D_Rt}\right]
I_0\left(\frac{RR_0}{2D_Rt}\right).
$$

Compare the radial histogram with this solution before boundaries are felt. Then test reflecting boundaries separately by verifying mass conservation and convergence toward uniform $\Sigma_d$, whose radial sampling probability is proportional to $R$

Also verify after every diffusion call that $x=x_{\rm ref}$, $z=\pi/2$, $\ell_\theta=0$, and the intended $v_R,v_\phi$ closure is preserved exactly

Suggested model: `test_diffusion_1d`

### 7. Optical depth and radiation

For prescribed $\Sigma_d(R)$, compare the GPU prefix sum with direct cell integration of

$$
\tau(R)
=\int_{Y_{\min}}^R
\kappa(R')
\frac{\Sigma_d(R')}{\sqrt{2\pi}H_g(R')}
\,dR'.
$$

Then verify

- optically thin reduced gravity
- a prescribed attenuated profile $\beta e^{-\tau(R)}$
- midpoint opacity reconstruction during radial motion
- temporal convergence of a non-equilibrium radiation trajectory

Suggested model: `test_radiation_1d`

### 8. Poynting–Robertson drag

In the optically thin, slow-drift circular limit, verify

$$
R^2(t)
=R_0^2-\frac{4\beta GM_S}{C_{\rm LIGHT}}t.
$$

Also verify the one-step component damping rates $\gamma_{\rm PR}$ for $\ell_\phi$ and $2\gamma_{\rm PR}$ for $v_R$

Suggested model: `test_prdrag_1d`

### 9. Radial KNN geometry and collision kernels

Test both KD-tree and Morton against exhaustive one-dimensional brute force

For each query, verify

- valid neighbor count
- original particle identifiers
- squared radial distances
- farthest valid distance
- exact annular area
- zero backend overflows

Cover interior, inner-edge, outer-edge, both-edge, coincident-radius, equal-distance, sparse, and fewer-than-`N_K` cases

For physical collisions, verify the complete normalization

$$
\lambda_{ij}
=\frac{N_j\sigma_{ij}\Delta v_{ij}}
{A_{K,i}\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)}}.
$$

Also rerun constant, additive, product, and custom kernels, event mass conservation, fragmentation bounds, and convergence with `CFL_COL`, `N_K`, and `N_P`

Suggested model: `test_collision_1d` plus radial cases in `test_knn`

### 10. Boundary behavior

Use deterministic trajectories to verify

- absorption at both transport boundaries
- represented mass loss equal to absorbed particle weight
- reflection at both diffusion boundaries
- no reactivation of parked particles
- convergence of boundary-loss time under timestep refinement

### 11. Restart and file semantics

Restart transport, diffusion, and collision runs and verify

- particle state and RNG continuation match uninterrupted runs under the existing reproducibility contract
- inactive coordinates are relocked on load
- `velocity_x` is physical $v_\phi$ in files and returns to internal $\ell_\phi$ on load
- radial density and optical-depth file sizes are exactly `N_Y*sizeof(real)`

### 12. Coupled feature matrix

At minimum exercise

- transport only
- transport plus diffusion
- transport plus radiation
- transport plus diffusion plus radiation
- `PR_EFFECT`
- `VISC_FLOW`
- collision only
- transport plus collisions
- transport, diffusion, radiation, and collisions together
- analytic gas and imported gas for every supported relevant combination
- both KD-tree and Morton collision backends

Coupled cases without a closed analytical solution should be tested by timestep, particle-number, and operator-splitting convergence rather than by declaring one numerical run ground truth

## Cross-dimensional equivalence test

Run the same axisymmetric disk in two representations

1. radial only with `N_X == 1 && N_Z == 1`
2. radial–azimuthal with `N_X > 1 && N_Z == 1` and azimuthally uniform gas and dust

For deterministic axisymmetric transport, paired particles with identical $R$, $v_R$, and $\ell_\phi$ should follow the same radial trajectories after inactive $x$ is ignored

For deposition, radiation, diffusion, and collisions, finite-particle azimuthal noise prevents general bytewise agreement. Use either deterministic ring-balanced particles or compare ensemble statistics. For radiation, ring-average the radial–azimuthal optical depth in the comparison case so the test measures dimensional reduction rather than azimuthal sampling noise

Compare

- radial $\Sigma_d$
- total mass and boundary losses
- $v_R$, $v_\phi$, and $\ell_\phi$ statistics
- local analytic and imported Stokes numbers
- optical depth and radiation acceleration
- radial diffusion moments and distributions
- collision rates and size distributions

The existing radial–azimuthal, radial–polar where retained, full-3D, and KNN suites must continue to pass unchanged

## Implementation sequence

1. add radial geometry cases to the verification constants without changing production code
2. add exact midplane geometry helpers and inactive-state assignments
3. add the explicit radial SSA and diffusion paths
4. correct the imported surface-density Stokes branch
5. correct imported-$\Sigma_g$ collision turbulence
6. add the exact annular collision measure
7. add radial-line KD-tree and Morton tests
8. add radial grid, orbit, drag, diffusion, radiation, P-R, collision, import, boundary, and restart cases
9. add one production-like radial model with a model-local `const_defs.cuh`
10. run the complete radial feature matrix
11. run the complete pre-existing swarm suite after every production change
12. perform the cross-dimensional radial versus radial–azimuthal comparison
13. update `numerics_swarm.md` and `testset_swarm.md` only after the implementation and native CUDA evidence exist

## Deliberately deferred strict vertical averaging

The first radial model will not vertically average every dynamical coefficient

A strict column-dynamical model would require vertical quadrature of gas drag, gas target velocities, gravity, radiation attenuation, temperature-dependent collision velocities, and possibly size-dependent vertical profiles

For example, a Gaussian well-mixed column with Epstein drag can produce an effective stopping time different from the midplane value. Introducing such factors silently would change the established vertically integrated radial–azimuthal model

Strict vertical averaging should therefore remain a separate physical extension with its own equations and tests

## Completion criteria

Radial-only swarm support is production-ready only when

1. $R=y$, $Z=0$, $x=x_{\rm ref}$, $z=\pi/2$, and $\ell_\theta=0$ are enforced exactly in every relevant path
2. `dustdens` and imported `gasdens` have consistent surface-density dimensions
3. imported Stokes and turbulent Reynolds scalings use the external $\Sigma_g$
4. radial collisions use exact annular area and exact KD-tree/Morton neighbor sets
5. every enabled physics module passes its radial analytical or statistical test
6. restart and output preserve the radial state contract
7. the radial model agrees with the azimuthally uniform radial–azimuthal reference at the expected deterministic or Monte Carlo tolerance
8. every pre-existing swarm verification case remains passing

Until all eight conditions hold, `N_X == 1 && N_Z == 1` should be treated as an implementation target rather than a validated production geometry

## References

- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Ormel & Cuzzi (2007), [turbulent relative velocities](https://arxiv.org/abs/astro-ph/0702303)
- Zsom & Dullemond (2008), [representative-particle coagulation](https://arxiv.org/abs/0807.5052)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](https://doi.org/10.1016/0019-1035(79)90050-2)
