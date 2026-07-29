# Future radial-only swarm model

## Purpose

This document specifies how to support a radial-only swarm model without changing the physical meaning or numerical behavior of the existing radial-azimuthal and full-3D models

The radial model uses

$$
N_X=1,
\qquad
N_Y>1,
\qquad
N_Z=1
$$

and represents a vertically integrated, azimuthally symmetric disk

The agreed closure is **vertically integrated density with midplane dynamics**

- dust and gas grid densities are surface densities
- representative particles evolve radial position, radial velocity, and azimuthal angular momentum at the midplane
- the inactive azimuthal and polar coordinates remain in the particle structure for interface and checkpoint compatibility
- vertical density profiles are reconstructed analytically only where opacity or collision rates require a volume-density closure

This is not a strict vertical average of every force

## State represented by one particle

Each representative particle describes a complete axisymmetric dust column or annulus rather than one localized azimuthal particle

The dynamically meaningful state is

$$
R,
\qquad
v_R,
\qquad
\ell_\phi=Rv_\phi
$$

The inactive coordinates satisfy

$$
x=\frac{X_{\min}+X_{\max}}{2},
\qquad
z=\frac{\pi}{2},
\qquad
\ell_\theta=0
$$

Azimuth is spatially inactive, but $v_\phi$ and $\ell_\phi$ must remain dynamical because they determine orbital support, radial centrifugal acceleration, gas drag, and Poynting-Robertson drag

The existing `swarm` structure and particle-file layout should be retained so the new geometry does not create a separate binary interface

## Vertically integrated density

The evolved radial density is

$$
\Sigma_d(R,t)
=
\int_{-\infty}^{+\infty}\rho_d(R,Z,t)\,dZ
$$

The well-mixed vertical closure is

$$
\rho_d(R,Z,t)
=
\frac{\Sigma_d(R,t)}{\sqrt{2\pi}H_g(R)}
\exp\left[-\frac{Z^2}{2H_g^2(R)}\right]
$$

All grain species use the gas scale height in this closure, independent of size and Stokes number

Midplane dynamics evaluate gas drag, pressure-supported rotation, gravity, radiation pressure, and collision velocities at $Z=0$

## Existing code that already reduces correctly

### Grid measure and deposition

The existing mesh helpers use radial measure power $d=2$ when `N_Z == 1` and the complete azimuthal extent $2\pi$ when `N_X == 1`

The exact radial cell area is therefore

$$
A_i
=
2\pi\frac{R_{i+1}^2-R_i^2}{2}
=
\pi\left(R_{i+1}^2-R_i^2\right)
$$

Particle initialization, total-mass integration, and density deposition already use this measure, so `dustdens` becomes $\Sigma_d$ in the radial model

### Initialization

The existing initializer samples the radial probability distribution proportional to

$$
2\pi R\Sigma_d(R)\,dR
$$

It also locks $z=\pi/2$ and $\ell_\theta=0$ when the polar dimension is inactive

The current steady no-backreaction drift initialization remains applicable at the midplane

### Transport and timestep

At $z=\pi/2$ and $\ell_\theta=0$, the existing staggered semi-analytic equations reduce to radial motion coupled to azimuthal angular momentum

The orbital timestep limit

$$
\Delta t\lesssim\frac{\mathrm{CFL}_{\rm dyn}}{\Omega_K}
$$

must remain active even though azimuth is not a spatial dimension because it resolves radial epicyclic and centrifugal dynamics

### Radial diffusion

The current radial stochastic increment is

$$
dR
=
\left[
\frac{dD_R}{dR}
+
\frac{D_R}{R}
\right]dt
+
\sqrt{2D_R}\,dW
$$

which corresponds to cylindrical density diffusion

$$
\frac{\partial\Sigma_d}{\partial t}
=
\frac{1}{R}
\frac{\partial}{\partial R}
\left(
R D_R\frac{\partial\Sigma_d}{\partial R}
\right)
$$

Azimuthal and vertical stochastic increments remain disabled

### Viscous gas flow

The existing `N_Z == 1` prescription already uses the vertically integrated radial gas velocity and can be retained unchanged

### Radiation and optical depth

For `N_Z == 1`, the opacity deposition reconstructs the midplane dust density through

$$
\rho_{d,0}(R)
=
\frac{\Sigma_d(R)}{\sqrt{2\pi}H_g(R)}
$$

The radial prefix sum then produces the midplane optical depth used by the radiation force

This is consistent with the adopted midplane dynamical closure

### Poynting-Robertson drag

The existing P-R damping remains valid at the midplane

$$
\gamma_{\rm PR}
=
\frac{\beta GM_S}{cR^2}
$$

Azimuthal angular momentum is damped at $\gamma_{\rm PR}$ and radial velocity at $2\gamma_{\rm PR}$

### Collision vertical overlap

For a physical collision kernel, the existing well-mixed overlap thickness

$$
\sqrt{2\pi\left(H_{g,i}^2+H_{g,j}^2\right)}
$$

correctly converts a surface-neighbor measure into an effective volume measure when the collision velocity is evaluated using the midplane closure

### Operator composition and I/O

The existing composition

$$
C^{1/2}D^{1/2}TD^{1/2}C^{1/2}
$$

remains valid

Particle files should continue storing all three positions and physical linear velocities, with inactive values fixed at $x=(X_{\min}+X_{\max})/2$, $z=\pi/2$, and $v_\theta=0$

## Mandatory production corrections

### Imported gas density contract

In every vertically integrated model, imported `gasdens` must mean gas surface density $\Sigma_g$

The current imported-position sampler already assumes this because it multiplies `gasdens` by cell area, but the current Stokes-number helper treats the same field as volume density

For `N_Z == 1`, the local Stokes number must be

$$
\mathrm{St}(R,s)
=
\mathrm{STOKES}_0
\frac{s}{S_0}
\frac{\Sigma_0}{\Sigma_g(R)}
$$

For `N_Z > 1`, retain the volume-density prescription

$$
\mathrm{St}(R,Z,s)
=
\mathrm{STOKES}_0
\frac{s}{S_0}
\frac{\rho_{g,0}H_{g,0}}
{\rho_g(R,Z)H_g(R)}
$$

This branch is dimensionally necessary and should also correct imported gas behavior in the existing vertically integrated radial-azimuthal model

### Exact radial collision measure

The current radial-only KNN measure clips a one-dimensional interval and multiplies it by the circumference at the query particle, which is exact away from radial boundaries but approximate near them

For `N_X == 1 && N_Z == 1`, use

$$
R_{\rm lo}=\max(Y_{\min},R-d_K),
\qquad
R_{\rm hi}=\min(Y_{\max},R+d_K)
$$

and normalize by the exact accessible annular area

$$
A_K
=
\pi\left(R_{\rm hi}^2-R_{\rm lo}^2\right)
$$

The existing 2D and 3D neighborhood-cap calculations should remain unchanged

### Radial verification support

The common swarm verification constants currently require an active azimuthal grid, so the test harness cannot compile a radial-only case

Replace the global test restriction with test-specific geometry requirements and add dedicated radial models under `qav/swarm/`

## Recommended isolation measures

### Infer geometry from dimensions

Do not add an independent `RADIAL_1D` configuration flag

The code should infer three compile-time properties from the grid dimensions

- axisymmetric grid when `N_X == 1`
- vertically integrated grid when `N_Z == 1`
- radial-only grid when `N_X == 1 && N_Z == 1`

This prevents a flag from contradicting the configured mesh

### Add an explicit radial SSA path

The current general SSA path calculates temporary azimuthal and polar coordinates and later resets them through the boundary helper

Although the final result is currently correct, a dedicated radial branch should set

$$
x_1=x_i,
\qquad
z_1=\frac{\pi}{2},
\qquad
x_j=x_i,
\qquad
z_j=\frac{\pi}{2},
\qquad
\ell_{\theta,j}=0
$$

and advance only $R$, $v_R$, and $\ell_\phi$

The general equations should remain untouched for every other geometry

### Preserve operator-specific boundaries

The current policies should initially remain unchanged

- radial transport absorbs particles crossing the domain boundaries
- radial diffusion reflects stochastic boundary crossings
- inactive azimuth and polar coordinates remain locked

Any future change to these policies should be treated as a separate physical decision rather than part of enabling radial geometry

### Document density semantics

Configuration output and numerical documentation should state that

- `dustdens` and imported `gasdens` are surface densities when `N_Z == 1`
- `dustdens` and imported `gasdens` are volume densities when `N_Z > 1`
- `velocity_x` remains physical $v_\phi$ even when azimuth is spatially inactive

## Implementation sequence

1. add a radial production model with a model-local `const_defs.cuh`
2. add inferred compile-time geometry properties
3. add radial-only analytical tests that initially expose the unsupported behavior
4. correct the vertically integrated imported-gas Stokes-number branch
5. add the exact radial annular collision measure
6. add the explicit inactive-coordinate SSA path
7. update density and imported-field documentation
8. run every radial test and every existing swarm test after each production change
9. compare an azimuthally uniform radial-azimuthal model against the radial-only model
10. declare radial production support only after the complete regression matrix passes

## Required verification cases

### Geometry and deposition

Deposit known ring masses and verify

$$
\Sigma_{d,i}
=
\frac{M_i}{\pi(R_{i+1}^2-R_i^2)}
$$

Check total represented mass and radial interpolation at both boundaries

### Circular orbit

Initialize

$$
v_R=0,
\qquad
\ell_\phi=\sqrt{GM_SR}
$$

and verify constant radius with temporal convergence

### Frozen drag

Use constant gas targets and stopping time and compare $v_R$ and $\ell_\phi$ with their exact exponential relaxation solutions

Test both analytic gas and imported gas

### Steady radial drift

Compare initialization and evolved motion with the no-backreaction steady-drift solution, first with zero radial gas velocity and then with `VISC_FLOW`

### Radial diffusion

For constant diffusivity and a narrow ring away from boundaries, verify

$$
\left\langle R^2(t)\right\rangle
=
R_0^2+4D_Rt
$$

Also compare the complete radial distribution with the cylindrical ring-diffusion Green function and verify statistical convergence with particle count and timestep

### Optical depth and radiation

Compare the computed prefix sum with direct integration of

$$
\rho_{d,0}(R)
=
\frac{\Sigma_d(R)}{\sqrt{2\pi}H_g(R)}
$$

Then test optically thin reduced gravity and one analytically prescribed attenuated force profile

### Poynting-Robertson drag

In the slow-drift circular limit, verify

$$
R^2(t)
=
R_0^2-
\frac{4\beta GM_S}{c}t
$$

### Collision geometry and kernels

- compare radial KNN neighbor sets with brute force
- test particles near the inner boundary, outer boundary, and with a neighborhood spanning the complete radial domain
- verify the exact annular measure
- rerun constant, additive, product, and physical custom-kernel tests
- verify Gaussian vertical-overlap normalization and total represented mass conservation

### Imported gas response

Provide a known $\Sigma_g(R)$ profile and verify the inverse surface-density Stokes scaling

A prescribed gap should increase the local Stokes number by the exact inverse depletion factor

### Coupled feature matrix

At minimum, exercise

- transport only
- transport and diffusion
- transport and radiation
- transport, diffusion, and radiation
- P-R drag
- viscous gas flow
- collisions with and without transport
- imported gas with representative supported physics combinations

### Cross-dimensional regression

Run the same azimuthally uniform disk as

1. a radial-azimuthal model with `N_X > 1` and identical data in every azimuthal cell
2. a radial-only model with `N_X == 1`

At matched radial resolution and Monte Carlo sampling, compare

- radial surface density
- total mass and boundary losses
- radial and azimuthal velocity statistics
- local Stokes number
- optical depth and radiation acceleration
- diffusion moments
- collision rates and size distributions

The existing radial-azimuthal and 3D verification suite must remain unchanged and continue to pass

## Deliberately deferred strict vertical averaging

The first radial model will not vertically average every force

A strict column-dynamical model would additionally require vertical quadrature for drag, gas target velocities, gravity, radiation attenuation, and height-dependent collision velocities

For example, Gaussian well-mixed dust with Epstein drag would give

$$
t_{s,\rm eff}=\sqrt{2}\,t_{s,0}
$$

rather than the adopted midplane stopping time

Such a formulation should be developed as a separate physical extension after the midplane-closure radial model is verified, because silently introducing these averages would change the established vertically integrated radial-azimuthal model

## Completion criteria

Radial-only swarm support can be considered production-ready when

1. the imported gas density contract is dimensionally consistent
2. radial collision neighborhoods use exact annular area
3. inactive coordinates remain exactly locked through initialization, evolution, output, and restart
4. every enabled physics module passes its radial analytical test
5. the radial model agrees with the azimuthally uniform radial-azimuthal reference within deterministic or Monte Carlo tolerances
6. every pre-existing radial-azimuthal and full-3D swarm verification case still passes
