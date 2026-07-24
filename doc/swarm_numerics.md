# Lagrangian swarm numerics

## Scope and status

The active representative-particle implementation is under the root `inc/` and `src/` directories.
It supports analytic or imported gas, semi-analytic particle transport, stochastic turbulent
diffusion, radiation pressure, particle-to-grid diagnostics, and representative-particle
coagulation/fragmentation.

The serious source-level defects identified by the 2026-07-22 and 2026-07-23 audits, plus the
follow-up A1–B4 and C3 findings, have been corrected in the active tree. Those corrections have
not yet received the native CUDA analytical and convergence campaign already available for the
fluid solver. Source inspection is therefore stronger than runtime validation for this branch.

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

When `N_Z == 1`, every particle remains at $z=\pi/2$ and $\ell_\theta=0$. When `N_X == 1`, one
stored azimuthal cell represents the complete axisymmetric ring and particles are centered in the
inactive coordinate.

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

## Dust normalization and initialization

The physical dust profile starts from

$$
\Sigma_d(R)=Z_{\rm metal}\Sigma_g(R)
$$

before the shared-style Gaussian edge convolution. `get_dust_mass()` integrates that initialized
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
$v_Z=-\mathrm{St}\,\Omega_K Z$. `VISC_ACCRETION` replaces the zero gas radial velocity with the
analytic viscous prescription and requires `DIFFUSION`.

Imported gas initialization samples the imported $\rho_g\epsilon$ field using the same exact
cell-measure construction. Active azimuthal cells are sampled with `_get_dx()`; an inactive
azimuth is centered. Every dynamically evaluated local Stokes number uses the imported gas density.
`STOKES_0` is anchored to the analytical reference midplane through

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

## Semi-analytic transport

Particle trajectories use a staggered semi-analytic drag integrator. Coordinate drifts account for
the angular-momentum storage convention and the changing spherical radius. At the force stage,
frozen-coefficient drag is integrated exponentially, so a short stopping time is not an explicit
stability restriction.

The analytic force model includes:

- central gravity, optionally reduced by attenuated radiation pressure
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

A random displacement is a spatial redistribution, not an impulse. The kernel reconstructs the
particle's Cartesian velocity before moving it and projects that unchanged velocity into the new
local spherical basis afterward. This is internally consistent, but it is not the same momentum
closure as the fluid's donor-angular-momentum diffusion; density-only fluid–swarm diffusion
comparisons are valid until a common momentum equation is derived.

## Representative-particle collisions

Collision neighborhoods are built with a cuKD tree. Periodic azimuthal images and locally planar
accessible-volume corrections handle periodic seams and physical boundaries. For `N_X == 1`, the
neighborhood measure is multiplied by $2\pi R$, restoring the missing axisymmetric dimension.

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
vertical overlap closure.

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

Coagulation uses

$$
s_{\rm new}=(s_i^3+s_j^3)^{1/3}
$$

and adjusts the represented grain number to conserve the representative mass. The fragmentation
law is a configured modeling choice rather than a universal fragment distribution. Dimensionless
constant, additive, and product kernels are available for analytical tests; the additive kernel is
$K=m_i+m_j$.

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
- diffusion is periodic in azimuth and reflecting in radial and polar directions

An absorbed representative is parked at `y=0`, assigned zero stored velocity, and skipped by
subsequent transport, diffusion, deposition, opacity, timestep, and collision calculations.

## File and restart semantics

Science files store physical linear velocity $(v_\phi,v_r,v_\theta)$. Loading converts it back to
the internal angular variables before device evolution. Grain size and represented grain number
are included only for multisize builds, matching the declared binary dtype.

Random states are deterministically reinitialized on resume rather than serialized. A resumed
stochastic run therefore has the correct distribution but is not the same random realization as
an uninterrupted run.

## Current limitations and required tests

- Native CUDA analytical tests are still required for particle orbits, stiff drag, diffusion
  Green's functions, settling equilibrium, optical depth, radiation acceleration, boundaries, and
  all flag combinations
- Collision tests must cover constant, additive, and product kernels; physical
  $\sigma\Delta v/V$ scaling; `CFL_COL`; neighbor count; and particle-number convergence
- The current fluid and swarm diffusion-momentum closures differ and should not be compared as the
  same velocity equation
- A zero integrated dust mass, `N_K == 1`, or a polar domain reaching a coordinate singularity is
  not a scientifically meaningful configuration and is not fully guarded
- The locally planar KNN boundary-cap correction is asymptotically consistent, not an exact
  curved-boundary intersection
- Replacing the KD tree with spatial hashing is only a future performance option. It should be
  considered after profiling and must reproduce neighbor statistics and collision rates before
  adoption; earlier memory and schedule estimates were speculative

## References

- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Ormel & Cuzzi (2007), [turbulent relative velocities](https://arxiv.org/abs/astro-ph/0702303)
- Zsom & Dullemond (2008), [representative-particle coagulation](https://arxiv.org/abs/0807.5052)
- Gillespie (1977), [stochastic reaction simulation](https://doi.org/10.1021/j100540a008)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
