# Swarm verification cases

The canonical detailed description is `doc/testset_swarm.md`. This file remains a compact index
beside the implementation.

## Implemented CUDA cases

| Model | Production calculation exercised | Analytical or statistical truth |
|---|---|---|
| `test_grid_2d` | 2D mass deposition, density conversion, optical-depth deposition, radial integration, azimuthal mean | one equal-mass particle at every interpolation centroid gives exactly known cell density and cumulative optical depth |
| `test_grid_3d` | full-3D spherical deposition geometry and optical-depth chain | same centroid construction with exact $r^2\,dr\,d\Omega$ cell measures |
| `test_orbit_2d` | complete non-radiative SSA transport kernel | a pressure-free circular Kepler orbit returns to its initial state after $2\pi$ and provides a timestep-convergence sequence |
| `test_drag_2d` | frozen-coefficient semi-analytic gas drag | $\ell(t)=\ell_g+(\ell_0-\ell_g)e^{-t/t_s}$ at the known midpoint radius |
| `test_diffusion_2d` | azimuthal cylindrical SDE and velocity reprojection | zero mean, angular variance $2D\Delta t/R^2$ evaluated at $R=1$, and invariant Cartesian velocity |
| `test_diffusion_3d` | cylindrical radial and vertical SDE mapped through spherical storage | $E[\Delta R]=D\Delta t/R$, $E[\Delta Z]=0$, both variances $2D\Delta t$, and invariant Cartesian velocity |
| `test_radiation_2d` | radiation midpoint split without P-R drag | exact frozen-midpoint angular relaxation, radiation-modified radial force response, and completed radial drift |
| `test_prdrag_2d` | combined gas and P-R exponential response | exact component-dependent exponential responses with $k_x=t_s^{-1}+\beta GM/(cR^2)$ and $k_y=t_s^{-1}+2\beta GM/(cR^2)$ |
| `test_collision_2d` | collision helper kernels in a radial-azimuthal disk | exact disk area/circular-cap measure and constant, additive, and product kernel numerators |
| `test_collision_3d` | collision helper kernels in full 3D | exact interior, radial-cap, and polar-cap ball measures plus the same three analytic kernel numerators |

Diffusion tests use six-standard-error acceptance bands for sample means and variances.  Increasing `--res` raises the
ensemble as $N_P=16N^2$ and therefore tightens those statistical bounds without pretending that stochastic trajectories
should converge pointwise. The $N=32$ and $N=64$ ensembles do not resolve the prepared radial Itô-drift signal within six
standard errors; use $N\ge128$ when interpreting that specific mean shift.

## Important remaining tests

The first implementation deliberately stops short of claiming complete swarm validation.  The following require additional
test-local drivers or controlled input datasets:

- KD-tree neighbor identities and periodic-image queries checked against brute force
- event-level coagulation and fragmentation distributions over many frozen collision batches
- analytic constant/additive/product Smoluchowski moment evolution
- initialization CDFs, finite-sample mass normalization, and multisize conditional vertical distributions
- imported-gas interpolation in space and time, including the analytical Stokes anchor
- checkpoint/restart bitwise continuation including random states
- transport boundary absorption and half-disk reflection statistics
- secular reduced-gravity orbit and P-R inspiral over many orbital periods
- coupled transport-diffusion-radiation and transport-diffusion-collision operator tests

These omissions are coverage gaps, not accepted evidence that the corresponding calculations are correct.
