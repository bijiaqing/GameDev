# Swarm verification cases

The canonical detailed description is `doc/testset_swarm.md`. This file remains a compact index
beside the implementation.

## Implemented HIP/ROCm cases

| Model | Production calculation exercised | Analytical or statistical truth |
|---|---|---|
| `test_grid_1d` | radial annular deposition and optical-depth chain | exact annular surface density, cumulative optical depth, and inactive-coordinate invariants |
| `test_grid_2d` | 2D mass deposition, density conversion, optical-depth deposition, radial integration, azimuthal mean | one equal-mass particle at every interpolation centroid gives exactly known cell density and cumulative optical depth |
| `test_grid_3d` | full-3D spherical deposition geometry and optical-depth chain | same centroid construction with exact $r^2\,dr\,d\Omega$ cell measures |
| `test_orbit_1d` | radial-only SSA transport | circular Kepler equilibrium with exact inactive coordinates |
| `test_orbit_2d` | complete non-radiative SSA transport kernel | a pressure-free circular Kepler orbit returns to its initial state after $2\pi$ and provides a timestep-convergence sequence |
| `test_drag_1d` | radial-only frozen gas drag | exact exponential angular and radial response with inactive-coordinate invariants |
| `test_viscflow_1d` | vertically integrated viscous gas flow in radial geometry | exact Kanagawa radial target and steady no-backreaction dust drift at several radii |
| `test_drag_2d` | frozen-coefficient semi-analytic gas drag | $\ell(t)=\ell_g+(\ell_0-\ell_g)e^{-t/t_s}$ at the known midpoint radius |
| `test_diffusion_1d` | radial cylindrical SDE | exact Itô mean and variance plus invariant physical velocity and inactive coordinates |
| `test_diffusion_2d` | azimuthal cylindrical SDE and velocity reprojection | zero mean, angular variance $2D\Delta t/R^2$ evaluated at $R=1$, and invariant Cartesian velocity |
| `test_diffusion_3d` | cylindrical radial and vertical SDE mapped through spherical storage | $E[\Delta R]=D\Delta t/R$, $E[\Delta Z]=0$, both variances $2D\Delta t$, and invariant Cartesian velocity |
| `test_initial_3d` | continuous finite-domain size-conditioned initialization | independent Gaussian containment and radial CDFs for endpoint and interpolated midpoint sizes, represented-mass closure, exact bounds, and invariance under changes to simulation $N_Z$ |
| `test_restart_2d` | production particle and hipRAND checkpoint save/load | a two-step diffusion trajectory must match save-clobber-reload continuation, including a byte-identical RNG file reload, byte-identical final positions, and roundoff-limited velocity serialization |
| `test_radiation_1d` | radial-only radiation midpoint response | exact frozen radiation response and inactive-state invariants |
| `test_radiation_2d` | radiation midpoint split without P-R drag | exact frozen-midpoint angular relaxation, radiation-modified radial force response, and completed radial drift |
| `test_prdrag_1d` | radial-only gas and P-R damping | exact component-dependent exponential response and inactive-state invariants |
| `test_prdrag_2d` | combined gas and P-R exponential response | exact component-dependent exponential responses with $k_x=t_s^{-1}+\beta GM/(cR^2)$ and $k_y=t_s^{-1}+2\beta GM/(cR^2)$ |
| `test_collision_1d` | exact radial annular KNN measure and collision kernel helpers | interior and boundary-clipped annular areas plus constant, additive, and product numerators |
| `test_collision_2d` | collision helper kernels in a radial-azimuthal disk | exact disk area/circular-cap measure and constant, additive, and product kernel numerators |
| `test_collision_3d` | collision helper kernels in full 3D | exact interior, radial-cap, and polar-cap ball measures plus the same three analytic kernel numerators |
| `test_import_1d` | imported surface-density coupling | exact Stokes and turbulent Reynolds scaling from external $\Sigma_g$ |
| `test_boundary_1d` | radial-only endpoint boundary helpers | exact radial absorption, repeated radial reflection, and inactive-coordinate locking |
| `test_boundary_2d` | narrow-wedge radial–azimuthal endpoint boundary helpers | exact periodic wrapping, radial absorption/reflection, and inactive polar state |
| `test_boundary_3d` | full-disk 3D endpoint boundary helpers | exact azimuthal wrapping and radial/polar absorption or reflection |
| `test_boundary_half` | half-disk 3D endpoint boundary helpers | exact lower-polar absorption and reflecting upper midplane |
| `test_knn` | KD-tree and adaptive-Morton exact KNN search, radial-line embedding, block-parallel sorted Morton top-$K$, periodic-image correctness, and compact production boundary ghosts | independent brute-force neighbor identities, adversarial boundary cases, overflow checks, and production-relevant timing/memory records |

The default KNN run uses $N_P=10^5$. Passing `--knn-full` to the common suite extends both the
ordinary and wedge matrices to $N_P=10^6$ and $10^7$. The focused radial run passed its
$N_P=10^5$ collinear benchmark and edge checks on 2026-07-31. A fresh complete `--knn-full` run is
still required to promote the $N_P=10^6$ and $10^7$ radial cases into the active manifest.

When selected through `--group radial`, `test_knn` runs only the collinear physical-1D ordinary
case and the radial-line and inactive-particle edge checks. The Morton implementation still embeds
the line as $(R,0)$ in its two-coordinate container, but the result is named `radial_1d_*` and
records `physical_dimension: 1`. These focused records are isolated under
`qav/swarm/out/test_knn/radial/`.

Diffusion tests use six-standard-error acceptance bands for sample means and variances.  Increasing `--res` raises the
ensemble as $N_P=16N^2$ and therefore tightens those statistical bounds without pretending that stochastic trajectories
should converge pointwise. The $N=32$ and $N=64$ ensembles do not resolve the prepared radial Itô-drift signal within six
standard errors; use $N\ge128$ when interpreting that specific mean shift.

## Important remaining tests

The first implementation deliberately stops short of claiming complete swarm validation.  The following require additional
test-local drivers or controlled input datasets:

- event-level coagulation and fragmentation distributions over many frozen collision batches
- analytic constant/additive/product Smoluchowski moment evolution
- monodisperse initialization and initial drift velocities in the complete production path
- imported-gas interpolation in time and full trajectory coupling beyond the isolated radial Stokes/Reynolds test
- within-step transport and stochastic boundary-event convergence
- secular reduced-gravity orbit and P-R inspiral over many orbital periods
- coupled transport-diffusion-radiation and transport-diffusion-collision operator tests

These omissions are coverage gaps, not accepted evidence that the corresponding calculations are correct.
