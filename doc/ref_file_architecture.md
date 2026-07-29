# Fluid–swarm architecture and consistency contract

## Objective and present state

GameDev contains two dust representations in one project:

- `fluid`: an Eulerian pressureless dust fluid
- `swarm`: Lagrangian representative particles
- `share`: representation-independent infrastructure and physical definitions

The two representations solve related physical models but require different data structures,
drivers, numerical operators, and tests. A single executable should select one representation at
build time; it should not carry both main loops behind a large forest of preprocessor branches.

The merged layout is active at the repository root:

- `inc/fluid/` and `src/fluid/` contain the Eulerian solver
- `inc/swarm/` and `src/swarm/` contain the Lagrangian solver
- `inc/share/` is reserved for representation-independent interfaces and currently contains no active interface
- `src/share/` is reserved for representation-independent kernels and currently contains no production source
- `mod/` contains production models and `qav/` contains representation-specific tests
- the root `Makefile` builds either representation according to the selected model's `DUST_REPR`

The two standalone swarm generations remain frozen under `legacy/` as recovery references, not as
active build trees.

## Build and driver design

Every model directory in `mod/`, `qav/fluid/`, or `qav/swarm/` provides `flags.mk`. A model that
changes numerical or physical constants may additionally provide `const_defs.cuh`; otherwise it
inherits the selected representation's header. The model directory is the first include-search
location, so an optional local header overrides the representation defaults without `MODEL_*`
compiler macros.

Each `flags.mk` contains exactly one

```make
DUST_REPR = fluid
```

or

```make
DUST_REPR = swarm
```

The build selects:

- one representation include directory
- one representation source/object list
- one driver

Fluid models additionally select `FLUID_SWEEP := thread` or `FLUID_SWEEP := block`. The choice
changes the CUDA line-kernel implementation and object directory; fluid tests also use it to
separate their output artifacts. Both variants use the same fluid state, driver composition, and
numerical equations.

`src/fluid/fluid_runtime.cu` and `src/swarm/swarm_runtime.cu` may both define the C++ function
`main`. They are separate translation units and only the selected one is linked into a target.
This keeps representation-specific allocations, time integration, restart behavior, and optional
physics legible.

Current commands from the repository root are:

```bash
make MODEL=fluid_fiducial
make MODEL=swarm_fiducial
```

## Common physical contract

Both representations use

$$
(x,y,z)=(\phi,r,\theta),
\qquad
R=y\sin z,
\qquad
Z=y\cos z.
$$

Both internal states use

$$
(\ell_\phi,v_r,\ell_\theta)
=(Rv_\phi,v_r,rv_\theta),
$$

while science files store $(v_\phi,v_r,v_\theta)$.

The common 2D interpretation is a vertically integrated radial–azimuthal disk:

$$
N_Z=1:\quad \Sigma_d\ [M/L^2],
$$

and the common 3D interpretation is a spherical volume:

$$
N_Z>1:\quad \rho_d\ [M/L^3].
$$

The common radial measure power is

$$
d=2+(N_Z>1).
$$

An inactive stored azimuth in an axisymmetric swarm calculation represents the complete $2\pi$
ring and must include that measure in deposition and collision normalization.

Both representations close the unresolved vertical structure of an `N_Z == 1` disk by assuming
all dust species share the gas scale height $H_g$. Models with `N_Z > 1` instead initialize
explicit size-dependent vertical density profiles.

The following physical definitions must remain algebraically identical:

- $G$, stellar mass, reference radius, and scalar precision
- Keplerian frequency
- gas surface density and aspect ratio
- exact vertically isothermal hydrostatic stratification
- pressure-support parameter
- analytic gas density
- viscosity and local alpha
- monodisperse Stokes scaling
- metallicity-based initial dust surface density
- radiation startup taper
- optical-depth outer-face convention
- optional viscous gas accretion

Both representations now use `STOKES_0`, `_get_stokes`, `_get_gas_strat`, `SCHMIDT_X`, `DT_MAX`,
`NB_G`, `CFL_DYN`, and local internal primitive names `lx`, `vy`, and `lz`.

## Intentional representation differences

Numerical algorithms should not be unified merely because they solve related equations.

| Fluid-owned | Swarm-owned |
|---|---|
| finite-volume density and momentum | representative-particle state |
| PPM reconstruction and HLL fluxes | semi-analytic trajectories |
| invariant-domain flux limiting | particle boundary mapping |
| FARGO orbital advection | particle/grid interpolation and deposition |
| SSPRK directional transport | cuRAND stochastic paths |
| Crank–Nicolson line solves | cylindrical diffusion SDE |
| fluid vacuum recovery | cuKD collision neighborhoods |
| conservative grid reductions | collision propensities and outcomes |

The diffusion bases are intentionally different:

$$
\text{fluid: diagonal in spherical }(x,y,z),
$$

$$
\text{swarm: diagonal in cylindrical }(\phi,R,Z).
$$

They represent different anisotropic tensors in 3D unless meridional diffusion is isotropic.
Cross-representation tests must either select an isotropic tensor or account for this distinction.

Imported time-dependent gas, multisize grains, and collisions are swarm-only capabilities at this
stage. Their absence from the fluid branch is not a consistency bug.

## Boundary and coupling contract

Both representations use periodic azimuth. Transport permits material to leave the radial domain
and uses the configured full-disk or `HALFDISK` polar policy. Diffusion uses zero-normal-flux
boundaries. The implementations differ:

- the fluid expresses transport boundaries through face fluxes and diffusion boundaries through
  CN matrix coefficients
- the swarm absorbs transport exits and reflects stochastic diffusion steps

The radiation force in both branches uses the same smoothstep startup and the same cumulative
outer-face optical-depth convention. The fluid samples at a cell center; the swarm continuously
interpolates to a particle.

For analytic gas, both initialize the no-backreaction steady radial/azimuthal drift. In 3D, their
settling state reflects their different diffusion coordinate bases. Imported-gas swarm
initialization uses the imported density for its local Stokes number, anchored by the analytical
reference values $\Sigma_0$ and $H_{g,0}$. Its initial gas velocity and pressure support remain
analytic, so exact equilibrium with an arbitrary imported velocity field remains
representation-specific work.

## Code suitable for `share`

The following components are safe candidates once their model-parameter inputs are made explicit:

| Component | Suggested shared responsibility |
|---|---|
| scalar type and universal constants | base definitions |
| CUDA error macros | host CUDA error reporting |
| binary file primitives | typed read/write and frame naming |
| common progress output | representation-neutral messages |
| grid spacing, faces, and exact measures | spherical mesh primitives |
| coordinate and velocity transforms | spherical/cylindrical/Cartesian conversions |
| analytic gas profiles | $\Omega_K$, $h_g$, $\eta$, stratification, $\Sigma_g$, $\rho_g$ |
| viscosity and viscous gas velocity | common disk physics |
| radiation taper | time-dependent scalar policy |
| radial optical-depth prefix sum | integration of an already constructed local increment |
| convolved radial surface profile | host initialization utility |

The monodisperse Stokes equation can share a common core. The swarm must retain adapters for grain
size and imported gas.

The local optical-depth increment remains representation-specific. The fluid kernel consumes the
Eulerian density field directly, whereas the swarm kernel normalizes extinction weight previously
deposited by representative particles. Both implementations therefore reside in their respective
`src/fluid/` and `src/swarm/` branches.

The two `optdepth_csum.cu` kernels currently perform the same radial prefix sum, but they remain
branch-local because their declarations belong to the representation kernel interfaces. They
should only be consolidated after a representation-neutral kernel interface exists; duplicating
this small kernel is clearer than labeling a branch-dependent translation unit as shared.

## Dependency rules

The shared include graph should flow in one direction:

```text
base constants and scalar types
    -> grid and coordinate geometry
    -> analytic physical profiles
    -> shared host or kernel interfaces
    -> fluid or swarm adapters
    -> representation kernels and main
```

Shared headers must not include a representation header. Representation headers may include shared
headers. Model constants should enter through an explicit selected configuration header rather
than by putting both branches' headers on the same include path.

The current swarm and fluid `param_grid.cuh` and `param_phys.cuh` files reuse the same include-guard
names. This is harmless in the present one-representation-per-build design because the Makefile
places only the selected branch on the include path. A future shared translation unit must first
extract their common layer or give the remaining adapters representation-qualified guards; it must
not include both current headers directly.

Both builds now explicitly request C++17. Keep one common NVCC base configuration after migration.
The launch-count formulas should eventually use exact ceiling division,

```cpp
(count + TPB - 1) / TPB
```

rather than always launching one extra block. This is a cleanup and efficiency item, not a current
correctness defect because every kernel guards its index.

## Open cross-representation decisions

### Diffusion momentum

The fluid carries donor $(\ell_\phi,v_r,\ell_\theta)$ with diffusive mass. The swarm preserves
Cartesian velocity during a random displacement. For an outward displacement, one keeps
$\ell_\phi$ approximately fixed and the other keeps $v_\phi$ fixed. They can therefore agree in
density and disagree in momentum.

A final closure should begin from

$$
\boldsymbol J=-\boldsymbol D\nabla\rho_d
$$

and derive a momentum equation consistent with that selected density flux, including coordinate
geometry and Galilean invariance. Huang & Bai provide useful structure for concentration diffusion,
but the equation must be re-derived for density diffusion. Until then, matched diffusion tests
should compare density separately from velocity.

### Perturbations and sampling noise

The fluid uses a reproducible deterministic azimuthal perturbation. The swarm has finite-sample
shot noise. Production clumping calculations need not erase this physical representation
difference, but a controlled comparison needs the same analytic probability perturbation plus
multiple swarm seeds or enough particles to quantify the expected $N_P^{-1/2}$ uncertainty.

### Validation parity

The fluid has operator and coupled analytical tests. `qav/swarm/` is still empty. Validation parity
is not complete until the swarm has:

- single-particle orbit and stiff-drag tests
- smooth ensemble advection projected onto the common grid
- cylindrical diffusion and settling tests
- optical-depth and radiation tests
- boundary mass-accounting tests
- constant, additive, product, and physical collision tests
- matched fluid–swarm tests with sampling uncertainty

## Remaining consolidation sequence

The merged one-representation-per-build structure is complete. Remaining work should proceed in
this order:

1. add a representation-neutral base/configuration interface without changing either numerical method
2. move common grid, coordinate, gas, file, and CUDA utilities behind that interface
3. reproduce standalone swarm behavior with analytical tests under `qav/swarm/`
4. add cross-representation tests for the common physical contract
5. remove duplicated representation utilities only after both test families pass
6. remove `legacy/` only after the user confirms that the active tree is the accepted recovery point

The old spatial-hashing proposal is not part of this migration contract. The current cuKD
implementation should first be profiled at representative scale; any alternative must demonstrate
equivalent neighbor statistics and collision rates before performance claims matter.

## References

- Huang & Bai (2022), [multifluid dust algorithms and momentum diffusion](https://arxiv.org/abs/2206.01023)
- Strang (1968), [symmetric operator splitting](https://doi.org/10.1137/0705041)
- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
