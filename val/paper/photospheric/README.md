# Photospheric dust transport

This campaign runs one photospheric transport problem with both dust representations, the
Lagrangian swarm and the Eulerian fluid, so that the two formulations can be compared at production
resolution. Both evolve a single dust species under stellar gravity, gas drag, radiation pressure
attenuated by the dust's own radial optical depth, and density diffusion with a Stokes-dependent
diffusivity
([swarm radiation](../../../doc/numerics_swarm.md#62-radiation-pressure-and-optical-depth) and
[diffusion](../../../doc/numerics_swarm.md#7-diffusion);
[fluid radiation](../../../doc/numerics_fluid.md#62-radiation-pressure-and-optical-depth) and
[diffusion](../../../doc/numerics_fluid.md#7-diffusion)). The routine suites test these operators
in isolation, for example the [radiation-pressure
orbit](../../../doc/testsets.md#32-radiation-pressure-orbit),
[swarm diffusion](../../../doc/testsets.md#4-swarm-diffusion), the fluid [optical
depth](../../../doc/testsets.md#132-optical-depth) and [attenuated
radiation](../../../doc/testsets.md#133-attenuated-radiation), and the fluid [coupled
composition](../../../doc/testsets.md#14-fluid-coupled-composition); this campaign combines them in
one disk and compares the representations. Project terms are defined in the
[glossary](../../../doc/README.md#glossary).

## Contents

- [At a glance](#at-a-glance)
- [Models](#models)
- [Setup](#setup)
- [Production code and overrides](#production-code-and-overrides)
- [Build and run](#build-and-run)
- [Outputs](#outputs)
- [Analysis](#analysis)
- [Limits](#limits)

## At a glance

| Item | Value |
|---|---|
| Representations | swarm (`swarm/`) and fluid (`fluid/`) |
| Geometry | vertically integrated [radial–azimuthal](../../../doc/numerics_basis.md#24-supported-geometries) quarter-disk wedge (`N_Z = 1`) |
| Backends | CUDA (`sm_80`; `sm_90` for H200) and ROCm (`gfx942`); fluid sweep `block` on both |
| Models | one per representation; no `MODEL` argument |
| Planned runs | 4: 2 representations $\times$ 2 backends |
| Resolution | swarm: `N_P = 1000000000` on a `(1536, 1024, 1)` deposition mesh; fluid: `(4096, 3072, 1)` |
| Main flags | swarm: `TRANSPORT`, `DIFFUSION`, `RADIATION`, `SAVE_DENS`, `CODE_UNIT`; fluid: `DIFFUSION`, `RADIATION` |
| Analysis | none included |
| Output | `val/paper/photospheric/<representation>/out/<backend>/` |

## Models

Each representation directory treats its own `src/` as the model directory, so no `MODEL` argument
is needed.

| Representation | Directory | `flags.mk` selections |
|---|---|---|
| Swarm | `swarm/` | `DUST_REPR := swarm`, `TRANSPORT`, `DIFFUSION`, `RADIATION`, `SAVE_DENS`, `CODE_UNIT` |
| Fluid | `fluid/` | `DUST_REPR := fluid`, `FLUID_SWEEP := block`, `DIFFUSION`, `RADIATION` |

Collisions, imported gas, viscous gas flow, concentration diffusion, and Poynting–Robertson drag are
not enabled. The swarm is monodisperse (`MULTISIZE` is off), and its `CODE_UNIT` flag acts only
with `COLLISION`, so it has no effect here. The fluid always transports, so it needs no `TRANSPORT`
flag; `FLUID_SWEEP := block` fixes the [sweep](../../../doc/numerics_fluid.md#102-parallel-mapping)
on both backends, whereas the root default on CUDA is `thread`.

## Setup

### Shared parameters

Both representations use code units with $G=M_\star=R_0=1$ and the same physical parameters:

| Parameter | Value |
|---|---|
| Domain | $0.5\le R\le1.5$, azimuthal wedge $0\le\phi\le\pi/2$, vertically integrated (`N_Z = 1`) |
| Gas | `SIGMA_0 = 1e-2`, `IDX_P = -0.5`, `IDX_Q = 0`, `ASPR_0 = 0.05` |
| Turbulence | `ALPHA = 1e-4`, all Schmidt numbers 1 |
| Dust | `METAL_Z = 1e-2`, `STOKES_0 = 1e-3` |
| Radiation | `BETA_0 = 10`, `KAPPA_0 = 5e4`, ramp time `T_BETA` $=2\pi$ |
| Timestep limits | `DT_MAX = 0.1`, `CFL_DYN = 0.45` |
| Output | `DT_OUT` $=2\pi$, `SAVE_MAX = 20`, final time $40\pi$ |

`KAPPA_0` is an opacity with dimensions of area per mass. `STOKES_0` is the reference midplane value
at `R_0`, not a spatially constant Stokes number: $\mathrm{St}(R)=\mathrm{St}_0(R/R_0)^{1/2}$
([Stokes number](../../../doc/numerics_basis.md#4-stopping-time-and-stokes-number)). Radiation
switches on smoothly over `T_BETA`
([radiation ramp](../../../doc/numerics_basis.md#81-radiation-ratio-and-startup-ramp)).

### Representation-specific parameters

| Parameter | Swarm | Fluid |
|---|---|---|
| Mesh | `(1536, 1024, 1)` deposition mesh | `(4096, 3072, 1)` |
| Representatives | `N_P = 1000000000`, equal represented masses, one size | not applicable |
| Particle output | every tenth frame (`LIN_BASE = 10`) | not applicable |
| Fluid limiter | not applicable | `POS_LIMIT = 0.9`, `RHO_VAC = 1e-30` |

`POS_LIMIT` and `RHO_VAC` set the fluid's [positivity
subcycling](../../../doc/numerics_fluid.md#72-cranknicolson-solve) and vacuum state.

### Initialization

Both root host initializers construct the same
[edge-tapered](../../../doc/numerics_basis.md#62-edge-taper) dust surface density:
$Z\Sigma_0(R/R_0)^{p}$ with sources on $0.6\le R/R_0\le1.4$ (at least $0.1R_0$ inside each edge),
convolved with a Gaussian of standard deviation $0.025R_0$. Each
tabulates the profile on its own `N_Y + 1`-point radial axis, so the swarm table has 1025 points
and the fluid table 3073. Both use the same gas surface-density and pressure-support prescriptions
and start from the steady drag-coupled drift velocities
([steady drift](../../../doc/numerics_basis.md#7-steady-drift-velocity)).

The swarm reuses the root initializer: it samples radius with probability proportional to
$`R\,\Sigma_d(R)`$ and azimuth uniformly, with equal represented masses and no size distribution.
The fluid interpolates the profile onto its mesh and multiplies each azimuthal column by
$1+10^{-10}\xi$, where $\xi$ is a standard normal deviate shared across radius.

### Optical depth

Both optical-depth calculations use the 2D well-mixed closure $`\rho_d=\Sigma_d/(\sqrt{2\pi}\,H_g)`$
and integrate $\kappa_0\rho_d$ radially outward from the inner boundary
([radial optical depth](../../../doc/numerics_basis.md#82-radial-optical-depth)). The swarm deposits
each particle's extinction with the gas scale height at its own radius; the fluid uses cell-center
scale heights.

### Timestep policies

`DT_MAX` and `CFL_DYN` are shared, but the timestep criteria differ
([swarm](../../../doc/numerics_swarm.md#102-timestep-control),
[fluid](../../../doc/numerics_fluid.md#82-timestep-control)). The swarm keeps absolute orbital and
cell-crossing bounds and explicit stochastic-displacement bounds; its particle drift has no
[FARGO](../../../doc/numerics_fluid.md#55-fargo-azimuthal-transport) residual-step policy, whereas
fluid FARGO advection limits the step by the residual azimuthal motion. On the swarm mesh the
orbital bound at the inner edge is

```math
\Delta t=0.45\,\frac{(\pi/2)/1536}{\Omega_K(0.5)}=1.62703\times10^{-4}.
```

Both schemes evaluate radiation at their intermediate state: the swarm reconstructs it at the
midpoint of its staggered transport step, and the fluid rebuilds the optical depth at the midpoint
of its symmetric composition ([swarm](../../../doc/numerics_swarm.md#101-operator-composition),
[fluid](../../../doc/numerics_fluid.md#81-operator-composition)).

## Production code and overrides

Each representation `Makefile` includes the root `Makefile`, sets `MODEL := src` with its own
directory as the model root, and otherwise compiles production sources.

- `swarm/src/const_defs.cuh` is a copy of the root swarm constants with the domain, mesh, `N_P`, gas
  indices, `KAPPA_0`, output schedule, and `LIN_BASE` changed to the values above. The swarm has no
  source override: initialization, transport, diffusion, deposition, and radiation are the root
  code.
- `fluid/src/const_defs.cuh` is a copy of the root fluid constants with the domain, mesh, gas
  indices, `ALPHA`, and `SAVE_MAX` changed to the values above.
- `fluid/src/init_rho_calc.cu` is the root density initializer with one change: the relative
  amplitude of the azimuthal density perturbation is $10^{-10}$ instead of the root's 0.1, so the
  fluid starts from an essentially axisymmetric profile.

## Build and run

Run every command from the repository root. The root defaults select `GPU_TARGET=sm_80` for CUDA
and `gfx942` for ROCm.

On CUDA (A100 target; use `sm_90` for H200, and in the executable path), build and run the swarm:

```bash
make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
```

```bash
val/paper/photospheric/swarm/obj/cuda/sm_80/gamedev
```

then the fluid:

```bash
make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
```

```bash
val/paper/photospheric/fluid/obj/cuda/sm_80/gamedev
```

On ROCm, build and run the swarm:

```bash
make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
```

```bash
val/paper/photospheric/swarm/obj/rocm/gfx942/gamedev
```

then the fluid:

```bash
make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
```

```bash
val/paper/photospheric/fluid/obj/rocm/gfx942/gamedev
```

| Item | Path under `val/paper/photospheric/` |
|---|---|
| Executable and objects | `<representation>/obj/<backend>/<target>/` |
| Output | `<representation>/out/<backend>/` |

Runs with the same representation and backend share an output directory, even when compiled for
different GPU targets. A saved frame index passed to the executable resumes a run as described in
[Restarting a simulation](../../../README.md#restarting-a-simulation); the swarm can resume only
from a frame with a particle checkpoint (0, 10, or 20).

## Outputs

The swarm writes `dustdens` and `optdepth` at every frame 0–20 (one double per cell, about 12.6 MB
each), `particle` and `rngstate` checkpoints at frames 0, 10, and 20, and `variables.txt`. A
particle checkpoint holds six doubles per representative, 48 GB, and an RNG checkpoint one backend
state per representative, 48 GB on CUDA, so a complete CUDA swarm run writes about 290 GB.

The fluid writes `dustdens`, `dustvelx`, `dustvely`, `dustvelz`, and `optdepth` at every frame 0–20,
plus `variables.txt`. Each field holds one double per cell, about 101 MB, so a complete fluid run
writes about 10.6 GB. File formats are described in
[Output files](../../../README.md#output-files).

Estimated swarm array sizes at one billion representatives, in decimal GB and not measured peak
usage ([memory footprint](../../../doc/numerics_swarm.md#126-memory-footprint)):

| Storage | Size |
|---|---|
| Particle state (six doubles per particle) | 48 GB on the device and 48 GB in pinned host memory |
| Dynamics-rate array | 8 GB on the device |
| RNG state, enabled by `DIFFUSION` | one backend state per particle on the device (48 GB with CUDA's 48-byte `curandState`) |
| Fresh-start coordinate scratch | three coordinate arrays, 24 GB on the device and 24 GB pinned, freed before evolution |
| One particle snapshot | 48 GB; 144 GB for the three snapshots, plus matching RNG checkpoints |

RNG checkpoint I/O uses a bounded 64 MiB host staging buffer rather than a second complete RNG-state
array. Mesh fields and reduction workspaces add further, smaller storage.

## Analysis

No analysis script is included. Compare the initial azimuthally averaged surface density, enclosed
mass, optical depth, and radiation acceleration, then their evolution at matching output times.
Compare the attenuation itself, not only the mean optical depth: matching mean optical depth does
not guarantee matching mean attenuation, since
$\langle e^{-\tau}\rangle\ne e^{-\langle\tau\rangle}$.

## Limits

The campaign sets up one physical problem in both representations with production sources and
only the overrides above. It does not establish accuracy, which requires matched-source outputs and
timestep and resolution convergence.

- **Statistical agreement only.** The swarm and fluid differ in their extinction sampling (particle
  radius versus cell center), their initial-profile tables, and their meshes, so their agreement is
  statistical and subject to mesh discretization.
- **Timestep restriction.** The swarm's orbital bound sets its step. Relaxing it requires checking
  force and opacity sampling and radial transport accuracy, and larger orbital advances must not
  bypass the intermediate-state radiation coupling.

The campaign sources enter the validation source fingerprint, but a campaign run does not qualify
the validation suites; the rules are in [What qualifies a
result](../../README.md#what-qualifies-a-result).
