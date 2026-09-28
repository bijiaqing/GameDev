# Photospheric dust transport

## Purpose

This campaign runs the same photospheric transport problem with the Lagrangian swarm and the
Eulerian fluid representations so that the two formulations can be compared. Both evolve a single
dust species under stellar gravity, gas drag, attenuated radiation pressure, and density diffusion
with Stokes-dependent diffusivity. Collisions, imported gas, viscous gas flow, concentration
diffusion, and Poynting–Robertson drag are not enabled.

## Models

Each formulation directory treats its own `src/` as the model directory, so no `MODEL` argument is
needed.

| Formulation | Directory | `flags.mk` selections |
|---|---|---|
| Swarm | `swarm/` | `DUST_REPR := swarm`, `TRANSPORT`, `DIFFUSION`, `RADIATION`, `SAVE_DENS`, `CODE_UNIT` |
| Fluid | `fluid/` | `DUST_REPR := fluid`, `FLUID_SWEEP := block`, `DIFFUSION`, `RADIATION` |

Shared parameters, in code units with $G=M_\star=R_0=1$:

| Parameter | Value |
|---|---|
| Domain | $0.5\le R\le1.5$, azimuthal wedge $0\le\phi\le\pi/2$, vertically integrated (`N_Z = 1`) |
| Gas | `SIGMA_0 = 1e-2`, `IDX_P = -0.5`, `IDX_Q = 0`, `ASPR_0 = 0.05` |
| Turbulence | `ALPHA = 1e-4`, all Schmidt numbers 1 |
| Dust | `METAL_Z = 1e-2`, `STOKES_0 = 1e-3` |
| Radiation | `BETA_0 = 10`, `KAPPA_0 = 5e4`, ramp time `T_BETA` $=2\pi$ |
| Timestep limits | `DT_MAX = 0.1`, `CFL_DYN = 0.45` |
| Output | `DT_OUT` $=2\pi$, `SAVE_MAX = 20`, final time $40\pi$ |

`KAPPA_0` is an opacity with dimensions of area per mass. `STOKES_0` is the reference midplane
value at `R_0`, not a spatially constant Stokes number: $\mathrm{St}(R)=\mathrm{St}_0(R/R_0)^{1/2}$.
Radiation ramps on over `T_BETA`.

| Parameter | Swarm | Fluid |
|---|---|---|
| Mesh | `(1536, 1024, 1)` deposition mesh | `(4096, 3072, 1)` |
| Representatives | `N_P = 1000000000`, equal represented masses, one size | not applicable |
| Particle output | every tenth frame (`LIN_BASE = 10`) | not applicable |
| Fluid limiter | not applicable | `POS_LIMIT = 0.9`, `RHO_VAC = 1e-30` |

## Initialization

The root swarm and fluid host initializers construct the same Gaussian-smoothed dust surface
density: $Z\Sigma_0(R/R_0)^{p}$ on $0.6\lt R/R_0\lt 1.4$, convolved with a Gaussian of standard
deviation $0.025R_0$. Both use the same gas surface-density and pressure-support prescriptions and
initial steady drag-coupled velocities.

The swarm model reuses the root initializer, sampling radius with probability proportional to
$R\,\Sigma_d(R)$ and uniform azimuth, with equal represented masses and no size distribution. The
fluid override `fluid/src/init_rho_calc.cu` interpolates this profile onto the mesh and adds
azimuthal Gaussian density perturbations of relative amplitude $10^{-10}$.

Both optical-depth calculations use the 2D closure $\rho_d=\Sigma_d/(\sqrt{2\pi}\,H_g)$ and
integrate $\kappa_0\rho_d$ radially from the inner boundary. Swarm extinction is deposited using
each particle's gas scale height; fluid extinction uses cell-center scale heights. Agreement is
therefore statistical and subject to mesh discretization.

## Timestep policies

`DT_MAX` and `CFL_DYN` are shared, but the timestep criteria differ. The swarm retains absolute
orbital and cell-crossing bounds and explicit stochastic-displacement bounds; its particle drift has
no FARGO residual-step policy, whereas fluid FARGO advection limits the step by the residual
azimuthal motion. For the swarm mesh the orbital bound at the inner edge is

$$
\Delta t=0.45\,\frac{(\pi/2)/1536}{\Omega_K(0.5)}=1.62703\times10^{-4}.
$$

Relaxing this restriction requires checking force and opacity sampling and radial transport
accuracy. Both schemes evaluate radiation at their intermediate state; larger orbital advances must
not bypass that coupling.

## Build and run

Run from the repository root. Each formulation `Makefile` includes the root `Makefile`, selects its
local `src/` overrides, and otherwise compiles production sources.

CUDA (A100 target; use `sm_90` for H200 and its executable path):

```sh
make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
val/paper/photospheric/swarm/obj/cuda/sm_80/gamedev

make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
val/paper/photospheric/fluid/obj/cuda/sm_80/gamedev
```

ROCm:

```sh
make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/photospheric/swarm/obj/rocm/gfx942/gamedev

make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/photospheric/fluid/obj/rocm/gfx942/gamedev
```

| Item | Path under `val/paper/photospheric/` |
|---|---|
| Executable and objects | `<formulation>/obj/<backend>/<target>/` |
| Output | `<formulation>/out/<backend>/` |

Runs with the same formulation and backend share an output directory, even when compiled for
different GPU targets. A saved frame index passed to the executable resumes a run as described in
[Restarting a simulation](../../../README.md#restarting-a-simulation); the swarm can resume only
from a frame with a particle checkpoint (0, 10, or 20).

## Outputs

The swarm writes `dustdens` and `optdepth` at every frame (about 12.6 MB each) and `particle` and
`rngstate` checkpoints at frames 0, 10, and 20.

The fluid writes `dustdens`, `dustvelx`, `dustvely`, `dustvelz`, and `optdepth` at every frame
0–20, plus `variables.txt`. Each field holds one double per cell, about 101 MB, so a complete fluid
run writes about 10.6 GB. File formats are described in
[Output files](../../../README.md#output-files).

Estimated swarm array sizes at one billion particles, in decimal GB and not measured peak usage:

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
Matching mean optical depth alone does not guarantee matching mean attenuation, since
$\langle e^{-\tau}\rangle\ne e^{-\langle\tau\rangle}$.

## Evidence boundary

Matched-source outputs and timestep and resolution convergence are required to
establish accuracy.
