# Photospheric dust transport

Both formulations evolve a single dust species under stellar gravity, gas drag,
and attenuated radiation pressure. Diffusion, collisions, imported gas, viscous
gas flow, and Poynting–Robertson drag are disabled.

`STOKES_0=1e-4`, `BETA_0=10`, and `KAPPA_0=5e4` are shared. These are code-unit
values; opacity has dimensions area/mass. `STOKES_0` is the reference midplane
value at `R_0`, not a spatially constant Stokes number:
`St(R)=STOKES_0*(R/R_0)^0.5`. Radiation ramps on over `T_BETA=2*pi` time units.

## Initialization and comparison

The root fluid and particle host initializers construct the same Gaussian-smoothed
dust surface density: `METAL_Z*SIGMA_0*(R/R_0)^IDX_P` on `0.6<R/R_0<1.4`,
convolved with a Gaussian of standard deviation `0.025*R_0`. Both use the same
gas surface-density and pressure-support prescriptions (`IDX_P=-0.5`, `IDX_Q=0`,
`ASPR_0=0.05`) and initial steady drag-coupled velocities.

The fluid override retains azimuthal Gaussian density perturbations of amplitude
`1e-10`. The particle model reuses the root initializer, sampling radius with
probability proportional to `R*Sigma_d(R)` and uniform azimuth, with equal
represented masses. It uses `N_P=1000000000`; no size distribution is enabled.

Both optical-depth calculations use the 2D closure
`rho_d=Sigma_d/(sqrt(2*pi)*H_g)` and integrate `KAPPA_0*rho_d` radially from
the inner boundary. Particle extinction is deposited using each particle's gas
scale height; fluid extinction uses cell-center scale heights. Agreement is
therefore statistical and subject to mesh discretization. Compare initial
azimuthally averaged surface density, enclosed mass, optical depth, and radiation
acceleration. Matching mean optical depth alone does not guarantee matching mean
attenuation, since `mean(exp(-tau)) != exp(-mean(tau))`.

## Build and run

Run from the repository root. The root Makefile selects local `src/` overrides
and otherwise compiles production sources. No `MODEL` argument is needed.

CUDA (A100 target; use `sm_90` for H200 and its executable path):

```sh
make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
val/paper/photospheric/fluid/obj/cuda/sm_80/gamedev

make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=cuda GPU_TARGET=sm_80
val/paper/photospheric/swarm/obj/cuda/sm_80/gamedev
```

ROCm:

```sh
make -C val/paper/photospheric/fluid -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/photospheric/fluid/obj/rocm/gfx942/gamedev

make -C val/paper/photospheric/swarm -j8 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/photospheric/swarm/obj/rocm/gfx942/gamedev
```

Outputs go to `<formulation>/out/<backend>/`. Objects and executables go to
`<formulation>/obj/<backend>/<target>/`. Runs with the same formulation and
backend share an output directory, even when compiled for different GPU targets.

At one billion particles, explicit device arrays peak at about 80.2 GB during
initialization (about 56.2 GB afterward), plus runtime/reduction overhead. Pinned
host arrays peak at about 72.2 GB. Each particle snapshot contains 48 GB.
Density and optical depth are saved at every output; particles are saved initially
and every tenth output (indices 0, 10, and 20), requiring 144 GB before mesh
outputs. These are array-size estimates in decimal GB, not measured peak usage.

Build/source/output/clean routing was checked with CUDA and ROCm dry runs for
both formulations. Native GPU compilation and numerical execution still need
to be checked on the target machines.
