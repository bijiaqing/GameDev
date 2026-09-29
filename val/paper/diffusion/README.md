# Particle diffusion: density and concentration

This campaign tests whether the production swarm diffusion operator, run for many steps inside the
production runtime, reproduces the analytical log-normal solution of radial density and
concentration diffusion at four constant Stokes numbers. It exercises the Lagrangian dust model in
the [radial-only closure](../../../doc/README.md#glossary): the production `diffusion_pos` kernel
with its cylindrical [Itô drift](../../../doc/guide_swarm.md#72-eulermaruyama-step), the
Stokes-suppressed diffusivity, the gas-density-gradient term of [concentration
diffusion](../../../doc/guide_swarm.md#73-concentration-diffusion), the reflecting radial
boundaries, the backend random-number streams, and the [operator
composition](../../../doc/guide_swarm.md#101-operator-composition) that applies diffusion as two
half-steps around transport. The routine [swarm diffusion
tests](../../../doc/guide_tests.md#4-swarm-diffusion) check one step from prescribed positions; this
campaign checks the distribution that a thousand composed steps produce. The operator is derived in
[Diffusion](../../../doc/guide_swarm.md#7-diffusion), and project terms are defined in the
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
| Representation | swarm, radial-only closure (`N_X = N_Z = 1`) |
| Backend | CUDA or ROCm; no neighbor search (collisions are disabled) |
| Sweep | diffusion mode (density, concentration) × Stokes number $\mathrm{St}=0,0.1,1,10$ |
| Models | 8: `density_st{0,0p1,1,10}` and `concentration_st{0,0p1,1,10}` |
| Planned runs | 8 per backend target, one per model |
| Representatives | `N_P = 1048576` ($2^{20}$) per run |
| Main flags | `TRANSPORT`, `DIFFUSION`, `CONST_ST`, `CODE_UNIT`; `DIFFUSE_CONCENTRATION` in the concentration models |
| Reference | analytical log-normal solution ([Analysis](#analysis)) |
| Analysis entry point | none included; compare with the reference in [Analysis](#analysis) |
| Output location | `val/paper/diffusion/out/<model>/<backend>/<target>/` |

## Models

Each model fixes one diffusion mode and one Stokes number:

| Stokes number | Density model | Concentration model |
|---|---|---|
| $\mathrm{St}=0$ | `density_st0` | `concentration_st0` |
| $\mathrm{St}=0.1$ | `density_st0p1` | `concentration_st0p1` |
| $\mathrm{St}=1$ | `density_st1` | `concentration_st1` |
| $\mathrm{St}=10$ | `density_st10` | `concentration_st10` |

Each `mod/<model>/const_defs.cuh` sets only `DIFFUSION_STOKES` and includes the shared
`src/const_defs.cuh`. Every `flags.mk` selects `DUST_REPR := swarm`, `MODEL_PARENT := ../src`,
`TRANSPORT`, `DIFFUSION`, `CONST_ST`, and `CODE_UNIT`; only the concentration models add
`DIFFUSE_CONCENTRATION`, which switches from [density to concentration
diffusion](../../../doc/guide_basis.md#52-density-and-concentration-diffusion). `TRANSPORT` is
required because `DIFFUSION` without it is rejected at compile time (`inc/swarm/swarm_kern.cuh`).
`CODE_UNIT` acts only with `COLLISION` and has no effect here.

## Setup

### Parameters

| Parameter | Value |
|---|---|
| Units | $G=M_\star=R_0=1$, so $\Omega_K(R_0)=1$ |
| Representatives | `N_P = 1048576` ($2^{20}$), equal represented mass, single size `S_0 = 1` |
| Mesh | `(N_X, N_Y, N_Z) = (1, 128, 1)`, radial-only |
| Radial domain | $e^{-3}\le R\le e^{3}$ (`Y_MIN`, `Y_MAX`), reflecting for diffusion |
| Gas | $\Sigma_g\propto R^{p}$ with $p=-1$ (`IDX_P`); `ASPR_0 = 0.05`, `IDX_Q = 0.5` |
| Viscosity and Schmidt number | constant `ALPHA = 0.04`, `SCHMIDT_R = 1` (`SCHMIDT_X = SCHMIDT_Z = 1` are inactive) |
| Stokes number | `STOKES_0 = DIFFUSION_STOKES`, held fixed in space by `CONST_ST` |
| Initial ring | $\ln(R/R_0)\sim\mathcal N(0,0.05^2)$ (`RING_LOG_WIDTH`), zero velocity |
| Random streams | ring drawn with `RING_INIT_SEED = 17`; evolution uses the production seed 1 |
| Timestep | fixed `DT_MAX` $=1+\mathrm{St}^2$ |
| Output | `DT_OUT` $=100(1+\mathrm{St}^2)$, `SAVE_MAX = 10`, `LIN_BASE = 1` |
| Other constants | `SIGMA_0 = 1`, `METAL_Z = 0.01`, `RHO_0 = 1`; `CFL_DYN = 0.45` is recorded but unused |

`SIGMA_0` and `METAL_Z` set the gas and dust normalization; neither enters the drift or the noise
of the diffusion step.

### Diffusivity

The chosen gas profile makes the radial dust diffusivity a pure power of radius. With the aspect
ratio $h=h_0(R/R_0)^{(q+1)/2}$ and the viscosity $\nu=\alpha h^2R^2\Omega_K$ of the [shared disk
model](../../../doc/guide_basis.md#35-viscosity), $q=0.5$ gives $\nu=\alpha h_0^2R^2=10^{-4}R^2$,
and the [directional diffusivity](../../../doc/guide_basis.md#51-directional-diffusivities) is

```math
D(R)=aR^2,\qquad a=\frac{10^{-4}}{1+\mathrm{St}^2}.
```

`CONST_ST` removes the spatial variation of the Stokes number, so the Stokes part of the
diffusivity gradient vanishes.

### Initial ring and random streams

The initial state is a narrow log-normal ring at $R_0$ with zero velocity. The local
`particle_init` draws $\ln(R/R_0)$ for particle $i$ from its own stream (seed
`RING_INIT_SEED = 17`, subsequence $i$), so the ring does not consume the evolution streams. The
evolution uses the production per-particle streams, seed 1 with subsequence equal to the particle
index ([random streams](../../../doc/guide_swarm.md#129-random-streams)). The azimuth and polar
angle stay at their inactive values, $\phi=0$ and $\theta=\pi/2$.

### Timestep and output schedule

Every model takes 1,000 nominal steps of `DT_MAX` to a final time $1000(1+\mathrm{St}^2)$ in code
units and saves the initial state plus ten outputs. Each step applies diffusion over
$\Delta t/2$, the no-op transport over $\Delta t$, and diffusion over $\Delta t/2$ again.
Scaling the step and the output interval by $1+\mathrm{St}^2$ keeps the **diffusion age** $at$ (the
dimensionless spreading time) and the number of steps per unit diffusion age equal across Stokes
numbers; each step advances $at$ by $10^{-4}$, and the final age is $at=0.1$.

## Production code and overrides

All runtime, diffusion, boundary, RNG, and output code comes from the root `src/swarm/` and
`inc/swarm/`. The campaign `src/` directory, inherited through `MODEL_PARENT := ../src`, replaces
production files of the same name ([source and header
overrides](../../../README.md#source-and-header-overrides)) and contains only:

- `const_defs.cuh`: the constants above.
- `particle_init.cu`: the initial log-normal ring, drawn from independent streams.
- `dyn_rate_calc.cu`: a fixed step `DT_MAX` in place of the dynamics CFL policy of [timestep
  control](../../../doc/guide_swarm.md#102-timestep-control); the runtime still shortens the
  last step to reach each output time.
- `ssa_transport.cu`: a no-op deterministic transport kernel, which isolates diffusion while keeping
  the production operator composition.

## Build and run

Run the commands from the repository root. Build and run one model on ROCm:

```bash
make -C val/paper/diffusion -j8 MODEL=concentration_st1 GPU_BACKEND=rocm GPU_TARGET=gfx942
```

```bash
val/paper/diffusion/obj/concentration_st1/rocm/gfx942/gamedev
```

For CUDA use `GPU_BACKEND=cuda GPU_TARGET=sm_80`, or the actual architecture, and the
`cuda/<target>` executable. All eight models can be run sequentially:

```bash
(
  set -e
  for model_dir in val/paper/diffusion/mod/*; do
    model="${model_dir##*/}"
    make -C val/paper/diffusion -j8 MODEL="$model" GPU_BACKEND=cuda GPU_TARGET=sm_80
    "val/paper/diffusion/obj/$model/cuda/sm_80/gamedev"
  done
)
```

The campaign `Makefile` includes the root `Makefile` and redirects its model, object, executable,
and output paths into this directory. A run without arguments starts fresh and replaces existing
frames; `gamedev <frame>` resumes from a saved frame as described in [Restarting a
simulation](../../../README.md#restarting-a-simulation).

## Outputs

Builds for different GPU targets keep separate executables and output directories:

| Item | Path under `val/paper/diffusion/` |
|---|---|
| Executable | `obj/<model>/<backend>/<target>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<target>/` |
| Output | `out/<model>/<backend>/<target>/` |

Both `obj/` and `out/` are ignored by Git. Each output directory contains `variables.txt`,
`particle_00000.dat` through `particle_00010.dat`, and a matching
[RNG-state checkpoint](../../../doc/README.md#glossary) `rngstate_<frame>.dat` for every particle
frame. No mesh fields are written. Each particle record holds six doubles, 48 bytes: position
(azimuth, radius, polar angle) and physical velocity, with no size fields. One particle file is
about 50 MB ($2^{20}\times48$ bytes), so the eleven particle files of a model take about 550 MB. The
record layout is written in the `[SWARM_DTYPE]` section of `variables.txt`; see [Output
files](../../../README.md#output-files) and [Reading output with
Python](../../../README.md#reading-output-with-python).

## Analysis

The reference is the exact solution of the target equation for a log-normal initial ring. Let
$\chi=0$ for density diffusion and $\chi=1$ for concentration diffusion. Both models obey

```math
\partial_t\Sigma_d=\frac{1}{R}\,\partial_R\left[RD\left(\partial_R\Sigma_d-\chi\,\Sigma_d\,\partial_R\ln\Sigma_g\right)\right].
```

The production radial stochastic drift is $(3+\chi p)aR$ and its noise amplitude is
$`\sqrt{2a}\,R`$. Itô's formula therefore gives a Gaussian log radius $u=\ln(R/R_0)$ with

```math
\bar u(t)=(2+\chi p)\,at,\qquad \sigma_u^2(t)=0.05^2+2at .
```

The surface density per total dust mass $M_d$ is

```math
\frac{\Sigma_d}{M_d}=\frac{1}{2\pi R^2\sqrt{2\pi\sigma_u^2}}\exp\left[-\frac{(u-\bar u)^2}{2\sigma_u^2}\right].
```

No analysis script is included in this directory. Compare empirical CDFs, log-radius moments, and
area-averaged density bins with this solution, choosing $\chi$ from the model's
`DIFFUSE_CONCENTRATION` setting. At the final age $at=0.1$ the mean log radius is $0.2$ for density
and $0.1$ for concentration diffusion, and $\sigma_u\approx0.45$.

## Limits

The campaign tests constant-St suppression, the cylindrical Itô drift, and the gas-density-gradient
term of concentration diffusion over many composed steps. It does not establish:

- agreement near the walls: the reference solution is for an unbounded domain, while the
  simulation reflects at $R=e^{\pm3}$; at $at=0.1$ the boundaries lie more than six standard
  deviations of $u$ from the mean, so their effect is small but not zero;
- accuracy for publication without timestep and particle-number convergence, which the campaign
  does not include;
- spatial derivatives of St, azimuthal or vertical diffusion, or three-dimensional diffusion;
- accuracy of the current source on either backend without native CUDA and ROCm builds and fresh
  results from that source; results from a different source snapshot must be kept separate.

The rules for when a result qualifies are in [What qualifies a
result](../../README.md#what-qualifies-a-result).
