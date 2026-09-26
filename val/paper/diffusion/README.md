# Particle diffusion: density and concentration

## Purpose

This campaign tests the production swarm diffusion operator against an analytical log-normal
solution for radial density and concentration diffusion at four constant Stokes numbers. It
exercises the production `diffusion_pos` kernel, its cylindrical Itô drift, the Stokes-suppressed
diffusion coefficient, the gas-density-gradient term of concentration diffusion, the reflecting
radial boundaries, the backend random-number streams, and the runtime composition that applies
diffusion as two half-steps of $\Delta t/2$ around the transport stage of each runtime step.

## Models

| Mode | $\mathrm{St}=0$ | $\mathrm{St}=0.1$ | $\mathrm{St}=1$ | $\mathrm{St}=10$ |
|---|---|---|---|---|
| Density | `density_st0` | `density_st0p1` | `density_st1` | `density_st10` |
| Concentration | `concentration_st0` | `concentration_st0p1` | `concentration_st1` | `concentration_st10` |

Each `mod/<model>/const_defs.cuh` sets only `DIFFUSION_STOKES` and includes the shared
`src/const_defs.cuh`. Every `flags.mk` selects `DUST_REPR := swarm`, `MODEL_PARENT := ../src`,
`TRANSPORT`, `DIFFUSION`, `CONST_ST`, and `CODE_UNIT`; only the concentration models add
`DIFFUSE_CONCENTRATION`. `TRANSPORT` is required because `DIFFUSION` without it is rejected at
compile time.

| Parameter | Value |
|---|---|
| Units | $G=M_\star=R_0=1$, so $\Omega_K(R_0)=1$ |
| Representatives | `N_P = 1048576` ($2^{20}$), equal represented mass, single size `S_0 = 1` |
| Mesh | `(N_X, N_Y, N_Z) = (1, 128, 1)`, radial-only |
| Radial domain | $e^{-3}\le R\le e^{3}$ (`Y_MIN`, `Y_MAX`), reflecting |
| Gas | $\Sigma_g\propto R^{p}$ with $p=-1$ (`IDX_P`); `ASPR_0 = 0.05`, `IDX_Q = 0.5` |
| Viscosity and Schmidt number | `ALPHA = 0.04`, `SCHMIDT_R = 1` |
| Stokes number | `STOKES_0 = DIFFUSION_STOKES`, held fixed in space by `CONST_ST` |
| Initial ring | $\ln(R/R_0)\sim\mathcal N(0,0.05^2)$ (`RING_LOG_WIDTH`), zero velocity |
| Random streams | ring drawn with `RING_INIT_SEED = 17`; evolution uses the production seed 1 |
| Timestep | fixed `DT_MAX` $=1+\mathrm{St}^2$ |
| Output | `DT_OUT` $=100(1+\mathrm{St}^2)$, `SAVE_MAX = 10` |

With $h=h_0(R/R_0)^{(q+1)/2}$ and $\nu=\alpha h^2R^2\Omega_K$, these constants give
$\nu=\alpha h_0^2R^2=10^{-4}R^2$ and the radial dust diffusivity

$$
D(R)=aR^2,\qquad a=\frac{10^{-4}}{1+\mathrm{St}^2}.
$$

Every model takes 1,000 nominal steps to a final time $1000(1+\mathrm{St}^2)$ in code units and
saves the initial state plus ten outputs. Scaling the step and output interval by $1+\mathrm{St}^2$
keeps the diffusion age $at$ and the number of steps per diffusion age equal across Stokes numbers;
the final age is $at=0.1$.

## Production code and overrides

All runtime, diffusion, boundary, RNG, and output code comes from the root `src/swarm/` and
`inc/swarm/`. The campaign `src/` directory contains only:

- `const_defs.cuh`: the constants above.
- `particle_init.cu`: the initial log-normal ring, drawn from independent streams.
- `dyn_rate_calc.cu`: a fixed step `DT_MAX` in place of the dynamics CFL policy; the runtime still
  shortens the last step to reach each output time.
- `ssa_transport.cu`: a no-op deterministic transport kernel, which isolates diffusion while keeping
  the production operator composition.

## Build and run

From the repository root:

```sh
make -C val/paper/diffusion -j8 MODEL=concentration_st1 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/diffusion/obj/concentration_st1/rocm/gfx942/gamedev
```

For CUDA use `GPU_BACKEND=cuda GPU_TARGET=sm_80` (or the actual architecture) and the
`cuda/<target>` executable. All eight models can be run sequentially:

```sh
(
  set -e
  for model_dir in val/paper/diffusion/mod/*; do
    model="${model_dir##*/}"
    make -C val/paper/diffusion -j8 MODEL="$model" GPU_BACKEND=cuda GPU_TARGET=sm_80
    "val/paper/diffusion/obj/$model/cuda/sm_80/gamedev"
  done
)
```

The campaign `Makefile` includes the root `Makefile` with these paths:

| Item | Path under `val/paper/diffusion/` |
|---|---|
| Executable | `obj/<model>/<backend>/<target>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<target>/` |
| Output | `out/<model>/<backend>/<target>/` |

Builds for different GPU targets keep separate executables and output directories. A run without
arguments starts fresh and replaces existing frames; `gamedev <frame>` resumes from a saved frame as
described in [Restarting a simulation](../../../README.md#restarting-a-simulation).

## Outputs

Each output directory contains `variables.txt`, `particle_00000.dat` through `particle_00010.dat`,
and a matching `rngstate_<frame>.dat` for every particle frame. No mesh fields are written. Each
particle record holds six doubles, 48 bytes: position (azimuth, radius, polar angle) and physical
velocity, with no size fields; one particle file is about 50 MB. The record layout is written in
the `[SWARM_DTYPE]` section of `variables.txt`; see [Output files](../../../README.md#output-files)
and [Reading output with Python](../../../README.md#reading-output-with-python).

## Analysis

Let $\chi=0$ for density diffusion and $\chi=1$ for concentration diffusion. Both models obey

$$
\partial_t\Sigma_d=\frac{1}{R}\,\partial_R\left[RD\left(\partial_R\Sigma_d-\chi\,\Sigma_d\,\partial_R\ln\Sigma_g\right)\right].
$$

The production radial stochastic drift is $(3+\chi p)aR$ and its noise amplitude is $\sqrt{2a}\,R$.
Itô's formula therefore gives a Gaussian log radius $u=\ln(R/R_0)$ with

$$
\bar u(t)=(2+\chi p)\,at,\qquad \sigma_u^2(t)=0.05^2+2at .
$$

The surface density per total dust mass $M_d$ is

$$
\frac{\Sigma_d}{M_d}=\frac{1}{2\pi R^2\sqrt{2\pi\sigma_u^2}}\exp\left[-\frac{(u-\bar u)^2}{2\sigma_u^2}\right].
$$

No analysis script is included in this directory. Compare empirical CDFs, log-radius moments, and
area-averaged density bins with this solution, choosing $\chi$ from the model's
`DIFFUSE_CONCENTRATION` setting.

## Evidence boundary

The reference solution is for an unbounded domain, while the simulation reflects at $R=e^{\pm3}$.
At the final age $at=0.1$ the log-radius standard deviation is about 0.45, so the boundaries lie
more than six standard deviations from the mean and their effect is small. Timestep and
particle-number convergence are still required for a publication accuracy claim. These cases test
constant-St suppression and the gas-density-gradient term; they do not test spatial derivatives of
St, azimuthal or vertical diffusion, or three-dimensional diffusion. Native CUDA and ROCm builds and
fresh results from the current source are required before claiming accuracy; results from a
different source snapshot must be kept separate.
