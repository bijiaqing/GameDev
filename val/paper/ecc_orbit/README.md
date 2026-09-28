# Eccentric Kepler orbit: timestep convergence

This campaign measures how the error of the production semi-analytic swarm trajectory integrator
converges with the timestep in its exact zero-drag limit, using a single particle on a planar
eccentric Kepler orbit for 100 orbital periods. It exercises the Lagrangian dust model through the
production runtime, the `ssa_transport` kernel with its first drift stage and force evaluation of
the [staggered update](../../../doc/numerics_swarm.md#51-staggered-update), and the transport
boundaries. The routine [eccentric-orbit test](../../../doc/testsets.md#31-eccentric-kepler-orbit)
checks the same zero-drag specialization with its own test driver over a resolution sweep; this
campaign runs it inside the production runtime loop over many orbits. Finite-drag accuracy is a
separate test ([drag path](../../../doc/testsets.md#34-drag-path)). Project terms are defined in
the [glossary](../../../doc/README.md#glossary).

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
| Representation | swarm, planar (`N_Z = 1`) with active azimuth |
| Backend | CUDA or ROCm; no neighbor search (collisions are disabled) |
| Sweep | fixed timestep $P/N$ with $N=80,160,320,640$ steps per period |
| Models | 4: `orbit_dt_p80`, `orbit_dt_p160`, `orbit_dt_p320`, `orbit_dt_p640` |
| Planned runs | 4 per backend target, one per model |
| Representatives | `N_P = 1` |
| Main flags | `TRANSPORT`, `CODE_UNIT` |
| Reference | exact Kepler solution ([Analysis](#analysis)) |
| Analysis entry point | none included; compare with the reference in [Analysis](#analysis) |
| Output location | `val/paper/ecc_orbit/out/<model>/<backend>/<target>/` |

## Models

Each model fixes the number of steps per orbital period $P$:

| Model | Timestep | Nominal steps in $100P$ | Steps per output |
|---|---|---:|---:|
| `orbit_dt_p80` | $P/80$ | 8,000 | 4 |
| `orbit_dt_p160` | $P/160$ | 16,000 | 8 |
| `orbit_dt_p320` | $P/320$ | 32,000 | 16 |
| `orbit_dt_p640` | $P/640$ | 64,000 | 32 |

Each `mod/<model>/const_defs.cuh` sets only `ORBIT_STEPS_PER_PERIOD` and includes the shared
`src/const_defs.cuh`. Every `flags.mk` selects `DUST_REPR := swarm`, `MODEL_PARENT := ../src`,
`TRANSPORT`, and `CODE_UNIT`; radiation, diffusion, and collisions are not enabled, and the header
override described in [Production code and overrides](#production-code-and-overrides) removes gas
drag. `CODE_UNIT` acts only with `COLLISION` and has no effect here.

## Setup

### Parameters

| Parameter | Value |
|---|---|
| Orbit | $GM=a=1$, $e=0.5$, period $P=2\pi$ (`ORBIT_PERIOD`) |
| Initial state | pericenter $R=0.5$, $\phi=0$, $v_R=0$, stored specific angular momentum $\ell=\sqrt{3}/2$ |
| Representatives | `N_P = 1` |
| Domain | full azimuth $-\pi\le\phi\le\pi$, $0.3\le R\le 2$, planar (`N_Z = 1`) |
| Mesh | `(N_X, N_Y, N_Z) = (8, 16, 1)`; no mesh fields are written |
| Timestep | fixed `DT_MAX` $=P/N$ with $N=$ `ORBIT_STEPS_PER_PERIOD` |
| Output | `DT_OUT` $=P/20$, `SAVE_MAX = 2000`, `LIN_BASE = 1`: 20 outputs per period for $100P$ |
| Unused constants | `ASPR_0 = 0.05`, `IDX_P = 1`, `IDX_Q = -1`, `SIGMA_0 = 1`, `METAL_Z = 0.01`, `RHO_0 = 1`, `STOKES_0 = 1`, `CFL_DYN = 0.45` |

### Orbit and initial state

The particle starts at pericenter, $R=a(1-e)=0.5$, where the Kepler speed is $\sqrt3$. The local
`particle_init` stores the tangential slot as the angular momentum $\ell=Rv_\phi=\sqrt{0.75}$ with
zero radial velocity. The orbit reaches apocenter at $R=1.5$, inside the radial domain, so no
transport boundary absorbs the particle.

### Unused constants

The gas constants (`ASPR_0`, `IDX_P`, `IDX_Q`, `SIGMA_0`, `METAL_Z`, `RHO_0`, `STOKES_0`) are
present only because the production host interface requires them; the zero-drag update ignores
them. `CFL_DYN` is recorded for completeness, since the local `dyn_rate_calc` fixes the step.

### Timestep policy

Each output interval contains an integer number of nominal steps, $N/20$. Floating-point
accumulation of the step may leave tiny remainder steps, because the production runtime shortens
the last step to reach each output time exactly.

## Production code and overrides

The runtime, the `ssa_transport` kernel, the first drift `_ssa_substep_1`, the force term, and the
boundaries come from the root `src/swarm/` and `inc/swarm/`. The campaign `src/` directory,
inherited through `MODEL_PARENT := ../src`, replaces production files of the same name ([source and
header overrides](../../../README.md#source-and-header-overrides)) and contains only:

- `const_defs.cuh`: the constants above.
- `particle_init.cu`: the pericenter initial state.
- `dyn_rate_calc.cu`: a fixed step `DT_MAX` in place of the production CFL policy of [timestep
  control](../../../doc/numerics_swarm.md#102-timestep-control); the runtime still shortens the
  last step to reach each output time.
- `_transport.cuh`: includes the production `inc/swarm/_transport.cuh` and replaces only its second
  SSA stage, `_ssa_substep_2`, with a call to the production `_ssa_advance<true>` helper in the
  exact zero-drag limit, which no finite Stokes number can express. It contains no copied
  integration stages and asserts `N_Z == 1`.

## Build and run

Run the commands from the repository root. Build and run one model on CUDA:

```bash
make -C val/paper/ecc_orbit -j8 MODEL=orbit_dt_p80 GPU_BACKEND=cuda GPU_TARGET=sm_80
```

```bash
./val/paper/ecc_orbit/obj/orbit_dt_p80/cuda/sm_80/gamedev
```

Use your actual GPU architecture. For ROCm use `GPU_BACKEND=rocm`, the matching `GPU_TARGET` (for
example `gfx942`), and the `rocm/<target>` executable. All four models can be run sequentially:

```bash
(
  set -e
  for model_dir in val/paper/ecc_orbit/mod/orbit_dt_p*; do
    model_name="${model_dir##*/}"
    make -C val/paper/ecc_orbit -j8 MODEL="$model_name" \
      GPU_BACKEND=cuda GPU_TARGET=sm_80
    "val/paper/ecc_orbit/obj/$model_name/cuda/sm_80/gamedev"
  done
)
```

The campaign `Makefile` includes the root `Makefile` and redirects its model, object, executable,
and output paths into this directory.

## Outputs

Builds for different GPU targets keep separate executables and outputs:

| Item | Path under `val/paper/ecc_orbit/` |
|---|---|
| Executable | `obj/<model>/<backend>/<target>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<target>/` |
| Output | `out/<model>/<backend>/<target>/` |

Both `obj/` and `out/` are ignored by Git. Each run writes `variables.txt` and
`particle_00000.dat` through `particle_02000.dat`. Each particle file holds one 48-byte record of
six doubles, about 96 kB for all 2,001 files: position (azimuth $\phi$, radius $R$, polar angle)
and physical velocity ($v_\phi$, $v_R$, $v_\theta$). No RNG-state checkpoints are written, since
neither diffusion nor collisions are enabled. See [Output files](../../../README.md#output-files)
for the record layout recorded in `variables.txt`.

## Analysis

The reference is the exact Kepler orbit through the initial pericenter. No postprocessing or
reference solver is included in this directory. Compare each snapshot at its output time $t$ with
the exact solution: solve

$$
u-e\sin u=\frac{2\pi t}{P}
$$

for the eccentric anomaly $u$, then use

$$
X=a(\cos u-e),\qquad Y=a\sqrt{1-e^2}\,\sin u,
$$

with the simulated Cartesian position $X=R\cos\phi$, $Y=R\sin\phi$. Also monitor the specific
energy $E=\tfrac12(v_R^2+v_\phi^2)-GM/R$, whose exact value is $-GM/(2a)=-1/2$, and the specific
angular momentum $Rv_\phi=\sqrt{3}/2$. Energy error alone does not measure phase accuracy. The
staggered update is second order for smooth forcing
([accuracy](../../../doc/numerics_swarm.md#111-accuracy)), so compare the observed order across the
four timesteps with two.

## Limits

The campaign tests the zero-drag SSA specialization inside the production runtime over 100
periods. It does not establish:

- drag coupling, radiation pressure, or out-of-plane motion;
- accuracy of the current source from outputs of another source snapshot;
- accuracy on either backend without native CUDA and ROCm builds and fresh runs from the current
  source.

The rules for when a result qualifies are in [What qualifies a
result](../../README.md#what-qualifies-a-result).
