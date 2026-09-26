# Particle orbital-motion test

## Purpose

This campaign measures the timestep convergence of the production semi-analytic swarm trajectory
integrator in its exact zero-drag limit, using a single particle on a planar eccentric Kepler orbit.
It exercises the production runtime, the `ssa_transport` kernel with its first drift stage and force
evaluation, and the boundaries. Finite-drag accuracy is a separate test.

## Models

| Model | Timestep | Duration | Nominal steps | Steps per output |
|---|---|---|---:|---:|
| `orbit_dt_p80` | $P/80$ | $100P$ | 8,000 | 4 |
| `orbit_dt_p160` | $P/160$ | $100P$ | 16,000 | 8 |
| `orbit_dt_p320` | $P/320$ | $100P$ | 32,000 | 16 |
| `orbit_dt_p640` | $P/640$ | $100P$ | 64,000 | 32 |

Each `mod/<model>/const_defs.cuh` sets only `ORBIT_STEPS_PER_PERIOD` and includes the shared
`src/const_defs.cuh`. Every `flags.mk` selects `DUST_REPR := swarm`, `MODEL_PARENT := ../src`,
`TRANSPORT`, and `CODE_UNIT`; radiation, diffusion, and collisions are not enabled, and the
override below removes gas drag.

| Parameter | Value |
|---|---|
| Orbit | $GM=a=1$, $e=0.5$, period $P=2\pi$ |
| Initial state | pericenter $R=0.5$, $\phi=0$, $v_R=0$, stored specific angular momentum $\ell=\sqrt{3}/2$ |
| Representatives | `N_P = 1` |
| Domain | full azimuth $-\pi\le\phi\le\pi$, $0.3\le R\le 2$, planar (`N_Z = 1`) |
| Mesh | `(N_X, N_Y, N_Z) = (8, 16, 1)`; no mesh fields are written |
| Timestep | fixed `DT_MAX` $=P/N$ with $N=$ `ORBIT_STEPS_PER_PERIOD` |
| Output | `DT_OUT` $=P/20$, `SAVE_MAX = 2000` |

Gas constants (`ASPR_0`, `IDX_P`, `IDX_Q`, `SIGMA_0`, `STOKES_0`) are present only because the
production host interface requires them; the zero-drag update ignores them. Output intervals contain
an integer number of nominal steps, although floating-point accumulation may cause tiny remainder
steps in the production runtime.

## Production code and overrides

The campaign `src/` directory contains only:

- `const_defs.cuh`: the constants above.
- `particle_init.cu`: the pericenter initial state.
- `dyn_rate_calc.cu`: a fixed step `DT_MAX` in place of the production CFL policy; the runtime still
  shortens the last step to reach each output time.
- `_transport.cuh`: includes the production `inc/swarm/_transport.cuh` and replaces only its second
  SSA stage, `_ssa_substep_2`, with a call to the production `_ssa_advance<true>` helper in the
  exact zero-drag limit, which no finite Stokes number can express. It contains no copied
  integration stages.

## Build and run

From the repository root, on a CUDA machine:

```bash
make -C val/paper/ecc_orbit -j8 MODEL=orbit_dt_p80 \
  GPU_BACKEND=cuda GPU_TARGET=sm_80
./val/paper/ecc_orbit/obj/orbit_dt_p80/cuda/gamedev
```

Use your actual GPU architecture. For ROCm use `GPU_BACKEND=rocm`, the matching `GPU_TARGET`, and
the `rocm` executable. All four models can be run sequentially:

```bash
(
  set -e
  for model_dir in val/paper/ecc_orbit/mod/orbit_dt_p*; do
    model_name="${model_dir##*/}"
    make -C val/paper/ecc_orbit -j8 MODEL="$model_name" \
      GPU_BACKEND=cuda GPU_TARGET=sm_80
    "val/paper/ecc_orbit/obj/$model_name/cuda/gamedev"
  done
)
```

The campaign `Makefile` includes the root `Makefile` with these paths:

| Item | Path under `val/paper/ecc_orbit/` |
|---|---|
| Executable | `obj/<model>/<backend>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<target>/` |
| Output | `out/<model>/<backend>/` |

Both `obj/` and `out/` are ignored by Git. Builds for different GPU targets of one backend share the
executable and output paths.

## Outputs

Each run writes `variables.txt` and `particle_00000.dat` through `particle_02000.dat`. Each file
holds one 48-byte record of six doubles: position (azimuth $\phi$, radius $R$, polar angle) and
physical velocity ($v_\phi$, $v_R$, $v_\theta$). No RNG checkpoints are written. See
[Output files](../../../README.md#output-files) for the record layout recorded in `variables.txt`.

## Analysis

No postprocessing or reference solver is included in this directory. Compare each snapshot at its
output time $t$ with the exact Kepler solution: solve

$$
u-e\sin u=\frac{2\pi t}{P}
$$

for the eccentric anomaly $u$, then use

$$
X=a(\cos u-e),\qquad Y=a\sqrt{1-e^2}\,\sin u,
$$

with the simulated Cartesian position $X=R\cos\phi$, $Y=R\sin\phi$. Also monitor the specific
energy $E=\tfrac12(v_R^2+v_\phi^2)-GM/R$, whose exact value is $-GM/(2a)=-1/2$, and the specific
angular momentum $Rv_\phi=\sqrt{3}/2$. Energy error alone does not measure phase accuracy.

## Evidence boundary

The campaign tests the zero-drag SSA specialization only; it does not test drag coupling, radiation
pressure, or out-of-plane motion. Outputs from another source snapshot are not evidence for the
current source. Native CUDA and ROCm builds and fresh runs from the current source are required
before claiming accuracy.
