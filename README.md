# GameDev

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

GameDev is a GPU research code for dust evolution in protoplanetary disks. One repository and one
Makefile build it for either NVIDIA CUDA or AMD HIP/ROCm, and each build selects one of two
independent dust representations:

- an Eulerian, pressureless **dust fluid** for conservative continuum evolution
- a Lagrangian **dust swarm** for particle trajectories, stochastic diffusion, and
  representative-particle collisions

Both representations evolve dust in a prescribed gas disk with one-way gas-to-dust coupling. They
share coordinate and physical conventions but own separate headers, kernels, and runtimes.
Simulation setups, units, resolutions, physical switches, and output cadence are compile-time
choices of a model directory, so GameDev is a research code rather than a packaged application.
Read the numerical and test-set guides before using results for scientific analysis.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Building](#building)
- [Configuring a model](#configuring-a-model)
- [Running](#running)
- [Restarting a simulation](#restarting-a-simulation)
- [Output files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Validation](#validation)
- [Reproducibility](#reproducibility)
- [Current limitations](#current-limitations)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation and license](#citation-and-license)

## Overview

### Dust representations

| Capability | Eulerian fluid | Lagrangian swarm |
|---|---|---|
| State | cell-averaged density and momentum | weighted computational representatives |
| Grain sizes | one fixed species | monodisperse or multisize |
| Velocity closure | one velocity per cell | multiple velocities can coexist locally |
| Transport | conservative finite-volume sweeps | semi-analytic particle trajectories |
| Diffusion | spherical finite-volume density diffusion | cylindrical stochastic displacement |
| Radiation pressure | optional | optional, with optional Poynting–Robertson drag |
| Collisions | not implemented | optional coagulation and fragmentation |
| Imported gas fields | not implemented | optional |
| Gas backreaction, dust self-gravity | not implemented | not implemented |

The fluid branch is most natural while the dust velocity remains approximately single valued. The
swarm branch retains trajectory crossing and a size distribution, at the cost of sampling noise and
more expensive neighborhood-based collisions. The two models are complementary rather than
interchangeable discretizations of every physical closure.

### Numerical methods at a glance

- **Fluid:** finite-volume PPM reconstruction with HLL fluxes and invariant-domain limiting, integer
  FARGO orbital shifts, SSPRK(3,3) sweeps, exponentially weighted drag and force sources,
  Crank–Nicolson diffusion with a conservative donor-momentum closure, and a palindromic operator
  composition. Smooth transport is second order in time and space.
- **Swarm:** staggered semi-analytic drag and gravity trajectories, radiation pressure and optional
  Poynting–Robertson drag, Itô Euler–Maruyama diffusion, a frozen-bath continuous-time collision
  chain with exact top-$K$ neighbors from a KD tree or an adaptive Morton hierarchy, and Strang
  composition of the enabled operators. Stochastic diffusion and collisions require statistical
  convergence tests rather than a pointwise order claim.

The equations, discretizations, accuracy statements, and GPU implementation details are in
[`doc/fluid_numeric.md`](doc/fluid_numeric.md) and [`doc/swarm_numeric.md`](doc/swarm_numeric.md).

### Coordinates and geometries

The computational coordinates are spherical,

$$
(x,y,z)=(\phi,r,\theta),
\qquad
R=y\sin z,
\qquad
Z=y\cos z,
$$

where $R$ and $Z$ are cylindrical radius and height. The mesh is uniform in $x$ and $z$ and
logarithmic in $y$, and arrays are stored $x$-first,
$\mathrm{index}=i_x+i_yN_X+i_zN_XN_Y$. With `N_Z == 1` the evolved density is a vertically
integrated surface density; with `N_Z > 1` it is a volume density.

| Geometry | Grid condition | Fluid | Swarm | Interpretation |
|---|---|:---:|:---:|---|
| radial-only | `N_X == 1`, `N_Z == 1` | no | yes | axisymmetric, vertically integrated midplane dynamics |
| radial–azimuthal | `N_X > 1`, `N_Z == 1` | yes | yes | vertically integrated disk plane |
| radial–polar | `N_X == 1`, `N_Z > 1` | no | yes | axisymmetric resolved vertical structure |
| full 3D | `N_X > 1`, `N_Z > 1` | yes | yes | resolved spherical dynamics |

The fluid branch always requires an active azimuthal dimension. Every three-dimensional fluid or
swarm model requires `DIFFUSION`, which supports the vertically resolved dust layer.

## Requirements

- GNU Make and a C++17 host compiler supported by the GPU toolkit
- **CUDA:** a CUDA-capable NVIDIA GPU and the CUDA toolkit (`nvcc`, Thrust, cuRAND). The default
  target is `GPU_TARGET=sm_80` (Ampere, for example A100).
- **ROCm:** `hipcc`, the HIP runtime, hipRAND, rocThrust, and hipCUB, and an AMD GPU supported by
  `GPU_TARGET`. The default target is `gfx942` (MI300A).
- Python 3.10 or later with NumPy for the validation runners, validators, and analysis scripts

Both backends compile with `-O2 -std=c++17` and without fast-math, because finite-only assumptions
would invalidate the production NaN and Inf guards. There is no installation step.

## Quick start

Run all commands from the repository root. Build and run the fiducial fluid model:

```bash
make MODEL=fluid_fiducial GPU_BACKEND=cuda
```

```bash
mod/fluid_fiducial/gamedev
```

Build and run the fiducial swarm model on ROCm:

```bash
make MODEL=swarm_fiducial GPU_BACKEND=rocm
```

```bash
mod/swarm_fiducial/gamedev
```

The fiducial models are production-scale examples (a $1024\times1024$ fluid grid and $10^7$ swarm
representatives), not lightweight demonstrations. For a short installation check, run the
validation workflow in quick mode:

```bash
python3 -B val/run_all.py --backend cuda --quick
```

## Building

### Make variables

`make MODEL=<name>` builds one model. The other variables can be given on the command line or, for
the model-level settings, in the model's `flags.mk`; command-line values take precedence.

| Variable | Values and default | Effect |
|---|---|---|
| `MODEL` | directory name, required | model to build; see [Configuring a model](#configuring-a-model) |
| `GPU_BACKEND` | `cuda` (default) or `rocm` | selects `nvcc` or `hipcc` and the backend mappings in `inc/gpu.cuh` |
| `GPU_TARGET` | `sm_80` on CUDA, `gfx942` on ROCm | GPU architecture passed to the compiler |
| `FLUID_SWEEP` | `thread` on CUDA, `block` on ROCm | fluid line solver: one thread per line, or one cooperative block per line |
| `COLLISION_SEARCH` | `kdtree` on CUDA, `morton` on ROCm | swarm collision-neighbor search |
| `CUDA_FLAGS`, `ROCM_FLAGS` | empty | extra compiler flags appended only for that backend |
| `RESOURCE_REPORT` | empty | on ROCm, any value prints per-kernel register and LDS usage |

The two fluid sweeps implement the same numerical method; their relative speed depends on the
resolution and GPU, so check both on the intended machine. On gfx942 the thread sweep exceeds a
linker limit at the fiducial `N_X = 1024`, which is why ROCm defaults to the block sweep.

### Executables, objects, and cleaning

A production model's executable is written beside its flags as `mod/<MODEL>/gamedev`, and its
objects go to `obj/<MODEL>/<fluid|swarm>/<backend>/<sweep or search>/<target>/`. Validation models
keep both below `val/fluid/obj/` or `val/swarm/obj/`, and their output below `val/*/out/`.
Each object directory keeps a configuration stamp, so a change of command-line definitions,
compiler flags, include or source selection, or output path recompiles the affected objects;
switching between configurations reuses their cached objects and always relinks the requested
executable. Remove one model's executable and objects, or everything generated:

```bash
make MODEL=fluid_fiducial clean
```

```bash
make clean
```

Build messages report the selected model, representation, constant header, header overrides,
fluid sweep or collision search, and for every compiled object which source file won the
model-over-production search order.

## Configuring a model

### Model directories

Production models live under `mod/`. A model directory needs a `flags.mk` and may add
`const_defs.cuh` and override files:

```text
mod/
└── my_model/
    ├── flags.mk          # required: representation and feature flags
    └── const_defs.cuh    # optional: complete replacement of the default constants
```

`MODEL` is looked up under `mod/`, `val/fluid/mod/`, and `val/swarm/mod/` and must resolve to
exactly one directory. `flags.mk` lists `DUST_REPR`, then `FLUID_SWEEP`, `COLLISION_SEARCH`, and
`MODEL_PARENT` as needed, a blank line, and one compile-time feature flag per line:

```make
DUST_REPR := swarm

GPU_FLAGS += -DTRANSPORT
GPU_FLAGS += -DRADIATION
GPU_FLAGS += -DSAVE_DENS
GPU_FLAGS += -DCODE_UNIT
```

### Fluid feature flags

Transport and the local drag, gravity, and geometric sources are always active in the fluid branch.

| Flag | Effect |
|---|---|
| `DIFFUSION` | spherical dust diffusion with Stokes-dependent diffusivity; required in 3D |
| `DIFFUSE_CONCENTRATION` | diffuse the dust-to-gas concentration instead of the dust density; takes effect only with `DIFFUSION` |
| `RADIATION` | attenuated radiation pressure |
| `VISC_FLOW` | prescribed viscous gas radial flow; requires `DIFFUSION` |
| `CONST_NU` | constant kinematic viscosity instead of constant $\alpha$ |
| `HALF_DISK` | 3D upper half disk with a reflecting midplane (`Z_MAX` must be $\pi/2$) |

### Swarm feature flags

| Flag | Effect |
|---|---|
| `TRANSPORT` | particle dynamics |
| `DIFFUSION` | stochastic position diffusion with Stokes-dependent diffusivity; requires `TRANSPORT`, required in 3D |
| `DIFFUSE_CONCENTRATION` | add the gas-density-gradient drift of concentration diffusion; takes effect only with `DIFFUSION` |
| `RADIATION` | attenuated radiation pressure; requires `TRANSPORT` |
| `PR_EFFECT` | first-order Poynting–Robertson drag; requires `RADIATION` |
| `VISC_FLOW` | viscous gas radial flow; requires `DIFFUSION`, excludes `IMPORTGAS` |
| `COLLISION` | representative-particle coagulation and fragmentation; requires `MULTISIZE` |
| `MULTISIZE` | store and evolve individual grain sizes and represented grain counts |
| `IMPORTGAS` | read gridded gas density and velocity fields; excludes `CONST_ST` |
| `CONST_ST` | hold the Stokes number fixed |
| `CONST_NU` | constant kinematic viscosity instead of constant $\alpha$ |
| `SAVE_DENS` | deposit and save the swarm density on the mesh |
| `CODE_UNIT` | code-unit collision prescription instead of physical gas microphysics |
| `HALF_DISK` | 3D upper half disk with a reflecting midplane |
| `LOGTIMING` | logarithmic output times; collision-only, excludes `TRANSPORT`, `SAVE_DENS`, and `LOGOUTPUT` |
| `LOGOUTPUT` | linear output times but particle checkpoints only at logarithmic frame indices |
| `COL_DIAGNOSTICS` | write collision-controller and event diagnostics |

A swarm model enables at least one of `TRANSPORT` and `COLLISION`. A collision build uses exactly
one search, chosen with `COLLISION_SEARCH := kdtree` (exact KD tree) or `COLLISION_SEARCH := morton`
(adaptive Morton index with periodic ghosts). The collision integrator is always the frozen-bath
chain; the removed flags `BERNOULLI`, `KNN_CACHE`, and `COL_CHAIN` stop the build with an error.
The dependencies marked "requires" or "excludes" are compile-time errors from
`inc/swarm/swarm_kern.cuh`, `inc/fluid/fluid_kern.cuh`, or the constant headers.

### Constants

Grid dimensions and extents, disk and dust parameters, particle count, KNN settings, output
cadence, and launch widths are `constexpr` constants in the representation's default header,
[`inc/fluid/const_defs.cuh`](inc/fluid/const_defs.cuh) or
[`inc/swarm/const_defs.cuh`](inc/swarm/const_defs.cuh). To change any of them, copy the default
into the model directory and edit the copy; the model's `const_defs.cuh` replaces the default
entirely. Constants a user most often changes are:

| Constant | Meaning |
|---|---|
| `N_X`, `N_Y`, `N_Z` and `X_MIN` … `Z_MAX` | grid size and extent in $(\phi,r,\theta)$ |
| `N_P` | number of swarm representatives |
| `SAVE_MAX`, `DT_OUT` | number of output frames and the output interval |
| `LIN_BASE`, `LOG_BASE` | swarm particle-checkpoint stride and logarithmic base (with `LOGTIMING` or `LOGOUTPUT`) |
| `CFL_DYN`, `DT_MAX` | dynamics Courant factor and timestep ceiling |
| `COAG_KERNEL` | `0`–`2` normalized synthetic collision kernels, `3` physical cross section and relative velocity |
| `N_K`, `H_SEARCH` | retained collision neighbors and the search cap in gas scale heights |

The numerical guides list every constant with its role. Do not add preprocessor parameters to the
default headers only to configure one model.

### Source and header overrides

A model-local `.cu` file replaces the production translation unit of the same name, and a
model-local header wins include lookup over the production header of the same name. ROCm uses a
model `.cu` as HIP when it is backend neutral or selects its API with `GAMEDEV_CUDA` and
`GAMEDEV_ROCM`; a model `.hip` file may instead provide a ROCm-only implementation. Directory
priority decides before the extension: a model `.cu` wins over a production `.hip`, and within one
directory ROCm prefers `.hip`. `MODEL_PARENT := <parent>` inherits the files of another model
directory, with the child's files taking priority. Overrides suit controlled experiments and
analytical tests; physics meant for every model belongs in the branch source.

## Running

### Console output and failure behavior

A run starts from frame 0 and advances to `SAVE_MAX`. The fluid branch prints each step's frame
index, timestep, and elapsed and total time, plus a `[CFL]` line naming the rate-limiting cell. The
swarm branch prints the frame index `idx`, the total time `clock_sim`, the time within the current
output interval `clock_out`, and, when enabled, the dynamics (`count_dyn`, `dt_dyn`) and collision
(`count_col`, `dt_col`) step counts and sizes. Both print `NNN/SAVE_MAX finished on <date>` after
each frame.

The runtimes check the evolved state for non-finite values: the fluid branch after its operators,
the swarm branch before every collision-neighbor search. A non-finite cell or particle, a failed
file read, a failed swarm write, or any GPU runtime or kernel error prints a message beginning with
`Error:` and exits with a nonzero status, so a batch job fails visibly instead of writing corrupted
frames. A failed fluid frame write prints `Error:` and the run continues. Internal consistency
failures of the collision chain, the Morton index, the controller, and initialization throw a C++
exception instead; the runtime then terminates with the exception's message and a nonzero status.
[`doc/swarm_numeric.md`](doc/swarm_numeric.md#116-failure-behavior) and
[`doc/fluid_numeric.md`](doc/fluid_numeric.md#84-finite-state-checks-and-failure-behavior) list the
checks.

### GPU memory

The fluid branch keeps about eight double-precision device arrays per cell (density, momenta,
primitive velocities, and CFL rates), nine with `RADIATION`; the block sweep adds a
twelve-field workspace, roughly 96 bytes per cell. A swarm representative costs 48 bytes of state,
64 with `MULTISIZE`, plus one RNG state with `DIFFUSION` or `COLLISION`. Collisions add roughly
$146+4N_K$ bytes per representative, dominated by the neighbor cache, plus the search index: about
1.15 kB per representative, or 11.5 GB for the default $10^7$ representatives at `N_K = 200`.
[`doc/swarm_numeric.md`](doc/swarm_numeric.md#114-memory-footprint) gives the complete estimate.

## Restarting a simulation

Pass a saved frame index to the executable:

```bash
mod/swarm_fiducial/gamedev 10
```

The restart time is reconstructed from the model's output schedule. The fluid branch reloads
density and physical linear velocities, converts them to its internal angular state, and rebuilds
momentum and optical depth. The swarm branch reloads its particle checkpoint and, when diffusion or
collisions use random numbers, the matching RNG-state checkpoint, so a swarm restart frame must be
one at which a particle checkpoint was written; with `LIN_BASE > 1` or `LOGOUTPUT`, mesh-only
frames cannot be resumed. A restart continues writing frames into the same output directory and
does not rewrite `variables.txt`. Restart files are not an interchange format between versions or
backends: raw swarm RNG state depends on the backend's RNG layout.

## Output files

Production output goes to `out/<MODEL>/`; the path is fixed at compile time and survives cleaning.
Frames use five-digit indices such as `00000`. Binary arrays hold native `double` values in the
writing host's byte order without a header, and `variables.txt` records the parameters needed to
interpret them in INI format.

| Branch | Files per frame | Notes |
|---|---|---|
| fluid | `dustdens`, `dustvelx`, `dustvely`, `dustvelz`, `optdepth` (with `RADIATION`) | `N_X*N_Y*N_Z` values, $x$-first; velocities are physical linear components |
| swarm | `particle`, `rngstate` (with `DIFFUSION` or `COLLISION`), `dustdens` (with `SAVE_DENS`), `optdepth` (with `RADIATION`) | mesh fields every frame; particle and RNG checkpoints at frame 0 and then every `LIN_BASE` frames, at powers of `LOG_BASE` with `LOGOUTPUT`, or every frame with `LOGTIMING` |

`dustdens` is a surface density for `N_Z == 1` and a volume density for `N_Z > 1`.
`particle_*.dat` stores physical positions and linear velocities and, with `MULTISIZE`, grain size
and represented grain count; its exact record layout is the `[SWARM_DTYPE]` section of
`variables.txt`. Builds with `COL_DIAGNOSTICS` also write `collision_chain_<frame>.json` and
`collision_local_<timestamp>.jsonl`.

With `IMPORTGAS`, the swarm runtime reads `gasdens_*.dat`, `gasvelx_*.dat`, `gasvely_*.dat`, and
`gasvelz_*.dat` for the starting frame and every later frame from the output directory, and a fresh
start reads `epsilon_00000.dat` to shape the initial dust sampling. Imported values must be finite
and nonnegative, with a finite positive total dust mass.

### Reading output with Python

Take the dimensions and the particle layout from `variables.txt`:

```python
from configparser import ConfigParser
from pathlib import Path

import numpy as np

output = Path("out/fluid_fiducial")
config = ConfigParser()
config.read(output / "variables.txt")
nx, ny, nz = (int(config["PARAMETERS"][key]) for key in ("N_X", "N_Y", "N_Z"))

dustdens = np.fromfile(output / "dustdens_00000.dat", dtype=np.float64).reshape(nz, ny, nx)
```

```python
particle_dtype = np.dtype(list(config["SWARM_DTYPE"].items()))
particles = np.fromfile(Path("out/swarm_fiducial") / "particle_00000.dat", dtype=particle_dtype)
```

For the swarm example, read that model's `variables.txt` into `config` first. The reshaped fluid
array is indexed `[iz, iy, ix]`. For output written on a machine with a different byte order, give
NumPy an explicitly byte-ordered dtype.

## Troubleshooting

| Symptom | Check |
|---|---|
| `MODEL is not defined` | run `make MODEL=<directory-name>` from the repository root |
| model not found or ambiguous | the name must occur exactly once under `mod/`, `val/fluid/mod/`, or `val/swarm/mod/` |
| `DUST_REPR is not defined` | every `flags.mk` sets `DUST_REPR := fluid` or `swarm` |
| feature-dependency `#error` | check the flag constraints in [Configuring a model](#configuring-a-model) |
| unsupported GPU architecture | pass the correct `GPU_TARGET` for the installed GPU and rebuild |
| an unexpected source is compiled | read the `Compiling ...` lines and look for model-local files that shadow production names |
| a failing or hanging kernel is hard to locate | rebuild with `CUDA_FLAGS=-DCUDA_SYNC_TRACE` or `ROCM_FLAGS=-DHIP_SYNC_TRACE`; every kernel check then synchronizes and prints the kernel name |
| no files in the expected output directory | the output path is compiled in; check `out/<MODEL>/` for the model named during the build |
| a swarm restart diverges immediately | the `particle_*` and `rngstate_*` files must come from the same frame and build |
| validation fails only on a new GPU or toolkit | archive the environment, rebuild cleanly, and separate tolerance-level rounding from a failed invariant or convergence criterion |

## Validation

The validation suites under `val/` run production kernels against analytical, statistical, and
invariant references for both representations. A complete native campaign for one backend is

```bash
python3 -B val/run_all.py --backend cuda
```

[`val/README.md`](val/README.md) explains the campaign, its outputs, and the CUDA–ROCm comparison;
[`doc/fluid_testset.md`](doc/fluid_testset.md) and [`doc/swarm_testset.md`](doc/swarm_testset.md)
state what each test establishes and its acceptance criteria. The scientific campaigns under
`val/paper/` have their own READMEs.

## Reproducibility

For every scientific run, retain:

- the repository commit or an exact source archive
- the complete model directory, including `flags.mk`, `const_defs.cuh`, and local overrides
- the backend, `GPU_TARGET`, `FLUID_SWEEP` or `COLLISION_SEARCH`, compiler and toolkit versions,
  driver, and GPU model
- `variables.txt`, runtime logs, the initial frame, any restart frame, and validation metrics
- the RNG-state checkpoints of stochastic swarm calculations

Byte-for-byte agreement is not expected after changing the GPU architecture, compiler,
optimization flags, fluid sweep, or KNN tie-breaking. Compare deterministic fields first at short,
roundoff-sensitive times and then by error norms and conserved quantities; compare stochastic or
chaotic calculations through ensemble distributions of scientifically relevant observables.

## Current limitations

- The gas is prescribed and receives no dust backreaction, and neither representation includes
  dust self-gravity.
- The fluid branch is pressureless and monodisperse and cannot represent multistream velocity
  distributions after trajectory crossing; its diffusive momentum closure is a documented modeling
  approximation.
- Swarm diffusion and collisions are stochastic and need particle-number and ensemble convergence,
  not only mesh convergence; the frozen-bath tolerance and the fixed neighbor reservoir need
  convergence checks for each scientific use.
- Collision neighbor searches use a local planar metric with a documented search-radius limit.
- Runs use one GPU; multi-GPU domain decomposition is not implemented.
- Checkpoints and RNG-state files are not portable between CUDA and ROCm, and the backend and GPU
  target can change rounding and long-time trajectories.
- Long-time, imported-gas, extreme-vacuum, and large-production collision regimes are less
  thoroughly exercised than the analytical core.

The numerical guides document representation-specific limitations in detail.

## Repository layout

```text
.
├── Makefile           # model, representation, and GPU-backend build rules
├── inc/
│   ├── gpu.cuh        # CUDA/HIP API mappings and GPU error checks
│   ├── fluid/         # fluid headers
│   └── swarm/         # swarm headers, including the collision chain and KNN searches
├── src/
│   ├── fluid/         # fluid kernels and runtime, one kernel per file
│   └── swarm/         # swarm kernels and runtime, one kernel per file
├── mod/               # production model setups
├── val/               # validation suites, scientific campaigns, and campaign utilities
├── doc/               # numerical and test-set guides
├── obj/               # generated objects (not tracked)
└── out/               # generated production output (not tracked)
```

The ignored `lab/` directory may hold experimental work; it is not part of the production or
validation interface.

## Documentation

| Document | Purpose |
|---|---|
| [`doc/README.md`](doc/README.md) | documentation map, conventions shared by both representations, and authority order |
| [`doc/fluid_numeric.md`](doc/fluid_numeric.md) | fluid equations, initialization, numerics, accuracy, and implementation |
| [`doc/swarm_numeric.md`](doc/swarm_numeric.md) | swarm equations, mass weighting, trajectories, diffusion, collisions, and neighbor search |
| [`doc/fluid_testset.md`](doc/fluid_testset.md) | fluid validation cases, references, and acceptance criteria |
| [`doc/swarm_testset.md`](doc/swarm_testset.md) | swarm validation cases, statistical references, and acceptance criteria |
| [`val/README.md`](val/README.md) | running, archiving, and comparing the validation suites |

Where prose and code disagree, the current source and matching machine-readable results take
precedence.

## Contributing

When changing a numerical method or physical prescription:

1. keep the fluid and swarm implementations owned by their own branches
2. update the corresponding `doc/*_numeric.md` guide with the equation, assumptions, and reference
3. add or update an analytical, statistical, boundary, or regression test under `val/`
4. archive enough metrics and environment information to support the new claim
5. update the relevant `doc/*_testset.md` summary
6. keep the established names, formatting, and coordinate conventions of the affected branch

Record surviving invariants, limitations, and regression requirements in the guides rather than a
history of resolved work.

## Citation and license

A formal software citation is not yet provided. Until one is, cite the repository version or commit
used for a calculation and record the model constants, compile-time flags, GPU backend, toolkit,
compiler, target architecture, and relevant validation results.

GameDev's original code is distributed under the [MIT License](LICENSE). Copyright 2026 Jiaqing Bi.

GameDev includes modified portions of [cudaKDTree](https://github.com/ingowald/cudaKDTree),
copyright 2018-2023 Ingo Wald, under the [Apache License 2.0](inc/swarm/kdtree/Apache-2.0.txt),
which is retained beside the bundled implementation. Files under `inc/swarm/kdtree/cubit/` derive
from [cudaBitonic](https://github.com/ingowald/cudaBitonic), copyright 2018-2023 Ingo Wald, under
the same license. The original copyright and license notices are retained in the bundled sources.
