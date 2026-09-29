# GameDev

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

GameDev is a GPU research code for dust evolution in protoplanetary disks. One repository and one
Makefile build it for either NVIDIA CUDA or AMD HIP/ROCm, and each build selects one of two
independent dust representations:

- a Lagrangian **dust swarm** for particle trajectories, stochastic diffusion, and
  representative-particle collisions
- an Eulerian, pressureless **dust fluid** for conservative continuum evolution

Both representations evolve dust in a prescribed gas disk with one-way gas-to-dust coupling. They
share the disk model of [`doc/guide_basis.md`](doc/guide_basis.md) but own separate headers,
kernels, and runtimes. Simulation setups, units, resolutions, physical switches, and output
cadence are compile-time choices of a model directory, so GameDev is a research code rather than a
packaged application. This file is the user guide: it covers building, configuring, running,
restarting, and reading output. Read the numerical and test-set guides listed in
[Documentation](#documentation) before using results for scientific analysis; project terms are
defined in the [glossary](doc/README.md#glossary).

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

| Property | Lagrangian swarm | Eulerian fluid |
|---|---|---|
| State | weighted computational representatives | cell-averaged density and momentum |
| Grain sizes | monodisperse or multisize | monodisperse (one grain size) |
| Velocity at one position | several representatives may cross with different velocities | single valued |
| Transport | semi-analytic particle trajectories | conservative finite-volume sweeps |
| Diffusion | cylindrical stochastic displacement | spherical finite-volume diffusion |
| Radiation pressure | optional, with optional Poynting–Robertson drag | optional |
| Collisions | optional coagulation and fragmentation | not implemented |
| Imported gas fields | optional | not implemented |
| Gas backreaction, dust pressure, dust self-gravity | not implemented | not implemented |

The swarm keeps trajectory crossing and a size distribution, at the cost of sampling noise and
more expensive neighborhood-based collisions. The fluid is most natural while the dust velocity
stays approximately single valued. The two models complement each other; they are not
interchangeable discretizations of every physical closure.

### Numerical methods at a glance

- **Swarm:** staggered semi-analytic drag and gravity trajectories, radiation pressure and optional
  Poynting–Robertson drag, Itô Euler–Maruyama diffusion, and Strang composition of the enabled
  operators ([`doc/guide_swarm.md`](doc/guide_swarm.md)). Collisions use a frozen-bath
  continuous-time event chain with exact top-$`K`$ neighbors from a KD tree or an adaptive Morton
  hierarchy ([swarm collisions](doc/guide_swarm.md#8-collisions)). Stochastic diffusion and
  collisions need statistical convergence tests rather than a pointwise order claim.
- **Fluid:** finite-volume PPM reconstruction with HLL fluxes and invariant-domain limiting, integer
  FARGO orbital shifts, SSPRK(3,3) sweeps, exponentially weighted drag and force sources,
  Crank–Nicolson diffusion with a conservative donor-momentum closure, and a palindromic operator
  composition ([`doc/guide_fluid.md`](doc/guide_fluid.md)). PPM is nominally third order in
  space, and the composed scheme is second order overall for smooth transport.

### Coordinates and geometries

The computational coordinates are spherical, $(x,y,z)=(\phi,r,\theta)$, with a mesh uniform in
azimuth and polar angle and logarithmic in radius
([coordinates](doc/guide_basis.md#21-computational-coordinates)). With `N_Z == 1` the evolved
density is a vertically integrated surface density; with `N_Z > 1` it is a volume density. The
swarm supports all four geometries (radial-only, radial–azimuthal, radial–polar, and full 3D); the
fluid needs an active azimuth (`N_X > 1`) and so supports only radial–azimuthal and full 3D
([supported geometries](doc/guide_basis.md#24-supported-geometries)).

## Requirements

- GNU Make and a C++17 host compiler supported by the GPU toolkit
- **CUDA:** a CUDA-capable NVIDIA GPU and the CUDA toolkit (`nvcc`, Thrust, cuRAND). The default
  target is `GPU_TARGET=sm_80` (Ampere, for example A100).
- **ROCm:** `hipcc`, the HIP runtime, hipRAND, rocThrust, and hipCUB, and an AMD GPU supported by
  `GPU_TARGET`. The default target is `gfx942` (MI300A).
- Python 3.10 or later with NumPy for the validation runners, validators, and analysis scripts;
  `python3 -m pip install -r requirements.txt` installs the Python packages

Both backends compile with `-O2 -std=c++17` and without fast-math, because finite-only assumptions
would invalidate the production NaN and Inf guards. There is no installation step.

## Quick start

Run all commands from the repository root. Build and run the fiducial swarm model on ROCm:

```bash
make MODEL=swarm_fiducial GPU_BACKEND=rocm
```

```bash
mod/swarm_fiducial/gamedev
```

Build and run the fiducial fluid model on CUDA:

```bash
make MODEL=fluid_fiducial GPU_BACKEND=cuda
```

```bash
mod/fluid_fiducial/gamedev
```

The fiducial models are production-scale examples ($10^7$ swarm representatives and a
$1024\times1024$ fluid grid), not lightweight demonstrations. For a short installation check, run
the validation workflow in quick mode:

```bash
python3 -B val/run_all.py --backend cuda --quick
```

## Building

### Make variables

`make MODEL=<name>` builds one model. The other variables can be given on the command line or, for
the model-level settings, in the model's `flags.mk`; command-line values take precedence.

| Variable | Values and default | Effect |
|---|---|---|
| `MODEL` | directory name, required | model to build; see [Model directories](#model-directories) |
| `GPU_BACKEND` | `cuda` (default) or `rocm` | selects `nvcc` or `hipcc` and the backend mappings in `inc/gpu.cuh` |
| `GPU_TARGET` | `sm_80` on CUDA, `gfx942` on ROCm | GPU architecture passed to the compiler |
| `GPU_COMPILER` | `nvcc` on CUDA, `hipcc` on ROCm | compiler executable |
| `COLLISION_SEARCH` | `kdtree` on CUDA, `morton` on ROCm | swarm collision-neighbor search: exact KD tree or adaptive Morton index |
| `FLUID_SWEEP` | `thread` on CUDA, `block` on ROCm | fluid line solver: one thread per line, or one cooperative block per line |
| `CUDA_FLAGS`, `ROCM_FLAGS` | empty | extra compiler flags appended only for that backend |
| `RESOURCE_REPORT` | empty | on ROCm, any value prints per-kernel register and LDS usage |

**Search and sweep selection.** A default applies only when neither the command line nor the
model's `flags.mk` sets the variable. Any value other than the two listed stops the build with
`COLLISION_SEARCH must be kdtree or morton` or `FLUID_SWEEP must be thread or block`.

- `COLLISION_SEARCH` is read only in swarm builds with `-DCOLLISION`. The Makefile turns it into
  the internal macro `COLLISION_KDTREE` or `COLLISION_MORTON`; both GPU backends accept either
  search. Objects of the two searches go to separate directories, so switching never reuses
  objects compiled for the other search. `variables.txt` records the choice as `COLLISION_SEARCH`,
  with `MORTON_LEAF_TARGET` and `MORTON_MAX_LEVEL` for the Morton search.
  [Choosing a search](doc/guide_swarm.md#96-choosing-a-search) compares them.
- `FLUID_SWEEP=block` defines `FLUID_BLOCK_SWEEP`. The two sweeps implement the same numerical
  method with different work decomposition and memory, and their objects also go to separate
  directories. Their relative speed depends on the resolution and GPU, so check both on the
  intended machine ([Choosing a sweep](doc/guide_fluid.md#107-choosing-a-sweep)). On gfx942 the
  thread sweep exceeds a linker limit at the fiducial `N_X = 1024`, which is why ROCm defaults to
  the block sweep. `mod/fluid_fiducial/flags.mk` sets `FLUID_SWEEP := thread`, so pass
  `FLUID_SWEEP=block` on the command line to build it for ROCm.

Validation models accept further variables (`RES`, `CFL`, `SHIFT`, `OUT_TAG`, `VAL_SCOPE`, and
others) that the validation runners set; [`val/README.md`](val/README.md) describes the runners.

### Executables, objects, and cleaning

A production model's executable is written beside its flags as `mod/<MODEL>/gamedev`, and its
objects go to `obj/<MODEL>/<swarm|fluid>/<backend>/[<search or sweep>/]<target>/`; the search
level exists only for collision builds. Validation models write their executable to
`val/<swarm|fluid>/obj/<MODEL>/<backend>/gamedev` and their objects below the same directory.
Each object directory keeps a configuration stamp, so a change of command-line definitions,
compiler flags, include or source selection, or output path recompiles the affected objects;
switching between configurations reuses their cached objects and always relinks the requested
executable.

Remove one model's executable and the objects of the selected configuration (backend, search or
sweep, and target):

```bash
make MODEL=swarm_fiducial clean
```

Remove all objects (including those of the validation models and `val/paper/` campaigns) and all
production executables; output directories are kept:

```bash
make clean
```

Build messages report the backend, target, model, constant header (when no model header replaces
it), header overrides, representation, and collision search or fluid sweep, and for every
compiled object which source file won the model-over-production search order.

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

`MODEL` is looked up under `mod/`, `val/swarm/mod/`, and `val/fluid/mod/` and must resolve to
exactly one directory. `flags.mk` lists `DUST_REPR`, then `COLLISION_SEARCH`, `FLUID_SWEEP`, and
`MODEL_PARENT` as needed, a blank line, and one compile-time feature flag per line:

```make
DUST_REPR := swarm

GPU_FLAGS += -DTRANSPORT
GPU_FLAGS += -DRADIATION
GPU_FLAGS += -DSAVE_DENS
GPU_FLAGS += -DCODE_UNIT
```

The flags decide which equations exist in an executable; they do not switch operators on or off
during a run. Every "requires" and "excludes" entry below is a compile-time `#error` or
`static_assert` in `inc/swarm/swarm_kern.cuh`, `inc/fluid/fluid_kern.cuh`, or the default constant
header, so a violating model fails to build.

### Swarm feature flags

| Flag | Effect | Requires or excludes |
|---|---|---|
| `TRANSPORT` | integrate particle trajectories with drag, gravity, and any enabled radiative force | a model needs `TRANSPORT`, `COLLISION`, or both |
| `DIFFUSION` | add cylindrical stochastic position diffusion with Stokes-dependent diffusivity | requires `TRANSPORT`; required by `N_Z > 1` |
| `DIFFUSE_CONCENTRATION` | diffuse the dust-to-gas concentration instead of the dust density | requires `DIFFUSION` |
| `RADIATION` | deposit and accumulate optical depth and add attenuated radiation pressure | requires `TRANSPORT` |
| `PR_EFFECT` | add first-order Poynting–Robertson drag | requires `RADIATION` |
| `VISC_FLOW` | prescribe the viscous radial gas flow | requires `DIFFUSION`; excludes `IMPORTGAS` |
| `COLLISION` | evolve representative-particle coagulation and fragmentation with the frozen-bath chain | requires `MULTISIZE` and one `COLLISION_SEARCH` |
| `MULTISIZE` | store and evolve each representative's grain size and represented grain count | none |
| `IMPORTGAS` | read gridded gas density and velocity fields ([Imported gas input](#imported-gas-input)) | excludes `CONST_ST` and `VISC_FLOW` |
| `CONST_ST` | fix the Stokes number at $`\mathrm{St}_0\,s/S_0`$, keeping only its grain-size factor | excludes `IMPORTGAS` |
| `CONST_NU` | use a constant kinematic viscosity instead of a constant $\alpha$ | acts only with `DIFFUSION` or `COLLISION` |
| `CODE_UNIT` | use the code-unit collision calibration instead of physical gas microphysics | acts only with `COLLISION` |
| `HALF_DISK` | 3D upper half disk with a reflecting midplane | with `N_Z > 1`, `Z_MAX` must be $\pi/2$ (checked at run time) |
| `SAVE_DENS` | deposit and save the swarm density on the mesh | excluded by `LOGTIMING` |
| `LOGTIMING` | logarithmic output times, for collision-only runs | excludes `LOGOUTPUT`, `TRANSPORT`, and `SAVE_DENS` |
| `LOGOUTPUT` | linear output times, particle checkpoints only at logarithmic frame indices | excludes `LOGTIMING` |
| `COL_DIAGNOSTICS` | write collision-controller and event diagnostics ([File contents](#file-contents)) | acts only with `COLLISION` |

The reasons for the rules: diffusion and radiation run inside the dynamics step; the drag uses the
attenuated radiation ratio; the viscous flow needs $\nu$, and imported gas already prescribes the
velocity; imported density determines the Stokes number; collision outcomes change grain size and
represented number; an executable must evolve something; logarithmic timing serves collision-only
runs; and a resolved vertical layer needs vertical mixing. The `N_Z > 1` rule is a `static_assert`
in the default header `inc/swarm/const_defs.cuh`, so a model `const_defs.cuh` that replaces the
header should keep it. The collision integrator is always the frozen-bath chain: the flags
`BERNOULLI`, `KNN_CACHE`, and `COL_CHAIN` stop the build with an error. Compile-time checks on the
collision constants are listed in [collision parameters](doc/guide_swarm.md#24-parameters).

### Fluid feature flags

Transport and the local drag, gravity, and geometric sources are always compiled in the fluid
branch and have no flag.

| Flag | Effect | Requires or excludes |
|---|---|---|
| `DIFFUSION` | add the three directional diffusion operators with Stokes-dependent diffusivity and their momentum closure; use the diffusive balance in the initial polar velocity | required by `N_Z > 1` |
| `DIFFUSE_CONCENTRATION` | diffuse the dust-to-gas concentration instead of the dust density | requires `DIFFUSION` |
| `RADIATION` | construct optical depth, add attenuated radiation pressure, and write `optdepth` frames | none |
| `VISC_FLOW` | use the viscous radial gas velocity in the source update, initialization, and CFL rate | requires `DIFFUSION` |
| `CONST_NU` | use a constant kinematic viscosity instead of a constant $\alpha$ | acts only with `DIFFUSION` |
| `HALF_DISK` | 3D upper half disk with a reflecting midplane | with `N_Z > 1`, `Z_MAX` must be $\pi/2$ (compile time); no effect with `N_Z == 1` |

The `N_Z > 1` and `HALF_DISK` rules are `static_assert`s in the default header
`inc/fluid/const_defs.cuh`, which also checks the grid and `CFL_DYN`
([fluid parameters](doc/guide_fluid.md#24-parameters)). A model `const_defs.cuh` replaces those
checks together with the rest of the header, so it should keep them.

### Constants

Grid dimensions and extents, disk and dust parameters, particle count, KNN settings, output
cadence, and launch widths are `constexpr` constants in the representation's default header,
[`inc/swarm/const_defs.cuh`](inc/swarm/const_defs.cuh) or
[`inc/fluid/const_defs.cuh`](inc/fluid/const_defs.cuh). To change any of them, copy the default
into the model directory and edit the copy; the model's `const_defs.cuh` replaces the default
entirely. Constants a user most often changes are:

| Constant | Meaning |
|---|---|
| `N_X`, `N_Y`, `N_Z` and `X_MIN` … `Z_MAX` | grid size and extent in $(\phi,r,\theta)$ |
| `N_P` | number of swarm representatives |
| `SAVE_MAX`, `DT_OUT` | final frame index and the output interval |
| `LIN_BASE`, `LOG_BASE` | swarm particle-checkpoint stride, or logarithmic base with `LOGTIMING` or `LOGOUTPUT` |
| `CFL_DYN`, `DT_MAX` | dynamics Courant factor and timestep ceiling |
| `COAG_KERNEL` | `0`–`2` normalized synthetic collision kernels, `3` physical cross section and relative velocity |
| `N_K`, `H_SEARCH` | retained collision neighbors and the search cap in gas scale heights |

Every constant, its default, and its role are listed in [swarm
parameters](doc/guide_swarm.md#24-parameters), [collision
parameters](doc/guide_swarm.md#24-parameters), and [fluid
parameters](doc/guide_fluid.md#24-parameters). Do not add preprocessor parameters to the default
headers only to configure one model.

### Source and header overrides

A model-local `.cu` file replaces the production translation unit of the same name, and a
model-local header wins include lookup over the branch header of the same name in `inc/swarm/` or
`inc/fluid/`. ROCm uses a model `.cu` as HIP when it is backend neutral or selects its API with
`GAMEDEV_CUDA` and `GAMEDEV_ROCM`; a model `.hip` file may instead provide a ROCm-only
implementation. Directory priority decides before the extension: a model `.cu` wins over a
production `.hip`, and within one directory ROCm prefers `.hip`. `MODEL_PARENT := <parent>`
inherits the files of another model directory under `mod/`, with the child's files taking
priority. Validation models also search `val/<swarm|fluid>/src/` right after their own directory.
A `flags.mk` may add header directories with `MODEL_INCLUDE_DIRS` and extra object names with
`_OBJ_MOD`. Overrides suit controlled experiments and analytical tests; physics meant for every
model belongs in the branch source.

## Running

A run starts from frame 0 and advances to frame `SAVE_MAX`; pass a frame index to resume instead
([Restarting a simulation](#restarting-a-simulation)).

### Console output

The swarm branch prints a column header at the start of every output interval and one line per
dynamics step, or per interval in collision-only runs. The columns are the frame being computed
`idx`, the total time `clock_sim`, and the time within the interval `clock_out`; with `TRANSPORT`,
the dynamics step count and size `count_dyn` and `dt_dyn`; and with `COLLISION`, the collision step
count and size `count_col` and `dt_col`, preceded by the collision time within the dynamics step
`clock_dyn` when transport is also on.

The fluid branch prints one header and then one line per global step: the index of the last saved
frame `idx_from`, the step `dt`, and the times `clock_out` and `clock_sim`.

Both branches print `NNN/SAVE_MAX finished on <date>` after each saved frame, including the
starting frame.

**Trace builds.** Rebuilding with `CUDA_FLAGS=-DCUDA_SYNC_TRACE` or `ROCM_FLAGS=-DHIP_SYNC_TRACE`
makes every kernel check synchronize and print `[CUDA] completed <kernel>` or
`[HIP] completed <kernel>`, so an asynchronous fault is attributed to the failing kernel. In the
fluid branch, trace builds also print a `[CFL] cell=(i,j,k)` line naming the rate-limiting cell,
with its radius, velocities, rate, and step, before every advection substep and after each opening
diffusion direction. Ordinary builds skip this printout.

### Failure behavior

The runtimes stop rather than continue from an invalid state; neither retries or rolls back a
step, and the most recent saved frame remains a valid restart point. Where checks run is described
in [swarm finite-state checks](doc/guide_swarm.md#127-finite-state-and-error-checks) and [fluid
finite-state checks](doc/guide_fluid.md#105-finite-state-checks). Messages go to standard error.
`EXIT_FAILURE` is a normal nonzero exit; "abort" means an uncaught C++ exception, which the runtime
does not catch, so the process terminates through the C++ runtime with the exception's message and
a nonzero status.

| Condition | Message | Exit |
|---|---|---|
| Both: failed GPU runtime call or kernel launch | `Error: CUDA failure at <file>:<line> during <call>: <reason> (<code>)`, or `HIP failure` on ROCm | `EXIT_FAILURE` |
| Swarm: invalid resume argument | `Error: Invalid resume file number: <arg>` | `EXIT_FAILURE` |
| Fluid: invalid resume argument | `Error: invalid resume frame number: <arg>` | `EXIT_FAILURE` |
| Swarm: missing restart file, or one whose size differs from the expected array (for example after changing `N_P`, `MULTISIZE`, or the RNG layout) | `Error: Failed to load file: <path>` | `EXIT_FAILURE` |
| Fluid: missing or wrongly sized restart file | `Error: failed to load <field> frame <n>` | `EXIT_FAILURE` |
| Swarm: missing or wrongly sized imported gas or `epsilon` file | `Error: Failed to load gas data files for frame <n>` | `EXIT_FAILURE` |
| Swarm: failed output write | `Error: Failed to save file: <path>` | `EXIT_FAILURE` |
| Fluid: failed output write | `Error: failed to save <field> frame <n>` | `EXIT_FAILURE` |
| Fluid: failed write of `variables.txt` | `Error: failed to save simulation parameters` | `EXIT_FAILURE` |
| Swarm: nonfinite particle state before a collision search | `Error: non-finite particle state before collision search at particle <p>` | `EXIT_FAILURE` |
| Fluid: nonfinite cell found by the full-grid check | `Error: non-finite simulation state at cell (<i>,<j>,<k>)` | `EXIT_FAILURE` |
| Fluid: nonfinite cell found by a CFL evaluation | `Error: non-finite dust state detected by CFL validation at cell (<i>,<j>,<k>)` | `EXIT_FAILURE` |
| Fluid, ROCm block sweep: diffusion line needs more LDS than the device allows | `Error: <kernel> requires <n> dynamic LDS bytes ...` | `EXIT_FAILURE` |
| Swarm: `HALF_DISK` with `N_Z > 1` and `Z_MAX` $\ne\pi/2$ | `HALF_DISK requires Z_MAX = pi/2` | abort, before evolution |
| Fluid: `HALF_DISK` with `N_Z > 1` and `Z_MAX` $\ne\pi/2$ | `HALF_DISK requires its reflecting outer polar boundary at Z_MAX = pi/2` | build fails (`static_assert`) |
| Swarm: initialized dust profile with no mass in the domain | `initialized dust profile has zero vertical mass`, `initialized dust profile has zero mass in the domain`, or `failed to select an initialized vertical interval` | abort, before evolution |
| Swarm: negative or nonfinite imported density, dust-to-gas ratio, or dust mass | `invalid imported gas density at cell <i>`, `invalid imported dust-to-gas ratio at cell <i>`, `nonfinite imported dust mass at cell <i>`, or `imported dust profile has zero or nonfinite total mass` | abort, before evolution |
| Swarm: Morton traversal-stack overflow or Morton allocation failure | `Morton traversal stack overflow in col_cache_get`, or `<operation>: <GPU error>` | abort |
| Swarm: invalid collision-chain or controller state | `local collision chain error <N>`, `local collision continuation limit exceeded`, or a controller message | abort |
| Swarm, `COL_DIAGNOSTICS`: diagnostics file cannot be opened or written | `cannot open local collision diagnostics` or `cannot write local collision diagnostics` | abort |

The chain error codes, the controller messages, and their limits are listed in [collision error
checks](doc/guide_swarm.md#127-finite-state-and-error-checks). The swarm branch does not check
the write of `variables.txt`. Controller audit overshoots, including persistent ones, are recorded
but never stop a run. Negative fluid densities are not failures: transport converts a negative
low-order density to exact vacuum, and diffusion cannot produce one.

### GPU memory

A swarm representative costs 48 bytes of state, 64 with `MULTISIZE`, plus one RNG state with
`DIFFUSION` or `COLLISION`. Collisions add roughly $146+4N_K$ bytes per representative, dominated
by the neighbor cache, plus the search index. A transported multisize physical-kernel model with
diffusion at the default `N_K = 200` needs about 1.15 kB per representative in total, or 11.5 GB
for the default $10^7$ representatives. The fluid branch keeps eight double-precision device arrays
per cell (density, three conserved momenta, three primitive velocities, and the CFL rate), nine
with `RADIATION`; the block sweep adds a twelve-field workspace, 96 bytes per cell. The complete
estimates are in [swarm memory](doc/guide_swarm.md#126-memory-footprint), [collision
memory](doc/guide_swarm.md#126-memory-footprint), and [fluid
memory](doc/guide_fluid.md#104-memory-footprint).

## Restarting a simulation

Pass a saved frame index to the executable:

```bash
mod/swarm_fiducial/gamedev 10
```

The run resumes at the output time of that frame ([Output times and
checkpoints](#output-times-and-checkpoints)) and continues writing frames `n+1` to `SAVE_MAX` into
the same output directory. `variables.txt` is written only by a fresh start; a restart does not
rewrite it.

- **Swarm.** The runtime reloads the particle checkpoint and, with `DIFFUSION` or `COLLISION`, the
  matching RNG-state checkpoint (an **RNG-state checkpoint** stores each representative's raw
  backend random-number state). A swarm restart frame must therefore be one at which a particle
  checkpoint was written; with `LIN_BASE > 1` or `LOGOUTPUT`, mesh-only frames cannot be resumed.
  With `IMPORTGAS` it also reloads the gas fields of the restart frame. The collision controller
  and neighbor geometry are rebuilt, so no other state is needed.
- **Fluid.** The runtime reloads density and the physical linear velocities, converts them to its
  internal angular state, rebuilds momentum and, with `RADIATION`, optical depth, and checks the
  restored state for nonfinite values. The time is reset to $`n\,\mathrm{DT\_OUT}`$. The fluid holds
  no random state.

A resumed run is not bitwise identical to an uninterrupted one, because the checkpoints store
linear rather than angular velocity; [swarm restart
semantics](doc/guide_swarm.md#128-output-and-restart-semantics) and [fluid restart
semantics](doc/guide_fluid.md#106-output-and-restart-semantics) explain why. Restart files are
not an interchange format between versions or backends: the raw swarm RNG state depends on the
backend's RNG layout, so CUDA checkpoints resume only on CUDA and ROCm checkpoints only on ROCm.

## Output files

Production output goes to `out/<MODEL>/`; the path is fixed at compile time and survives cleaning.
Frame numbers are zero padded to at least five digits, such as `00000`. Binary arrays hold native
`double` values in the writing host's byte order without a header, and `variables.txt` records the
parameters needed to interpret them in INI format.

### Output times and checkpoints

A frame is an output index $`n=0,\dots,\mathrm{SAVE\_MAX}`$. Each branch shortens the last step of
an output interval so that the saved state lies exactly at the output time; no interpolation is
used. Output times are linear,

```math
t_n=n\,\mathrm{DT\_OUT},
```

in the swarm branch by default and in the fluid branch. With the swarm flag `LOGTIMING` they are
logarithmic,

```math
t_0=0,
\qquad
t_n=\mathrm{DT\_OUT}\,\mathrm{LOG\_BASE}^{\,n}\quad(n\ge1).
```

A restart from frame $n$ resumes at $t_n$.

Mesh fields are written at every frame. Swarm particle checkpoints (with their RNG states) are
written at frame 0 and then

- at every frame with `LOGTIMING`;
- at frame indices that are integer powers of `LOG_BASE` (1, `LOG_BASE`, `LOG_BASE`², …) with
  `LOGOUTPUT`;
- at every `LIN_BASE`-th frame otherwise.

A frame without a particle checkpoint is a **mesh-only frame**. Skipping a checkpoint skips neither
the evolution nor the mesh diagnostics at that time.

### File contents

| Branch | Files per frame | Notes |
|---|---|---|
| swarm | `particle`, `rngstate` (with `DIFFUSION` or `COLLISION`), `dustdens` (with `SAVE_DENS`), `optdepth` (with `RADIATION`) | `particle` and `rngstate` only at checkpoint frames |
| fluid | `dustdens`, `dustvelx`, `dustvely`, `dustvelz`, `optdepth` (with `RADIATION`) | velocities are physical linear components $v_\phi$, $v_r$, $v_\theta$ |

Each file is named `<field>_<frame>.dat`. Mesh fields hold `N_X*N_Y*N_Z` values stored $`x`$-first,

```math
\mathrm{index}=i_x+i_yN_X+i_zN_XN_Y.
```

`dustdens` is a surface density for `N_Z == 1` and a volume density for `N_Z > 1`. `optdepth` is the
cumulative optical depth at the outer radial cell faces, rebuilt from the saved state.
`particle_*.dat` stores physical positions and linear velocities and, with `MULTISIZE`, grain size
and represented grain count; its exact record layout is the `[SWARM_DTYPE]` section of
`variables.txt`. `rngstate_*.dat` is raw backend memory. Builds with `COL_DIAGNOSTICS` also write
`collision_chain_<frame>.json` (the controller schedule of each output interval) and one
`collision_local_<timestamp>.jsonl` per run (timing and event statistics of every collision
operator), described in [collision
diagnostics](doc/guide_swarm.md#1210-collision-diagnostics-output).

### Imported gas input

With `IMPORTGAS`, the swarm runtime reads `gasdens_*.dat`, `gasvelx_*.dat`, `gasvely_*.dat`, and
`gasvelz_*.dat` for the starting frame and every later frame from the output directory, and a fresh
start reads `epsilon_00000.dat` to shape the initial dust sampling. Each file must hold exactly
`N_G` doubles in the same layout as the mesh output. Imported values must be finite and
nonnegative, with a finite positive total dust mass.

### Reading output with Python

Take the particle layout and the dimensions from `variables.txt`:

```python
from configparser import ConfigParser
from pathlib import Path

import numpy as np

output = Path("out/swarm_fiducial")
config = ConfigParser()
config.read(output / "variables.txt")

particle_dtype = np.dtype(list(config["SWARM_DTYPE"].items()))
particles = np.fromfile(output / "particle_00000.dat", dtype=particle_dtype)
```

```python
output = Path("out/fluid_fiducial")
config = ConfigParser()
config.read(output / "variables.txt")
nx, ny, nz = (int(config["PARAMETERS"][key]) for key in ("N_X", "N_Y", "N_Z"))

dustdens = np.fromfile(output / "dustdens_00000.dat", dtype=np.float64).reshape(nz, ny, nx)
```

The second example reuses the imports of the first. The reshaped fluid array is indexed
`[iz, iy, ix]`. For output written on a machine with a different byte order, give NumPy an
explicitly byte-ordered dtype.

## Troubleshooting

| Symptom | Check |
|---|---|
| `MODEL is not defined` | run `make MODEL=<directory-name>` from the repository root |
| model not found or ambiguous | the name must occur exactly once under `mod/`, `val/swarm/mod/`, or `val/fluid/mod/` |
| `DUST_REPR is not defined` | every `flags.mk` sets `DUST_REPR := swarm` or `fluid` |
| feature-dependency `#error` or `static_assert` | check the flag rules in [Configuring a model](#configuring-a-model) |
| unsupported GPU architecture | pass the correct `GPU_TARGET` for the installed GPU and rebuild |
| an unexpected source is compiled | read the `Compiling ...` lines and look for model-local files that shadow production names |
| a failing or hanging kernel is hard to locate | rebuild with `CUDA_FLAGS=-DCUDA_SYNC_TRACE` or `ROCM_FLAGS=-DHIP_SYNC_TRACE`; every kernel check then synchronizes and prints the kernel name |
| no `[CFL]` line naming the limiting fluid cell | the line is printed only by trace builds (`CUDA_SYNC_TRACE` or `HIP_SYNC_TRACE`); see [Console output](#console-output) |
| no files in the expected output directory | the output path is compiled in; check `out/<MODEL>/` for the model named during the build |
| a swarm restart diverges immediately | the `particle_*` and `rngstate_*` files must come from the same frame and build |
| validation fails only on a new GPU or toolkit | archive the environment, rebuild cleanly, and separate tolerance-level rounding from a failed invariant or convergence criterion |

## Validation

The validation suites under `val/` run production kernels against analytical, statistical, and
invariant references for both representations. A complete native campaign for one backend is

```bash
python3 -B val/run_all.py --backend cuda
```

[`val/README.md`](val/README.md) explains [the campaign](val/README.md#running-a-complete-campaign),
its outputs, and [the CUDA–ROCm comparison](val/README.md#comparing-cuda-and-rocm);
[`doc/guide_tests.md`](doc/guide_tests.md) states what each test establishes and its acceptance
criteria. The scientific campaigns under `val/paper/` have their own READMEs.

## Reproducibility

For every scientific run, retain:

- the release version ([`CHANGELOG.md`](CHANGELOG.md)) and repository commit, or an exact source
  archive
- the complete model directory, including `flags.mk`, `const_defs.cuh`, and local overrides
- the backend, `GPU_TARGET`, `COLLISION_SEARCH` or `FLUID_SWEEP`, compiler and toolkit versions,
  driver, and GPU model
- `variables.txt`, runtime logs, the initial frame, any restart frame, and validation metrics
- the RNG-state checkpoints of stochastic swarm calculations

Byte-for-byte agreement is not expected after changing the GPU architecture, compiler,
optimization flags, KNN search or tie-breaking, or fluid sweep. Changing the KNN search can change
partner ordering and random-number consumption even when the neighbor set is identical. Compare
deterministic fields first at short, roundoff-sensitive times and then by error norms and conserved
quantities; compare stochastic or chaotic calculations through ensemble distributions of
scientifically relevant observables.

## Current limitations

- The gas is prescribed and receives no dust backreaction, and neither representation includes
  dust self-gravity.
- Swarm diffusion and collisions are stochastic and need particle-number and ensemble convergence,
  not only mesh convergence; the frozen-bath tolerance and the fixed neighbor reservoir need
  convergence checks for each scientific use.
- Collision neighbor searches use exact Cartesian distances within the search cap `H_SEARCH`; only
  the boundary correction of the KNN measure is locally planar.
- The fluid is pressureless and monodisperse and cannot represent multistream velocity
  distributions after trajectory crossing; its diffusive momentum closure is a documented modeling
  approximation.
- Runs use one GPU; multi-GPU domain decomposition is not implemented.
- Checkpoints and RNG-state files are not portable between CUDA and ROCm, and the backend and GPU
  target can change rounding and long-time trajectories.
- Long-time, imported-gas, extreme-vacuum, and large-production collision regimes are less
  thoroughly exercised than the analytical core.

The representation-specific limitations are indexed in [swarm
limitations](doc/guide_swarm.md#113-known-limitations), [collision
limitations](doc/guide_swarm.md#113-known-limitations), and [fluid
limitations](doc/guide_fluid.md#92-known-limitations).

## Repository layout

```text
.
├── Makefile           # model, representation, and GPU-backend build rules
├── CHANGELOG.md       # contents of each release
├── requirements.txt   # Python packages for the validation and analysis scripts
├── inc/
│   ├── gpu.cuh        # CUDA/HIP API mappings and GPU error checks
│   ├── swarm/         # swarm headers, including the collision chain and KNN searches
│   └── fluid/         # fluid headers
├── src/
│   ├── swarm/         # swarm kernels and runtime, one kernel per file
│   └── fluid/         # fluid kernels and runtime, one kernel per file
├── mod/               # production model setups
├── val/
│   ├── swarm/, fluid/ # validation models (mod/), shared drivers and validators (src/),
│   │                  #   generated results (out/) and objects (obj/)
│   ├── *.py           # campaign, archive-check, and comparison utilities
│   └── paper/         # scientific campaigns with their own READMEs
├── doc/               # shared disk model, numerical guides, and validation guide
├── obj/               # generated objects (not tracked)
└── out/               # generated production output (not tracked)
```

- `inc/swarm/` with `src/swarm/` and `inc/fluid/` with `src/fluid/` own the two representations.
  Kernels launched by the runtimes mostly have one source file each, named after the kernel.
- `inc/gpu.cuh` maps the CUDA and HIP runtime, random-number, and Thrust APIs and defines the
  `GPU_CHECK` and `GPU_KERNEL_CHECK` error checks, so both backends compile the same numerical
  sources.
- A build selects one GPU backend and, where applicable, one collision search or one fluid sweep.
  Validation outputs keep backend-separated paths so that CUDA and ROCm results can coexist for
  comparison; checkpoints are not portable between backends.

The ignored `lab/` directory may hold experimental work; it is not part of the production or
validation interface.

## Documentation

The guides under `doc/` state the shared disk model, the equations and algorithms of each
representation, and the validation evidence behind this user guide. [The documentation
map](doc/README.md#documentation-map) lists every guide, says which one owns each fact, and holds
the [glossary](doc/README.md#glossary) and the [authority order](doc/README.md#authority-order) for
resolving disagreements between prose and code.

## Contributing

When changing a numerical method or physical prescription:

1. keep the swarm and fluid implementations owned by their own branches
2. update the guide that owns the equation, assumptions, and reference (see
   [Maintaining the documentation](doc/README.md#maintaining-the-documentation))
3. add or update an analytical, statistical, boundary, or regression test under `val/`
4. archive enough metrics and environment information to support the new claim
5. update the relevant test in `doc/guide_tests.md`, following the same maintenance rules
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
