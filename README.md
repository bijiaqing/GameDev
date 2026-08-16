# GameDev

**GameDev: GPU-Accelerated ModEl for Dust EVolution**

GameDev is a GPU research code for dust evolution in protoplanetary disks. One build system selects
either the NVIDIA CUDA or AMD HIP/ROCm backend. Both backends provide two independent numerical
representations:

- an Eulerian, pressureless dust-fluid solver for conservative continuum evolution
- a Lagrangian dust-swarm solver for particle trajectories, stochastic diffusion, and optional
  representative-particle collisions

Both branches use prescribed gas disks and one-way gas-to-dust coupling. They share coordinate and
physical conventions, but intentionally own separate headers, kernels, runtimes, and optical-depth
pipelines. A build selects exactly one representation through the model's `flags.mk` file.

GameDev is an actively developed scientific code rather than a packaged application. Simulation
setups, units, resolutions, physical switches, and output cadence are compile-time model choices.
Read the relevant numerical and verification guides before using results for scientific analysis.

## Contents

- [Highlights](#highlights)
- [Dust representations](#dust-representations)
- [Coordinates and supported geometries](#coordinates-and-supported-geometries)
- [Numerical methods and accuracy](#numerical-methods-and-accuracy)
- [Requirements](#requirements)
- [Building and running](#building-and-running)
- [Configuring a model](#configuring-a-model)
- [Restarting a simulation](#restarting-a-simulation)
- [Output files](#output-files)
- [Reading output with Python](#reading-output-with-python)
- [Verification](#verification)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Current limitations](#current-limitations)
- [Contributing](#contributing)
- [Citation and license](#citation-and-license)

## Highlights

- CUDA and HIP/ROCm implementations selected from one repository and one Makefile
- logarithmic spherical-radial grids with radial-only, disk-plane, meridional, and full-3D support
  where permitted by the selected representation
- gas drag, stellar gravity, disk geometry, radiation pressure, and optional viscous gas flow
- density diffusion in the fluid branch and stochastic position diffusion in the swarm branch
- conservative finite-volume PPM/HLL/FARGO transport for the fluid branch
- semi-analytic staggered particle dynamics for the swarm branch
- optional Poynting-Robertson drag, multisize dust, coagulation, and fragmentation in the swarm
  branch
- exact top-$K$ collision-neighborhood searches through either a bundled KD tree or an adaptive
  Morton index with periodic ghost records
- model-local constants and source overrides without modifying production implementation files
- analytical, statistical, boundary, backend, and regression suites under `qav/`
- restart files that store physical linear velocities in both representations

## Dust representations

| Capability | Eulerian fluid | Lagrangian swarm |
|---|---|---|
| State | cell-averaged density and momentum | weighted computational representatives |
| Grain sizes | one fixed species | monodisperse or multisize |
| Velocity closure | one velocity per cell | multiple velocities can coexist locally |
| Transport | conservative finite-volume sweeps | semi-analytic particle trajectories |
| Diffusion | spherical finite-volume density diffusion | cylindrical stochastic displacement |
| Radiation pressure | optional | optional |
| Poynting-Robertson drag | not implemented | optional |
| Collisions | not implemented | optional coagulation and fragmentation |
| Imported gas fields | not implemented | optional |
| Gas backreaction | not implemented | not implemented |
| Dust self-gravity | not implemented | not implemented |

The fluid branch is most natural while the dust velocity remains approximately single-valued. The
swarm branch retains trajectory crossing and a size distribution, at the cost of sampling noise and
more expensive neighborhood-based collision calculations. The two models are complementary rather
than interchangeable discretizations of every physical closure.

## Coordinates and supported geometries

The computational coordinates are spherical:

$$
(x,y,z)=(\phi,r,\theta),
\qquad
R=y\sin z,
\qquad
Z=y\cos z,
$$

where $R$ and $Z$ are cylindrical radius and height. The mesh is uniform in $x$ and $z$ and
logarithmic in $y$. Array storage is $x$-first:

$$
\mathrm{index}=i_x+i_yN_X+i_zN_XN_Y.
$$

When `N_Z == 1`, the evolved density is a vertically integrated surface density. When `N_Z > 1`,
it is a volume density on the resolved spherical grid.

| Geometry | Grid condition | Fluid | Swarm | Interpretation |
|---|---|---:|---:|---|
| radial-only | `N_X == 1`, `N_Z == 1` | no | yes | axisymmetric, vertically integrated, midplane dynamics |
| radial-azimuthal | `N_X > 1`, `N_Z == 1` | yes | yes | vertically integrated disk plane |
| radial-polar | `N_X == 1`, `N_Z > 1` | no | yes | axisymmetric resolved vertical structure |
| full 3D | `N_X > 1`, `N_Z > 1` | yes | yes | resolved spherical dynamics |

The fluid branch always requires an active azimuthal dimension. Any three-dimensional fluid or
swarm model currently requires diffusion to support the vertically resolved dust layer.

## Numerical methods and accuracy

### Fluid branch

The fluid solver evolves pressureless dust mass and momentum with:

- finite-volume Piecewise Parabolic Method reconstruction
- HLL fluxes for the pressureless Riemann problem
- integer FARGO orbital shifts plus residual azimuthal transport
- invariant-domain flux limiting for density and momentum consistency
- SSPRK(3,3) integration within directional transport sweeps
- exponentially weighted source integration for drag, gravity, geometry, and radiation
- Crank-Nicolson diffusion with Thomas or Sherman-Morrison line solves, positivity subcycling, and
  conservative donor-momentum fluxes
- a palindromic Strang-style composition of diffusion, transport, and source operators

PPM is nominally third-order in smooth one-dimensional reconstruction, but boundaries, limiting,
directional splitting, and operator composition reduce the appropriate global smooth-solution claim
to second order. Shocks, contacts, vacuum interfaces, and extrema are deliberately limited and need
not retain that order.

Two CUDA sweep implementations are available:

- `FLUID_SWEEP := thread` uses one thread per directional line and line-sized local work arrays
- `FLUID_SWEEP := block` uses one CUDA block per line with explicit workspace and cooperative
  reconstruction

They implement the same numerical method. Their relative performance depends on resolution and GPU
architecture, so equivalence and timing should be checked on the intended machine.

### Swarm branch

The swarm solver uses:

- staggered semi-analytic drag, gravity, and geometric trajectory updates
- midpoint radiation pressure and optional Poynting-Robertson drag
- cylindrical Ito Euler-Maruyama diffusion followed by coordinate reprojection
- frozen-snapshot Bernoulli collision batches for representative particles
- exact top-$K$ nearest-neighbor candidates from either a KD tree or an adaptive Morton hierarchy
- Strang composition of the enabled transport, diffusion, and collision operators

Smooth deterministic trajectory evolution is second-order in time. Stochastic diffusion and
representative-particle collisions require weak/statistical convergence tests rather than a
deterministic pointwise order claim.

Full equations, finite-volume measures, initial mass normalization, boundary policies, source
quadrature, collision estimators, and literature references are documented in
[`doc/fluid_numeric.md`](doc/fluid_numeric.md) and
[`doc/swarm_numeric.md`](doc/swarm_numeric.md).

## Requirements

The CUDA backend requires:

- a CUDA-capable NVIDIA GPU
- the CUDA toolkit, including `nvcc`, Thrust, and cuRAND
- GNU Make
- a C++17-compatible host compiler supported by the installed CUDA toolkit
- Python 3 and NumPy for the verification runners and validators

The CUDA backend uses `GPU_TARGET=sm_80`, `-O2`, `--use_fast_math`, and `-std=c++17` by default.
Set `CUDA_MATH=precise` to omit `--use_fast_math` for matched-arithmetic verification. `sm_80`
targets NVIDIA Ampere GPUs such as the A100.

The ROCm backend requires `hipcc`, HIP Runtime, hipRAND, rocThrust, hipCUB, and an AMD GPU supported
by the selected `GPU_TARGET`; `gfx942` is the current MI300A target. ROCm correctness builds omit
blanket `-ffast-math` because finite-only assumptions can invalidate the production NaN/Inf guards.
The numerical guides document backend-dependent arithmetic and execution details, the test guides
record native validation, and [`qav/rocm/bench/README.md`](qav/rocm/bench/README.md) defines the
MI300A profiling and performance workflow.

There is no installation step. Executables are written below `bin/MODEL/GPU_BACKEND/`.

## Building and running

All commands below are run from the repository root.

### Fluid example

Build the fiducial fluid model:

```bash
make MODEL=fluid_fiducial GPU_BACKEND=cuda GPU_TARGET=sm_80
```

Run it:

```bash
bin/fluid_fiducial/cuda/gamedev
```

### Swarm example

Build the fiducial swarm model:

```bash
make MODEL=swarm_fiducial GPU_BACKEND=cuda GPU_TARGET=sm_80
```

Run it:

```bash
bin/swarm_fiducial/cuda/gamedev
```

### ROCm examples

Select the AMD backend from the same repository root:

```bash
make MODEL=fluid_fiducial GPU_BACKEND=rocm GPU_TARGET=gfx942
bin/fluid_fiducial/rocm/gamedev

make MODEL=swarm_fiducial GPU_BACKEND=rocm GPU_TARGET=gfx942
bin/swarm_fiducial/rocm/gamedev
```

Object files, executables, production output, and QA output contain the backend name, so switching
backends cannot reuse incompatible artifacts.

The supplied fiducial models are production-scale examples, not lightweight demonstrations. In
particular, the default fluid grid is large and the default swarm model contains many
representatives. Use the verification quick runs for a short installation check.

### Cleaning a build

Remove one model executable and its active object directory:

```bash
make MODEL=fluid_fiducial clean
```

Remove all generated object files:

```bash
make clean
```

Build messages report the selected model, representation, constant header, overridden headers,
fluid sweep, or collision-search backend as applicable. Each compiled line also reports which
source file won the model-over-production search order.

## Configuring a model

Production models live under `mod/`. A minimal model directory contains a `flags.mk` file:

```text
mod/
└── my_model/
    ├── flags.mk
    └── const_defs.cuh    # optional
```

The build searches `mod/`, `qav/comm/fluid/`, and `qav/comm/swarm/` for the requested `MODEL`.
The name must resolve to exactly one directory.

### Fluid model flags

A minimal fluid configuration is:

```make
DUST_REPR := fluid
FLUID_SWEEP := thread
```

Optional compile-time switches are added through `NVCC`:

```make
NVCC += -DDIFFUSION
NVCC += -DRADIATION
```

| Setting | Meaning |
|---|---|
| `FLUID_SWEEP := thread` | line-per-thread transport and diffusion kernels |
| `FLUID_SWEEP := block` | cooperative line-per-block kernels with explicit workspace |
| `DIFFUSION` | enable spherical dust-density diffusion |
| `RADIATION` | enable attenuated radiation pressure |
| `VISC_FLOW` | prescribe viscous gas radial flow; requires `DIFFUSION` |
| `CONST_NU` | use constant kinematic viscosity instead of constant alpha |
| `HALF_DISK` | use the supported half-disk polar configuration |

Transport and local source evolution are intrinsic to the fluid solver and do not have independent
feature flags.

### Swarm model flags

A basic transported swarm configuration is:

```make
DUST_REPR := swarm

NVCC += -DTRANSPORT
NVCC += -DRADIATION
NVCC += -DSAVE_DENS
NVCC += -DCODE_UNIT
```

| Flag or setting | Meaning |
|---|---|
| `TRANSPORT` | enable particle dynamics |
| `DIFFUSION` | enable stochastic position diffusion; requires `TRANSPORT` |
| `RADIATION` | enable attenuated radiation pressure; requires `TRANSPORT` |
| `PR_EFFECT` | add Poynting-Robertson drag; requires `RADIATION` |
| `VISC_FLOW` | use viscous gas radial flow; requires `DIFFUSION` and excludes `IMPORTGAS` |
| `COLLISION` | enable representative-particle collisions; requires `MULTISIZE` |
| `MULTISIZE` | store and evolve individual grain sizes and represented grain counts |
| `IMPORTGAS` | read gridded gas density and velocity fields |
| `CONST_ST` | hold the Stokes number fixed; incompatible with `IMPORTGAS` |
| `CONST_NU` | use constant kinematic viscosity instead of constant alpha |
| `SAVE_DENS` | deposit and save the swarm density on the mesh |
| `CODE_UNIT` | use the code-unit collision prescription rather than physical gas microphysics |
| `HALF_DISK` | use the supported half-disk polar configuration |
| `LOGTIMING` | use logarithmic collision-only timing; incompatible with transport and density output |
| `LOGOUTPUT` | use logarithmic particle-output cadence; incompatible with `LOGTIMING` |
| `COLLISION_SEARCH := kdtree` | use the bundled exact KD-tree collision search |
| `COLLISION_SEARCH := morton` | use the adaptive Morton search with periodic ghost records |

Collision-enabled builds must select exactly one search backend. The Makefile translates
`COLLISION_SEARCH` into the corresponding internal backend macro.

### Constants

If a model needs different physical parameters, grid dimensions, particle count, cadence, or CUDA
launch settings, place a complete `const_defs.cuh` in the model directory. The build gives this file
priority over the backend and representation default:

- [`inc/cuda/fluid/const_defs.cuh`](inc/cuda/fluid/const_defs.cuh)
- [`inc/rocm/fluid/const_defs.cuh`](inc/rocm/fluid/const_defs.cuh)
- [`inc/cuda/swarm/const_defs.cuh`](inc/cuda/swarm/const_defs.cuh)
- [`inc/rocm/swarm/const_defs.cuh`](inc/rocm/swarm/const_defs.cuh)

Models that do not need different constants should omit the local header and inherit the selected
representation's defaults. Do not add preprocessor parameters to the production constant headers
solely to configure one model.

### Source and header overrides

A model-local `.cu` file replaces a same-named production translation unit. A model-local header
with the same relative name similarly wins compiler include lookup. This is useful for controlled
experiments and analytical verification cases, but production physics should remain in the branch
source when it is intended for every model.

A model may also set `MODEL_PARENT := parent_name` to inherit model files from another directory
under `mod/`; files in the child model keep priority.

## Restarting a simulation

Pass a saved frame index to the executable:

```bash
bin/fluid_fiducial/cuda/gamedev 10
```

or:

```bash
bin/swarm_fiducial/cuda/gamedev 10
```

The physical restart time is reconstructed from the model's output schedule. The fluid branch loads
density and physical linear velocities, converts velocity files back to its internal angular state,
and rebuilds conserved momentum and optical depth. The swarm branch loads its particle checkpoint
and, when diffusion or collisions use random numbers, the matching CUDA random-state checkpoint.

Restart files are not designed as a cross-version interchange format. In particular, raw swarm RNG
state depends on the CUDA state layout used to produce it.

## Output files

Production output is written to:

```text
out/<MODEL>/
```

Frames use five-digit indices such as `00000`. Unless a model-specific interface states otherwise,
binary arrays contain native `double` values in the writing host's byte order and without an
external container header. The companion `variables.txt` records the model metadata needed to
interpret the output.

### Fluid output

Typical files are:

```text
variables.txt
dustdens_00000.dat
dustvelx_00000.dat
dustvely_00000.dat
dustvelz_00000.dat
optdepth_00000.dat    # with RADIATION
```

Each field contains `N_X*N_Y*N_Z` values in $x$-first order. `dustdens` is surface density for
`N_Z == 1` and volume density for `N_Z > 1`. Velocity files contain physical linear components even
though the GPU evolution uses specific angular momenta in the azimuthal and polar slots.

### Swarm output

Typical files are:

```text
variables.txt
particle_00000.dat
rngstate_00000.dat    # with COLLISION or DIFFUSION
dustdens_00000.dat    # with SAVE_DENS
optdepth_00000.dat    # with RADIATION
```

`particle_*.dat` stores physical positions and linear velocities. In a multisize build it also
stores grain size and represented grain count. The exact structured dtype is written in the
`[SWARM_DTYPE]` section of `variables.txt`; use that description rather than assuming a fixed
record size.

With `IMPORTGAS`, the swarm runtime expects matching input frames named `gasdens_*.dat`,
`gasvelx_*.dat`, `gasvely_*.dat`, `gasvelz_*.dat`, and `epsilon_*.dat` in the configured output
location. See the swarm numerical guide for calibration and interpolation semantics.

## Reading output with Python

A fluid mesh field can be loaded with NumPy and reshaped so that indexing remains `[iz, iy, ix]`:

```python
from pathlib import Path

import numpy as np

nx, ny, nz = 1024, 1024, 1
path = Path("out/fluid_fiducial/cuda/dustdens_00000.dat")

dustdens = np.fromfile(path, dtype=np.float64)
dustdens = dustdens.reshape(nz, ny, nx)
```

Use the dimensions recorded for the actual model rather than copying the example values. If output
was written on a machine with a different byte order, give NumPy an explicitly byte-ordered dtype.

For a swarm checkpoint, reconstruct the structured dtype from `variables.txt`:

```python
from configparser import ConfigParser
from pathlib import Path

import numpy as np

output = Path("out/swarm_fiducial/cuda")
config = ConfigParser()
config.read(output / "variables.txt")

particle_dtype = np.dtype([
    (name, dtype) for name, dtype in config["SWARM_DTYPE"].items()
])
particles = np.fromfile(output / "particle_00000.dat", dtype=particle_dtype)
```

This preserves the monodisperse/multisize record distinction and exposes fields by the names written
by the simulation.

## Verification

Verification models, validators, generated metrics, and standalone numerical checks live under
`qav/`, separate from production models in `mod/`.

### Short installation checks

Run the quick fluid suite:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group all --res 32 64 --quick
```

Run the quick swarm suite:

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group all --res 32 64 --quick
```

These commands still compile multiple model variants; they are regression suites rather than a
single smoke-test executable.

### Full fluid suite

```bash
python3 qav/cuda/fluid/test_common/run_suite.py \
    --group all \
    --res 32 64 128 256
```

Available groups are `transport`, `diffusion`, `source`, `radiation`, `ring`, and `sweep`. The
complete analytical matrix contains 85 builds/runs at the four standard resolutions. Thread/block
equivalence is a separate group:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --sweep-dim all
```

The implementation-independent algorithm checks can be run without CUDA:

```bash
python3 qav/comm/fluid/mock/algorithm_checks.py
```

### Full swarm suite

```bash
python3 qav/cuda/swarm/test_common/run_suite.py \
    --group all \
    --res 32 64 128 256
```

Available groups are `radial`, `grid`, `transport`, `diffusion`, `initialization`, `radiation`,
`boundary`, `collision`, and `knn`. The standard analytical/statistical matrix contains 51
resolution builds, followed by the KNN and production-backend checks selected by the runner.

Run only the KNN group:

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group knn --res 32
```

Or run the standalone KNN harness directly:

```bash
python3 qav/cuda/swarm/test_knn/run.py
python3 qav/cuda/swarm/test_knn/run.py --full
```

ROCm uses the corresponding `qav/rocm/fluid/` and `qav/rocm/swarm/` runners and accepts
`--target gfx942`. Generated results are written below `qav/logs/REPRESENTATION/BACKEND/`. They are
ignored by Git and are not part of the source test set. Metadata, metrics, manifests, environment,
build, and run records use JSON; compact numerical field arrays remain binary. Consult the test guides before
interpreting a pass: deterministic norms, convergence orders, conservation tolerances, statistical
tests, and KNN topology criteria are intentionally different.

The current evidence retained in the repository is summarized in
[`doc/README.md`](doc/README.md). Historical tables are not a substitute for rerunning the relevant
suite after numerical, compiler, architecture, or model changes.

## Repository layout

```text
.
├── Makefile                  # model, representation, and GPU-backend build rules
├── README.md                 # project entry point
├── LICENSE                   # MIT license
├── inc/
│   ├── comm/                 # backend-neutral headers by representation
│   ├── cuda/                 # complete CUDA-owned headers by representation
│   └── rocm/                 # complete ROCm-owned headers by representation
├── src/
│   ├── comm/                 # backend-neutral translation units by representation
│   ├── cuda/                 # CUDA-owned translation units
│   └── rocm/                 # ROCm-owned translation units
├── mod/                      # production model configurations
├── qav/
│   ├── comm/                 # backend-neutral test definitions by representation
│   ├── cuda/                 # CUDA test drivers and backend-specific cases
│   ├── rocm/                 # ROCm test drivers, failure cases, and profiling
│   ├── tool/                 # backend-neutral QA and comparison utilities
│   └── logs/                 # generated, backend-separated QA records and fields
├── doc/                      # canonical numerical and development documentation
├── obj/                      # generated model-specific object files
└── out/                      # generated production outputs
```

Experimental work may use the ignored `lab/` directory. It is not part of the production or
verification interface.

## Documentation

| Document | Purpose |
|---|---|
| [`doc/README.md`](doc/README.md) | canonical document map, repository conventions, and evidence status |
| [`doc/fluid_numeric.md`](doc/fluid_numeric.md) | fluid equations, initialization, numerics, state semantics, references, and limitations |
| [`doc/swarm_numeric.md`](doc/swarm_numeric.md) | swarm equations, mass weighting, transport, diffusion, collisions, KNN methods, and limitations |
| [`doc/fluid_testset.md`](doc/fluid_testset.md) | fluid analytical cases, validators, commands, and retained results |
| [`doc/swarm_testset.md`](doc/swarm_testset.md) | swarm analytical/statistical cases, KNN checks, commands, and retained results |
| [`doc/naming.md`](doc/naming.md) | source formatting, naming, coordinates, fields, and branch-ownership conventions |

For numerical behavior, current production source and machine-readable test results take precedence
over prose. The authority order and documentation maintenance policy are stated in
[`doc/README.md`](doc/README.md).

## Reproducibility

For every scientific run, retain at least:

- the repository commit or an exact source archive
- the complete model directory, including `flags.mk`, `const_defs.cuh`, and local overrides
- the selected `FLUID_SWEEP` or `COLLISION_SEARCH` backend
- the exact `nvcc` command options, CUDA toolkit, host compiler, driver, and GPU model
- `variables.txt`, runtime logs, initial frame, restart frame if used, and validation metrics
- random-state checkpoints for stochastic swarm calculations

Byte-for-byte agreement is not generally expected after changing GPU architecture, compiler,
optimization flags, sweep implementation, or KNN topology tie-breaking. Deterministic fields should
first be compared at roundoff-sensitive short times and then by error norms and conserved
quantities. Stochastic or chaotic calculations require ensemble distributions and scientifically
relevant observables.

## Troubleshooting

| Symptom | Check |
|---|---|
| `MODEL is not defined` | run `make MODEL=<directory-name>` from the repository root |
| model not found or ambiguous | ensure the name occurs exactly once under `mod/`, `qav/comm/fluid/`, or `qav/comm/swarm/` |
| missing `flags.mk` | every model requires a local `flags.mk` containing `DUST_REPR` |
| unsupported GPU architecture | replace the Makefile's `-arch=sm_80` with the target CUDA architecture and rebuild |
| feature-dependency compile error | review the fluid or swarm flag constraints in [Configuring a model](#configuring-a-model) |
| unexpected source file is compiled | read the `Compiling ...` line and check model-local files that shadow production names |
| stale model behavior after changing flags | run `make MODEL=<name> clean` before rebuilding |
| no files in the expected output directory | output paths are embedded at compile time; check the model printed during the build and `out/<MODEL>/` |
| swarm restart diverges immediately | confirm the matching `particle_*` and `rngstate_*` files came from the same frame and build |
| validation fails only on a new GPU/toolkit | archive the environment, rerun from a clean build, and distinguish tolerance-level rounding from a failed invariant or convergence criterion |

## Current limitations

- The gas is prescribed and never receives dust drag backreaction
- Neither representation includes dust self-gravity
- The fluid branch is pressureless, monodisperse, and cannot represent multistream velocity
  distributions after trajectory crossing
- Fluid diffusion uses a conservative donor-momentum closure that remains a documented modeling
  approximation
- Swarm diffusion and collisions are stochastic and require particle-number and ensemble
  convergence, not only mesh convergence
- The collision timestep can become globally restrictive in dense or strongly clumped regions
- Collision KNN searches use a local planar metric with a documented search-radius validity limit
- Multi-GPU domain decomposition is not implemented
- CUDA and ROCm are selected from one build tree, but checkpoints and vendor RNG-state files are
  intentionally not portable between them
- `--use_fast_math`, backend choice, and CUDA architecture can change rounding and long-time
  trajectories; reproducibility claims must record the build environment
- Some long-time, imported-gas, extreme-vacuum, and large-production collision regimes remain less
  thoroughly exercised than the analytical core

The detailed validation boundary and representation-specific caveats are maintained in the two
numerical and test-set guides.

## Contributing

When changing a numerical method or physical prescription:

1. keep the fluid and swarm implementations independently owned by their respective branches
2. update the corresponding `doc/numerics_*.md` guide with the equation, assumptions, and reference
3. add or update an analytical, statistical, boundary, or regression test under `qav/`
4. archive sufficient metrics and environment information to support the new claim
5. update the relevant `doc/testset_*.md` evidence summary
6. follow [`doc/naming.md`](doc/naming.md) for names, comments, includes,
   formatting, and coordinate conventions

Avoid recording resolved work as a permanent audit diary. Distill surviving invariants,
limitations, and regression requirements into the canonical guides.

## Citation and license

A formal software citation is not yet provided. Until one is added, cite the repository version or
commit used for a calculation and record the model constants, compile-time flags, CUDA toolkit,
GPU architecture, and verification results relevant to the run.

GameDev's original code is distributed under the [MIT License](LICENSE). Copyright 2026 Jiaqing Bi.

GameDev includes modified portions of
[cudaKDTree](https://github.com/ingowald/cudaKDTree), copyright 2018-2023 Ingo Wald, under the
[Apache License 2.0](inc/cuda/swarm/kdtree/Apache-2.0.txt). The license is retained beside both
backend implementations. Files under the CUDA and ROCm `swarm/kdtree/cubit/` directories derive from
[cudaBitonic](https://github.com/ingowald/cudaBitonic), copyright 2018-2023 Ingo Wald, under the
same license. The original copyright and license notices are retained in the bundled source files.
