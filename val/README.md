# Numerical validation

This is a fresh source-only validation tree. No result, executable, object file,
or qualification record has been carried over from `val_stale/`. That directory
is an archive and is not a dependency of these tests.

## Layout

- `fluid/mod/`, `swarm/mod/`: numerical setups and model-specific analytical references.
- `fluid/src/`, `swarm/src/`: test drivers, validators, and suite runners.
- `val_config.py`: case lists, output paths, and source fingerprints.
- `run_all.py`: combined fluid + swarm campaign for one backend and fluid sweep.
- `check_archive.py`: require complete, passing native numerical results.
- `compare_backends.py`: compare CUDA/ROCm archives; stochastic diffusion is judged by each native analytical test.

Runs create `fluid/out/<model>/<backend>/<sweep>/` and
`swarm/out/<model>/<backend>/`. Builds create `fluid/obj/` and `swarm/obj/`.
Fluid CUDA sweep directories are `thread_precise` and `block_precise`;
ROCm uses `thread` and `block`. Collision-chain records use `groups/chain/`.
`paper/` and previous output archives are intentionally not restored.

## What is under test

ROCm collision-chain cases inherit the root parallel rate/chain reductions and
128-thread collision launches, including when test constants specify a different
CUDA width. Saved `COL_BATH_TPB` is the actual launch width. Their N_K=200 exercises
non-power-of-two reductions. KNN tests inherit the root ROCm private KD-tree heap
and its 64-query launch width; Morton retains compacted queries in the production
runtime. No test-local copies of these optimizations are needed. Transport-only
and fluid tests do not use these collision changes. CUDA-only Smoluchowski
campaigns retain their existing CUDA path and tolerance sweeps.

The retained matrix has 19 fluid models and 19 particle models (including KNN
and four collision-chain models). Model constants and initial conditions are
necessarily test-specific. The operators being validated come from current
`inc/` and `src/`, with these explicit specializations:

- Fluid transport/diffusion drivers prescribe fields and operator ordering and call production kernels.
  Constant-D diffusion uses production `CONST_NU` and the tracer limit `St=0`;
  the coupled ring retains finite-Stokes diffusivity. These are density-diffusion
  tests, not complete coverage of concentration diffusion or imported gas.
- Fluid drag prescribes gas velocity and force endpoints. It calls the production
  `_get_drag_weights` helper; Python integrates the forced drag equation with
  60-digit decimal arithmetic. This isolates drag quadrature, not the whole
  disk-dependent source kernel. Coupled ring tests also call that source kernel.
- Particle orbit and constant-drag tests call production `_ssa_advance` stages.
  Orbit tests select the zero-drag coefficient limit with production gravity;
  the drag-path test prescribes stopping time, radial gas velocity and force.
  They no longer carry separate copies of the kick/drift algorithm. Physical
  gas interpolation and size-to-Stokes conversion are outside these specializations.
- Particle diffusion calls production `diffusion_pos`; its references include
  finite-Stokes suppression and diffusivity-gradient drift. Initialization uses
  the current production host initializer.
- Physical collision references use query-local differential radial/azimuthal
  drift, capped settling, turbulence and Brownian motion. Periodic image
  identity affects the neighbor search; it no longer changes the prescribed
  pair speed at a fixed query environment.
- Collision-chain models use the current production runtime and local controller,
  with `COL_DIAGNOSTICS` enabled for their short runs. Their tests cover represented
  mass, positivity, active collision/fragmentation, continuation behavior and
  repeatability. They do not independently establish population-distribution accuracy.
- KNN drivers use current KD-tree/Morton implementations with independent exact
  neighbor references. The optional topology target compares the Morton hierarchy
  with a CPU oracle. These drivers are retained as accuracy tests, not performance records.

Thresholds for retained numerical checks have not been loosened. The removed
swarm runtime/host clones and fluid viscosity wrapper are not needed. No local
changes have been made to production collision physics or diffusion equations.
The production SSA stages and fluid drag weights were extracted into existing
headers so the tests and simulation can share those implementations.

## Four independent jobs

Use the same source checkout on both clusters. Requirements: Python 3.10+ with
NumPy, Make, a C++17 CUDA/ROCm toolchain, and a GPU allocation. The prior targets
are A100 `sm_80` and AMD `gfx942`; select the target matching the allocated GPU.
Each fluid job runs thread then block; each swarm job covers both searches.
The runners clean each tested build configuration before compiling. No global
`cleanall` is needed or safe while another job is building.

CUDA fluid:

```bash
set -e
export GPU_BACKEND=cuda PYTHONDONTWRITEBYTECODE=1
for sweep in thread block; do
    FLUID_SWEEP="$sweep" python3 -B val/fluid/src/run_suite.py --group all --target sm_80 --res 32 64 128 256
    python3 -B val/check_archive.py --backend cuda --component fluid --fluid-sweep "${sweep}_precise" --output "check_archive_cuda_${sweep}.json"
done
```

CUDA swarm:

```bash
set -e
export GPU_BACKEND=cuda PYTHONDONTWRITEBYTECODE=1
python3 -B val/swarm/src/run_suite.py --group all --target sm_80 --res 32 64 128 256
python3 -B val/swarm/src/run_suite.py --group chain --target sm_80
python3 -B val/check_archive.py --backend cuda --component swarm --output check_archive_cuda_swarm.json
```

ROCm fluid:

```bash
set -e
export GPU_BACKEND=rocm PYTHONDONTWRITEBYTECODE=1
for sweep in thread block; do
    FLUID_SWEEP="$sweep" python3 -B val/fluid/src/run_suite.py --group all --target gfx942 --res 32 64 128 256
    python3 -B val/check_archive.py --backend rocm --component fluid --fluid-sweep "$sweep" --output "check_archive_rocm_${sweep}.json"
done
```

ROCm swarm:

```bash
set -e
export GPU_BACKEND=rocm PYTHONDONTWRITEBYTECODE=1
python3 -B val/swarm/src/run_suite.py --group all --target gfx942 --res 32 64 128 256
python3 -B val/swarm/src/run_suite.py --group chain --target gfx942
python3 -B val/check_archive.py --backend rocm --component swarm --output check_archive_rocm_swarm.json
```

Run these from the repository root. Keep the source checkout alongside results;
standalone suite runs do not create `run_all` source fingerprints. Download
`val/fluid/out/`, `val/swarm/out/`, and `val/check_archive_*.json`, excluding `obj/`.
A rerun replaces that model/backend/sweep's results. All validation Python entry
points disable bytecode generation; no `__pycache__` is needed.

## Qualification status

No fresh native CUDA or ROCm test has run in this reconstructed tree. Local path,
import, and host arithmetic checks are not GPU compilation or numerical qualification.
Do not reuse old PASS records as evidence for this version.
