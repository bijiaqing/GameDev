# Numerical validation

The numerical suites use current production operators. Any historical results retained separately as
`val_stale/` are not dependencies or qualification records for this tree; that archive is not
required in the checkout. Native runs generate results and disposable build products in the paths
below.

## Layout

- `fluid/mod/`, `swarm/mod/`: numerical setups and model-specific analytical references.
- `fluid/src/`, `swarm/src/`: test drivers, validators, and suite runners.
- `val_config.py`: case lists, output paths, and source fingerprints.
- `run_all.py`: complete campaign for one backend: the fluid matrix for both sweeps, the swarm
  matrices once, and one campaign record per sweep.
- `check_archive.py`: require complete, passing native numerical results.
- `compare_backends.py`: compare CUDA/ROCm archives; stochastic diffusion is judged by each native
  analytical test.

Runs create `fluid/out/<model>/<backend>/<sweep>/` and
`swarm/out/<model>/<backend>/`. Builds create `fluid/obj/` and `swarm/obj/`.
Fluid CUDA sweep directories are `thread_precise` and `block_precise`;
ROCm uses `thread` and `block`. Collision-chain records use `groups/chain/`.
`paper/` contains separate scientific campaigns: eccentric orbits, diffusion, photospheric
transport, Smoluchowski kernels, and disk coagulation. Their local READMEs define commands
and evidence boundaries; they are not stages of `run_all.py`.

## What is under test

ROCm collision-chain cases inherit the root parallel rate/chain reductions.
The root default is 128 threads on ROCm and 64 on CUDA; an explicit model
`COL_BATH_TPB` takes precedence on either backend. These chain tests retain their
explicit 256-thread width. Saved `COL_BATH_TPB` is the actual launch width.
Their N_K=200 exercises
non-power-of-two reductions. KNN tests inherit the root ROCm private KD-tree heap
and its 64-query launch width; Morton retains compacted queries in the production
runtime. No test-local copies of these optimizations are needed. Transport-only
and fluid tests do not use these collision changes. CUDA-only Smoluchowski
campaigns retain their existing CUDA path and tolerance sweeps.

The retained matrix has 19 fluid models and 19 particle models (including KNN
and four collision-chain models). Model constants and initial conditions are
necessarily test-specific. The operators being validated come from current
`inc/` and `src/`, with these explicit specializations:

- Fluid transport/diffusion drivers prescribe fields and operator ordering and call production
  kernels. Constant-D diffusion uses production `CONST_NU` and the tracer limit `St=0`; the coupled
  ring retains finite-Stokes diffusivity. These are density-diffusion tests, not complete coverage
  of concentration diffusion or imported gas.
- Fluid drag prescribes gas velocity and force endpoints. It calls the production
  `_get_drag_weights` helper; Python integrates the forced drag equation with
  60-digit decimal arithmetic. This isolates drag quadrature, not the whole
  disk-dependent source kernel. Coupled ring tests also call that source kernel.
- Particle orbit and constant-drag tests call production `_ssa_advance` stages.
  Orbit tests select the zero-drag coefficient limit with production gravity;
  the drag-path test prescribes stopping time, radial gas velocity and force.
  They carry no separate copy of the kick/drift algorithm. Physical
  gas interpolation and size-to-Stokes conversion are outside these specializations.
- Particle diffusion calls production `diffusion_pos`; its references include
  finite-Stokes suppression and diffusivity-gradient drift. Initialization uses
  the current production host initializer.
- Physical collision references use query-local differential radial/azimuthal
  drift, capped settling, turbulence and Brownian motion. Periodic image
  identity affects the neighbor search but not the prescribed pair speed at a
  fixed query environment.
- Collision-chain models use the current production runtime and local controller,
  with `COL_DIAGNOSTICS` enabled for their short runs. Their tests cover represented
  mass, positivity, active collision/fragmentation, continuation behavior and
  repeatability. They do not independently establish population-distribution accuracy.
- KNN drivers use current KD-tree/Morton implementations with independent exact
  neighbor references. The optional topology target compares the Morton hierarchy
  with a CPU oracle. These drivers are retained as accuracy tests, not performance records.

Test-specific constants and drivers never modify production collision physics
or diffusion equations. The SSA stages (`_ssa_advance` in `inc/swarm/_transport.cuh`)
and fluid drag weights (`_get_drag_weights` in `inc/fluid/fluid_kern.cuh`) are
shared by the simulation and the tests rather than copied.

## Complete campaign

Run the native campaigns on both clusters at the same time, from the same source commit:

```bash
python3 -B val/run_all.py --backend cuda
```

```bash
python3 -B val/run_all.py --backend rocm
```

Each run validates the fluid matrix with the thread and then the block sweep, runs the swarm
matrix and collision-chain group once, checks the archive for each sweep, and writes
`val/run_all_<backend>_thread.json` and `val/run_all_<backend>_block.json`; the swarm stages appear
in both records. `--fluid-sweep thread` or `--fluid-sweep block` restricts the fluid matrix to one
sweep. Afterwards, copy one backend's `val/fluid/out/`, `val/swarm/out/`, and `val/*.json` into a
separate directory on the other system, excluding `obj/`, and compare each sweep:

```bash
python3 -B val/compare_backends.py --cuda-root CUDA_VAL_COPY --cuda-sweep thread_precise --rocm-sweep thread
```

```bash
python3 -B val/compare_backends.py --cuda-root CUDA_VAL_COPY --cuda-sweep block_precise --rocm-sweep block
```

`run_all.py --compare` instead reruns the native campaign and compares it with a counterpart copied
into this checkout's own `val/` directory.

## Four independent jobs

Use the same source checkout on both clusters. Requirements: Python 3.10+ with
NumPy, Make, a C++17 CUDA/ROCm toolchain, and a GPU allocation. The default targets
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

Current numerical-suite qualification requires fresh native CUDA and ROCm archives tied
to this source. Local path, import, and host arithmetic checks are not GPU compilation or
numerical qualification. A successful individual paper-model run does not qualify these suites.
Do not reuse old PASS records as evidence for this version.

No automatic static-check stage is part of the campaign. Keep block-source provenance and
separate native production-build qualification pending until their evidence is available.
