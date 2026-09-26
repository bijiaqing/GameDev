# Numerical validation

The validation suites compile the current production kernels with test-specific constants and
drivers and judge them against analytical, statistical, and invariant references. This file
explains how to run, archive, and compare them. What each case establishes, and its acceptance
criteria, are in [`doc/fluid_testset.md`](../doc/fluid_testset.md) and
[`doc/swarm_testset.md`](../doc/swarm_testset.md).

## Requirements

- the GPU toolchain and GPU of the backend under test (see the top-level
  [`README.md`](../README.md#requirements)); the default targets are `sm_80` and `gfx942`, so pass
  `--target` for other GPUs
- Python 3.10 or later with NumPy
- one GPU allocation per backend; a complete campaign takes several hours

Run every command from the repository root. The validation scripts disable Python bytecode
generation, so no `__pycache__` directories are written.

## Layout

| Path | Contents |
|---|---|
| `fluid/mod/`, `swarm/mod/` | validation models: `flags.mk`, constants, drivers, and per-model runners and validators |
| `fluid/src/`, `swarm/src/` | shared drivers, suite runners, and validators |
| `val_config.py` | case lists, groups, output paths, and the source fingerprint |
| `run_all.py` | complete native campaign for one backend |
| `check_archive.py` | require a complete, passing native archive |
| `compare_backends.py` | compare a CUDA archive with a ROCm archive |
| `paper/` | scientific campaigns with their own READMEs; not part of `run_all.py` |

Results go to `fluid/out/<model>/<backend>/<sweep>/` and `swarm/out/<model>/<backend>/`, and builds
to `fluid/obj/` and `swarm/obj/`. The fluid sweep directories are `thread_precise` and
`block_precise` on CUDA and `thread` and `block` on ROCm. Suite manifests are under
`fluid/out/_suite/` and `swarm/out/_suite/`, collision-chain records under `groups/chain/`. A rerun
replaces the results of that model, backend, and sweep.

## Running a complete campaign

On each system, inside a GPU job:

```bash
python3 -B val/run_all.py --backend cuda
```

```bash
python3 -B val/run_all.py --backend rocm
```

A campaign runs, in order, the fluid matrix with the thread and then the block sweep, the swarm
matrix, the swarm collision-chain group, and an archive check for each sweep. It writes one
campaign record per sweep, `val/run_all_<backend>_thread.json` and
`val/run_all_<backend>_block.json`, each holding the source fingerprint, every stage's command and
output, and the overall result; the shared swarm stages appear in both. The run prints
`COMPLETE VALIDATION: PASS` and exits with status zero only when every stage passes.

| Option | Effect |
|---|---|
| `--fluid-sweep thread` or `block` | validate only one fluid sweep (default: both) |
| `--target TARGET` | GPU architecture for other GPUs (default `sm_80` or `gfx942`) |
| `--res N ...` | resolution ladder (default `32 64 128 256`) |
| `--quick` | only the two coarsest resolutions; a workflow check, not qualification |
| `--compare` | also require and compare a counterpart archive copied into this `val/` directory |

The runners clean each build configuration before compiling it, so campaigns for different
backends can share one checkout. Do not run `make clean` while a campaign is building.

## Comparing CUDA and ROCm

Run the two native campaigns concurrently from the same commit. Afterwards, copy one system's
`val/fluid/out/`, `val/swarm/out/`, and `val/*.json`, excluding `obj/` directories, into a separate
directory on the other system and compare each sweep, for example on the ROCm system:

```bash
python3 -B val/compare_backends.py --cuda-root CUDA_VAL_COPY --cuda-sweep thread_precise --rocm-sweep thread
```

```bash
python3 -B val/compare_backends.py --cuda-root CUDA_VAL_COPY --cuda-sweep block_precise --rocm-sweep block
```

`CUDA_VAL_COPY` is the directory that received the copy. The comparison requires both campaign
records to carry the same source fingerprint, compares deterministic metrics within numerical
tolerances (except diagnostics that carry single-precision rounding, such as the collision-physics
KNN-measure errors, which each backend bounds natively), and judges stochastic cases by each
backend's own analytical test rather than by matching random streams. Alternatively, copy the
counterpart into this checkout's own `val/` directory and rerun the native campaign with
`run_all.py --compare`.

## Running part of the suites

The suite runners run one group; set the backend in the environment:

```bash
GPU_BACKEND=rocm FLUID_SWEEP=block python3 -B val/fluid/src/run_suite.py --group diffusion --target gfx942
```

```bash
GPU_BACKEND=cuda python3 -B val/swarm/src/run_suite.py --group chain --target sm_80
```

| Runner | Groups |
|---|---|
| `val/fluid/src/run_suite.py` | `all`, `equilibrium`, `transport`, `diffusion`, `source`, `radiation`, `coupled` |
| `val/swarm/src/run_suite.py` | `all`, `transport`, `diffusion`, `initialization`, `collision`, `knn`, `chain` |

The swarm `all` group does not include `chain`; `run_all.py` runs both. A single model runs through
its own runner, for example `GPU_BACKEND=cuda python3 -B val/fluid/mod/test_x_diffusion_2d/run.py`.
`val/check_archive.py --backend <backend> --component fluid|swarm|all --fluid-sweep thread|block`
checks an existing archive (the sweep maps to the backend's archive directory, `thread_precise` or
`block_precise` on CUDA), and `val/swarm/src/run_suite.py --rebuild-manifest` rebuilds the swarm
suite manifest from downloaded component manifests without rerunning.

Standalone suite runs do not write the `run_all` campaign records, so a cross-backend comparison
of their results needs `--ignore-source-fingerprint` and does not certify a common source.

## What qualifies a result

A validation result qualifies only the source it was produced from. The source fingerprint hashes
the `Makefile` and every C, C++, CUDA, HIP, make-fragment, and Python source under `inc/`, `src/`,
and `val/`, except `val/tools/` and the generated `out/`, `obj/`, `logs/`, and `temp/` directories,
so any other file of these kinds present in the checkout, tracked or not, changes it. Qualification
requires fresh native CUDA and ROCm campaigns with matching source fingerprints and a passing
comparison. Local builds, import checks, and host arithmetic are not GPU validation, a single model
run or a scientific campaign under `paper/` does not qualify the suites, and results from earlier
sources are not evidence for the current one.
