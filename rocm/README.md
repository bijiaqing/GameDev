# GameDev ROCm backend

This tracked top-level directory is a self-contained HIP/ROCm reproduction of the GameDev fluid and swarm models. It intentionally does not include, modify, or compile any CUDA source file, and it does not yet attempt to select CUDA and ROCm from one shared build system

It may be copied by itself to a compute cluster and used as that deployment's project root. In that
layout, run commands directly from the copied directory rather than from a nested `rocm/` path. The
structural checker treats the root MIT license and CUDA-tree parity checks as development-repository
metadata when the canonical parent tree is present; the bundled KD-tree Apache license remains part
of the standalone copy

The initial hardware target is AMD Instinct MI300A, whose HIP offload target is `gfx942`. Other AMDGPU targets can be selected with `AMDGPU_TARGET` or the QA runners' `--target` option

## Current implementation

- all production fluid and swarm translation units are present as `.hip` files
- CUDA Runtime calls are translated to HIP Runtime calls
- cuRAND device states and samples are translated to hipRAND
- CUB operations are translated to hipCUB and Thrust operations are compiled through rocThrust
- both the KD-tree and Morton/ghost KNN backends are available through `COLLISION_SEARCH=kdtree|morton`
- fluid thread-line and block-line sweeps remain selectable through `FLUID_SWEEP=thread|block`
- the block diffusion path checks its requested dynamic LDS against the device-reported limit;
  exact static-plus-dynamic near-limit qualification remains pending
- the complete analytical fluid, analytical swarm, boundary, collision, and KNN QA trees are reproduced under `qav/`
- precise floating-point semantics are retained during correctness bring-up so NaN and Inf guards remain active

The physical equations and discretizations are unchanged. Differences in floating-point reduction order, wavefront scheduling, random-number streams, and library algorithms mean that ROCm results are expected to agree in norms, invariants, convergence order, and distributions rather than byte for byte with CUDA output

`TPB=32` is deliberately retained from the validated numerical baseline. It does not rely on warp-synchronous behavior, but it occupies only half of a 64-lane AMD wavefront; testing 64, 128, and 256 threads per block is a pending performance step

## Requirements

The cluster environment must provide:

- a ROCm installation with `hipcc`, HIP Runtime, hipRAND, rocThrust, and hipCUB headers
- an AMD GPU visible to HIP; `gfx942` is the default target for MI300A
- GNU Make
- Python 3.9 or newer with NumPy for the validators
- `amd-smi` or `rocm-smi` for optional environment recording

GPU diagnostic commands are optional; a missing diagnostic utility is recorded as unavailable and does not fail a numerical test

## Build and run production models

From this directory:

```bash
make MODEL=fluid_fiducial AMDGPU_TARGET=gfx942
mod/fluid_fiducial/gamedev

make MODEL=swarm_fiducial AMDGPU_TARGET=gfx942
mod/swarm_fiducial/gamedev
```

Add `RESOURCE_REPORT=1` to a build command to print AMD kernel register, scratch, and LDS usage during compilation

Collision models can select either search backend:

```bash
make MODEL=test_collision_3d RES=32 COLLISION_SEARCH=morton AMDGPU_TARGET=gfx942
make MODEL=test_collision_3d RES=32 COLLISION_SEARCH=kdtree AMDGPU_TARGET=gfx942
```

The default fluid sweep in this ROCm tree is the block-line implementation. A model may override it in `flags.mk`, or it can be selected on the command line:

```bash
make MODEL=fluid_fiducial FLUID_SWEEP=block AMDGPU_TARGET=gfx942
make MODEL=fluid_fiducial FLUID_SWEEP=thread AMDGPU_TARGET=gfx942
```

## Verification

First run the host-side structural check, which does not require ROCm:

```bash
python3 tools/static_check.py
```

Then follow [qav/README.md](qav/README.md) on an AMD node. A recommended first native sequence is:

```bash
python3 qav/fluid/test_common/run_suite.py --group source --res 8 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group grid --res 32 --quick --target gfx942
python3 qav/swarm/test_common/run_suite.py --group knn --res 32 --target gfx942
```

Only after those compile-and-run checks pass should the full fluid and swarm matrices be launched

## Native-validation boundary

The static checker passes from the tracked `rocm/` location. Native ROCm 7.2.4 campaigns on a
`gfx942` MI300A have completed the full analytical fluid and swarm matrices, the native restart
regression, ordinary and adversarial KD-tree/Morton suites, periodic and wedge searches, and all six
production collision-backend links. The direct CUDA-versus-ROCm fluid comparison also passes its
conserved-field and density-weighted velocity criteria; all shared swarm records and the KNN
component pass their corresponding cross-backend gates.

These results establish the implemented numerical portability baseline, not final production
performance. Deliberate NaN/Inf failure-path tests, greater-than-48-KiB LDS launch qualification,
AMD profiler evidence, and repeated production-scale timings remain pending. The detailed evidence,
tolerances, commands, and remaining procedures are maintained in
[`../doc/future_rocm_support.md`](../doc/future_rocm_support.md).

Raw RNG-state checkpoint files are backend-specific and must not be exchanged between CUDA and ROCm runs. GameDev does not support cross-backend restart, including continuation from ordinary physical field files

## Source synchronization

`tools/sync_from_cuda.py` documents and automates the mechanical source-copy stage from the parent
CUDA tree. Running it with its required `--force-overwrite` acknowledgement replaces the generated
`inc/`, `src/`, `mod/`, and `qav/` trees, then restores every hand-maintained path registered in
`tools/rocm_preserve.txt`. Newly introduced ROCm adaptations must be added to that manifest before
the next refresh. The script never writes to the CUDA project

The repository [README](../README.md#citation-and-license) records KD-tree and bitonic-sort
attribution, and [inc/swarm/kdtree/Apache-2.0.txt](inc/swarm/kdtree/Apache-2.0.txt) retains their
Apache License 2.0 text

## License

GameDev's original code is distributed under the repository [MIT License](../LICENSE). The modified
cudaKDTree and cudaBitonic portions retain the Apache License 2.0 identified above

## ROCm references

- [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP compiler targets, including `gfx942`](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html)
- [hipRAND documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)
- [rocThrust HIP execution policies](https://rocm.docs.amd.com/projects/rocThrust/en/latest/reference/rocThrust-hip-policies.html)
- [hipCUB documentation](https://rocm.docs.amd.com/projects/hipCUB/en/latest/)
