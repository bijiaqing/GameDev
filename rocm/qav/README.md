# ROCm verification suites

The tracked ROCm backend carries local copies of the complete fluid and swarm QA suites so it can be validated without invoking or modifying the CUDA project

The analytical cases use the same mathematical references and pass criteria as the established CUDA tests, while compiling and executing only the `.hip` sources in this tree. The KNN group additionally compares the HIP KD-tree and Morton implementations against brute force, adversarial boundary cases, and one another

On an MI300A node, run a short bring-up first:

```bash
cd rocm
python3 tools/static_check.py
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 --quick --target gfx942
```

After the smoke tests pass, run the full matrices:

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
python3 qav/fluid/test_common/run_suite.py --group sweep --target gfx942
```

Fluid results are written below `qav/fluid/out/`; swarm and KNN results are written below `qav/swarm/out/`. Each suite records the HIP compiler, ROCm configuration, AMD device information, target architecture, numerical metrics, and an aggregate manifest

The ROCm-native checkpoint regression can be run independently with:

```bash
python3 qav/swarm/test_common/run_suite.py --group restart --res 32 --target gfx942
```

After matching CUDA and ROCm archives are available, compare their common scientific metrics with:

```bash
python3 qav/compare_backends.py --component all
```

When `rocm/` remains inside the complete repository, the comparator automatically finds the sibling
CUDA `qav/` tree. If the ROCm directory is deployed as a standalone cluster project, supply the CUDA
archive explicitly:

```bash
python3 qav/compare_backends.py \
    --component all \
    --cuda-qav /path/to/cuda-project/qav \
    --rocm-qav "$PWD/qav"
```

Deterministic metrics use configurable numerical tolerances. Stochastic diffusion records instead
require each vendor backend to pass its own analytical confidence criteria because CUDA and hipRAND
streams are not expected to be identical.
