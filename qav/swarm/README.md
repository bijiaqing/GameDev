# Swarm tests

The swarm verification suite exercises production CUDA kernels and device helpers through test-local runtime drivers.  It
does not replace or modify files under `inc/swarm/` or `src/swarm/`.

Run the quick matrix from the repository root with:

```bash
python3 qav/swarm/test_common/run_suite.py --group all --quick
```

Run the full default matrix with:

```bash
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256
```

Available groups are `grid`, `transport`, `diffusion`, `radiation`, `collision`, and `knn`. Use `--build-only` to check all
CUDA configurations without executing them. Analytical outputs are written under `qav/swarm/out/`; KNN benchmark records
are written under `qav/swarm/test_knn/out/`.

The analytical matrix performs 25 builds: four resolutions for the five grid, orbit, and diffusion convergence cases,
plus one build for each of the five resolution-independent algebra cases. The KNN group adds four CUDA drivers, links the
production collision sources once per backend, and runs the $10^5$-particle ordinary, wedge, and narrow-wedge matrices.
`--quick` reduces the analytical part to 15 builds; `--knn-full` adds the million-particle KNN matrix.

See `doc/testset_swarm.md` for the complete setups, analytical solutions, pass criteria, and
remaining coverage; `test_common/TEST_CASES.md` is the compact implementation index.
