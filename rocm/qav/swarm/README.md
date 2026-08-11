# Swarm tests

The swarm verification suite exercises production HIP/ROCm kernels and device helpers through test-local runtime drivers.  It
does not replace or modify files under `inc/swarm/` or `src/swarm/`.

Run the quick matrix from the repository root with:

```bash
python3 qav/swarm/test_common/run_suite.py --group all --quick --target gfx942
```

Run only the new radial prescription and its radial-line KNN checks with

```bash
python3 qav/swarm/test_common/run_suite.py --group radial --res 32 64 128 256 --target gfx942
```

For this group, `test_knn` enters a radial-only path: it links only the physical
1D collision model with both search backends, runs the radial-line and inactive-particle
edge checks, and skips the generic 2D/3D ordinary, periodic, and wedge matrices. Its
records are written under `qav/swarm/out/test_knn/radial/`, separately from the
general KNN matrix.

Run the full default matrix with:

```bash
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256 --target gfx942
```

Available groups are `radial`, `grid`, `transport`, `diffusion`, `initialization`, `restart`, `radiation`, `boundary`, `collision`, and `knn`. Use `--build-only` to check all
HIP/ROCm configurations without executing them. All analytical and KNN outputs are written under
`qav/swarm/out/`, with KNN records collected in `qav/swarm/out/test_knn/`.

All active runners write compiler and GPU metadata as `environment.json`. Analytical HIP/ROCm drivers
also write `meta_N*.json`, raw binary `*.dat` fields, and validator-generated
`metrics_N*.json`; KNN runners store configuration and metrics together in result JSON files and
index each matrix with `manifest.json` and the complete KNN group with `suite_manifest.json`.
The dispatcher writes `qav/swarm/out/manifest.json` after every model and prints an explicit final
suite result.

If component directories are copied from a cluster without the final aggregate record, rebuild it
locally without rerunning HIP executables:

```bash
python3 qav/swarm/test_common/run_suite.py --group all --res 32 64 128 256 --rebuild-manifest
```

The current analytical matrix performs 52 builds: four resolutions for the nine grid, orbit, diffusion, and initialization refinement cases,
plus one build for each of the sixteen resolution-independent algebra, boundary, and restart cases. The KNN group adds four standalone HIP/ROCm drivers,
links production collision sources in 1D, 2D, and 3D with both backends, and runs the
$10^5$-particle ordinary, wedge, and narrow-wedge matrices.
`--quick` reduces the analytical part to 34 builds; `--knn-full` adds the million- and ten-million-particle KNN matrices.

See `doc/testset_swarm.md` for the complete setups, analytical solutions, pass criteria, and
remaining coverage; `test_common/TEST_CASES.md` is the compact implementation index.
