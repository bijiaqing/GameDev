# Swarm tests

The swarm verification suite exercises production kernels and device helpers through backend-local
runtime drivers. Shared test definitions live here; CUDA and ROCm drivers live under
`qav/cuda/swarm/` and `qav/rocm/swarm/`.

Run the quick matrix from the repository root with:

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group all --quick
```

Run only the new radial prescription and its radial-line KNN checks with

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group radial --res 32 64 128 256
```

For this group, `test_knn` enters a radial-only path: it links only the physical
1D collision model with both search backends, runs the radial-line and inactive-particle
edge checks, and skips the generic 2D/3D ordinary, periodic, and wedge matrices. Its
records are written under `qav/logs/swarm/BACKEND/groups/radial/test_knn/`, separately from the
general KNN matrix.

Run the full default matrix with:

```bash
python3 qav/cuda/swarm/test_common/run_suite.py --group all --res 32 64 128 256
```

Available groups are `radial`, `grid`, `transport`, `diffusion`, `initialization`, `radiation`, `boundary`, `collision`, and `knn`. Use `--build-only` to check all
selected-backend configurations without executing them. All analytical and KNN outputs are written
under `qav/logs/swarm/BACKEND/`, with KNN records collected in
`qav/logs/swarm/BACKEND/test_knn/`.

All active runners write compiler and GPU metadata as `environment.json`. Analytical GPU drivers
also write `meta_N*.json`, raw binary `*.dat` fields, and validator-generated
`metrics_N*.json`; KNN runners store configuration and metrics together in result JSON files and
index each matrix with `manifest.json` and the complete KNN group with `suite_manifest.json`.
The complete dispatcher updates `qav/logs/swarm/BACKEND/manifest_all.json` after every model.
Focused groups write all their records below `qav/logs/swarm/BACKEND/groups/GROUP/`, and direct
wrappers use `groups/manual/`, so neither can overwrite any part of the complete archive.

The current analytical matrix performs 51 builds: four resolutions for the nine grid, orbit, diffusion, and initialization refinement cases,
plus one build for each of the fifteen resolution-independent algebra and boundary cases. The KNN group adds four standalone GPU drivers,
links production collision sources in 1D, 2D, and 3D with both backends, and runs the
$10^5$-particle ordinary, wedge, and narrow-wedge matrices.
`--quick` reduces the analytical part to 33 builds; `--knn-full` adds the million- and ten-million-particle KNN matrices.

See `doc/swarm_testset.md` for the complete setups, analytical solutions, pass criteria, and
remaining coverage; `test_common/TEST_CASES.md` is the compact implementation index.
