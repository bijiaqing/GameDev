# Eriksson disk coagulation campaign

Two models, eight independent fresh runs: alpha_1e-3 / alpha_1e-4 × CUDA / ROCm × KD-tree / Morton.

## Layout

- `mod/<model>/`: alpha, final output index and flags.
- `src/`: shared constants and one host initialization override.
- `obj/<model>/<backend>/<search>/gamedev`: executable; root build rules place
  intermediate objects in backend/search/target-specific subdirectories under obj.
- `out/<model>/<backend>/<search>/`: simulation outputs, isolated for all eight runs.

## Physical setup and timing

| Parameter | Value |
|---|---|
| alpha_1e-3 | alpha=1e-3; 125 intervals; final time 12,500 yr |
| alpha_1e-4 | alpha=1e-4; 250 intervals; final time 25,000 yr |
| Output cadence | Initial frame 0, then every 100 yr; particle and dust-density outputs |
| Representatives / neighbors | 1,048,576 / N_K=256, including owner |
| Refresh tolerance | COL_BATH_EPS=0.08 |
| Initial grain diameter / density | 1e-4 cm (0.5 micron radius) / 1 g cm^-3 |
| Initial dust/gas mass ratio | 0.01, tracing gas within the finite domain |
| Star | 1 solar mass |
| Domain | Spherical r=5--50 au, theta=pi/2 +/- 0.2 |
| Mesh | (1,128,64), axisymmetric |
| Gas surface density | 1410.4014065096128 g cm^-2 at 1 au, proportional to R^-1 |
| Gas temperature | 209.7926358245702 K at 1 au, proportional to R^-1/2 |
| Fragmentation threshold | 100 cm/s |
| Diffusion | Dust/gas concentration; nu/[Sc*(1+St^2)], Sc=1 |

All dimensional simulation quantities are CGS; one year is 31557600 seconds.
St varies with current grain size and local gas conditions. CODE_UNIT and CONST_ST
are not enabled. There is no artificial global partner reshuffling.

## Root reuse and essential overrides

All runtime and device sources come directly from the root, including dynamics,
St-dependent concentration diffusion, query-local collision velocities, cached
rates, sticking/erosion grouping, adaptive size bins, local refreshes, and optimized
KD-tree/Morton implementations. No old runtime, collision header, or backend tree
is copied. No collision JSONL diagnostics are enabled.

`src/swarm_host.cuh` includes the root host header and replaces only initmass_calc
and rand_disk_poly. It samples dust proportional to the production hydrostatic gas
density, rather than the root's settled distribution, and normalizes mass using a
Simpson integral over the same finite spherical domain. The initial-profile helper
is merged into this header. Root monodisperse size sampling gives the exact monomer
diameter. All other host and output helpers are reused.

The previous campaign's physical constants, 16x8 spatial controller groups,
32 adaptive size bins, one-year maximum collision interval, one-year maximum
dynamics timestep, and 100-year output strategy are retained. COL_BATH_EPS changes
from 0.02 to 0.08; fixed size-bin limits are removed in favor of current production
moving bins. The tolerance controls refreshes and is not an 8% accuracy guarantee.

These setups preserve our previous comparison's physical ingredients, not identical
MCDUST equations. In particular, production hydrostatic gas stratification, fixed
spherical boundaries, relative-velocity prescriptions and Gaussian diffusion noise
are retained. No MCDUST gas-density floor or extra noise normalization is imported.

## Fresh runs

From the repository root. Each command pair is one independent GPU run. No restart
argument is supplied. Upload this directory and the current root Makefile/inc/src;
no val_stale or lab files are required. Avoid rerunning into an existing output
path unless intentionally replacing its results.

### CUDA

**alpha_1e-3, kdtree**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/paper/eriksson/obj/alpha_1e-3/cuda/kdtree/gamedev
```

**alpha_1e-3, morton**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/paper/eriksson/obj/alpha_1e-3/cuda/morton/gamedev
```

**alpha_1e-4, kdtree**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-4 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/paper/eriksson/obj/alpha_1e-4/cuda/kdtree/gamedev
```

**alpha_1e-4, morton**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-4 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/paper/eriksson/obj/alpha_1e-4/cuda/morton/gamedev
```

### ROCM

**alpha_1e-3, kdtree**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=kdtree &&
val/paper/eriksson/obj/alpha_1e-3/rocm/kdtree/gamedev
```

**alpha_1e-3, morton**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=morton &&
val/paper/eriksson/obj/alpha_1e-3/rocm/morton/gamedev
```

**alpha_1e-4, kdtree**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-4 GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=kdtree &&
val/paper/eriksson/obj/alpha_1e-4/rocm/kdtree/gamedev
```

**alpha_1e-4, morton**

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-4 GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=morton &&
val/paper/eriksson/obj/alpha_1e-4/rocm/morton/gamedev
```

## Verification status

All eight make dry runs select only the constants/host overrides and root numerical
sources, with distinct executable/output paths. Host checks verified domain support,
the sampled distribution against the hydrostatic density, radial sample moments,
vertical symmetry, mass-integral convergence, and the requested parameters.
Native CUDA/ROCm compilation and fresh scientific results are still required.
No stale output, old plots, archived reports or validation scaffolding is copied.
