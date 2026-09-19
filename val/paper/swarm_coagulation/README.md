# Optimized radial–vertical coagulation campaign

Eight independent fresh runs: weak/strong turbulence × CUDA/ROCm × KD-tree/Morton.
The two model directories select alpha and duration; every configuration uses
`common/`. There is no lab dependency at build/run time. Root `inc/` and `src/`
are unchanged by this campaign update.

## Physical setup

| Setting | Value |
|---|---|
| Weak / strong alpha | 1e-4 / 1e-3 |
| Weak / strong outputs | 250 / 125 intervals after initial output 0 |
| Output spacing | 100 years |
| Weak / strong end time | 25000 / 12500 years |
| Representative particles | 1048576 |
| Collision neighbors | N_K=256, including the owner |
| Refresh parameter | COL_BATH_EPS=0.02 |
| Initial grains | Fixed radius 0.5 micron; diameter 1e-4 cm |
| Material density | 1 g/cm^3 |
| Initial dust/gas ratio | 0.01, spatially well mixed in the finite domain |
| Star / domain | 1 solar mass; spherical r=5–50 au; theta=pi/2 ± 0.2 |
| Mesh | (1,128,64), axisymmetric |
| Gas surface density | 1410.4014065096128 g/cm² at 1 au; exponent -1 |
| Temperature | 209.7926358245702 K at 1 au; exponent -1/2 |
| Fragmentation threshold | 100 cm/s |

All dimensional code/output quantities are CGS. The code stores grain diameter,
not radius. `INIT_SMIN=INIT_SMAX=1e-4 cm`; sizes are initialized exactly with
`std::fill`. The existing normalization assigns represented grain numbers so
the total dust mass equals the integral over the simulated domain.

Monomer provenance: the retained MCDUST comparison
`lab/collision_fix/results/mcdust_comparison/README.md` records that its output 0
contains only 0.5-micron-radius grains. The supplied MCDUST directory was no
longer available locally during this update; its setup was not freshly reread.

### Dust-to-gas diffusion

The diffusive mass flux is

    J_d = -rho_g D grad(rho_d/rho_g).

All eight configurations enable `DIFFUSE_CONCENTRATION` and use the root diffusion
implementation (no model-local diffusion source overrides).

The cylindrical stochastic update uses D_R=nu/[SCHMIDT_R*(1+St²)] and
D_Z=nu/[SCHMIDT_Z*(1+St²)], with nu=alpha*c_s*H and unit Schmidt numbers.
St is evaluated at the particle location using its current diameter. The radial
drift is dD_R/dR + D_R/R + D_R*d(ln rho_g)/dR; the vertical drift is
dD_Z/dZ + D_Z*d(ln rho_g)/dZ. Diffusivity derivatives include the local Stokes
number gradients at fixed grain size. Noise variance remains 2D dt, with no
additional MCDUST normalization factor. The gas-density derivatives are analytic derivatives of
this model's spherical hydrostatic gas profile. The dynamics timestep controller
includes the same additional drift. Cartesian velocities are preserved during
diffusion; existing boundary handling is retained.

This aligns the diffused quantity and initial grain size with the MCDUST
comparison. It does not make all gas, velocity, boundary, or collision
prescriptions identical to MCDUST.

## Optimized implementation and consolidation

- Generalized compacted Morton search: sorted-prefix merging, lower-half
  selection, skip checks, and candidate compaction; 64 threads per query.
  Buffer sizes derive from N_K. Duplicate-image queries retain the full-sort
  branch when the retained set exceeds half the available workspace.
- KD-tree shared heaps: 16 threads at N_K=256; fewer columns for larger heaps
  keep heap storage within 32 KiB per block. Both KD-tree launch sites agree
  with the heap layout.
- Parallel collision-rate and event-chain reductions, query-environment/rate
  caches, local change-based refreshes, narrow sticking/erosion grouping, and
  the existing fragmentation/erosion physics are retained from the tested lab
  workflow. The per-continuation event cap remains 32.
- `collision_physics.cuh` combines the five small query-environment, refresh,
  diagnostic-event, sticking and erosion helper headers. `_col_chain.cuh`
  contains the shared GPU collision kernels.
- `swarm_runtime.cu` owns the scheduler/workspace and the former contents of
  `local_evolve.inc`; the separate `.inc`, `local_gpu.cuh` and
  `local_schedule.hpp` files are removed. `swarm_runtime.hip` includes the same
  runtime body with HIP API mappings.
- Morton headers are shared across CUDA/ROCm using conditional backend sections.
  Diffusion likewise has one implementation and a thin HIP entry file.
- Collision diagnostics are retained; this change does not alter their cadence.

## Fresh run commands

Upload the entire updated suite, including `common/morton/` and `common/kdtree/`,
plus the current root Makefile, `inc/`, and `src/`. Replace the server's old
`common/` directory so deleted helper files are not retained.

Before starting this campaign, move any existing output directory aside **once**
on the server. Do not perform this while a suite run is active:

```sh
mv val/paper/swarm_coagulation/outputs val/paper/swarm_coagulation/outputs_previous_campaign
```

Run that only when `outputs` exists and the destination is unused. These are
fresh starts: no checkpoint argument. Old output-20 restart benchmarks are not
the initial conditions for this campaign.

### CUDA

Each block is one independent GPU run.

**weak_turbulence, kdtree**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/paper/swarm_coagulation/obj/bin/weak_turbulence/cuda/kdtree/gamedev
```

**weak_turbulence, morton**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/paper/swarm_coagulation/obj/bin/weak_turbulence/cuda/morton/gamedev
```

**strong_turbulence, kdtree**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/paper/swarm_coagulation/obj/bin/strong_turbulence/cuda/kdtree/gamedev
```

**strong_turbulence, morton**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/paper/swarm_coagulation/obj/bin/strong_turbulence/cuda/morton/gamedev
```

### ROCM

Each block is one independent GPU run.

**weak_turbulence, kdtree**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=kdtree &&
val/paper/swarm_coagulation/obj/bin/weak_turbulence/rocm/kdtree/gamedev
```

**weak_turbulence, morton**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=morton &&
val/paper/swarm_coagulation/obj/bin/weak_turbulence/rocm/morton/gamedev
```

**strong_turbulence, kdtree**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=kdtree &&
val/paper/swarm_coagulation/obj/bin/strong_turbulence/rocm/kdtree/gamedev
```

**strong_turbulence, morton**

```sh
make -C val/paper/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=rocm GPU_TARGET=gfx942 COLLISION_SEARCH=morton &&
val/paper/swarm_coagulation/obj/bin/strong_turbulence/rocm/morton/gamedev
```

## Paths

- Outputs: `val/paper/swarm_coagulation/outputs/<model>/<backend>/<search>/`.
- Executables: `val/paper/swarm_coagulation/obj/bin/<model>/<backend>/<search>/gamedev`.
- Build products: `val/paper/swarm_coagulation/obj/`.

## Root search promotion

KD-tree shared heaps and generalized Morton merge/compaction now come from
`inc/swarm/{kdtree,morton}/`; the suite's duplicate search headers were removed.
Both backends use the same source. All cached and uncached KD-tree query launches
use the heap's compile-time column count. Root retains its configured neighbor
count (200 by default); this suite retains 256. Root Morton defaults to 64 threads.
