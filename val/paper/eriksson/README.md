# Eriksson disk coagulation campaign

This campaign asks how monomer dust grows and fragments in a turbulent disk when the complete
production swarm physics acts together: transport, Stokes-dependent concentration diffusion, and the
physical collision kernel with sticking, erosion, and fragmentation. It follows the physical
ingredients of the coagulation comparison of Eriksson et al. at two turbulence levels, and runs each
with both [neighbor searches](../../../doc/guide_swarm.md#9-nearest-neighbor-search) on both GPU
backends. The routine swarm suite checks the ingredients one at a time, namely the [collision
rates](../../../doc/guide_tests.md#6-swarm-collision-rates), the [neighbor
search](../../../doc/guide_tests.md#7-swarm-neighbor-search), and the [collision
chain](../../../doc/guide_tests.md#8-swarm-collision-chain); the
[Smoluchowski campaigns](../../../doc/guide_tests.md#9-swarm-coagulation-campaigns) test the chain
against synthetic kernels. This campaign is the physical-kernel application that neither covers.
Project terms are defined in the [glossary](../../../doc/README.md#glossary).

## Contents

- [At a glance](#at-a-glance)
- [Models](#models)
- [Setup](#setup)
- [Production code and overrides](#production-code-and-overrides)
- [Build and run](#build-and-run)
- [Outputs](#outputs)
- [Analysis](#analysis)
- [Limits](#limits)

## At a glance

| Item | Value |
|---|---|
| Representation | swarm (`DUST_REPR := swarm`), multisize, physical CGS units |
| Geometry | axisymmetric [radial–polar](../../../doc/guide_basis.md#24-supported-geometries), mesh `(N_X, N_Y, N_Z) = (1, 128, 64)` |
| Backends and searches | CUDA and ROCm, each with `COLLISION_SEARCH=kdtree` and `morton` |
| Models | 2: `alpha_1e-3` and `alpha_1e-4` |
| Planned runs | 8: 2 models $\times$ 2 searches $\times$ 2 backends |
| Flags | `TRANSPORT`, `DIFFUSION`, `DIFFUSE_CONCENTRATION`, `COLLISION`, `MULTISIZE`, `SAVE_DENS` |
| Representatives | `N_P = 1048576` |
| Analysis | none included |
| Output | `val/paper/eriksson/out/<model>/<backend>/<search>/` |

## Models

The two models differ only in the turbulence strength $\alpha$ and the number of outputs.

| Model | $\alpha$ | Outputs (`SAVE_MAX`) | Final time |
|---|---|---:|---|
| `alpha_1e-3` | $10^{-3}$ | 125 | 12,500 yr |
| `alpha_1e-4` | $10^{-4}$ | 250 | 25,000 yr |

Each `mod/<model>/const_defs.cuh` sets `COAG_ALPHA` (which becomes `ALPHA`) and `COAG_OUTPUTS`
(which becomes `SAVE_MAX`) and includes the shared `src/const_defs.cuh`. Both `flags.mk` files are
identical: they select `DUST_REPR := swarm`, `MODEL_PARENT := ../src`, and the flags listed in [At
a glance](#at-a-glance). `CODE_UNIT` and `CONST_ST` are not enabled, so collisions use the molecular
gas constants and the Stokes number varies with the current grain size and local gas conditions.

## Setup

### Disk and grains

All dimensional values are in CGS units, and `CODE_UNIT` must stay disabled because the Brownian
relative velocity needs the molecular gas constants.

| Parameter | Value |
|---|---|
| Units | CGS; one year is 31,557,600 s, `R_0` is 1 au |
| Star | 1 solar mass |
| Domain | spherical $5\le r\le 50$ au, $\theta=\pi/2\pm0.2$, full azimuth $-\pi\le\phi\le\pi$ in one cell |
| Mesh | `(N_X, N_Y, N_Z) = (1, 128, 64)` |
| Gas surface density | `SIGMA_0 = 1410.4014065096128` g cm<sup>-2</sup> at 1 au, $\propto R^{-1}$ (`IDX_P = -1`) |
| Gas temperature | 209.7926358245702 K at 1 au, $\propto R^{-1/2}$ (`IDX_Q = -0.5`); `ASPR_0 = 0.0288821994330985` |
| Gas molecules | mean mass 2.34 proton masses, cross section `X_SEC = 2e-15` cm<sup>2</sup> |
| Initial grains | monomers of diameter `S_0 = INIT_SMIN = INIT_SMAX = 1e-4` cm (0.5 micron radius), material density `RHO_0 = 1` g cm<sup>-3</sup> |
| Reference Stokes number | `STOKES_0` $=\pi\rho_0S_0/(4\Sigma_0)$, the Epstein midplane value of a monomer at 1 au |
| Initial dust-to-gas ratio | 0.01 (`METAL_Z`), tracing the hydrostatic gas within the finite domain |
| Turbulence | `ALPHA` from the model; `SCHMIDT_X = SCHMIDT_R = SCHMIDT_Z = 1` |
| Diffusion | concentration diffusion with $D=\nu/[\mathrm{Sc}(1+\mathrm{St}^2)]$, $\mathrm{Sc}=1$ |

The Stokes number scales from `STOKES_0` with the grain diameter, the midplane gas surface density,
and the exact spherical stratification ([Stokes
number](../../../doc/guide_basis.md#4-stopping-time-and-stokes-number)). With
`DIFFUSE_CONCENTRATION` the dust diffuses toward a uniform dust-to-gas ratio ([concentration
diffusion](../../../doc/guide_swarm.md#73-concentration-diffusion)).

### Collisions and neighbor search

The physical kernel (`COAG_KERNEL = 3`) evaluates pair rates from the neighbors of each owner and
applies sticking below `V_FRAG` and erosion or fragmentation above it ([collision
kernels](../../../doc/guide_swarm.md#841-collision-kernels), [outcome
channels](../../../doc/guide_swarm.md#851-outcome-channels)). Fragments never become smaller than
`INIT_SMIN`, the monomer diameter.

| Parameter | Value |
|---|---|
| Fragmentation threshold | `V_FRAG = 100` cm s<sup>-1</sup> |
| Neighbors | `N_K = 256`, including the owner; search cap `H_SEARCH = 1` gas scale height |
| Controller groups | 16 radial $\times$ 8 polar spatial groups (`COL_BIN_Y`, `COL_BIN_Z`; `COL_BIN_X = 1`), 32 adaptive size bins (`COL_BIN_S`) |
| Controller settings | `COL_BATH_EPS = 0.08`, `COL_BATH_ALPHA = 1e-3`, `COL_BIN_MIN = 64`, `COL_EVENT_CAP = 32` |
| Chain width | `COL_BATH_TPB`: 64 on CUDA, 128 on ROCm (the root defaults) |
| Maximum bath duration | `COL_BATH_MAX` = 1 yr |
| Morton parameters | `MORTON_TPB = 64`, `MORTON_LEAF_TARGET = 128`, `MORTON_MAX_LEVEL = 20` |

`src/const_defs.cuh` asserts `N_K == 256` and `COL_BATH_EPS == 0.08`, so all variants share these
benchmark settings. The root default is `COL_BATH_EPS = 0.02`. The controller groups, size bins, and
bath tolerance are explained in
[bath controller](../../../doc/guide_swarm.md#87-bath-controller).

### Initialization

Dust starts well mixed with the gas: positions are sampled by rejection in proportion to the
production hydrostatic gas density over the finite spherical domain, bounded by the volume-weighted
density at the inner radius and polar edge, where it is largest. All representatives are monomers.
The dust mass is `METAL_Z` times the gas mass in the domain, computed by a Simpson integral
($512\times512$ intervals) over both hemispheres. This replaces the root initializer, which samples
a size-dependent settled distribution from the
[edge-tapered](../../../doc/guide_basis.md#62-edge-taper) profile
([spatial sampling](../../../doc/guide_swarm.md#35-spatial-sampling)); the campaign applies no
edge taper.

The seeds are fixed by design: the host position generator uses seed 0 and the per-particle
random streams use device seed 1
([random streams](../../../doc/guide_swarm.md#129-random-streams)). Every run therefore starts
from the same particle state and random streams, so the eight runs form a controlled comparison in
which only the collision search and the GPU backend change.

### Timestep and output

| Parameter | Value |
|---|---|
| Timestep limits | `DT_MAX` = 1 yr, `CFL_DYN = 0.45` |
| Output cadence | every 100 yr (`DT_OUT`); particles at every frame (`LIN_BASE = 1`) |

## Production code and overrides

All runtime and device sources come directly from the root: dynamics, Stokes-dependent
concentration diffusion, query-local collision velocities, cached rates, sticking and erosion
grouping, adaptive size bins with the production moving-bin bounds, local refreshes, and the KD-tree
and Morton searches. No artificial global partner reshuffling is added.

The campaign `src/` directory contains only:

- `const_defs.cuh`: the constants above.
- `swarm_host.cuh`: includes the root host header and replaces only `initmass_calc` and
  `rand_disk_poly` with the hydrostatic sampling and Simpson mass integral of
  [Initialization](#initialization). Root monodisperse size sampling gives the exact monomer
  diameter; all other host and output helpers are reused.

On ROCm the root collision code specializes some kernels for this shape ([collision and search
kernels](../../../doc/guide_swarm.md#123-collision-and-search-kernels)). When compiled for
`gfx942`, the rate and chain kernels (`col_bath_rate`, `col_chain_run`) use wavefront reductions
for `N_K = 256` with `COL_BATH_TPB = 128`. On every ROCm target, the KD-tree neighbor-cache query
keeps a private per-thread heap with 64 queries per block. The Morton search uses the same query
kernel on both GPU backends.

## Build and run

Run every command from the repository root. Only this directory and the root `Makefile`, `inc/`,
and `src/` are required. The campaign `Makefile` defaults to `GPU_BACKEND=cuda`, to
`GPU_TARGET=sm_80` for CUDA or `gfx942` for ROCm, and to `COLLISION_SEARCH=kdtree` on both backends
(the root default on ROCm is `morton`), so name the search explicitly.

One run, for example the Morton search on CUDA, is built with

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton
```

and started with

```bash
val/paper/eriksson/obj/alpha_1e-3/cuda/morton/gamedev
```

The four runs of one backend run sequentially with the loop below; set `backend=rocm` and
`target=gfx942` for ROCm:

```bash
(
  set -e
  backend=cuda target=sm_80
  for model in alpha_1e-3 alpha_1e-4; do
    for search in kdtree morton; do
      make -C val/paper/eriksson -j8 MODEL="$model" GPU_BACKEND="$backend" \
        GPU_TARGET="$target" COLLISION_SEARCH="$search"
      "val/paper/eriksson/obj/$model/$backend/$search/gamedev"
    done
  done
)
```

Each executable run without arguments is one fresh run; rerunning into an existing output path
replaces its results. The paths are:

| Item | Path under `val/paper/eriksson/` |
|---|---|
| Executable | `obj/<model>/<backend>/<search>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<search>/<target>/` |
| Output | `out/<model>/<backend>/<search>/` |

The executable and output paths do not contain the GPU target, so a build for another target
replaces the executable of the same model, backend, and search, and its run writes into the same
output directory.

## Outputs

Every frame, including the initial frame 0, writes three files; the fresh start also writes
`variables.txt` with the parameters:

| File | Contents | Size |
|---|---|---|
| `dustdens_<frame>.dat` | one double per mesh cell | 65.5 kB |
| `particle_<frame>.dat` | eight doubles (64 bytes) per representative: position, physical velocity, grain diameter `par_size`, represented grain count `par_numr` | about 67 MB |
| `rngstate_<frame>.dat` | one backend RNG state per representative | about 50 MB on CUDA (48-byte `curandState`) |

A CUDA run of `alpha_1e-3` (126 frames) therefore writes about 15 GB and one of `alpha_1e-4` (251
frames) about 30 GB. `COL_DIAGNOSTICS` is not enabled, so no collision JSON or JSONL diagnostics are
written. The file formats are in [Output files](../../../README.md#output-files).

## Analysis

No analysis script is included in this directory. Compare the evolving mass-weighted size
distributions and dust-density fields between the two searches and the two GPU backends, and with
the Eriksson comparison at matching times.

## Limits

The campaign establishes that its eight builds use only the constants and host overrides above
with the root numerical sources, and that they have distinct executable and output paths for each
model, backend, and search. By itself it supports no scientific claim: that requires native CUDA
and ROCm builds and fresh runs from the current source.

- **Search and backend effects, not seed scatter.** Because the seeds are fixed, differences
  between the eight runs come only from the search (which stores the same neighbor set in a
  different order, see [search contract](../../../doc/guide_swarm.md#92-search-contract)) and
  from backend arithmetic. The campaign does not measure the spread between independent
  realizations.
- **No bath-tolerance convergence.** `COL_BATH_EPS` controls refreshes and is not an 8% accuracy
  guarantee. The campaign fixes it at 0.08 and uses one seed, so it does not provide the refinement
  of the bath tolerance and the comparison of independent seeds that a size-distribution result
  needs ([finite-bath convergence](../../../doc/guide_swarm.md#112-finite-bath-convergence)).
- **Not the MCDUST equations.** The setup follows the physical ingredients of the Eriksson
  comparison, not identical MCDUST equations. It keeps the production hydrostatic gas
  stratification, fixed spherical boundaries, relative-velocity prescriptions, and Gaussian
  diffusion noise, and it imports no MCDUST gas-density floor or extra noise normalization.

The campaign sources enter the validation source fingerprint, but a campaign run does not qualify
the validation suites; the rules are in [What qualifies a
result](../../README.md#what-qualifies-a-result).
