# Eriksson disk coagulation campaign

## Purpose

This campaign evolves monomer dust in a turbulent disk with the full production swarm physics:
dynamics, Stokes-dependent concentration diffusion, and the physical collision kernel with sticking,
erosion, and fragmentation. It follows the physical ingredients of the Eriksson coagulation
comparison, at two turbulence levels and with both collision-search backends on both GPU backends,
so that eight independent fresh runs can be compared.

## Models

| Model | $\alpha$ | Outputs (`SAVE_MAX`) | Final time |
|---|---|---:|---|
| `alpha_1e-3` | $10^{-3}$ | 125 | 12,500 yr |
| `alpha_1e-4` | $10^{-4}$ | 250 | 25,000 yr |

Each `mod/<model>/const_defs.cuh` sets `COAG_ALPHA` and `COAG_OUTPUTS` and includes the shared
`src/const_defs.cuh`. Both `flags.mk` files select `DUST_REPR := swarm`, `MODEL_PARENT := ../src`,
`TRANSPORT`, `DIFFUSION`, `DIFFUSE_CONCENTRATION`, `COLLISION`, `MULTISIZE`, and `SAVE_DENS`.
`CODE_UNIT` and `CONST_ST` are not enabled, so the Stokes number varies with the current grain size
and local gas conditions. Each model runs with `COLLISION_SEARCH=kdtree` and `morton` on CUDA and
ROCm.

| Parameter | Value |
|---|---|
| Units | CGS; one year is 31,557,600 s |
| Star | 1 solar mass |
| Domain | spherical $5\le r\le 50$ au, $\theta=\pi/2\pm0.2$, axisymmetric |
| Mesh | `(N_X, N_Y, N_Z) = (1, 128, 64)` |
| Gas surface density | 1410.4014065096128 g cm<sup>-2</sup> at 1 au, $\propto R^{-1}$ |
| Gas temperature | 209.7926358245702 K at 1 au, $\propto R^{-1/2}$; `ASPR_0 = 0.0288821994330985` |
| Gas molecules | mean mass 2.34 proton masses, cross section `X_SEC = 2e-15` cm<sup>2</sup> |
| Initial grains | monomers of diameter `S_0 = 1e-4` cm (0.5 micron radius), density 1 g cm<sup>-3</sup> |
| Initial dust-to-gas ratio | 0.01 (`METAL_Z`), tracing the hydrostatic gas within the finite domain |
| Diffusion | concentration diffusion with $D=\nu/[\mathrm{Sc}(1+\mathrm{St}^2)]$, $\mathrm{Sc}=1$ |
| Fragmentation threshold | `V_FRAG = 100` cm s<sup>-1</sup> |
| Representatives | `N_P = 1048576` |
| Neighbors | `N_K = 256`, including the owner; search radius `H_SEARCH = 1` gas scale height |
| Controller groups | 16 radial $\times$ 8 polar spatial groups, 32 adaptive size bins (`COL_BIN_S`) |
| Controller settings | `COL_BATH_EPS = 0.08`, `COL_BATH_ALPHA = 1e-3`, `COL_BIN_MIN = 64`, `COL_EVENT_CAP = 32` |
| Chain width | `COL_BATH_TPB`: 64 on CUDA, 128 on ROCm (the root defaults) |
| Maximum intervals | `COL_BATH_MAX = DT_MAX =` 1 yr |
| Output cadence | every 100 yr (`DT_OUT`); particles at every frame (`LIN_BASE = 1`) |

`src/const_defs.cuh` asserts `N_K == 256` and `COL_BATH_EPS == 0.08`, so all variants share the
benchmark settings. The root default is `COL_BATH_EPS = 0.02`. The tolerance controls refreshes and
is not an 8% accuracy guarantee.

## Production code and overrides

All runtime and device sources come directly from the root: dynamics, St-dependent concentration
diffusion, query-local collision velocities, cached rates, sticking and erosion grouping, adaptive
size bins with the production moving-bin bounds, local refreshes, and the KD-tree and Morton
searches. There is no artificial global partner reshuffling.

The campaign `src/` directory contains only:

- `const_defs.cuh`: the constants above.
- `swarm_host.cuh`: includes the root host header and replaces only `initmass_calc` and
  `rand_disk_poly`. It samples dust positions by rejection in proportion to the production
  hydrostatic gas density, rather than the root's settled distribution, and normalizes the mass with
  a Simpson integral over the same finite spherical domain. Root monodisperse size sampling gives
  the exact monomer diameter; all other host and output helpers are reused.

On ROCm, the root collision code specializes some kernels for this shape. When compiled for
`gfx942`, the rate and chain kernels use wavefront reductions for `N_K = 256` with
`COL_BATH_TPB = 128`; the KD-tree neighbor-cache query uses private per-thread heaps with 64 queries
per block. The Morton search uses the same query kernel on both GPU backends.

These setups follow the physical ingredients of the Eriksson comparison, not identical MCDUST
equations. Production hydrostatic gas stratification, fixed spherical boundaries,
relative-velocity prescriptions, and Gaussian diffusion noise are retained. No MCDUST gas-density
floor or extra noise normalization is imported.

## Build and run

From the repository root. Only this directory and the root `Makefile`, `inc/`, and `src/` are
required. The campaign `Makefile` defaults to `GPU_BACKEND=cuda`, to `GPU_TARGET=sm_80` for CUDA or
`gfx942` for ROCm, and to `COLLISION_SEARCH=kdtree` on both backends, so name the search explicitly.
One run, for example:

```bash
make -C val/paper/eriksson -j8 MODEL=alpha_1e-3 GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/paper/eriksson/obj/alpha_1e-3/cuda/morton/gamedev
```

The four runs of one backend, sequentially (set `backend=rocm` and `target=gfx942` for ROCm):

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

Each executable run without arguments is one independent fresh run; rerunning into an existing
output path replaces its results. The paths are:

| Item | Path under `val/paper/eriksson/` |
|---|---|
| Executable | `obj/<model>/<backend>/<search>/gamedev` |
| Objects | `obj/<model>/swarm/<backend>/<search>/<target>/` |
| Output | `out/<model>/<backend>/<search>/` |

## Outputs

Every frame, including the initial frame 0, writes `dustdens_<frame>.dat` (one double per mesh
cell), `particle_<frame>.dat`, and `rngstate_<frame>.dat`; `variables.txt` records the parameters.
A particle record holds eight doubles, 64 bytes: position, physical velocity, grain diameter
`par_size`, and represented grain count `par_numr`. One particle file is about 67 MB, and one RNG
checkpoint holds one backend RNG state per particle (about 50 MB with CUDA's 48-byte
`curandState`). A CUDA run of `alpha_1e-3` therefore writes about 15 GB and one of `alpha_1e-4`
about 30 GB. `COL_DIAGNOSTICS` is not enabled, so no collision JSON or JSONL diagnostics are
written. See [Output files](../../../README.md#output-files) for the file formats.

## Analysis

No analysis script is included in this directory. Compare the evolving mass-weighted size
distributions and dust-density fields between the two search backends and the two GPU backends, and
with the Eriksson comparison at matching times.

## Evidence boundary

The eight builds use only the constants and host overrides above with root numerical sources and
have distinct executable and output paths. The ROCm specializations were tuned on MI300A (`gfx942`)
at `N_K = 256` and `COL_BATH_EPS = 0.02`; their speedup at this campaign's 0.08 is unmeasured.
Native CUDA and ROCm builds and fresh scientific results from the current source are required
before the campaign supports a scientific claim.
