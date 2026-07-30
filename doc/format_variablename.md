# Naming and ownership reference

## Scope and status

This document defines the current naming contract for production files under:

- `inc/fluid/` and `src/fluid/`
- `inc/swarm/` and `src/swarm/`

Verification overrides may use additional test-local names, but production declarations, definitions, and call sites should follow this contract. The accepted fluid, swarm, and cross-model renames are applied in the active tree. Remaining work concerns code ownership and future backends rather than unresolved identifier spelling

Names describe physical meaning first and storage representation second. The two branches should use the same name only when the quantity has the same meaning

## Physical fields and state

| Meaning | Canonical spelling |
|---|---|
| persistent dust density | `dustdens`, `dev_dustdens` |
| local dust or gas density | `rhod`, `rhog` |
| dust or gas surface density | `sigma_d`, `sigma_g` |
| initial dust surface-density profile | `initdens`, and `dev_initdens` in the fluid device path |
| fluid conserved momenta | `dev_dustmomx/y/z`; local `mx/my/mz` |
| evolved angular/radial primitives | `dev_dustvelx/y/z`; local `lx/vy/lz` |
| physical spherical velocity | `vx/vy/vz` |
| physical cylindrical velocity | `vR/vZ` |
| gas velocity target | suffix `_g`, for example `vx_g`, `vR_g`, and `lz_g` |
| swarm representative state | `particle`, `dev_particle` |

`lx=Rv_\phi` and `lz=rv_\theta` are specific angular momenta even though persistent arrays retain the historical `dustvel*` interface. Science files contain physical linear velocity, and both branches convert to or from the internal angular representation at file boundaries

In a vertically integrated 2D fluid kernel, local `rhod` denotes the evolved surface density even though the compact spelling is retained

## Coordinates and geometry

- spherical coordinates use `x`, `y`, and `z`
- cylindrical coordinates use uppercase `R` and `Z`
- dimensionless gas aspect ratio uses `h_g`; physical scale heights use `H_g` and `H_d`
- finite-volume interfaces use `face`, not `edge`
- face helpers are `_get_yface` and `_get_zface`
- center helpers are `_get_ycent` and `_get_zcent`
- `dx` and `dz` are angular increments; `dy` is the logarithmic radial ratio
- `dx_len`, `dr`, `dr_i/dr_o`, and `dz_len` are physical lengths
- inner and outer locations use `_i` and `_o`
- Riemann states use `_L` and `_R`
- donor or upwind locations use `_up`

Fluid cells are flattened x-first:

```cpp
ix + iy*N_X + iz*N_X*N_Y
```

## Indices, counts, and pointer wrappers

- directional cell indices are `ix`, `iy`, and `iz`
- a flattened fluid cell uses `idx_cell`
- an azimuthal ring uses `idx_ring`
- a directional column uses `idx_col`
- an optical-depth ray uses `idx_ray`
- other indices use descriptive `idx_*` names such as `idx_face`, `idx_stencil`, `idx_sub`, and `idx_cfl_max`
- counts use `*_count`, for example `bin_count` and `sub_count`
- Thrust or pointer wrappers place the object first, for example `cfl_rate_ptr`, `dt_rate_ptr`, and `max_cfl_rate_ptr`

Avoid a generic `idx` when a function contains more than one index role. A swarm kernel with only one representative index may retain `idx`

## Suffixes and stages

- use `_sq` and `_cb` for squared and cubed values
- use descriptive `_old`, `_new`, and `_tmp` stages outside the semi-analytic particle integrator
- retain `_i`, `_1`, `_2`, and `_j` in SSA code because they follow the reference derivation
- merge a later qualifier with a one-letter directional tag, for example `lx_g1`, `grav_yold`, `diff_yi`, and `sinz_new`
- retain conventional compact PPM names `dq` and `q6`

## Drag, radiation, and forces

- stopping time is `ts`
- the dimensionless drag interval is `tau=dt/ts`, with `tau_sq` and `tau_cb`
- optical depth is `optdepth`, including `optdepth_i` and `optdepth_o`
- radiation-ramp variables are `taper_raw` and `beta_taper`
- Poynting-Robertson damping is `pr_rate` and uses `C_LIGHT`
- directional damping and response terms are `rate_x/y/z`, `relax_x/y/z`, and `resp_x/y/z`
- spherical force components are `grav_y`, `cent_y`, and `torq_z`
- viscous-flow algebraic components are `term_R` and `term_Z`

The active feature flags retain their source spellings, including `TRANSPORT`, `DIFFUSION`, `COLLISION`, `RADIATION`, `VISC_FLOW`, `PR_EFFECT`, `IMPORTGAS`, `MULTISIZE`, `CONST_ST`, and `CODE_UNIT`

## Diffusion names

The diffusion basis is intentionally representation-specific:

- swarm cylindrical diffusivities: `diff_x`, `diff_R`, `diff_Z` and `SCHMIDT_X/R/Z`
- swarm projections into spherical storage: `diff_y` and `diff_z`
- fluid spherical diffusivities: `diff_x`, `diff_y`, `diff_z` and `SCHMIDT_X/Y/Z`
- fluid face diffusivities: `diff_yi/diff_yo` and `diff_zi/diff_zo`
- diffusion substeps: `sub_count`, `inv_sub_count`, `idx_sub`, and `dt_sub`

Do not rename cylindrical swarm quantities to spherical `y/z` or spherical fluid operators to cylindrical `R/Z`

## Kernel and file conventions

Fluid directional kernels use:

- `advection_[xyz]th` and `diffusion_[xyz]th` for one-thread-per-line sweeps
- `advection_[xyz]bl` and `diffusion_[xyz]bl` for one-block-per-line sweeps

The direction plus two-letter sweep suffix keeps each kernel name at 13 characters. Corresponding translation-unit names match the kernels

The runtime entry files are `fluid_runtime.cu` and `swarm_runtime.cu`; each still defines the required C++ entry function `main`. Branch-local optical-depth files may share function names because only one representation is compiled into a model

Swarm collision backends are selected as `COLLISION_SEARCH=kdtree` or `COLLISION_SEARCH=morton`. Their private headers live under `inc/swarm/kdtree/` and `inc/swarm/morton/`; backend-specific identifiers use the corresponding `kdtree_*` or `morton_*` prefix

Parallel host spellings include `initdens_calc`, `initdens_lerp`, `save_host_binary`, `load_host_binary`, `save_sam_as_velocity`, `load_velocity_as_sam`, `frame_stream`, `time_now`, `duration`, and `remaining`

## Scratch storage

Scratch names must describe the value's current role even when lexical scopes let the compiler reuse memory:

- thread advection uses `work_rhod` and `work_x/y/z` for arrays whose role changes within a contained phase
- radial and polar thread sweeps use `face_work_rhod` and `face_work_x/y/z` for face-indexed reconstruction/correction storage
- diffusion solves use `rhod_rhs`, `upper_work`, and `cycle_work`
- diffusive transport distinguishes `mass_flux` from `moment_flux`
- block fluid fields use `dev_adv_work` and the `BlockAdvField` enumeration

## Intentional cross-model differences

Do not unify names that encode different algorithms or stored state:

- swarm `particle_init` versus fluid `init_rho_calc` and `init_vel_calc`
- swarm `diffusion_pos` versus fluid `diffusion_[xyz]{th,bl}`
- swarm `dev_dt_rate` versus fluid `dev_cfl_rate`
- swarm `total_dust_mass`, `mass_norm`, grain size, and represented-grain number, which have no monodisperse fluid counterpart
- fluid conserved `mx/my/mz`, vacuum threshold, positivity controls, and line workspaces, which have no particle-state counterpart
- swarm `rhog_0`, which is the imported-gas Stokes calibration anchor at `R_0`

## Independent ownership

Fluid and swarm do not share headers or translation units. Each branch owns its grid geometry, analytic disk helpers, radiation pipeline, file routines, GPU error handling, and any future CUDA/HIP wrapper

Equivalent physical definitions should retain parallel names and equations, but changes must be implemented and tested independently in both branches. Neither representation may include the other's state or helper headers

## Maintenance checks

For any public rename:

1. update declarations, definitions, call sites, model overrides, and verification overrides
2. search for the superseded spelling across `inc/`, `src/`, `mod/`, and `qav/`
3. preserve external science-file labels unless an intentional format change is documented
4. run `git diff --check` and representative Makefile dry runs
5. perform native CUDA builds for both representations when the same physical convention changes
