# Variable-naming reference

## Scope

This reference covers the active production fluid code in:

- `inc/fluid/*.cuh`
- `src/fluid/*.cu`

It also defines corresponding names in the active swarm code in:

- `inc/swarm/*.cuh`
- `src/swarm/*.cu`

Generated objects, production-model overrides, verification models, mock scripts, and test overrides are excluded except when a production-interface rename requires a matching test-harness update

## Status

The production naming audit is complete as of 2026-07-27

- all accepted fluid-only naming changes are applied
- all accepted cross-model names are aligned where the quantities have the same physical or numerical meaning
- representation-specific names remain different where the algorithms or stored states differ
- fluid and swarm optical-depth translation units are owned by their respective source branches
- no unresolved naming inconsistency remains in the audited production paths
- moving equivalent helpers into shared headers remains deferred because it changes code ownership rather than identifier spelling

The changes preserve the numerical formulas, field layout, kernel launch geometry, and external data labels

## Canonical naming rules

### Physical fields and local quantities

- persistent dust-density arrays use `dustdens` and `dev_dustdens`
- local dust and gas density variables use `rhod` and `rhog`; in a vertically integrated 2D fluid
  kernel, `rhod` stores the evolved surface density even though the local variable spelling is retained
- dust and gas surface densities use `sigma_d` and `sigma_g`
- the initial dust surface-density profile uses `initdens` and `dev_initdens`
- conserved fluid momentum uses `dev_dustmomx/y/z` and local `mx/my/mz`
- evolved fluid primitives use `dev_dustvelx/y/z` and local `lx/vy/lz`
- `lx` and `lz` are specific angular momenta even though persistent arrays retain the `dustvel*` interface names
- `vx/vy/vz` denote physical spherical linear velocities
- `vR/vZ` denote physical cylindrical linear velocities
- gas velocities use `_g`, as in `vx_g`, `vR_g`, `vy_g`, and `lz_g`

### Coordinates and grid geometry

- spherical coordinates use `x`, `y`, and `z`
- cylindrical coordinates use `R` and `Z`
- inner and outer locations use `_i` and `_o`
- Riemann left and right states use `_L` and `_R`
- donor or upwind locations use `_up`
- finite-volume interfaces use `face`, not `edge`
- face-coordinate helpers are `_get_yface` and `_get_zface`
- cell-center helpers are `_get_ycent` and `_get_zcent`
- `dx` and `dz` are angular increments, while `dy` is the logarithmic radial ratio
- `dx_len`, `dr`, `dr_i/dr_o`, and `dz_len` denote physical lengths
- dimensionless gas aspect ratio uses `h_g`, while physical scale heights use `H_g` and `H_d`

### Indices, counts, and pointers

- full flattened fluid-cell indices use `idx_cell`
- azimuthal-ring work items use `idx_ring`
- directional-column work items use `idx_col`
- radial optical-depth rays use `idx_ray`
- directional grid indices use `ix`, `iy`, and `iz`
- other indices use the `idx_*` prefix, such as `idx_face`, `idx_stencil`, `idx_aug`, and `idx_sub`
- counts use a descriptive `_count` suffix, such as `cell_count`, `bin_count`, `sub_count`, and `shift_count`
- pointer wrappers place the object before `_ptr`, such as `dt_rate_ptr`, `cfl_rate_ptr`, and `lx_ptr`

The flattened fluid-grid order remains:

```cpp
ix + iy*N_X + iz*N_X*N_Y
```

### Suffix construction

- use an underscore before a final one-letter physical tag, as in `lx_g`, `diff_x`, `grav_y`, and `torq_z`
- merge a later qualifier with that one-letter tag, as in `lx_g1`, `grav_yold`, `diff_yi`, and `sinz_new`
- use `_sq` and `_cb` for squared and cubed quantities
- retain descriptive `_old`, `_new`, and `_tmp` time stages outside SSA
- retain `_i`, `_1`, `_2`, and `_j` inside SSA because those suffixes follow the reference-paper notation

### Drag, radiation, and forces

- stopping time uses `ts`
- the dimensionless drag interval `dt/ts` uses `tau`
- its powers use `tau_sq` and `tau_cb`
- optical depth uses `optdepth`, including `optdepth_i` and `optdepth_o` at faces
- radiation ramp variables use `taper_raw` and `beta_taper`
- spherical source terms use `grav_y`, `cent_y`, and `torq_z`
- viscous-accretion algebraic components use `term_R` and `term_Z`

### Diffusion

- swarm cylindrical diffusivities use `diff_x`, `diff_R`, and `diff_Z`
- swarm projections onto spherical directions use `diff_y` and `diff_z`
- fluid spherical diffusivities use `diff_x`, `diff_y`, and `diff_z`
- inner and outer fluid face values use `diff_yi/diff_yo` and `diff_zi/diff_zo`
- fluid Schmidt numbers use `SCHMIDT_X/Y/Z`
- swarm Schmidt numbers use `SCHMIDT_X/R/Z`
- diffusion substeps use `sub_count`, `inv_sub_count`, `idx_sub`, and `dt_sub`

## Applied implementation record

### Density and dynamical variables

| Previous fluid spelling | Applied spelling |
|---|---|
| `dens`, `dens_prev/next/up/shift/work/rhs` | `rhod`, `rhod_prev/next/up/shift/work/rhs` |
| `face_dens`, `flux_dens`, `corr_dens` | `face_rhod`, `flux_rhod`, `corr_rhod` where the storage retains that physical role |
| `rho_mid` | `rhog_mid` |
| `vel_x/vel_y/vel_z`, `vel_R` | `vx/vy/vz`, `vR` |
| `vgas_x`, `vgas_R` | `vx_g`, `vR_g` |
| `vel_x_res`, `vel_z_diff` | `vx_res`, `vz_diff` |
| `stress_R`, `stress_Z` | `term_R`, `term_Z` |

### Source, radiation, and boundaries

| Previous fluid spelling | Applied spelling |
|---|---|
| `drag_h`, `drag_h2`, `drag_h3` | `tau`, `tau_sq`, `tau_cb` |
| `tau_i`, `tau_o` for optical depth | `optdepth_i`, `optdepth_o` |
| `force_weight_n` | `force_weight_old` |
| `grav_y_n/tmp/new` | `grav_yold/ytmp/ynew` |
| `cent_y_n/tmp/new` | `cent_yold/ytmp/ynew` |
| `torq_z_n/new` | `torq_zold/znew` |
| `speed_ib`, `speed_ob` | `speed_i`, `speed_o` |
| `flux_dens_ib`, `flux_mx/my/mz_ib` | `flux_rhod_i`, `flux_mx/my/mz_i` |
| real-valued `outflow` mask | direct conditional flux expression |

### PPM and finite-volume helpers

| Previous spelling | Applied spelling |
|---|---|
| `iface`, `icell` | `idx_face`, `idx_cell` |
| generic polynomial and stencil indices | `degree`, `idx_stencil`, `idx_aug`, and `idx_row/idx_col` |
| `iupL`, `iupR` | `idx_up_L`, `idx_up_R` |
| `qm1`, `q0`, `qp1`, `qp2` | `q_m1`, `q_0`, `q_p1`, `q_p2` |
| `qa`, `qb` | `q_L`, `q_R` |
| `_get_s_y`, `_get_s_z` | `_get_sy`, `_get_sz` |
| `_get_dr_cc_i`, `_get_dr_cc_o` | `_thread_dr_cent_i/o` and `_block_dr_cent_i/o` |
| `_ppm_faces_nonuniform`, `_ppm_face_value` | `_thread_ppm_faces`, `_thread_ppm_state` |

The standard PPM quantities `dq` and `q6` retain their conventional mathematical names

### Initial surface-density profile

| Previous fluid spelling | Applied spelling |
|---|---|
| `convpow_calc` | `initdens_calc` |
| `u_axis`, `v_axis` | `conv_u`, direct output to `initdens` |
| `n_bin` | `bin_count` |
| generic `i/j/k` loops | `idx_src`, `idx_dst` |
| `u_j`, `u_min/max`, `u_s` | `R_src`, `R_src_min/max`, `smooth` |
| `du`, `delta_u`, `sig_u` | `dR`, `delta_R`, `kernel_std` |
| `norm`, `kernel` | `kernel_norm`, `kernel_weight` |
| manual device interpolation | `initdens_lerp` |
| `rng`, `xi` | `rngstate`, `noise_x` |

The host-vector and device-pointer forms of `initdens_lerp` use the same physical argument first and differ only in their storage interface

### CFL, file I/O, and main loop

| Previous fluid spelling | Applied spelling |
|---|---|
| `ptr_cfl`, `ptr_lx` | `cfl_rate_ptr`, `lx_ptr` |
| `max_rate`, `max_it`, `idx_max` | `max_cfl_rate`, `max_cfl_rate_ptr`, `idx_cfl_max` |
| `save_binary`, `load_binary` | `save_host_binary`, `load_host_binary` |
| template `T`, integer `count` | `DataType`, `std::size_t count` |
| macro parameter `IDX` | `idx_file` |
| `num_len`, `len`, `t_curr` | `width`, `time_now` |
| resume streams `convert` and `ss` | `frame_stream` |
| `advance_x/y/z` | `advance_advection_x/y/z` |
| `recover_dust_velocity` | `sync_dust_state` |
| `time_interval`, `time_remain` | `duration`, `remaining` |
| `adv_interval`, `dt_cfl_begin` | `dt_adv`, `dt_cfl` |
| raw `taper` | `taper_raw` |

## Scratch-storage naming

Scratch arrays retain the existing memory-conscious reuse while avoiding names that claim the wrong physical dimensions

- `advection_xth` uses `work_rhod` and `work_x/y/z` for arrays that change from unshifted state to antidiffusive flux differences
- `advection_yth` and `advection_zth` use `face_work_rhod` and `face_work_x/y/z` for face-indexed storage that changes from reconstructed primitives to correction fluxes
- diffusion solvers keep `rhod_rhs`, `upper_work`, and `cycle_work` inside solver-only lexical scopes
- diffusion transport uses separate `mass_flux` and `moment_flux` arrays inside non-overlapping flux scopes

The lexical scopes expose the physical role at every use while allowing the CUDA compiler to reuse storage whose live ranges do not overlap

## Intentional cross-model differences

Do not unify the following names because they encode real representation or algorithm differences:

- swarm particle index `idx` versus fluid grid index `idx_cell`
- `particle/dev_particle` versus fluid fields `dustdens`, `dustvel*`, and `dustmom*`
- swarm cylindrical diffusion suffixes `R/Z` versus fluid spherical diffusion suffixes `y/z`
- swarm `dev_dt_rate` versus fluid `dev_cfl_rate`
- swarm `particle_init` versus fluid `init_rho_calc` and `init_vel_calc`
- swarm `diffusion_pos` versus fluid `diffusion_[xyz]{th,bl}`
- swarm `total_dust_mass`, `mass_norm`, `size`, and represented grain number, which have no single-species fluid counterparts
- fluid conserved `mx/my/mz`, which have no density-weighted particle-state equivalents
- fluid `RHO_VAC` and `POS_LIMIT`, which belong to Eulerian recovery and implicit diffusion
- swarm `rhog_0`, which is the analytical imported-gas calibration anchor at `R_0`
- persistent array names such as `dev_gas_velx`, the `swarm::velocity` member, and external labels `velocity_x/y/z`

The directional kernel and file names `advection_[xyz]{th,bl}` and `diffusion_[xyz]{th,bl}` retain their isolated direction letter and two-letter sweep suffix to satisfy the exact 13-character kernel-name convention

## Names suitable for common infrastructure

The following common names should be retained when representation-independent infrastructure is
extracted from the already merged project:

- grid helpers: `_get_dx/dy/dz`, `_get_yface`, `_get_zface`, `_get_ycent`, `_get_zcent`, `_get_mesh_dim`, `_get_vol_y`, `_get_vol_z`
- physical helpers: `_get_omegaK`, `_get_hg`, `_get_eta`, `_get_gas_strat`, `_get_sigma_g`, `_get_nu`, `_get_alpha`, `_get_visc_vel`
- source quantities: `_get_force_term`, `grav_y`, `cent_y`, `torq_z`
- radiation fields: `optdepth`, `dev_optdepth`, `beta`, `beta_taper`
- output state: `PATH`, `idx_from`, `idx_file`, `clock_sim`, `clock_out`
- host operations: `cuda_fail`, `frame_num`, `msg_output`, `save_sam_as_velocity`, `load_velocity_as_sam`, `save_host_binary`, `load_host_binary`, `save_variable`

## Deferred ownership work

These are structural merge tasks rather than naming defects:

1. move the swarm host-only `_get_ycent` and `_get_zcent` definitions into the eventual shared grid header
2. separate representation-specific code from the common portions of `param_grid.cuh` and `param_phys.cuh`
3. place the equivalent `_get_force_term` implementation in one shared physics header
4. retain branch-specific `optdepth_calc` kernels because their inputs and preprocessing differ
5. retain branch-local `optdepth_csum` translation units until a representation-neutral kernel interface exists

No identifier rename is required before undertaking these ownership changes

## Validation

The completed implementation was checked by:

- scanning the audited production paths for every superseded identifier listed above
- checking declaration and call-site consistency after public helper renames
- updating the verification harness only where the production `save_host_binary` interface required it
- running `git diff --check`
- running Makefile dry runs for `fluid_fiducial`, `verify_x_transport_2d`, and `verify_source_drag`

A native CUDA compilation was not available in the local environment because `nvcc` is not installed
