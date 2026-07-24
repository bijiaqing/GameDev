# Fluid variable-naming consistency review

## Scope

This review covers the active production fluid code in:

- `new/inc/fluid/*.cuh`
- `new/src/fluid/*.cu`
- `new/src/share/optdepth_calc.cu`
- `new/src/share/optdepth_csum.cu`

Generated objects, production models, verification models, mock scripts, and test overrides are excluded

This began as a naming-only audit and now also tracks the implementation status of the agreed changes

## Status

The recommended canonical naming table was applied to the production fluid code on 2026-07-24

- the encoded invalid-cell value is now `bad_cell` on the host and `dev_bad_cell` on the device
- index spaces, coordinates, face terminology, physical velocities, diffusivities, source forces, PPM weights, CFL arrays, file names, and element counts now follow the table below
- the changes are identifier-only and do not alter formulas, control flow, array layout, or kernel launch geometry
- the more detailed recommendations below remain pending unless explicitly marked as applied

## Summary

The persistent dust-field convention is already coherent:

- density: `dustdens`, `dev_dustdens`, and local `dens`
- conserved momentum: `dev_dustmomx/y/z` and local `mx/my/mz`
- evolved primitives: `dev_dustvelx/y/z` and local `lx/vy/lz`
- initial dust surface-density profile: `initdens` and `dev_initdens`

The canonical inconsistencies identified in the first audit have been resolved

The principal remaining detailed inconsistencies are:

1. several reused scratch arrays change physical meaning while retaining names such as `face_lx` or `dens_rhs`
2. detailed PPM loop and stencil names remain abbreviated
3. source-stage, drag, and boundary quantities still use a few mathematical or ambiguous suffixes
4. convolution loop variables and some main-loop helper names remain generic

## Recommended canonical naming

| Concept | Previous names | Applied names |
|---|---|---|
| full flattened cell index | `idx`, `ic` | `idx_cell` |
| azimuthal-ring work item | `idx` | `idx_ring` |
| directional-column work item | `idx` | `idx_col` |
| radial optical-depth ray | `idx` | `idx_ray` |
| directional indices | `ix`, `iy`, `iz` | retain |
| spherical coordinates | `yc`, `zc` | `y`, `z` |
| cylindrical coordinates | `Rc`, `Zc` | `R`, `Z` |
| face coordinates | `y0/y1`, `z0/z1` | `y_i/y_o`, `z_i/z_o` |
| stored dust primitives | `lx`, `vy`, `lz` | retain |
| physical dust velocities | `v_x`, `v_y`, `v_z`, `vz` | `vel_x`, `vel_y`, `vel_z` |
| physical gas velocity | `vg_x`, `vgas_R` | `vgas_x`, `vgas_R` |
| diffusivity | `Dx`, `Dy_i/o`, `Dz`, `Dz_i/o` | `diff_x`, `diff_y_i/o`, `diff_z`, `diff_z_i/o` |
| radial gravity | `Fy` | `grav_y` |
| radial centrifugal acceleration | `Fcy` | `cent_y` |
| polar geometric torque | `Tcz` | `torq_z` |
| PPM interface | `edge`, `face` | `face` |
| per-cell CFL array | `dev_cfl_rate` | `dev_cfl_rates` |
| PPM weights | `weight_y/z`, `dev_weight_y/z` | `ppm_weight_y/z`, `dev_ppm_weight_y/z` |
| encoded first invalid cell | `badstate`, `dev_badstate` | `bad_cell`, `dev_bad_cell` |
| file name | `fname`, `file_name` | `file_name` |
| element count | `number`, `n_cells`, `ncells` | `count`, `cell_count` as appropriate |

## Detailed findings

### 1. Flattened and reduced indices

**Status: applied for the canonical index spaces**

The sweep kernels now distinguish their reduced work-item index from the full flattened cell index:

- `advect_x_calc`, `diffus_x_calc`, and `cfl_rate_calc` use `idx_ring`
- `advect_y_calc`, `diffus_y_calc`, `advect_z_calc`, and `diffus_z_calc` use `idx_col`
- `optdepth_csum` uses `idx_ray`
- all full flattened indices use `idx_cell`

In `fluid_host.cuh`, `idx_ring` is used when a ring offset is passed to `thrust`, while cell loops form `idx_cell` directly

The existing flattened order is already correct and should remain:

```cpp
ix + iy*N_X + iz*N_X*N_Y
```

### 2. Coordinate values

**Status: applied for `y/z/R/Z` and face or donor coordinate suffixes**

Cell kernels now use `y` and `z` for spherical coordinates and `R` and `Z` for cylindrical coordinates, matching the physical helpers and swarm implementation

Add role suffixes only when multiple locations coexist:

- `y_i`, `y_o`, `R_i`, `R_o` for inner and outer faces
- `y_up`, `z_up`, `R_up` for donor coordinates
- `R_L` and `R_R` for Riemann states if needed

The coordinate forms `R_i`, `R_o`, and `R_up` are applied

The more specific changes `h_i/h_o` to `h_g_i/h_g_o` and `rho_mid` to `rhog_mid` remain pending

Retain the intentional distinction between dimensionless `h_g` and physical `H_g` or `H_d`

### 3. Cell faces versus edges

**Status: applied for face terminology; detailed PPM loop names remain pending**

The finite-volume interface terminology is now consistently `face`:

- `_ppm_face_uniform`
- `_ppm_faces_nonuniform`
- `face_dens/lx/vy/lz`
- `face_s` and `face_weight`

Use `idx_face` instead of `iface` and `idx_cell` instead of `icell`

The left/right suffix convention `_L` and `_R` is already consistent and should remain

### 4. Arrays whose physical meaning changes

Several local arrays are deliberately reused to limit CUDA local-memory demand, but their names become physically incorrect after reuse

In `advect_x_calc`:

- `dens`, `mx`, `my`, and `mz` initially hold the unshifted conserved state
- the same arrays later hold high-minus-low antidiffusive face fluxes

In `advect_y_calc` and `advect_z_calc`:

- `face_dens/lx/vy/lz` initially hold reconstructed PPM face primitives
- they later hold density and momentum-flux corrections
- in particular, `face_lx` later has momentum-flux dimensions rather than specific-angular-momentum dimensions

In `diffus_y_calc` and `diffus_z_calc`:

- `dens_rhs` initially holds the Crank-Nicolson density right-hand side
- it later holds one component of momentum flux

In `diffus_x_calc`:

- `cycle_work` initially holds the Sherman-Morrison cyclic correction
- it later holds one component of momentum flux
- `upper_work` initially holds the modified Thomas upper diagonal and later the diffusive mass flux

Recommended treatment:

1. retain storage reuse to avoid increasing CUDA local memory
2. separate non-overlapping roles into lexical scopes where the compiler can reuse storage
3. use role-correct names such as `corr_flux_dens`, `corr_flux_mx/my/mz`, `dens_rhs`, `mass_flux`, and `moment_flux`
4. where one buffer must visibly span several roles, use an honest neutral name such as `aux_work` rather than a physically incorrect name

This is the highest-value naming cleanup because the current names can lead to dimensional mistakes during later maintenance

### 5. Physical velocity names

**Status: applied**

`init_vel_calc` now uses `vgas_x`, `vel_R`, `vel_x/vel_y/vel_z`, and `vel_z_diff`

The CFL diagnostic uses `vel_x_res` and `vel_z`, while the device kernel retains `omega_res` for an angular velocity

Retain:

- `lx`, `vy`, and `lz` for the stored fluid primitives
- `vgas_R` for physical cylindrical-radial gas velocity
- `lx_g`, `vy_g`, and `lz_g` for gas targets expressed in stored-state units

This also aligns the fluid initializer with the swarm initializer

### 6. Diffusion coefficients and face quantities

**Status: canonical diffusivity names applied; detailed helper and substep names remain pending**

The diffusion kernels and velocity initializer now use descriptive lower-case names:

- `diff_x`
- `diff_y_i`, `diff_y_o`
- `diff_z`
- `diff_z_i`, `diff_z_o`

This preserves the required spherical `x/y/z` directional convention without relying on pure mathematical symbols

Use `idx_sub` instead of `i_sub` for the diffusion-substep index

The names `dx_len`, `dr_i/dr_o`, and `dz_len` correctly distinguish physical lengths from angular `dx/dz` and logarithmic ratio `dy`

The local helpers `_get_dr_cc_i` and `_get_dr_cc_o` are understandable but `cc` is not self-explanatory outside the file

Possible replacements are `_get_dr_cent_i` and `_get_dr_cent_o`

### 7. Source-force and drag names

**Status: base force names applied; time-stage and drag names remain pending**

`source_update.cu` now uses the same base force convention as the swarm transport code:

- `grav_y`
- `cent_y`
- `torq_z`

Its stage forms currently use `_n`, `_new`, and `_tmp`; normalize them to:

- `grav_y_old/new/tmp`
- `cent_y_old/new/tmp`
- `torq_z_old/new/tmp`

The source variable `drag_h` is the dimensionless ratio `dt/ts`, not a scale height

Rename:

- `drag_h` to `tau_drag`
- `drag_h2` and `drag_h3` to `tau_drag2` and `tau_drag3`
- `force_weight_n` to `force_weight_old`

Retain `ts` for stopping time because the same convention is used by the swarm transport integrator

### 8. Boundary suffixes

**Status: face-coordinate suffixes applied; boundary-speed and flux suffixes remain pending**

The radial and polar advection kernels use:

- `speed_ib` and `speed_ob`
- `flux_dens_ib`, `flux_mx_ib`, `flux_my_ib`, and `flux_mz_ib`

Use the established inner/outer suffixes:

- `speed_i` and `speed_o`
- `flux_dens_i`, `flux_mx_i`, `flux_my_i`, and `flux_mz_i`

The coordinate forms `y_i/y_o` and `z_i/z_o` are already applied

Retain `_L/_R` exclusively for left and right Riemann states

The real-valued `outflow` variable is a numerical mask rather than a Boolean

Either replace it directly with the conditional flux expression or name it `outflow_mask`

### 9. PPM helper parameters and loop indices

The PPM helpers now use `cell_count`, but their remaining loop and stencil names still mix:

- `iface` and `icell`
- generic `i`, `j`, `k`, `n`, `row`, and `col`
- `iupL` and `iupR`

Recommended names:

- `idx_face`
- `idx_cell`
- `degree` for the polynomial-moment loop
- `idx_stencil` for the four-cell stencil
- `idx_aug` for augmented-matrix columns where needed
- `idx_up_L` and `idx_up_R`

The mathematical PPM quantities `dq` and `q6` are standard and can remain

For the uniform four-cell interpolation stencil, use `q_m1`, `q_0`, `q_p1`, and `q_p2` instead of `qm1`, `q0`, `qp1`, and `qp2`

Use `q_L` and `q_R` instead of `qa` and `qb` in `_ppm_state_L/R`

### 10. Convolved initial-density profile

`convpow_calc` currently mixes the intended `u/v` notation with generic loop indices:

- `u_axis` and `v_axis`
- `n_bin`
- `i`, `j`, and `k`
- `u_j`
- `norm` and `kernel`

The previously selected convolution convention should be restored consistently:

- `conv_u` for the uniformly sampled input coordinate
- `conv_v` for the convolved output profile
- `bin_count`
- `idx_u`, `idx_src`, and `idx_dst`
- `u_src`
- `kernel_norm` and `kernel_weight`

Retain `du`, `u_min`, `u_max`, and the `u/v` notation because these are internal convolution coordinates rather than spherical `x/y/z`

Rename `sig_u` to `std_u` if it is intended to mean the Gaussian standard deviation

In `init_rho_calc`, use `idx_u` instead of `iu` and `noise_x` instead of `xi`

Retain `initdens` because it denotes the initial dust surface-density profile rather than the evolved volume density

### 11. CFL, PPM-weight, and invalid-cell arrays

**Status: canonical array names applied using `bad_cell` and `dev_bad_cell`; detailed pointer and scalar names remain pending**

The per-cell array is now `dev_cfl_rates`, while `cfl_rate` remains the scalar rate for one cell

The PPM arrays are now `ppm_weight_y/z` and `dev_ppm_weight_y/z`

The encoded first-invalid-cell value is now `bad_cell` and `dev_bad_cell`, with `idx_bad` retained after decoding

Detailed follow-up:

- use `cfl_max` instead of `max_rate`
- use `cfl_ptr` and `lx_ptr` instead of `ptr_cfl` and `ptr_lx`

The lambda name `validate_finite_state` and kernel name `inf_cell_flag` describe their operations and can remain

### 12. Host file interfaces

**Status: `idx_file`, `file_name`, and `count` applied; macro and message-format names remain pending**

`fluid_host.cuh` now uses the same host interface convention as the swarm code:

- `frame_num(int idx_file)`
- `file_name`
- `count`

The macro parameter `IDX` can still become `IDX_FILE` or `idx_file`

Within `msg_output`, `t_curr` and `len` can become `time_now` and `width`

### 13. Main-loop helper names

The main function contains three advection lambdas named `advance_x/y/z`

Because diffusion and source terms also advance the state, use names that identify the operator:

- `advance_advection_x`
- `advance_advection_y`
- `advance_advection_z`

`recover_dust_velocity` invokes `momentum_getv`, which also repairs near-vacuum conserved momentum

Use `sync_dust_state`

Other recommended local names:

- `time_remain` to `time_left`
- `adv_interval` to `dt_adv`
- `dt_cfl_begin` to `dt_cfl`
- raw `taper` to `taper_raw`, retaining `beta_taper` for the smoothed multiplicative factor

The established `clock_sim`, `clock_out`, and `idx_from` names are used consistently and can remain

## Intentional distinctions to retain

The following differences carry physical or numerical information and should not be unified:

- `dens` is dust density while `sigma_d` is dust surface density and `rhog` is gas volume density
- `h_g` is the dimensionless gas aspect ratio while `H_g` and `H_d` are physical scale heights
- `lx` and `lz` are specific angular momenta while `vy` is a linear spherical-radial velocity
- `mx` and `mz` are angular-momentum densities while `my` is linear-momentum density
- `vel_x` and `vel_z` are physical linear velocities used for initialization, diagnostics, and files
- `_i/_o` denotes inner/outer geometry, `_L/_R` denotes Riemann left/right states, and `_old/_new` denotes time states
- `dx` and `dz` are angular coordinate increments, `dy` is a logarithmic radial ratio, and `dx_len/dr/dz_len` are physical lengths
- `cn_lower/diag/upper` are matrix coefficients while `flux_*` quantities are physical finite-volume fluxes
- `tau_i/tau_o` are scalar optical depths while `optdepth/dev_optdepth` are stored fields

## Existing names suitable for future sharing

The following helper names already agree between the fluid and swarm implementations and should be retained:

- `_get_omegaK`
- `_get_hg`
- `_get_eta`
- `_get_gas_strat`
- `_get_nu`
- `_get_alpha`
- `_get_visc_vel`
- `_get_yedge`
- `_get_ycent`
- `_get_zedge`
- `_get_zcent`
- `_get_mesh_dim`
- `_get_vol_y`
- `_get_vol_z`

The shared optical-depth kernels already have matching public names:

- `optdepth_calc`
- `optdepth_csum`

Their internal index names should nevertheless follow the `idx_cell` and `idx_ray` convention above

## Suggested implementation order

1. normalize index spaces and flattened-cell names — canonical portion applied
2. normalize `y/z/R/Z` coordinates and physical velocity names — applied
3. normalize force, drag, and diffusivity names — force and diffusivity bases applied; drag and stage suffixes pending
4. unify PPM `face` terminology and helper interfaces — face terminology applied; detailed loop names pending
5. repair semantic names for reused advection and diffusion scratch arrays
6. normalize boundary suffixes
7. normalize host arrays, file interfaces, and main-loop lambdas
8. clean up convolution and small local-loop names

Each group should be applied separately with declaration/call-site scans after the change
