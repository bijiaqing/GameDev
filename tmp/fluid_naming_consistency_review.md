# Fluid variable-naming consistency review

## Scope

This review covers the active production fluid code in:

- `new/inc/fluid/*.cuh`
- `new/src/fluid/*.cu`
- `new/src/share/optdepth_calc.cu`
- `new/src/share/optdepth_csum.cu`

The cross-model extension also compares the active swarm code in:

- `inc/*.cuh`
- `src/*.cu`

Generated objects, production models, verification models, mock scripts, and test overrides are excluded

This began as a naming-only audit and now also tracks the implementation status of the agreed changes

## Status

The recommended canonical naming table was applied to the production fluid code on 2026-07-24

- the encoded invalid-cell value is now `bad_cell` on the host and `dev_bad_cell` on the device
- index spaces, coordinates, face terminology, physical-quantity base names, source forces, PPM weights, CFL arrays, file names, and element counts follow the applied table below
- compound suffixes after one-letter physical tags remain pending under finding 25
- the changes are identifier-only and do not alter formulas, control flow, array layout, or kernel launch geometry
- the more detailed recommendations below remain pending unless explicitly marked as applied
- a swarm–fluid consistency pass was added after the fluid-only findings, and the adopted swarm-side names are now being applied incrementally

## Summary

The persistent dust-field convention distinguishes storage names from local physical quantities:

- stored dust-density fields: `dustdens` and `dev_dustdens`
- local volume densities: `rhod` and `rhog`
- local surface densities: `sigma_d` and `sigma_g`
- conserved momentum: `dev_dustmomx/y/z` and local `mx/my/mz`
- evolved primitives: `dev_dustvelx/y/z` and local `lx/vy/lz`
- initial dust surface-density profile: `initdens` and `dev_initdens`

The canonical inconsistencies identified in the first fluid-only audit have been resolved, while the later cross-model `dens` to `rhod` alignment remains pending

The principal remaining detailed inconsistencies are:

1. several reused scratch arrays change physical meaning while retaining names such as `face_lx` or `dens_rhs`
2. detailed PPM loop and stencil names remain abbreviated
3. source-stage, drag, boundary, and qualified directional quantities still use ambiguous or isolated one-letter suffixes
4. convolution loop variables and some main-loop helper names remain generic
5. several equivalent swarm and fluid quantities still use different names or suffix conventions

## Pending and partially resolved findings

### 2. Coordinate values

**Status: applied for `y/z/R/Z` and face or donor coordinate suffixes**

Cell kernels now use `y` and `z` for spherical coordinates and `R` and `Z` for cylindrical coordinates, matching the physical helpers and swarm implementation

Add role suffixes only when multiple locations coexist:

- `y_i`, `y_o`, `R_i`, `R_o` for inner and outer faces
- `y_up`, `z_up`, `R_up` for donor coordinates
- `R_L` and `R_R` for Riemann states if needed

The coordinate forms `R_i`, `R_o`, and `R_up` are applied

The more specific changes `h_i/h_o` to `h_gi/h_go` and `rho_mid` to `rhog_mid` remain pending

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

### 5. Local dynamical-variable names

**Status: applied in swarm; fluid alignment remains pending**

Use the physical quantity and coordinate directly:

- stored spherical state: `lx`, `vy`, `lz`
- physical spherical velocity: `vx`, `vy`, `vz`
- physical cylindrical velocity: `vR`, `vZ`
- gas linear velocity: `vx_g`, `vy_g`, `vz_g`, `vR_g`, `vZ_g`
- gas targets in stored units: `lx_g`, `vy_g`, `lz_g`

Attach later qualifiers to the complete compact base:

- `vel_x_res` to `vxres`
- `vel_z_diff` to `vzdiff`
- staged forms such as `vel_R_new` and `vel_x_new` to `vR_new` and `vx_new`

The swarm implementation now follows this convention, including `vx_cart/vy_cart` for fixed Cartesian working components

Use `_sq` and `_cb` for squared and cubed quantities rather than numeric `2` or `3` suffixes; the swarm examples are `vg_sq`, `vrel_sq`, `dist_sq`, and `max_dist_sq`

Persistent interfaces such as `dev_gas_velx`, the `swarm::velocity` member, and file-schema strings `velocity_x/y/z` remain unchanged because they are arrays, storage members, or external field labels rather than local dynamical variables

### 6. Diffusion coefficients and face quantities

**Status: canonical diffusivity names applied; detailed helper and substep names remain pending**

The diffusion kernels and velocity initializer now use descriptive lower-case base names:

- `diff_x`
- `diff_y_i`, `diff_y_o`
- `diff_z`
- `diff_z_i`, `diff_z_o`

Merge the face qualifier with the one-letter direction suffix:

- `diff_y_i/diff_y_o` to `diff_yi/diff_yo`
- `diff_z_i/diff_z_o` to `diff_zi/diff_zo`

This preserves the required spherical `x/y/z` directional convention without isolating the direction letter between underscores

Use `idx_sub` instead of `i_sub` for the diffusion-substep index

The names `dx_len`, `dr_i/dr_o`, and `dz_len` correctly distinguish physical lengths from angular `dx/dz` and logarithmic ratio `dy`

The local helpers `_get_dr_cc_i` and `_get_dr_cc_o` are understandable but `cc` is not self-explanatory outside the file

Possible replacements are `_get_dr_cent_i` and `_get_dr_cent_o`

### 7. Source-force and drag names

**Status: base force names applied; fluid time-stage and drag names remain pending**

`source_update.cu` now uses the same base force convention as the swarm transport code:

- `grav_y`
- `cent_y`
- `torq_z`

Its non-SSA stage forms currently use `_n`, `_new`, and `_tmp`; normalize them without isolating the direction letter:

- `grav_yold/ymid/ynew`
- `cent_yold/ymid/ynew`
- `torq_zold/zmid/znew`

The source variable `drag_h` is the dimensionless ratio `dt/ts`, not a scale height

Rename:

- `drag_h` to `tau`
- `drag_h2` and `drag_h3` to `tau2` and `tau3`
- `force_weight_n` to `force_weight_old`

Reserve `tau` for a dimensionless drag interval and retain `ts` for stopping time

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
- use `cfl_rates_ptr` and `lx_ptr` instead of `ptr_cfl` and `ptr_lx`

The lambda name `validate_finite_state` and kernel name `inf_cell_flag` describe their operations and can remain

### 12. Host file interfaces

**Status: `idx_file`, `file_name`, and `count` applied; macro and message-format names remain pending**

`fluid_host.cuh` now uses the same host interface convention as the swarm code:

- `frame_num(int idx_file)`
- `file_name`
- `count`

The macro parameter `IDX` should become `idx_file` to match the swarm output macros

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

## Cross-model swarm–fluid consistency audit

### Overview

The active swarm and fluid implementations already agree on most names that encode physical meaning

The remaining cross-model differences are concentrated in host initialization, drag and optical-depth base names, and binary-I/O helpers

### Recommended cross-model canonical names

Rows already identical in the swarm model, fluid model, and recommended convention are omitted

| Concept | Swarm name | Fluid name | Recommended common convention |
|---|---|---|---|
| convolved initial surface-density routine | `initdens_calc` | `convpow_calc` | `initdens_calc` |
| local dust volume density | `rhod`, `log_rhod` | `dens` | `rhod`, `log_rhod` |
| physical linear velocity | `vx/vy/vz`, `vR/vZ` | `vel_x/vel_y/vel_z`, `vel_R` | `vx/vy/vz`, `vR/vZ` |
| gas linear velocity | `vx_g/vy_g/vz_g`, `vR_g` | `vgas_x`, `vgas_R` | `vx_g/vy_g/vz_g`, `vR_g/vZ_g` |
| dimensionless drag interval | `tau_1` | `drag_h` | `tau`, retaining paper-defined SSA stage suffixes |
| scalar optical depth | `optdepth` | `tau_i`, `tau_o` | `optdepth`, with `_i/_o` for face values |
| host binary writer/reader | `save_host_binary`, `load_host_binary` | `save_binary`, `load_binary` | `save_host_binary`, `load_host_binary` |
| binary template type | `DataType` | `T` | `DataType` |
| frame/output field width | `width` | `num_len`, `len` | `width` |
| completion timestamp | `time_now` | `t_curr` | `time_now` |
| Thrust view of a rate array | `dt_rates_ptr` | `ptr_cfl` | swarm: `dt_rates_ptr`; fluid: `cfl_rates_ptr` |
| output macro frame parameter | `idx_file` | `IDX` | `idx_file` |
| local CUDA random state | `rngstate` | `rng` | `rngstate` |
| resume-frame parser | `frame_stream` | `ss` | `frame_stream` |

Swarm-side status: `initdens_calc`, `initdens`, host-loop `y/z`, `diff_x/R/Z/y/z`, binary `count`, `num_str`, `width`, `time_now`, `dt_rates_ptr`, and `frame_stream` are adopted canonical names

`cfl_rates_ptr` remains fluid-only because the swarm model has `dev_dt_rates`, not a fluid transport-only `dev_cfl_rates` array

### 14. Initial surface-density profile

The same Gaussian-convolved physical profile is currently presented as two different operations:

- swarm: `initdens_calc(std::vector<real> &initdens)`
- fluid: `convpow_calc(real *initdens)`

The former swarm name incorrectly used `_get_` for a routine that fills an output array, while the fluid name still describes the convolution method instead of the resulting physical quantity

Use `initdens_calc` for both implementations

This name:

- identifies the resulting initial dust surface-density profile
- follows the existing `*_calc` convention for output-filling routines
- avoids exposing the convolution implementation in the public name
- has 13 characters including the underscore

Use `initdens` for the output profile in both models, retaining `sigma_d` for one local physical surface-density value

The swarm host code uses `rhod` and `log_rhod` for local dust volume density, while fluid kernels still use `dens`

Use:

- `rhod` for one local dust volume density
- `log_rhod` for its logarithm
- `rhog` and `rhog_mid` for local gas volume density
- `sigma_d` and `sigma_g` for local dust and gas surface densities

The internal convolution-coordinate cleanup already recommended for the fluid code should be applied to both profile generators:

- `conv_u`, `conv_v`
- `idx_src`, `idx_dst`
- `u_src`
- `kernel_std`, `kernel_norm`, `kernel_weight`

The swarm interpolation helper `_interp_convpow_profile` and the manual interpolation in `init_rho_calc` implement the same lookup role

A later sharing pass could use one helper such as `initdens_lerp` with the same boundary and interpolation convention in both models

### 15. Coordinates and grid helpers

Device-side swarm dynamics and fluid kernels already use the common convention:

- spherical coordinates: `x`, `y`, `z`
- cylindrical coordinates: `R`, `Z`
- directional grid indices: `ix`, `iy`, `iz`
- flattened grid index: `idx_cell`

Three swarm host loops in `swarm_host.cuh` previously used `yc` and `zc` for cell-center coordinates

Those locals now use `y` and `z` to match the fluid code and the rest of the swarm code

The logarithmic radial ratio previously used `d_y` inside `swarm_grid.cuh` and now consistently uses `dy`

The helpers `_get_ycent` and `_get_zcent` have identical names and formulas, but their placement differs:

- fluid: `param_grid.cuh` as `__host__ __device__`
- swarm: `swarm_host.cuh` as host-only helpers

Move the swarm definitions to `param_grid.cuh` and give them the same host/device interface when the headers are merged

This would make the grid-helper surface identical without changing any caller names

### 17. SSA stage suffixes and physical base names

**Status: SSA suffixes and gas-target base names applied; fluid drag base name remains pending**

The swarm SSA implementation deliberately uses `_i`, `_1`, `_2`, and `_j` to match the notation of the reference paper

Retain these suffixes in all SSA-related code rather than translating them to `_old`, `_mid`, or `_new`

The same rule makes `ts_1` a consistent stage-specific form of the stopping-time base name `ts`

The swarm physical base names and compound suffix spelling are aligned:

- `lx_g1/vy_g1/lz_g1` retain stage `1` while merging it with `_g`
- `drag_h` to `tau` in the fluid source integrator
- `tau_1` remains the SSA stage-specific dimensionless drag interval

Reserve `tau` for the dimensionless drag interval and `ts` for stopping time

### 18. Optical-depth scalar names

Both models use:

- `optdepth` and `dev_optdepth` for stored host/device fields
- `beta_taper` for the radiation ramp
- `beta` for the local radiation-to-gravity ratio

The interpolated scalar in `ssa_substep_2.cu` is named `optdepth`, while the fluid source kernel calls scalar face values `tau_i` and `tau_o`

Use `optdepth` for both scalar and stored optical depth, adding `_i/_o` when face values must be distinguished

Rename the fluid face values to `optdepth_i` and `optdepth_o`

This reserves `tau` unambiguously for the dimensionless drag interval

### 19. Host binary and frame interfaces

The swarm binary helpers have the safer and more explicit interface:

- `save_host_binary`
- `load_host_binary`
- `std::size_t count`
- `const DataType *data` for writing

The fluid equivalents use the generic names `save_binary/load_binary`, an `int count`, and a non-const write pointer

Use the swarm function names and safety properties in both models

The common interface should therefore use:

```cpp
template <typename DataType>
bool save_host_binary (const std::string &file_name, const DataType *data, std::size_t count);

template <typename DataType>
bool load_host_binary (const std::string &file_name, DataType *data, std::size_t count);
```

The frame and progress helpers also calculate the same quantities with different locals

Use:

- `num_str` instead of `str`
- `width` instead of `length`, `num_len`, or `len`
- `time_now` instead of `end_time` or `t_curr`

The fluid output macros use uppercase `IDX`, whereas the swarm macros use the descriptive lowercase `idx_file`

Use `idx_file` in both

### 20. Rate-array and pointer conventions

The rate arrays have different physical scopes:

- swarm `dev_dt_rates` includes particle motion, acceleration, gas targets, and explicit diffusion
- fluid `dev_cfl_rates` contains the Eulerian transport CFL rate

The array names should remain different because `dev_cfl_rates` is not a complete analogue of `dev_dt_rates`

Their Thrust wrapper names should nevertheless follow the same object-before-type convention:

- retain `dt_rates_ptr`
- use `cfl_rates_ptr` instead of `ptr_cfl`
- use `lx_ptr` instead of `ptr_lx`

The collision-only `rate_ptr` and `max_rate` are local to a different reduction and need not be forced into the dynamics names

### 21. Gas-density naming

Both physical helper sets use `rhog` for a local gas volume density, but the fluid `_get_rhog` helper currently names its local midplane density `rho_mid`

Use `rhog_mid`

The swarm imported-gas Stokes calibration uses `rhog_0` for the analytical density anchor at `R_0`

Because that quantity is a fixed reference rather than the possibly evolved local density at index zero, `rhog_ref` is more informative than `rhog_0`

Retain `gas_dens/dev_gas_dens` for imported grid arrays unless the external-gas interface is redesigned; they are persistent field names rather than local physical scalars

### 22. Helper action prefixes

Across both models, `_get_*` is most readable when a function directly returns one value without filling an output container

Current output-mutating exceptions include:

- swarm `_get_initdens_profile`
- swarm `_get_disk_cdf`

Use an action suffix for output-filling routines:

- `initdens_calc`
- `disk_cdf_calc`

Retain `_get_*` for returned physical or geometric values such as `_get_hg`, `_get_nu`, `_get_stokes`, and `_get_vol_y`

### 23. Common parameter ordering

The shared portion of `_get_stokes` is ordered differently:

- swarm: `(R, Z, size, h_g, ...)`
- fluid: `(R, Z, h_g)`

Use the common physical prefix `(R, Z, h_g)` in both models, then append representation-specific inputs

The swarm form would become:

```cpp
_get_stokes(R, Z, h_g, size, ...)
```

This preserves the same first three arguments for every call and keeps grain size and imported-gas interpolation data visibly swarm-specific

Kernel interfaces already follow the useful convention that `dt` is the final argument; retain it

### 24. Small local conventions

The swarm stochastic kernels use `rngstate`, matching `dev_rngstate`, while `init_rho_calc` calls its local cuRAND state `rng`

Use `rngstate` for a cuRAND state and reserve `rng` for a generator object only if a shorter generic name is genuinely useful

The two main functions also give the resume-frame string stream unrelated names:

- swarm: `convert`
- fluid: `ss`

Use `frame_stream` in both

### 25. Compound suffixes after one-letter physical tags

Use an underscore before a one-letter physical tag when it is the final suffix:

- gas targets: `lx_g`, `vy_g`, `lz_g`
- directional quantities: `diff_x`, `grav_y`, `torq_z`, `vR`

If another qualifier follows, merge it with that one-letter tag rather than isolating the tag between underscores

The swarm-side compound suffixes are applied:

- gas targets: `lx_g1`, `vy_g1`, and `lz_g1`
- SSA forces: `grav_y1/y2`, `cent_y1/y2`, and `torq_z1/z2`
- diffusion-position quantities: `vR_new`, `vx_new`, `sinz_new`, and `cosz_new`
- grid-measure helpers: `_get_sy` and `_get_sz`

The remaining fluid-side identifier groups are:

- non-SSA force stages: `grav_y_n/new/tmp`, `cent_y_n/new/tmp`, and `torq_z_n/new/tmp` to merged forms such as `grav_yn/ynew/ytmp`
- face diffusivities: `diff_y_i/o` and `diff_z_i/o` to `diff_yi/yo` and `diff_zi/zo`
- qualified velocities: `vel_x_res` and `vel_z_diff` to `vxres` and `vzdiff`
- diffusion substep quantity: `inv_n_sub` to `inv_nsub`
- grid-measure helpers: `_get_s_y` and `_get_s_z` to `_get_sy` and `_get_sz`

The directional kernel and file names `advect_x/y/z_calc` and `diffus_x/y/z_calc` also match the isolated-letter pattern

Do not rename those kernels mechanically because their current names were selected to satisfy the exact 13-character kernel-name convention

Treat that conflict as a separate naming decision before changing the kernel declarations, file names, Makefile objects, or call sites

The mathematical notation `delta_v_ij` appears only in a collision comment and is not counted as a code-identifier issue

## Suggested implementation order

1. normalize index spaces and flattened-cell names — canonical portion applied
2. normalize `y/z/R/Z` coordinates and local dynamical-variable names — applied in swarm; fluid alignment pending
3. normalize force and drag names — force names applied; fluid drag name pending; retain paper-defined SSA suffixes
4. unify PPM `face` terminology and helper interfaces — face terminology applied; detailed loop names pending
5. repair semantic names for reused advection and diffusion scratch arrays
6. normalize boundary suffixes
7. normalize host arrays, file interfaces, and main-loop lambdas
8. clean up convolution and small local-loop names
9. align the swarm host coordinates and logarithmic-ratio local with the fluid convention
10. unify initial-profile routine, profile-array, and local dust-density names
11. retain resolved `diff_*` names with cylindrical `R/Z` in swarm and spherical `y/z` in fluid
12. normalize non-SSA temporal suffixes while retaining `_i/_1/_2/_j` in SSA code
13. rename fluid scalar optical depth to `optdepth`, reserve `tau` for drag, and align binary interfaces, frame locals, and pointer suffixes
14. move literally identical grid, physics, force, and optical-depth-prefix helpers into shared headers
15. merge compound suffixes after one-letter physical tags, with directional kernel names handled as a separate length-constrained decision

Each group should be applied separately with declaration/call-site scans after the change

## Resolved and intentional naming

### Applied canonical naming

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
| radial gravity | `Fy` | `grav_y` |
| radial centrifugal acceleration | `Fcy` | `cent_y` |
| polar geometric torque | `Tcz` | `torq_z` |
| PPM interface | `edge`, `face` | `face` |
| per-cell CFL array | `dev_cfl_rate` | `dev_cfl_rates` |
| PPM weights | `weight_y/z`, `dev_weight_y/z` | `ppm_weight_y/z`, `dev_ppm_weight_y/z` |
| encoded first invalid cell | `badstate`, `dev_badstate` | `bad_cell`, `dev_bad_cell` |
| file name | `fname`, `file_name` | `file_name` |
| element count | `number`, `n_cells`, `ncells` | `count`, `cell_count` as appropriate |

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

### 16. Directional diffusivity names

**Status: resolved as an intentional coordinate distinction**

The swarm diffusion SDE and its timestep estimator now use `diff_x`, `diff_R`, `diff_Z`, `diff_y`, and `diff_z`

The fluid diffusion operators currently use `diff_x`, `diff_y_i/o`, and `diff_z_i/o`

These variables all represent a physical diffusivity with dimensions of length squared per time

The two models already use `diff_*` consistently:

- swarm cylindrical coefficients: `diff_x`, `diff_R`, `diff_Z`
- swarm projections onto the spherical mesh: `diff_y`, `diff_z`
- fluid spherical direction bases: retain `diff_x`, `diff_y`, and `diff_z`, with face qualifiers merged as described in finding 25

The helper `_get_diff_drift_R` also accepts `diff_R`

The directional distinction is resolved; the remaining `_y_i/_z_i` compound-suffix spelling is tracked separately in finding 25

Retain the coordinate-letter distinction:

- uppercase `R/Z` for cylindrical directions in swarm diffusion
- lowercase `x/y/z` for spherical-coordinate fluid diffusion

The different Schmidt-number suffixes are therefore intentional:

- swarm: `SCHMIDT_X`, `SCHMIDT_R`, `SCHMIDT_Z`
- fluid: `SCHMIDT_X`, `SCHMIDT_Y`, `SCHMIDT_Z`

### Already aligned and suitable for sharing

The following names already match and should not be changed merely to distinguish the representations:

- physical constants: `G`, `M_S`, `R_0`, `SIGMA_0`, `METAL_Z`, `ASPR_0`, `IDX_P`, `IDX_Q`, `STOKES_0`
- radiation parameters: `BETA_0`, `KAPPA_0`, `T_BETA`
- common timestep parameters: `DT_OUT`, `DT_MAX`, `CFL_DYN`
- mesh parameters: `N_X/Y/Z`, `X/Y/Z_MIN/MAX`, `N_G`, `NB_G/X/Y`, `TPB`
- physical helpers: `_get_omegaK`, `_get_hg`, `_get_eta`, `_get_gas_strat`, `_get_sigma_g`, `_get_nu`, `_get_alpha`, `_get_visc_vel`
- grid helpers: `_get_dx/dy/dz`, `_get_yedge`, `_get_zedge`, `_get_mesh_dim`, `_get_vol_y`, `_get_vol_z`
- stored dust variables: `lx`, `vy`, `lz`
- local forces: `grav_y`, `cent_y`, `torq_z`
- grid fields: `dustdens/dev_dustdens`, `optdepth/dev_optdepth`
- output state: `PATH`, `idx_from`, `idx_file`, `clock_sim`, `clock_out`
- host operations: `cuda_fail`, `frame_num`, `msg_output`, `save_sam_as_velocity`, `load_velocity_as_sam`, `save_variable`
- shared radiation kernel: `optdepth_csum` with `idx_ray` and `idx_cell`

The duplicated `_get_force_term` implementations already have identical names, arguments, outputs, and formulas and are a direct candidate for a shared physics header

The common portions of `param_grid.cuh` and `param_phys.cuh` are also candidates for literal sharing after representation-specific helpers are separated

### Intentional cross-model differences

Do not unify the following names because they expose real representation or algorithm differences:

- swarm particle index `idx` versus fluid grid index `idx_cell`
- `particle/dev_particle` versus Eulerian `dustdens`, `dustvel*`, and `dustmom*`
- swarm cylindrical diffusion suffixes `R/Z` versus fluid spherical suffixes `y/z`
- swarm `dev_dt_rates` versus fluid `dev_cfl_rates`
- swarm `particle_init` versus fluid `init_rho_calc` and `init_vel_calc`
- swarm `diffusion_pos` versus fluid `diffus_x/y/z_calc`
- swarm `total_dust_mass`, `mass_norm`, `size`, and represented grain number, which have no single-species fluid counterparts
- swarm `vZ` for physical settling versus fluid `vzdiff` for a spherical-polar diffusion-balance velocity
- fluid `mx/my/mz`, which are conserved density-weighted fields and have no particle-state counterparts
- fluid `RHO_VAC` and `POS_LIMIT`, which belong to Eulerian state recovery and implicit diffusion

### Intentional distinctions to retain

The following differences carry physical or numerical information and should not be unified:

- `rhod/rhog` are dust/gas volume densities while `sigma_d/sigma_g` are dust/gas surface densities
- `h_g` is the dimensionless gas aspect ratio while `H_g` and `H_d` are physical scale heights
- `lx` and `lz` are specific angular momenta while `vy` is a linear spherical-radial velocity
- `mx` and `mz` are angular-momentum densities while `my` is linear-momentum density
- `vx` and `vz` are local physical linear velocities while persistent file fields retain `velocity_x` and `velocity_z`
- outside SSA, `_i/_o` denotes inner/outer geometry, `_L/_R` denotes Riemann left/right states, and `_old/_new` denotes time states
- inside SSA, `_i/_1/_2/_j` retains the paper-defined integration-stage notation
- `dx` and `dz` are angular coordinate increments, `dy` is a logarithmic radial ratio, and `dx_len/dr/dz_len` are physical lengths
- `cn_lower/diag/upper` are matrix coefficients while `flux_*` quantities are physical finite-volume fluxes
- `optdepth`, `optdepth_i`, and `optdepth_o` are optical depths while `tau` is the dimensionless drag interval

### Existing names suitable for future sharing

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

Both models expose matching optical-depth stage names:

- `optdepth_calc`
- `optdepth_csum`

Only `optdepth_csum` currently has the same interface and calculation and is directly shareable

`optdepth_calc` has representation-specific inputs and preconditions:

- swarm converts a deposited extinction measure in place
- fluid constructs the increment directly from `dev_dustdens`

Retain branch-specific implementations unless a common input contract is introduced
