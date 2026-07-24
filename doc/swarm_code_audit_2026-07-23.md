# Swarm code audit: `inc/`, `src/`, and root `Makefile`

Date: 2026-07-23

Scope: line-by-line review of every file in `inc/` (excluding the vendored `inc/cukd/` library, whose API usage was checked but whose internals were not re-audited), every file in `src/`, and the root `Makefile`. The audit follows up on `.github/swarm_numerical_audit.md` (2026-07-22), whose 31 findings are marked resolved; the code has since been reorganized (`helpers_*.cuh`/`graffiti_*.cuh` → `_*.cuh`, `swarm_*.cuh`, `param_*.cuh`, `const_defs.cuh`), so this review re-derives the current state from source rather than trusting the earlier document.

The review focuses on (1) obvious code bugs, (2) physical and dimensional correctness of the calculations, and (3) consistency between comments and code. `mod/`, `legacy/`, and `new/` trees were excluded except where they determine whether a defect is active in a current configuration.

This machine has no CUDA compiler, so findings were established by source inspection and algebraic checks. Two external cross-checks were performed: the Ormel–Cuzzi turbulent relative-velocity implementation was compared term by term against the referenced DustPy source (`dustpy/std/dust.f90` at commit 48e6c05), and the `double3` memory layout was verified against NVIDIA's `vector_types.h` (CUDA 8.0).

## Status vocabulary

- **confirmed**: inconsistent without requiring an additional modeling assumption
- **conditional**: a real defect only when the named feature/configuration is enabled; inactive in current production models
- **order limitation**: internally meaningful but of lower formal order than one might expect
- **modeling decision**: the code does not define enough physics to select one uniquely correct answer
- **note**: not a correctness defect; robustness, portability, or documentation item

## Executive summary

The code is in good shape. The 2026-07-22 corrections are all present and correctly implemented: the cylindrical diffusion SDE has the right Itô drift, metric factors, and Cartesian-velocity transport; the SSA transport force terms are dimensionally correct; the collision path uses the frozen-snapshot race-free update with the exact Bernoulli probability and a controlled `CFL_COL` leap; the optical-depth chain is dimensionally consistent in both 2D and 3D; and the timestep controller covers mesh crossing, orbital frequency, acceleration displacement, and diffusion displacement.

The newly identified A1 and B1--B4 issues were corrected on 2026-07-24: imported-gas azimuth sampling now uses the mesh helper and locks an inactive azimuth, logarithmic output powers accumulate directly in `real`, axisymmetric physical collision neighborhoods include their `2πR` ring closure, the linear test uses the standard additive kernel, and the Makefile explicitly selects C++17.

The Ormel–Cuzzi regimes, the SSA drag/force solve, the diffusion SDE, the collision-rate normalization, and the opacity machinery were re-derived algebraically and are correct; details are in section E so future reviews can skip re-deriving them.

## A. Confirmed defects

### A1. **[resolved 2026-07-24] `rand_from_file` used the undeclared identifier `dx`

[`inc/swarm_host.cuh:645`](../inc/swarm_host.cuh#L645):

```cpp
pos_x[i] = X_MIN + dx*(static_cast<real>(idx_x) + random(rand_generator));
```

No `dx` (variable, macro, or global) exists anywhere in scope or in the included headers; the mesh-spacing helper is the function `_get_dx()`. Every `IMPORTGAS` configuration fails to compile at this line. This is the same class of configuration-dependent build defect as the previously fixed missing comma (2026-07-22 finding 29), and it is invisible to non-`IMPORTGAS` builds because the whole function is `#ifdef IMPORTGAS`.

For `N_X == 1`, the same line would also sample an artificial azimuthal distribution instead of locking the inactive coordinate. This matters in collision-only imported-gas models because no transport step recenters those particles.

**Resolution, 2026-07-24:** active azimuths use `_get_dx()` to sample within the selected cell, while `N_X == 1` places every particle at `0.5*(X_MIN+X_MAX)`.

## B. Conditional / latent defects

### B1. **[resolved 2026-07-24] `int_pow` overflowed 32-bit `int` for `LOGTIMING`

[`inc/swarm_host.cuh:697`](../inc/swarm_host.cuh#L697) accumulates `LOG_BASE^exp` in an `int`. With `LOG_BASE = 10` the result exceeds `INT_MAX` at frame index 10, producing signed-overflow garbage in `_get_dt_out` and in the resume-time `clock_sim`. It is used only under `LOGTIMING`, and the only current `LOGTIMING` model (`mod/test_colli_const`) uses `SAVE_MAX = 8`, so it is safe today but breaks silently if a log-timed run extends past `LOG_BASE^9`. Computing the power in `long long` or directly in `real` removes the class.

A related (even more remote) note: `N_T = 3*N_P` in [`inc/const_defs.cuh:194`](../inc/const_defs.cuh#L194) overflows `int` if `N_P` is ever raised above ~7.1e8 for wedge models.

**Resolution, 2026-07-24:** `int_pow` now accumulates directly in `real`, so no signed-integer overflow occurs before conversion to the simulation time type.

### B2. **[resolved 2026-07-24] The physical (`CUSTOM`) collision kernel lacked azimuthal closure when `N_X == 1`

For `N_X == 1` all particles are locked to a single azimuth, so `_get_ball_measure` ([`inc/_collision.cuh:51`](../inc/_collision.cuh#L51)) returned a 1D length (`N_Z == 1`) or a 2D area (`N_Z > 1`). The physical kernel rate `N_j σ Δv / measure` then had dimensions `L/T` instead of `1/T`: the `N_Z == 1` branch of `_get_col_rate_ij` supplies the vertical Gaussian-overlap divisor, but nothing supplied the azimuthal ring closure.

**Resolution, 2026-07-24:** after applying the local radial and polar boundary fractions, `_get_ball_measure` multiplies every reduced-dimensional axisymmetric neighborhood by `2πR`, with `R=y sin(z)`. A radial interval therefore represents a ring area, and a radial-polar neighborhood represents a ring volume. `COLLISION_UNIT_VOLUME` continues to replace this physical measure for dimensionless tests.

### B3. **[resolved 2026-07-24] Linear test kernel differed by a factor of two from its documented definition

[`inc/_collision.cuh:344-345`](../inc/_collision.cuh#L344) documented `K=m_i+m_j` but returned `0.5*(m_i+m_j)`, rescaling the collision time by two.

**Resolution, 2026-07-24:** the factor `0.5` was removed, selecting the standard additive kernel `K=m_i+m_j`. Future analytical validation must use this same time normalization.

### B4. **[resolved 2026-07-24] The root `Makefile` did not pin the C++ standard

The code uses structured bindings (`inc/swarm_grid.cuh`, `src/col_rate_calc.cu`), `if constexpr` (`inc/_collision.cuh`), and `std::filesystem` (`src/graffiti_main.cu`) — all C++17. Relying on the compiler default makes compatibility dependent on the CUDA and host toolchain.

**Resolution, 2026-07-24:** the root Makefile now adds `-std=c++17` to the common `NVCC` configuration used for compilation and linking.

## C. Documentation and comment inconsistencies

### C1. The 2026-07-22 audit's finding-26 resolution no longer describes the code

The document states that imported-gas models "recycle exits at both radial boundaries to just inside `Y_MAX` with a midplane Keplerian state". The current `_apply_transport_boundary` ([`inc/_transport.cuh:79`](../inc/_transport.cuh#L79)) contains no `IMPORTGAS` branch and **absorbs** particles at both radial boundaries for all models (confirmed against the git history of this file). The code is self-consistent — absorption is a valid explicit boundary choice — but the earlier document should not be read as a description of current behavior, and `IMPORTGAS` users should be aware that particles leaving the gas mesh are now permanently removed rather than recycled.

### C2. `mod/*/flags.mk` comments vs. compile-time behavior

The flag headers state "If TRANSPORT is off, RADIATION and DIFFUSION are inactive regardless of their flags". [`inc/swarm_kern.cuh:4-10`](../inc/swarm_kern.cuh#L4) instead aborts compilation with `#error`. The code behavior is the better one; the comments mislead. (`mod/` is outside the requested scope; noted for completeness.)

### C3. **[resolved 2026-07-24] Smaller cleanup findings

- `inc/swarm_host.cuh` retained the dead `rand_gaussian` and `rand_convpow` samplers and the private convolution helpers used only by `rand_convpow`
- `col_rate_calc` widened the single-precision KNN squared distance to `real` and then passed it back through `sqrtf`

**Resolution, 2026-07-24:** the unused samplers and their dead helper chain were removed from the active header; the independent copies under `legacy/` remain untouched. Collision radius conversion now uses the `real` overload of `sqrt`. The unused `_get_ball_measure` azimuth argument had already been removed with the B2 correction.

## D. Minor observations (not defects under current configurations)

- **Zero-initial-mass edge case**: with `METAL_Z = 0` (or a fully tapered-out profile), `get_dust_mass`, `rand_disk_mono`, `_get_disk_cdf`, and `rand_from_file` divide by a zero total mass and produce NaN CDFs. Only a misconfiguration path, but a guard with a clear error message would help.
- **Resume RNG**: `rngstate_init` uses a fixed seed at both fresh start and resume, so a resumed run's diffusion/collision random sequence restarts from the beginning rather than continuing the uninterrupted stream. Statistically harmless; worth knowing when comparing continuous and resumed runs.
- **`N_K = 1` unguarded**: `lambda_0` divides by `N_K - 1`. `N_K = 200` by default; only a misconfiguration note.
- **Boundary asymmetry (deliberate)**: transport absorbs at radial/polar exits while diffusion reflects (zero diffusive flux). This is a defensible combination, but it should be documented in the model description since it means particles can leave advectively but not diffusively.
- **Polar singularity**: force and velocity conversions contain `1/sin(z)`, and `_ssa_substep_1` divides by `sin(z_i)sin(z_1)`; configurations need `Z_MIN > 0` (true for all current models). Relatedly, the initialization's cylindrical-R profile support correctly leaves a vacuum wedge near the pole for `R < Y_MIN`; particles can later enter that region dynamically where no disk was initialized — geometrically consistent, worth a sentence in the docs.
- **`optdepth_depo` under `IMPORTGAS`** (conditional): its 2D vertical closure computes `stokes_mid` from the analytic `Σ_g ∝ R^p` scaling, while `_get_stokes` elsewhere uses the imported density. Only relevant for the untested `IMPORTGAS`+`RADIATION`+`DIFFUSION` combination.
- **`--use_fast_math`**: enables FTZ for floats and slightly-less-precise double division/sqrt. The KD-tree is already single-precision by design; physics impact is negligible, but bit-level reproducibility across compilers should not be expected.
- **Host vs. device cell-center convention**: host samplers evaluate cell densities at the geometric midpoint `_get_ycent` while device interpolation places cell values at the measure centroid `m_y` of `swarm_grid.cuh`. Both are first-order-consistent quadrature choices; the difference is O(cell width²) and immaterial.
- **fp-roundoff fallback in partner selection** (`src/col_event_run.cu:81-90`): if the re-summed cumulative propensity undershoots the sampled target by roundoff (probability ~1e-16 per event), the last valid candidate is used. Negligible and arguably preferable to dropping the event.

## E. Verified correct (with the key algebra, for future reviewers)

### E1. Mesh measures and interpolation (`inc/param_grid.cuh`, `inc/swarm_grid.cuh`)

- `_get_vol_y = y0^d(dy^d-1)/d` with `d = 2 + (N_Z>1)` is the exact annular/spherical radial measure; `_get_vol_z = cos(z0)-cos(z1)` is the exact polar measure; `_get_vol_x` returns `dx` or `2π` for an inactive azimuth, matching `get_dust_mass`.
- The cell-centroid locator `m_y = ln[(d/(d+1))(dy^{d+1}-1)/(dy^d-1)]/ln(dy)` is exactly the measure-weighted centroid of the cell in log coordinates, and the weights `(dy^{deci-m}-1)/(dy-1)` and `(...)/(1/dy-1)` hit 0/1 at the correct neighboring centroids; interpolation is linear in physical `y` between centroids, with constant extrapolation at the domain edges.
- Optical-depth outer-face weights `(dy - dy^deci)/(dy-1)` give the face value at `deci=1` and the previous cell's face value at `deci=0`; the first-cell factor `value *= 1 - frac_y` interpolates from physical zero at the inner face (the finding-13 fix), and `loc_y < 0 → 0`, out-of-bounds → `DBL_MAX` (radiation suppression) behave as documented.
- Azimuthal weights wrap periodically with the correct linear weights at both seams; polar weights clamp to the last cell at boundaries. No out-of-bounds stencil offset is reachable (the `HALFDISK` `loc_z` clamp lands in the already-handled edge branch).

### E2. Transport and SSA integrator (`inc/_transport.cuh`, `src/ssa_*.cu`)

- Stored-state conventions `position=(φ,r,θ)`, `velocity=(l_φ,v_r,l_θ)` are used consistently everywhere; `_get_force_term` terms check out: `F_y = -(1-β)GM/r²`, `Fc_y = l_φ²/(R²r) + l_θ²/r³` (spherical radial centrifugal), `Tc_z = l_φ²cosθ/(R²sinθ)` (spherical polar torque from `dl_θ/dt = v_φ²cotθ`).
- Pressure support: with `a = p - q/2 - 3/2`, direct expansion of centrifugal balance with the exact vertically isothermal stratification gives `v_φ²/v_K² = 1 + h²(a+q) + q(1 - R/r) + O(h⁴, (Z²/R²)²)`; the code's `sqrt(1-2η)` with `η = -0.5((p+q/2-3/2)h² + q(1-R/r))` is exactly this expression — the midplane limit `v_φ² = v_K²(1 + (a+q)h²)` reproduces the standard sub-Keplerian result.
- The staggered position updates are exact integrals of the coordinate velocities for linearly varying `r(t)`: `∫dt/r² = Δt/(y_i y_1)` (used for both `z` and `x`); the `1/(sin z_i sin z_1)` factor in the azimuthal update is the corresponding second-order approximation of `∫dt/sin²θ`.
- The drag step is an exact frozen-coefficient exponential integrator of `dv/dt = (v_g + F t_s - v)/t_s` at the midpoint, with a two-stage midpoint-momentum force re-evaluation (correct `t_s → ∞` limit `v_j ≈ v_i + F dt + (v_g - v_i)dt/t_s`).
- `particle_init.cu` reproduces the Nakagawa steady drift with the correct signs (re-derived): `v_R = (v_R,g + 2St(v_φ,g - v_K))/(1+St²)` (inward for sub-Keplerian gas), `v_φ,d = v_φ,g - St·v_R/2`, settling `v_Z = -St·Ω_K·Z`; cylindrical→spherical projections are correct.
- Boundary handling: periodic azimuth (locked to center for `N_X==1`), midplane reflection for `HALFDISK` (`z → π - z`, `l_z → -l_z`), absorption elsewhere; absorbed particles (`y=0`) are consistently skipped by every kernel, deposition, and the timestep reduction.
- `_get_visc_vel` (Kanagawa-style): re-derived from the steady angular-momentum equation including the vertical-shear viscous term; `stress_R = 3ν(dlnν/dlnR + dlnρ/dlnR + 1/2)`, `stress_Z = qν(1 + dlnρ/dlnZ)`, and the 2D limit `-3ν(dlnν/dlnR + dlnΣ/dlnR + 1/2)/R` all check out, as do the stratification-gradient factors `±f(1-f²)/h²` and `-(q+1)·strat`.

### E3. Diffusion (`inc/_diffusion.cuh`, `src/diffusion_pos.cu`)

- The cylindrical density-diffusion Itô drift `∂_R D + D/R = D(idx_diff + 1)/R` with `idx_diff = dlnν/dlnR` (0 for `CONST_NU`, `q+3/2` for α-viscosity) is correct for both the 2D surface-density and 3D volume-density interpretations of the split operator.
- Azimuthal `std = √(2Ddt)/R` carries the correct `1/R` metric; radial and vertical increments are physical lengths; the deterministic drift enters only the mean (the finding-3 error is gone).
- Negative-`R` remapping (`R → -R`, `φ → φ+π`), subsequent reflection at the domain boundaries, and re-projection of the preserved Cartesian velocity into the new local basis are all consistent with "spatial redistribution without impulse" (finding 6).
- The reflecting boundary loop converges for arbitrarily large kicks (alternating fold reflections).

### E4. Runtime timestep (`src/dt_rates_calc.cu`)

Every rate was checked dimensionally: orbital `Ω/CFL`; azimuthal crossing `|l_φ|/(R² dx CFL)`; radial `|v_r|/(dr CFL)` with the exact logarithmic cell width `dr = y_edge(dy-1)`; polar `|l_θ|/(y² dz CFL)`; imported-gas crossing at both snapshots; constant-acceleration displacement `√(a/(2·CFL·dr))` with the *net* radial force (correct cancellation for near-Keplerian orbits) and polar acceleration `|Tc_z|/y`; diffusion RMS `2D/(CFL²·Δ²)` and drift `|drift|/(CFL·Δ)` in every active direction, with the cylindrical diffusion tensor correctly projected onto spherical coordinates (`D_R sin²θ + D_Z cos²θ`, etc.). The radiation force bound uses the unattenuated `β` with the `INIT_SMIN` fragmentation floor — conservative and correct. The host-side `fmin(DT_MAX, ...)` ladder and the `max_rate = 0` (all-absorbed) degenerate case behave correctly.

### E5. Radiation and optical depth (`src/optdepth_*.cu`, `src/ssa_substep_2.cu`)

- Extinction weight: `m(s)·N_swarm·κ_0/(s/S_0)` is an area (`κ∝1/s` geometric); the 2D branch divides by each particle's own `√(2π)H_d` with `H_d = H_g√((α/Sc_Z)/St_mid)`, then `optdepth_calc`'s `(weight/area)·dr` is dimensionless for a mixed-size population (finding 12 stays fixed). The equal-area `_get_mass_weight` makes the per-swarm extinction exactly size-independent, matching the `idx_swarm = -1.5` sampler (verified `E[weight] = 1` analytically).
- The cumulative scan stores `τ` at outer faces from the inner boundary; interpolation against it (E1) is consistent, and the first cell rises from physical zero.
- `β = taper·β_0·e^{-τ}/(s/S_0)` with the smoothstep taper; attenuation and size scalings are correct; out-of-mesh `τ = DBL_MAX` cleanly suppresses the force.
- `sizeof(swarm)` question settled: CUDA's `vector_types.h` defines `double3` with **no** alignment attribute, so `sizeof(double3) = 24` and `sizeof(swarm) = 48` (64 with `MULTISIZE`) — the binary particle records exactly match the `[SWARM_DTYPE]` metadata (finding 30 does not recur).

### E6. Diffusion/transport/collision coupling (`src/graffiti_main.cu`)

Half collision / half diffusion / full transport / half diffusion / half collision is proper Strang splitting (finding 25). The `IMPORTGAS` incremental blend `blend = (target - frac)/(1 - frac)` reproduces exact linear interpolation in time with an exact-snapshot reset at each output boundary (finding 27). Collision-only and log-timed paths are consistent (`_get_dt_out` and the resume `clock_sim` agree within the `int_pow` range, B1).

### E7. Collisions (`inc/_collision.cuh`, `src/col_*.cu`)

- **Ormel–Cuzzi regimes**: all six branches of `_get_vrel_t` are algebraically identical to DustPy's `vrel_ormel_cuzzi_2007` (verified term by term, including the `St/(1+Re^{-1/2}/St) = St²/(St+Re^{-1/2})` rewrite in regime 2, the `h1+h2` combination in regime 3, the regime-4/5 forms, and the partial-fraction identity `1/(1+St_l)+1/(1+St_s) = (2+St_l+St_s)/(1+St_l+St_s+St_lSt_s)` in regime 6). The `y_s` polynomial coefficients, `v_gas² = 1.5αc_s²`, and `Re = αΣσ_H2/(2μm_H)` all match; regime ordering is exhaustive and gap-free for any `Re`.
- **Brownian speed** `√(8c_s²μm_H(m_i+m_j)/(πm_im_j))` with the `c_s` cap matches DustPy.
- **Physical kernel**: `N_j·π(s_i+s_j)²/4·Δv` (diameter cross-section, consistent with `S_0`/`par_size` being diameters and `_get_grain_mass = πρs³/6`); the `N_Z==1` vertical overlap `√(2π(H_i²+H_j²))` is exactly the Gaussian `∫ρ_iρ_j dz` factor; each particle's Stokes number uses its own `h_g(R)` (finding 23). Rates are inverse-time in 2D (after overlap) and 3D.
- **Dimensionless kernels**: with `lambda_0 = N_P/((N_K-1)M_D)` and unit volume, the expected representative rate is `λ_i = Σ_all N_j K_ij/M_D` and the total number decay is `dN/dt = -N²/(2M_D)` for the constant kernel — exactly Smoluchowski with `K/V = 1/M_D`, including the 1/2 (verified algebraically). The `V = M_D` test convention is internally consistent (B3 only concerns the linear kernel's extra 1/2).
- **Event machinery**: `1 - exp(-λdt)` Bernoulli probability with `dt ≤ CFL_COL/max λ` (finding 17); partner selection by exact inverse sampling of the frozen pair propensities; the read-only `dev_size_old`/`dev_numr_old` snapshot eliminates the partner-read race while positions/velocities are untouched by collisions (finding 20); KNN candidate lists are deterministic between the rate and event kernels, so the two cumulative sums agree.
- **Outcomes**: coagulation `s_k = (s_i³+s_j³)^{1/3}` with `N_i' = N_i s_i³/s_k³` conserves representative mass exactly and is the correct Zsom–Dullemond asymmetric update; fragmentation samples a specific `s_k = s_coag·u²` law (pdf `∝ s^{-1/2}`) with probability below `INIT_SMIN` piled onto a spike at the floor — a documented modeling choice, mass-conserving but not a standard fragment distribution.
- **KNN geometry**: periodic wedge images (`col_tree_init`) plus accessible-fraction ball caps near radial/polar boundaries (finding 22); zero-radius neighborhoods give zero rate; self and self-images are excluded; the `N_X==1` dimensional gap is B2.
- **Reynolds-number scalings**: `Re = Re_0(α/α_0)(Σ/Σ_0)` (code units) is the correct rescaling of the physical formula.

### E8. Initialization (`inc/swarm_host.cuh`, `src/particle_init.cu`)

- The joint `(y,z)` sampler draws cells from the exact `density·vol_y·vol_z` mass CDF and samples uniformly in `y^d` and `cos(z)` inside the cell — the exact Jacobians, applied once (findings 7, 10); `rand_from_file` uses the same construction on imported `ρ_g·ε` (subject to A1).
- `rand_disk_poly` conditions the vertical settling on each particle's size via `H_d(R) = H_g√(α_z/St_mid(R))` with `St_mid(R) = St_0(s/S_0)(R/R_0)^{-p}` — the density-diffusion settling equilibrium (finding 8) — and the log-mass CDF tabulation with max-log shifting is robust for strongly settled distributions.
- `rand_gamma_k2` inverts `F(x) = 1 - e^{-x}(1+x)` via the `W_{-1}` branch — a correct gamma(k=2) sampler for the linear-kernel test.
- `get_mass_norm`/`_get_grain_number` normalize the finite sample to `dust_mass` exactly, and `get_dust_mass` integrates the same tapered profile that the samplers draw from (consistent `M_D` = domain-mass convention, finding 11).
- Initial azimuthal angular momentum is cylindrical Keplerian, `l_φ = √(GMR)` with `R = y sin(z)` (finding 9), and velocities are stored as angular momenta while file I/O converts to/from linear velocities symmetrically.

## F. Suggested actions (in priority order)

1. Add an `IMPORTGAS` smoke build covering both active and inactive azimuths so A1 cannot regress.
2. Validate the B2 `2πR` closure with a uniform axisymmetric physical-kernel setup and compare it against an azimuthally resolved realization.
3. Restore the analytical constant, additive, and product kernel tests using the B3 additive-kernel normalization.
4. Update `.github/swarm_numerical_audit.md` finding 26 to describe the current absorbing radial boundary (C1), and the `flags.mk` comments (C2).
5. When a CUDA machine is available, build every supported flag combination and run the analytical/convergence battery; none of the source-level findings substitutes for those runs.

## Review limitations

- No CUDA compilation or execution was possible on this machine; all findings are from source inspection and algebra.
- `inc/cukd/` internals were not re-audited; only the call signatures (`HeapCandidateList` semantics, `cct::knn`, `buildTree`, traits) were checked against use.
- The collision test reference solutions live outside this repository, so B3 could not be fully closed.
- The `mod/` tree was read only to judge whether conditional defects are active; it was not audited.
