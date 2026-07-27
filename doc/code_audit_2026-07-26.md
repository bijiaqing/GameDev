# Merged fluid–swarm code audit: `inc/`, `src/`, `tst/`

Date: 2026-07-26

Scope: line-by-line review of the merged tree — `inc/fluid/`, `inc/swarm/`, `src/fluid/`,
`src/swarm/`, `src/share/`, the root `Makefile`, and the complete `tst/fluid/` verification
harness (CUDA driver, Python validator, runners, mock CPU checks, recorded outputs). The vendored
`inc/swarm/cukd/` library was checked only at its call sites. `legacy/` and `mod/` were treated as
frozen/configuration material and not audited. The review asked: (1) whether the calculations are
numerically, mathematically, and physically correct; (2) whether the renaming rework introduced
code errors; (3) whether code and `doc/` agree; (4) whether the tests are complete and test the
right things.

Method: source inspection plus independent algebraic re-derivation of every numerical formula
(PPM, HLL, FARGO, SSPRK, Crank–Nicolson, source quadrature, diffusion SDE, collision rates,
optical depth). No CUDA compiler is available on this machine; runtime claims rely on the recorded
artifacts under `tst/fluid/out/`, per the authority order in `doc/README.md`. Selected recorded
metrics were spot-checked against `doc/fluid_verification.md` and all agreed (85 JSON records, 22
`environment.txt` files, the documented step sequences, and mass-conservation figures). At the
time of the audit, the source-case validator still showed its previously documented `2.22e-5`
reference discrepancy; the follow-up below resolved it.

## Executive summary

The merged tree is in excellent shape. Both branches implement their documented numerics
faithfully, the renaming rework is consistent at every declaration/definition/call site I
checked, and the canonical documents describe the code accurately. The fluid verification suite
is genuinely analytical (finite-volume reference solutions, independent eigenvalue solvers),
exercises the production kernels rather than copies, and its recorded results pass.

No new correctness defects were found in production code. The substantive observations are:

- three small code/documentation mismatches found during the audit and resolved afterward (F1–F3);
- one previously recorded validator-side defect resolved afterward (F4);
- coverage gaps in an otherwise strong suite, mostly already acknowledged in
  `doc/fluid_verification.md` (section D);
- one acute performance caveat for the fiducial fluid grid that the documents mention but
  arguably understate (F5).

## F. Findings

### F1. (doc, trivial, resolved) Stale swarm path in `doc/swarm_numerics.md`

The scope section says the swarm implementation is "under the root `inc/` and `src/`
directories"; it lives under `inc/swarm/` and `src/swarm/`. Cosmetic, but this is the canonical
document a new reader opens first.

Resolution: the scope now names `inc/swarm/` and `src/swarm/` explicitly.

### F2. (doc, minor, resolved) The FARGO residual-CFL sentence is imprecise

`doc/fluid_numerics.md` states: "The CFL calculation uses the same ring mean and therefore
bounds the velocity actually seen by the residual solver." What the residual solver actually
advects is the displacement relative to the *nearest-integer shift frame*, not the ring mean:

- the CFL kernel bounds `|Ω − Ω̄|/dx` with `Ω̄` the ring arithmetic mean
  (`src/fluid/cfl_rate_calc.cu`);
- `advection_xth` then rounds `Ω̄ dt/dx` to the nearest integer `shift_count`, so each cell's
  true residual fraction is `|(Ω − Ω̄) + (Ω̄ − Ω_shift)| dt/dx ≤ CFL_DYN + 0.5`, i.e. up to 1.0.

The PPM tracing uses the true per-cell fraction, so the scheme always stays inside its one-cell
domain of dependence — the design is correct and safe, but the document overstates what the CFL
bound covers. A one-sentence clarification would close the gap.

Resolution: `doc/fluid_numerics.md` now states the separate ring-mean and half-cell frame bounds
and the resulting `CFL_DYN + 0.5 <= 1` tracing bound.

### F3. (code, minor, resolved) `IMPORTGAS` + `CONST_ST` was silently meaningless (swarm)

In `inc/swarm/param_phys.cuh`, the `CONST_ST` branch sits inside the analytic-gas `#else` of
`_get_stokes`, so an `IMPORTGAS`+`CONST_ST` build compiles but ignores `CONST_ST` and uses the
imported-density Stokes everywhere. This matches the documentation ("every dynamically evaluated
local Stokes number uses the imported gas density"), so it is consistent rather than wrong — but
the flag combination is now a no-op and would be better served by a compile-time rejection in
`swarm_kern.cuh` alongside the other combination guards, or an explicit sentence in
`swarm_numerics.md`.

Resolution: `inc/swarm/swarm_kern.cuh` now rejects `IMPORTGAS + CONST_ST` at compile time because
the imported density must determine the dynamically evaluated local Stokes number.

### F4. (validator, resolved) `source_exact` cancellation in `validate_case.py`

At audit time, `tst/fluid/verify_common/validate_case.py` computed the source reference with the
naive forms `1.0 - decay` and `ts*time - ts*ts*(1.0 - decay)`. At stiffness `h = 1e-6` these
cancelled to roughly the recorded `2.22e-5` componentwise maximum (`momz linf = 2.2206e-5`). The
production CUDA kernel already used the stable small-`h` series and was correct; the defect was
confined to the Python evaluator.

Resolution: the validator now mirrors the CUDA endpoint-force weights, including the cubic
small-$h$ series. Reanalysis of the unchanged native source output reduces the maximum error from
`2.2206e-5` to `2.2204e-16`, so the source case now passes.

### F5. (performance, understated in docs) Thread-local line arrays at the fiducial grid

`doc/fluid_numerics.md` lists the line-sized thread-local arrays as "can spill heavily at large
3D resolution". At the *current* fiducial 2D grid (`N_X = N_Y = 1024`) the situation is already
acute: `advection_xth` declares 21 line arrays of length `N_X`, corresponding to a 172 KB
source-level array footprint per thread if all storage remains distinct. The diffusion kernels
also declare several line arrays. Compiler lifetime analysis may overlap non-simultaneous arrays,
so neither the actual local stack size nor the performance impact can be established from source
counting alone. The verification grids (≤256) hide this concern. This is not a correctness issue —
every kernel is index-guarded and the recorded tests pass — but the first production-scale fluid
run on `mod/fluid_fiducial` should record `ptxas` resource usage and profiler local-memory traffic
before the block-parallel line-kernel work is prioritized.

#### F5 implementation proposal

Status: both implementations now exist under `src/fluid` and are selected at compile time with
`FLUID_SWEEP := thread` or `FLUID_SWEEP := block`. The block method assigns one block to each
directional line, reuses an explicit 11-field global advection workspace, keeps the verified serial
face-limiter order within each block, and moves diffusion line work into dynamic shared memory.
Native comparisons established numerical equivalence, a speedup at `1024^2`, and a slowdown at
`128^3`, so the model configuration must select the implementation appropriate to its grid.

1. Establish the real baseline with `ptxas` stack/spill reporting and Nsight Compute measurements
   for every line kernel at the fiducial resolution. Source-level array totals are not a substitute
   for compiler resource usage and local-memory traffic.
2. Allocate reusable representation-level scratch fields once in `fluid_runtime.cu`. A handful of
   full-grid work arrays is preferable to replicating a large line workspace for every CUDA thread,
   and avoids repeated allocation inside an operator step.
3. Convert advection to one block per line, with threads cooperatively handling cells. Perform the
   ring-mean reduction, circular FARGO shift, primitive recovery, face reconstruction, and flux
   divergence in separate synchronized passes. Tile only the short PPM halo in shared memory and
   keep larger intermediate fields in the reusable global workspace.
4. Treat the invariant-domain correction as the numerically delicate part. The initial parallel
   implementation may retain a serial correction pass per line to preserve the verified update.
   A later face-parallel flux-corrected-transport limiter should use per-cell admissible budgets and
   one shared coefficient per face so conservation and primitive bounds remain explicit.
5. Replace the serial Thomas and Sherman–Morrison work with batched block-parallel tridiagonal
   solves. Nonperiodic Y/Z systems can use a batched Thomas/PCR hybrid; periodic X diffusion retains
   the two tridiagonal solves and rank-one Sherman–Morrison correction, parallelized across rings
   and line cells. FFT is unnecessary.
6. Land one directional operator at a time. Require the complete 85-case fluid suite, the strong
   positivity-subcycling mock, mass conservation, and production-resolution profiler results after
   each replacement. Parallel reductions need tolerance-based rather than bitwise comparison
   because summation order will change.

## A. Verified correct (fluid branch)

All formulas below were re-derived independently and match the code.

- **Mesh and measures** (`inc/fluid/param_grid.cuh`): exact radial measure
  `y0^d(dy^d−1)/d`, polar measure `cos z_i − cos z_o`, area factor `y^{d−1}`, volume coordinates
  `y^d/d` and `−cos z` used by the PPM weight construction.
- **PPM** (`inc/fluid/_transport.cuh`, `inc/fluid/fluid_host.cuh`): the uniform 4-point bounded
  face value is the Colella–Woodward form; the nonuniform weights are solved from exact cubic
  moment constraints in the finite-volume coordinate (Gauss–Jordan with partial pivoting,
  boundary-adjacent faces fall back to two-cell linear weights); the parabola limiter and the
  upwind domain-of-dependence integrals are the standard CW84 forms; MOL stages correctly use
  zero tracing fraction.
- **HLL** (`inc/fluid/_transport.cuh`): correct two-wave pressureless flux with proper
  single-sided branches.
- **FARGO** (`src/fluid/advection_xth.cu`): ring-mean specific angular momentum, nearest-integer
  circular shift of *both* conserved and primitive states, residual-frame PPM transport with
  true per-cell tracing fractions; integer shifts are exact permutations.
- **Radial/polar advection** (`src/fluid/advection_yth.cu`, `advection_zth.cu`): exact
  conservative geometry (`A = y^{d−1}`, `V = Δy^d/d`; `sin z` faces, `Δ(−cos z)` volumes, the
  extra `1/y` in the polar operator); outflow-only boundary fluxes with inflow suppression;
  `HALFDISK` zero midplane flux; SSPRK(3,3) staging coefficients (3/4+1/4, 1/3+2/3) correct;
  the invariant-domain limiter enforces linear constraints in (ρ, m) with one shared scale per
  face, which preserves conservation exactly.
- **Crank–Nicolson diffusion** (`src/fluid/diffus_[xyz]_calc.cu`): matrix coefficients match
  the FV form in all three geometries; zero-flux physical boundaries; positivity subcycling
  implements the correct sufficient condition (`cn_i + cn_o ≤ POS_LIMIT < 1` per substep);
  Sherman–Morrison cyclic reduction is textbook (corner split, two Thomas solves, rank-one
  correction); the time-centred mass flux reconstructed from the CN coefficients is *identical*
  to the flux implied by the matrix update, so the donor-state momentum transport is exactly
  conservative — the documented provisional momentum closure is implemented as specified.
- **Source** (`src/fluid/source_update.cu`): exponential relaxation exact for frozen `t_s`;
  the old/new force weights equal the analytic linear-force quadrature in both the closed form
  and the `h < 1e-4` series (I verified all four series coefficients against the exact
  integrals); sequential `l_φ → l_θ → v_r` with force re-evaluation matches the doc; the
  optical-depth center rule `1/(√dy + 1)` is exact for τ linear in `y`.
- **Driver** (`src/fluid/fluid_runtime.cu`): the palindromic composition matches
  `doc/fluid_numerics.md` exactly; directional advection is subcycled with a fresh global CFL
  reduction before every launch; optical depth is reconstructed at the Strang midpoint;
  output-due clipping lands exactly on `DT_OUT`; resume rebuilds conserved momenta and optical
  depth correctly; non-finite-state validation is wired in at the documented points.
- **Initialization**: convolved surface profile matches the swarm's host copy; the 3D Gaussian
  embedding uses the density-diffusion `H_d = H_g√(α_z/St_mid)`; the perturbation is one
  reproducible Gaussian amplitude per azimuthal column (`curand_init(ix, 0, 0)`), as documented;
  the polar diffusion-balance velocity equals the settling-velocity projection at equilibrium
  (`D_z Z sin z / H_d² = St·Ω·Z sin z` when `H_d² = D_z/(St·Ω)`), so the 3D initial state is the
  correct discrete equilibrium up to the documented small transient.
- **`src/share/`**: `optdepth_calc` implements the well-mixed 2D closure `Σ_d/(√(2π)H_g)` with
  `Δτ = κ ρ dr` (cell width exact); `optdepth_csum` is the correct per-ray prefix sum.

## B. Verified correct (swarm branch)

The swarm branch is the previously audited code with mechanical renames; I re-checked every file
and confirmed both the renames and the follow-up fixes claimed in `doc/swarm_numerics.md`:

- `rand_from_file` now uses `_get_dx()` and centers an inactive azimuth (the A1 build defect is
  gone);
- `int_pow` computes in `real`, removing the 32-bit overflow for `LOGTIMING`;
- `_get_ball_measure` multiplies the reduced-dimensional measure by `2πR` for `N_X == 1`,
  restoring dimensional correctness of the physical kernel in axisymmetric models;
- the additive kernel is now exactly `K = m_i + m_j` (comment and code agree, doc agrees);
- the 2D collision vertical overlap and the 2D opacity closure both use the gas scale height
  `H_g` only (well-mixed), matching the cross-representation contract and the fluid side;
- `IMPORTGAS` Stokes evaluation is anchored to the analytic `Σ_0, H_{g,0}` reference midplane
  with a hard assert on the density pointer; `particle_init` receives and uses the imported
  density, while initial gas velocity/pressure support remain analytic — exactly as documented;
- `rngstate_<frame>.dat` is saved beside every particle checkpoint and restored on resume
  (`save_particle_data`/`load_particle_data`), so resumed stochastic runs continue the same
  streams;
- `get_total_dust_mass()` replaces the independent dust-mass parameter and is used by opacity,
  deposition, and the collision `lambda_0` normalization;
- the Makefile builds with `-std=c++17` and selects exactly one representation via `DUST_REPR`;
  the swarm build correctly omits `src/share/` (its own optical-depth kernels are used), and the
  fluid build includes them.

All SSA transport, diffusion SDE, timestep-rate, collision propensity/event, deposition, and
interpolation formulas remain as previously verified (spherical force terms, cylindrical Itô
drift, Cartesian-velocity transport across diffusion, frozen-snapshot Bernoulli leap, outer-face
optical-depth interpolation with the physical zero at the inner face).

## C. Naming-rework consistency

No rename-induced defects found. Spot-verified across declaration/definition/call sites in both
branches and the test harness: `_get_yface/_get_zface`, `_get_sy/_get_sz`,
`initdens_calc/initdens_lerp`, `save_host_binary/load_host_binary`,
`save_particle_data/load_particle_data`, `get_total_dust_mass`, `REYNOLDS_0`,
`randpos*/randsize`, `gas_blend`, `cell_measure`, `idx_ray/idx_ring/idx_col`, `idx_cell`,
`lx/vy/lz` primitive conventions, `grav_y/cent_y/torq_z`, `term_R/term_Z`, `diff_*` Schmidt
suffixes (`SCHMIDT_X/Y/Z` fluid vs `SCHMIDT_X/R/Z` swarm), and the kernel/file names. The
`PARAM_GRID_CUH`/`PARAM_PHYS_CUH` guard collision between `inc/fluid/` and `inc/swarm/`
(documented in `doc/architecture.md`) is harmless under the one-branch-per-build Makefile
(verified: only one representation include directory is passed per build).

## D. Test-suite assessment

### What is genuinely tested (and how well)

The suite is analytically honest: initial data are exact finite-volume cell averages (8-point
Gauss–Legendre in CUDA, 16-point in Python), radial diffusion eigenmodes come from an
independent RK4/Bessel solution rather than the CUDA matrix, the mock suite adds dense-matrix
and RK4-shooting references, and the validator mirrors the production index order, weights norms
by the true cell measures, masks velocity norms on analytical vacuum, and records environment
and step counts. Parameter sweeps cover integer and fractional FARGO shifts, two CFL values for
radial/polar transport, all three diffusion directions with their correct Jacobians, eight drag
stiffnesses over twelve orders of magnitude, three optical-depth powers, and four coupled ring
equilibria — 85 records, all present under `tst/fluid/out/` and consistent with
`doc/fluid_verification.md` (spot-checked: FARGO orders and roundoff mass, radial step sequences
`3, 5, 9, 16`, and ring mass roundoff). The corrected source reference now agrees at roundoff.

Component-swap coverage is better than it first appears: transverse momenta use *distinct*
constants (0.7, −0.15, 0.11), so an `momx/momz` transposition or index-order error would be
caught.

### Coverage gaps (ordered by importance)

1. **No swarm tests at all** (`tst/swarm/` is empty). Every swarm correction since 2026-07-22 —
   including the `2πR` axisymmetric measure, the new `H_g` overlap/opacity closures, the
   IMPORTGAS Stokes anchor, and rngstate restart — is currently validated only by source
   inspection. This is the largest validation asymmetry between the branches and is documented
   in both `architecture.md` and `swarm_numerics.md`.
2. **Boundary flux paths are exercised only trivially.** All transport tests keep their compact
   support away from the boundaries, so the outflow and inflow-suppression branches of
   `advect_y/z_calc` run only with vacuum/zero material. A manufactured case that actually
   crosses a boundary (and a `HALFDISK` midplane-reflection case) would close this. The docs
   already list explicit boundary regressions as pending; this note just underlines that the
   *flux formulas themselves* are currently unverified at the boundary.
3. **Finite attenuation is untested on CUDA.** Ring radiation cases use `κ = 0` by design, so
   `exp(−τ)` with a nonzero, spatially varying τ — including the `_interp_optdepth` center rule
   — has no CUDA coverage (mock T6 covers the quadrature on CPU). Correctly assigned to the
   planned 3D MMS suite in the docs.
4. **No restart-tolerance or flag-matrix regression.** The verification harness always starts
   fresh; the fluid resume path (`LOAD_DUSTDATA_TO_VRAM` + `momentum_setv` + optical-depth
   rebuild) is untested. Also documented as pending.
5. **CN positivity subcycling** has no CUDA-level case that actually triggers `sub_count > 1`;
   only mock T1d covers it on CPU. A single large-`dt` diffusion case would suffice.
6. **`--use_fast_math` sensitivity** is not measured (documented as pending).

The planned-but-unbuilt `verify_mms_3d` specification is sound: SymPy-derived residuals,
generated forcing at substep times, exact boundary updates, four flag configurations including
zero-diffusivity MMS variants under enabled `DIFFUSION`, and an independent validator. Nothing
in the current tree contradicts it.

### Validator status

- F4 (source evaluator cancellation) is resolved and the archived native output now agrees at
  roundoff.
- No other validator/production divergence found: the ring gas profile (`R^{2−4β}`), the
  `η = β/2` equilibrium, the radiation balance, and the diffusion damping rates in
  `validate_case.py` all match the production physics helpers algebraically.

## E. Follow-up priorities

F1–F4 are complete. Remaining priorities are:

1. Create `tst/swarm/` with at minimum: orbit/stiff-drag single-particle tests, a cylindrical
   diffusion Green's-function test, an optical-depth/β attenuation test, and a collision
   constant-kernel test — the four areas where the swarm branch has the freshest unverified
   corrections.
2. Add one boundary-crossing manufactured case and one restart-tolerance case to `tst/fluid/`.
3. Before the first `fluid_fiducial` production run, record compiler resource usage and profiler
   local-memory measurements for `advection_xth` at `N_X = 1024`, then decide whether the
   block-parallel line kernels should land first (F5).

## Review limitations

- No CUDA compilation or execution was possible here; all runtime evidence is the recorded
  `tst/fluid/out/` artifact set, which I verified for internal consistency and against the
  documents but did not regenerate. The F4 follow-up reanalyzed the unchanged native source output.
- `inc/swarm/cukd/` internals were not re-audited (unchanged since the previous audit).
- `mod/fluid_fiducial` and `mod/swarm_fiducial` inherit their representation `const_defs.cuh`
  defaults; they were sanity-checked but not audited as physical configurations.
