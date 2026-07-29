# Swarm verification-suite audit: `qav/swarm/` and `doc/testset_swarm.md` (2026-07-29)

Date: 2026-07-29

Scope: review of the newly prepared swarm verification suite — `doc/testset_swarm.md`,
`doc/numerics_swarm.md`, `doc/README.md`, the complete `qav/swarm/test_common/` harness
(`verification_main.cu`, `const_defs.cuh`, `validate_case.py`, `run_model.py`, `run_suite.py`,
`TEST_CASES.md`), all ten test models' `flags.mk` and `const_defs.cuh` shims, the Makefile's
`qav/` routing, and the new `PR_EFFECT` production code in `inc/swarm/_transport.cuh` that the
radiation tests exercise. The review asked: (1) whether the swarm tests build correct test
models, (2) whether their expected results are correct, and (3) whether the suite is complete
while remaining feasible.

Method: source inspection, independent algebraic re-derivation of every analytical and
statistical expectation, and direct execution of the suspect Python validator path (no CUDA
compiler is available on this machine; the reproduced failure is a pure-Python defect).

## Executive summary

The suite is well designed: every measured operation comes from production kernels or production
device helpers, the analytical preconditions are set up exactly, every independent expectation I
re-derived is correct, and the documented coverage boundaries are honest and complete. **One
fatal Python defect currently prevents the suite from producing any result**: a scoping bug in
`validate_case.py` raises `NameError` on the first grid validation and aborts the whole matrix.
It is a two-line mechanical fix. The P–R drag production code the suite depends on is correct
against its documentation.

## Resolution status after the audit

The following test-only corrections were applied on 2026-07-29 after this audit was delivered:

- F1 is fixed by keeping `norms()` independent and calculating `mass_relative_change` inside
  `analyze_grid()`, where `density` and `cell_volume` are defined
- grid mass conservation is included explicitly in the pass boolean
- the 3D collision case now checks a polar cap as well as the interior ball and radial cap
- the diffusion documentation now explains that the $N=32$ and $N=64$ ensembles cannot resolve
  the small radial Itô drift independently
- the compact 2D diffusion description now states that $2D\Delta t/R^2$ is angular variance and
  that the prepared case evaluates it at $R=1$
- the first native build exposed an additional feature-guard mismatch in `particle_init.cu`:
  `_get_grain_number()` depended on the multisize-only `_get_mass_weight()` despite being compiled
  for monodisperse models; the helper is now correctly enclosed by `#ifdef MULTISIZE`
- the first collision build exposed an include-order error in the test driver: it tested the
  model-local `TEST_COLLISION_*` selector before `const_defs.cuh` had defined it; the production
  collision header is now included under the compiler-provided `COLLISION` physics flag

Pure-Python synthetic grid and 3D collision outputs pass the corrected validator. Native CUDA
compilation and execution are still pending, so the audit's evidence limitation remains in force.

## F1. (confirmed, blocking) `norms()` / `mass_relative_change` scoping bug in the validator

`qav/swarm/test_common/validate_case.py`:

```python
def norms(values):
    mass_relative_change = float(np.sum(density*cell_volume) - 1.0)   # density, cell_volume
    ...                                                                # are module-undefined
```

- `norms()` references `density` and `cell_volume` as free variables. They exist only as locals
  inside `analyze_grid`; at module scope they are undefined, so **every call to `norms()` raises
  `NameError`**. Reproduced locally: `validate_case.norms(np.array([1.0, -2.0]))` →
  `NameError: name 'density' is not defined`.
- `analyze_grid` then returns `"mass_relative_change": mass_relative_change`, which is defined
  only inside `norms()` — a second `NameError` site, reached even if the first were fixed.

Effect: the grid validator crashes on the first model (`test_grid_2d`), and because
`run_suite.py` invokes each runner with `check=True`, the entire 25-build matrix aborts there.
No metrics JSON can currently be produced by any grid case; the other case analyzers never run
through the suite either (grid is first in `GROUPS`).

Fix (not applied, per instructions): compute the deposited mass inside `analyze_grid` — which
has both `density` and `cell_volume` in scope — and let `norms()` return only `l1/l2/linf`.
The intended check itself is correct: with unit total dust mass the requirement is
`abs(sum(density*cell_volume) - 1.0) < 5e-12`, matching `doc/testset_swarm.md`.

Consistency evidence that this suite has never completed: `obj/test_grid_2d/swarm/` and
`qav/swarm/out/test_grid_2d/swarm/` exist but are empty, and no `metrics_*.json` is archived —
consistent with the docs' "prepared, not passed" status and with this defect being present since
the harness was assembled.

## A. Test models are built correctly

- `verification_main.cu` exercises **production code for every measured operation**:
  - grid: `dustdens_init/depo/calc`, `optdepth_init/depo/calc/csum/mean` with unit total mass
    (`total_dust_mass = 1.0`), particles placed exactly at the production interpolation
    centroids (the same `ref_y` formula as `inc/swarm/swarm_grid.cuh`), so each deposits unit
    weight to its intended cell;
  - orbit: the complete non-radiative `ssa_transport` over `VERIFY_RES` equal steps;
  - drag/radiation/P–R: `ssa_transport` and the `ssa_substep_1/2` split with a zeroed optical
    depth and `beta_taper = 1.0`, eight grain sizes `0.05·2^i` spanning exactly
    `[INIT_SMIN, INIT_SMAX]`;
  - diffusion: one `diffusion_pos` step with `rngstate_init(dev_rngstate, 17)` (the documented
    seed) and velocities encoding the cylindrical `(0.7, 0.2, 0)` initial Cartesian state;
  - collision: a test-local one-thread kernel that only **calls** the production helpers
    `_get_ball_measure` and `_get_col_rate_ij` with the documented interior point
    `(y=1.0, a=0.2)` and radial-boundary point `(Y_MIN + a/4)`.
- `const_defs.cuh` sets every analytical precondition exactly: `IDX_P=2, IDX_Q=-1` gives `η=0`
  (gas = dust = `v_K` for the orbit and drag tests); `CONST_ST` with `STOKES_0=0.2` gives
  `St = 0.2·s`; `C_LIGHT=25`; `NU=0.02` under `CONST_NU`; Schmidt switches isolate exactly one
  diffusion direction per model; the `N_P`/`N` relations match the doc's resolution table
  (2D grid `N²`, 3D grid `4N²`, diffusion `16N²`, orbit 64, algebra cases 8 and 2).
- All `flags.mk` combinations satisfy the compile-time guards (`COLLISION→MULTISIZE`,
  `DIFFUSION→TRANSPORT`, `PR_EFFECT→RADIATION`, no `IMPORTGAS`+`CONST_ST`). The collision models
  define `CODE_UNIT` so no physical-unit constants are required.
- The Makefile routes `qav/swarm` models correctly (`MODEL_MATCHES`, `MODEL_INCLUDE_DIRS`, and
  `OUT_DIR = qav/swarm/out/<model>`), and the `TEST_RES` passthrough works as in the fluid suite.
- The new `PR_EFFECT` production implementation (`inc/swarm/_transport.cuh:221-277`) is correct
  against `numerics_swarm.md`: `k_x = k_z = t_s^{-1} + γ_PR`, `k_y = t_s^{-1} + 2γ_PR`,
  `γ_PR = β·GM/(C_LIGHT·y²)`, gas targets entering only through `u_g/t_s` (damping toward the
  stellar rest frame), `expm1` throughout, and the `PR_EFFECT requires RADIATION` guard.

## B. Expected results are correct (independently re-derived)

- **Grid**: centroid placement = production `ref_y`; expected density `m_p/ΔV` with exact
  measures (`dx·(y_o^d−y_i^d)/d·(cos z_i−cos z_o)`); the 2D well-mixed extinction closure
  `κ m_p/(√(2π)·H_g)` with `H_g = ASPR_0·R` (constant `h_g` since `IDX_Q=−1`) matches
  `optdepth_depo`; 3D uses the bare `κ m_p` weight; the cumulative sum is the correct inclusive
  outer-face prefix; `optdepth_mean` is idempotent on the uniform rings by construction.
- **Orbit**: `p=2, q=−1 ⇒ η=0 ⇒ v_φ,gas = v_K = v_φ,dust`, so drag vanishes identically and the
  only error is the SSA trajectory error; wrapping the azimuthal error to `[−π, π)` is correct;
  the reported refinement orders are temporal (`Δt = 2π/N`) and correctly carry **no pass
  threshold** beyond the `L∞ < 0.5` failure guard, as documented.
- **Drag/radiation/P–R**: the validator's frozen-coefficient solutions match the code's exact
  SSA algebra in both branches — `e^{-kh}` decay, `(1−e^{-kh})/k` response with gas target
  `u_g/t_s`, midpoint force `F_{y,1} = −(1−β) + ℓ_{x,1}²` re-evaluated with the midpoint
  angular momentum, and `y_j = 1 + ½ v_{y,j} Δt`; `γ_PR = β/25` at `y=1` matches
  `β·GM/(C_LIGHT·y²)`; the validator uses `expm1` consistently, so its `2e-13` tolerance is
  appropriate.
- **Diffusion**: `Var = 2DΔt = 8×10⁻⁴`, `E[ΔR] = DΔt/R = 4×10⁻⁴` (the `CONST_NU` Itô drift),
  `E[Δx]=E[ΔZ]=0`; the Cartesian-velocity-invariance checks reconstruct exactly the vector
  `diffusion_pos` preserves (verified against the kernel's projection formulas in 2D and 3D);
  the six-standard-error bands use the correct Gaussian factors (`σ/√N` for the mean,
  `σ²√(2/(N−1))` for the variance), and assessing by moments rather than trajectories is the
  right statistical design.
- **Collision**: interior measures (`πa²`, `4πa³/3`), circular-cap and spherical-cap
  subtractions, and the three kernel numerators (`λ_0N_j`, `λ_0N_j(m_i+m_j)`,
  `λ_0N_j m_i m_j` with the `πρ s³/6` grain masses) match `_get_ball_measure` and
  `_get_col_rate_ij` exactly. These cases are genuine regression tests: they would have caught
  the historical additive-kernel ½-factor and the missing `2πR` axisymmetric measure.

## C. Completeness and feasibility

- **Feasibility is good**: 25 small clean builds (15 with `--quick`), five selective groups,
  `--build-only`, per-run environment capture, and an independent validator. The documented
  build counts match the harness exactly (`FIXED_RESOLUTION` covers the five
  resolution-independent algebra cases).
- **Coverage is well aimed** at previously buggy or unverified machinery: exact deposition and
  optical-depth geometry, full SSA transport (orbit plus stiff responses including the new P–R
  algebra), the cylindrical diffusion SDE with Itô drift and Cartesian-velocity transport, and
  the collision measure/kernel algebra.
- **The documented remaining gaps are the right ones** and are consistent between
  `testset_swarm.md`, `TEST_CASES.md`, and `numerics_swarm.md`: KD-tree neighbor identities
  versus brute force (periodic images, wedge seams), event-level coagulation/Smoluchowski
  evolution with `CFL_COL` convergence, initialization CDFs and mass normalization, imported
  gas, boundary absorption/reflection statistics, bitwise restart including cuRAND state,
  secular reduced-gravity and P–R inspiral, settling equilibrium, `dyn_rate_calc` rates, finite
  attenuation `βe^{−τ}`, coupled operator tests, and a fast-math-off rerun. These are
  correctly presented as missing evidence, not passing claims.

Two completeness notes beyond the documents:

1. The 3D collision case exercises only the **radial** cap of `_get_ball_measure`; the polar-cap
   approximation `y·(z − Z_bound)` is currently untested (one additional query point near a polar
   boundary would cover it at negligible cost).
2. **Statistical power is resolution-skewed by design**: the Itô-drift signal `4×10⁻⁴` exceeds
   the 6σ mean band only for `N ≳ 128` (at `N=32` the band is `≈1.3×10⁻³`, three times the
   signal), and the variance precision tightens from ≈6.6% (`N=32`) to ≈0.8% (`N=256`). The
   `N=32/64` diffusion results should therefore not be quoted as evidence for the drift term —
   worth one sentence in `testset_swarm.md`.

Doc nit: `TEST_CASES.md` states the 2D diffusion variance as `2DΔt/R²`; that is the *angular*
variance, which coincides with `2DΔt` only at `R=1`. `testset_swarm.md`'s form
(`Δx = √(2DΔt)·ξ` at `R=1`) is the precise one.

## D. Recommendations (priority order)

1. Fix F1 (two-line scoping fix in `validate_case.py`), then run
   `python3 qav/swarm/test_common/run_suite.py --group all --quick` natively as a smoke test
   before the full matrix.
2. Add the one-sentence statistical-power note to `testset_swarm.md` (C2) and optionally the
   polar-cap query point to the collision kernel (C1).
3. Archive the resulting `metrics_*.json`, `environment.txt`, and full terminal output under
   `qav/swarm/out/` so the suite graduates from "prepared" to "passed" evidence, then tackle the
   documented gap list starting with the KD-tree brute-force comparison and the bitwise
   restart-reproducibility test (the two cheapest high-value additions).

## Review limitations

- No CUDA compilation or execution was possible here; the analytical and statistical
  expectations were verified algebraically, and the blocking defect was reproduced in pure
  Python.
- The fluid suite under `qav/fluid/` was not re-reviewed in this pass (only its relocation and
  the README's new authority order were checked).
- `doc/future_bestknn.md` and `doc/future_amdgpus.md` were noted as design notes for a future
  collision-neighborhood optimization and were not part of this review.
