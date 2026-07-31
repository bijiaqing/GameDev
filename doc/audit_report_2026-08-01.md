# Static code audit — GameDev protoplanetary-disk dust code

**Date:** 2026-08-01
**Auditor scope:** read-only inspection of `inc/`, `src/`, `mod/`, and `qav/`. No source line was
modified and no build was run. All findings below are derived from reading the current source and
comparing it against the canonical documents in `doc/`, the retained test definitions, and
independent standard references.

The audit covered the Eulerian dust-fluid branch (`inc/fluid/`, `src/fluid/`), the Lagrangian
representative-particle branch (`inc/swarm/`, `src/swarm/`), the collision neighbor-search
backends as exercised by `qav/swarm/test_knn/`, and the verification design in `qav/`.

## 1. Executive summary

No crash-level, indexing, or gross conservation defect was found in the audited kernels. The
pressureless PPM/HLL transport, the FARGO frame, the Crank–Nicolson diffusion, the exponential
drag source, the staggered semi-analytic swarm transport, the stochastic diffusion, the
Poynting–Robertson damping, the optical-depth construction, and the collision rate/event pipeline
are internally consistent, and each implemented formula checked against standard references
(FARGO/PPM/HLL, Nakagawa–Sekiya–Hayashi drift, Kanagawa viscous flow, Ormel–Cuzzi 2007 turbulent
velocities, DustPy) was found to be a faithful transcription.

A subsequent independent re-derivation corrected the original headline `_get_eta` finding. The
implemented expression is the correct midplane pressure-support closure for `N_Z == 1`, matching
the documented vertically integrated density with midplane dynamics, and is also the exact 3D
rotation law for the code's vertically isothermal hydrostatic density. The apparently missing
height-dependent pressure term cancels the corresponding weakening of cylindrical stellar
gravity (§2.1).

The vacuum-clipping item was a documentation discrepancy and has been resolved (§2.2). The
finite-3D polydisperse mass issue was narrower than first stated and is now also corrected: the
target domain mass and representative weights both include the size-dependent vertical
containment fraction (§2.3). The remaining items are numerical tests that either do not exist yet
or are weaker than the physics they are meant to guard (§4). Most missing verification is already
acknowledged in `doc/testset_fluid.md` and `doc/testset_swarm.md`; §4 additionally identifies gaps
that are not yet recorded there.

---

## 2. Detailed findings

### 2.1 `_get_eta` — pressure-support parameter (both branches, corrected false positive)

Location:
- `inc/fluid/param_phys.cuh:18–20`
- `inc/swarm/param_phys.cuh:50–52`

```cpp
real _get_eta (real R, real Z, real h_g)
{ return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z))); }
```

It is used to set the pressure-supported gas rotation
$v_{\phi,g}=v_K\sqrt{1-2\eta}$ in
`src/fluid/init_vel_calc.cu`, `src/fluid/source_update.cu`,
`src/swarm/particle_init.cu`, and `inc/swarm/_transport.cuh` (`_ssa_substep_2`), so every
initial-condition, drift, and drag computation in both branches inherits its value.

#### (a) 2D (`N_Z == 1`) midplane closure

At the midplane ($Z=0$) the formula reduces to

$$
\eta_{\rm code}^{2D}=\eta_{\rm code}^{\rm mid}
=-\frac{h_g^2}{2}\left(p+\frac{q}{2}-\frac32\right).
$$

Because

$$
\rho_{g,\rm mid}=\frac{\Sigma_g}{\sqrt{2\pi}H_g},
\qquad
H_g\propto R^{(q+3)/2},
$$

the midplane pressure satisfies

$$
\frac{d\ln P_{\rm mid}}{d\ln R}
=p+\frac q2-\frac32.
$$

GameDev deliberately evolves surface density while using gas drag, gravity, radiation, and
particle dynamics at the midplane when `N_Z == 1`. The implemented expression is therefore the
correct closure. The alternative $-(h_g^2/2)(p+q)$ follows from a vertically averaged gas momentum
equation with column pressure $\Sigma_g c_s^2$; adopting it would change the model rather than fix
the present one. The well-mixed unresolved dust profile determines the density/opacity closure and
does not require a column-averaged gas orbital velocity. The previously claimed factor-of-1.9 drift
error is consequently not present.

#### (b) Exact 3D height dependence

Writing $u=R/\sqrt{R^2+Z^2}=\sin\theta$, the exact fixed-height pressure gradient of the
vertically isothermal disk gives (direct differentiation of
$\rho_g\propto R^{p}H_g^{-1}\exp[(u-1)/h_g^2]$, $c_s^2\propto R^{q}$)

$$
h_g^2\frac{\partial\ln P}{\partial\ln R}
=\left(p+\frac q2-\frac32\right)h_g^2
+q(1-u)+(1-u^3).
$$

The cylindrical component of stellar gravity contributes $u^3$ in units of the midplane
Keplerian gravity. Radial force balance therefore gives

$$
\frac{v_{\phi,g}^2}{v_K^2}
=u^3+\left(p+\frac q2-\frac32\right)h_g^2
+q(1-u)+(1-u^3)
=1+\left(p+\frac q2-\frac32\right)h_g^2+q(1-u)
=1-2\eta_{\rm code}.
$$

The $(1-u^3)$ term identified in the original audit is real in the pressure gradient, but it
cancels the $u^3$ weakening of radial gravity exactly. Treating these as two independent missing
terms caused the original false positive. The current 3D `_get_eta` is exact for the assumed
vertically isothermal hydrostatic density in a point-mass potential.

**Conclusion:** no source correction is required. The derivation should be retained in the
canonical numerical documentation, and independent nonzero-$(p,q)$ tests remain useful regression
coverage for both the midplane and 3D formulas.

### 2.2 Post-update vacuum clipping documentation (resolved)

- Location: `src/fluid/advection_yth.cu:153` (and the identical clip in
  `advection_xth.cu`, `advection_zth.cu`, and the `*bl` block versions).

The code retains this final safety fallback:

```cpp
if (rhod[iy] < 0.0) rhod[iy] = mx[iy] = my[iy] = mz[iy] = 0.0;
```

The invariant-domain limiter should make the fallback inactive under its CFL assumptions. It is
nevertheless appropriate to convert any residual negative floating-point density into an exact
vacuum so that it cannot generate an undefined velocity. `doc/numerics_fluid.md` now states that
the ordinary limited update is conservative and that the zero-density/zero-momentum operation is a
final floating-point safety fallback rather than part of the normal update. No source correction
is required.

### 2.3 Finite-3D multisize domain mass used the reference grain size (resolved in source)

- Location: `inc/swarm/swarm_host.cuh:161` (`get_total_dust_mass`).

The original size-proposal weight was already correct: $w(s)=1$ for the full-column equal-mass
proposal and $w(s)\propto s$ for the full-column equal-area proposal used with radiation. Together
with their corresponding sampled size distributions, both reproduce the intended physical
$dN/ds\propto s^{-3.5}$ spectrum. Density and opacity deposition use represented physical mass,
not an unweighted representative count.

The omitted factor was the size-dependent mass contained by the finite domain. The old
`get_total_dust_mass()` evaluated the
domain integral using `_get_init_rhod(sigma_d,R,Z,S_0)`. This was correct for a monodisperse model
and had no size dependence when `N_Z == 1`, where `_get_init_rhod` returns the surface density
directly. It was a convention-dependent approximation only in a finite 3D polydisperse domain,
because the fraction of each size's settled vertical profile lying inside the polar boundaries
depends on size.

The corrected code directly tabulates

$$
I(s)=\int_{\mathcal D}\rho_d(\boldsymbol{x}\mid s)\,dV,
\qquad
M_{\rm domain}=\int f_M(s)I(s)\,ds,
$$

where $f_M(s)$ is the normalized full-column mass spectrum. It evaluates $I(s)$ on the same
logarithmic size axis as the conditional spatial CDF, integrates $f_MI$ over size, and uses
finite-sample raw weights

$$
\widetilde W_i=w(s_i)I(s_i),
\qquad
W_i=M_{\rm domain}\frac{\widetilde W_i}{\sum_j\widetilde W_j},
\qquad
N_i=\frac{W_i}{m_g(s_i)}.
$$

The finite normalization makes $\sum_iW_i=M_{\rm domain}$ to roundoff. Without radiation,
$w=1$; with radiation, $w\propto s$. This reduces exactly to the previous normalization for
monodisperse, vertically integrated, imported-common-profile, and vertically complete domains.

The first correction still evaluated the vertical Gaussian at simulation-cell centers. That made
both the containment weights and conditional sampler internally consistent but resolution-dependent
when $H_d\ll y\Delta z$. The active correction instead intersects a fixed cylindrical radius with
the spherical radial-polar domain, integrates the resulting one or two truncated-Gaussian intervals
analytically, samples height by inverse Gaussian CDF, and tabulates only the remaining radial
marginal on an auxiliary grid independent of `N_Z`. The broad-size finite-polar-domain regression
described in T2b is now implemented as `test_initial_3d`, including an intermediate-size population
that exercises interpolation between adjacent size knots; native CUDA execution is pending.

The correction also exposed an independent prefactor error in the old domain integral: it summed
only over radial and polar cells but multiplied by one azimuthal cell width when `N_X > 1`. The
revised integral uses $N_X\Delta x=X_{\max}-X_{\min}$, while the axisymmetric `N_X == 1` branch
retains its complete $2\pi$ extent.

### 2.4 Other observations (informational)

- **FARGO residual at the PPM stability boundary.** With `CFL_DYN = 0.5` the residual PPM tracing
  fraction can reach exactly 1.0 cell (documented in `numerics_fluid.md`). This is legal for PPM
  but sits at the stability limit; a small safety margin (e.g. 0.45) would be safer for
  production runs with strong velocity shear.
- **3D `VISC_FLOW` branch is unverified.** Only the vertically-integrated 2D branch of
  `_get_visc_vel` (`-3ν(∂lnν+ p+1/2)/R`) has an analytical test (`test_viscflow_1d`). The 3D
  branch is a transcription of Kanagawa et al. (2017) with the radial gradient verified by hand in
  this audit, but the vertical-shear term is not covered by any test and should be checked against
  the paper.
- **`--use_fast_math`** reproducibility caveat is already documented; it is repeated here only
  because it affects every tolerance claim.

---

## 3. Verified-correct inventory (inspected and checked against references)

The following were read in full and found mathematically/numerically consistent (this is a
statement about the inspected code, not a substitute for the missing runtime cases in §4):

**Fluid (`inc/fluid`, `src/fluid`)**
- SSPRK(3,3) stage composition in all three `advection_*th` kernels (weights $1,\tfrac34,\tfrac13$
  correct).
- Bounded PPM face reconstruction (uniform 4-cell and nonuniform cubic-moment weights, two-cell
  boundary stencils) and the standard PPM monotonicity/upwind-tracing formulas.
- Pressureless HLL flux: upwind branches, vacuum branch (zero flux), and the compression branch all
  consistent with the exact pressureless Riemann structure.
- FARGO: nearest-integer ring shift, frame subtraction, residual angular-velocity transport in the
  rotating frame (the missing-$R$ concern resolves to the correct rotating-frame form), and
  periodic-shift indexing.
- Radial/polar finite-volume geometry: $dV_y=y^d dy/d$, $dV_z=\cos z_i-\cos z_o$ measures and
  face-area factors used consistently in the updates and CFL rates.
- Crank–Nicolson diffusion in all three directions: coefficients, Thomas / Sherman–Morrison
  solves, zero-flux and periodic boundary conditions, POS_LIMIT positivity subcycling, and the
  time-centred diffusive mass flux with donor-cell momentum transport (verified factor-for-factor
  for the azimuthal $1/R^2$ metric).
- Source operator: exact exponential drag relaxation, the drag-weighted old/new force quadrature
  (series coefficients $1/2-\tau/3+\tau^2/8-\tau^3/30$ and
  $1/2-\tau/6+\tau^2/24-\tau^3/120$ verified), centrifugal $\ell_\phi^2/(R^2y)+\ell_\theta^2/y^3$,
  polar torque $\ell_\phi^2\cos\theta/(R^2\sin\theta)$, and sequential $\ell_\phi\to\ell_\theta\to
  v_y$ force re-evaluation.
- Optical-depth quadrature ($\kappa\rho\,dr$), radial prefix sum, and the exact
  linear-in-$R$ cell-centre interpolation factor $1/(\sqrt{\Delta y}+1)$.
- Viscous radial velocity 2D branch and 3D radial-gradient term.

**Swarm (`inc/swarm`, `src/swarm`)**
- Staggered semi-analytic transport: midpoint drift with exact $\int dt/y^2=dt/(y_iy_1)$ for
  linear radial motion, half/full exponential drag response, and the documented
  $(\ell_\phi,v_r,\ell_\theta)$ storage.
- Poynting–Robertson rates: $\gamma=\beta GM_\star/(c y^2)$ on $\ell_\phi,\ell_\theta$ and
  $2\gamma$ on $v_y$, matching Burns–Lamy–Soter first-order form.
- Cylindrical stochastic diffusion: Itô and variable-diffusivity drifts
  ($D_R(q+5/2)/R$), $R^{-1}$ azimuthal noise, Cartesian-velocity preservation and reprojection,
  cylindrical→spherical mapping, and repeated-reflection boundaries.
- Ormel–Cuzzi (2007) turbulent relative velocity: **all six regimes, the $y_s$ polynomial, and
  the Reynolds-number closures (`Re=0.5\alpha\Sigma\sigma_{H_2}/M_{\rm mol}` physical,
  `Re=Re_0(\alpha/\alpha_0)(\Sigma/\Sigma_0)` code-unit) were compared line-by-line with the
  current DustPy Fortran source (`stammler/dustpy/dustpy/std/dust.f90`,
  `vrel_ormel_cuzzi_2007`) and agree exactly.
- Brownian relative velocity (physical-unit branch) matches DustPy.
- Collision rate/event: Bernoulli probability $1-e^{-\lambda\Delta t}$, frozen-snapshot partner
  sampling, the KNN-ball normalization $N_P/((N_K-1)M_{\rm dust})$, mass-conserving coagulation
  $s_k=(s_i^3+s_j^3)^{1/3}$, the 2D vertical-overlap factor $\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)}$,
  and the exact annular 1D accessible measure.
- Interpolation/deposition: cell-centroid radial weights, outer-face optical-depth weights
  (including the inner-boundary zero-$\tau$ factor), and exact-measure cell CDF sampling.
- Imported-gas Stokes anchoring ($\mathrm{St}\propto s\,\rho_{g,0}H_{g,0}/(\rho_gH_g)$) and
  analytic-gas Stokes scaling verified against Epstein-drift algebra.
- Multisize mass normalization (equal-mass / equal-area grain-number solutions sum to the target
  mass).

---

## 4. Numerical tests: what exists, what is missing

### 4.1 What the current suite already covers

**Fluid** (`qav/fluid/`): directional transport/diffusion convergence in all three dimensions and
both radial Jacobians, FARGO integer/fractional shifts, optical-depth quadrature, exponential
drag source (eight stiffness ratios), and coupled rotating-ring cases. A thread/block sweep
comparison harness exists (no archived JSON). The 3D MMS harness is **documented but not
implemented** (`qav/fluid/test_mms_3d/README.md`).

**Swarm** (`qav/swarm/`): grid deposition, optical depth, circular-orbit equilibrium, frozen
drag/radiation/P-R responses, viscous-flow initialization, stochastic diffusion moments (1D/2D/3D),
boundary-helper endpoint maps, collision-helper measures and constant/additive/product kernel
numerators, imported-gas Stokes/Reynolds scaling, and an extremely thorough KNN suite (brute-force
validated KD-tree and Morton, edge/periodic/wedge cases, $N_P=10^5$ archived).

### 4.2 Missing tests already recorded in the docs

The "Verification still required" sections of `doc/testset_fluid.md` and `doc/testset_swarm.md`
already list: full-3D MMS, settling equilibrium, boundary-crossing transport cases, restart
tolerance, POS_LIMIT multi-substep diffusion, finite-opacity radiation coupling, matched
thread/block artifacts, non-`--use_fast_math` runs (fluid); and regenerated collision-runtime
comparisons, complete frozen collision batches + `CFL_COL`/`N_K`/`N_P` convergence, Smoluchowski
moment evolution, initialization-CDF validation, imported-gas interpolation/anchoring,
boundary-event convergence, direct `dyn_rate_calc` tests, finite `e^{-\tau}` coupling,
long-term reduced-gravity and secular P-R inspiral, settling–diffusion equilibrium,
restart bitwise reproducibility, radial-vs-axisymmetric equivalence, and end-to-end operator
combinations (swarm). These should all be implemented; nothing in this audit contradicts them.

### 4.3 Gaps not yet recorded in the documentation (recommended additions)

**T1. Midplane pressure-support / radial-drift analytical test with non-vanishing `(p,q)` (fluid
and swarm).**
The present suite does not isolate `_get_eta` against an independent nonzero analytical value. Add
a 1D or 2D case with the fiducial $p=-1$, $q=-0.4$ (or another non-vanishing pair) that compares
the initialized gas rotation and steady dust drift with

$$
\eta_{\rm mid}=-\frac{h_g^2}{2}\left(p+\frac q2-\frac32\right)
$$

and the NSH drift formula. This is regression coverage for the documented midplane dynamical
closure, not a test of the column-pressure expression $-(h_g^2/2)(p+q)$.

**T2. Gas vertical-shear profile test (3D, fluid and swarm).**
Compare $v_{\phi,g}(R,Z)$ and the height-dependent drift against an independent evaluation that
separately evaluates the exact horizontal-gravity factor and fixed-$Z$ pressure gradient before
confirming their cancellation to the implemented expression in §2.1(b).

**T2b. Polydisperse finite-domain mass initialization test (3D swarm; implemented, native run pending).**
Use a broad size interval and polar boundaries that truncate a measurable fraction of at least one
size-dependent dust layer. Independently integrate $f_M(s)F(s)$, then verify the initialized sum
$\sum_i m_g(s_i)N_i$, the size-binned represented masses, and the conditional spatial CDFs. Repeat
with multiple simulation values of `N_Z`, including $H_d\ll y\Delta z$, and require the containment
and sampled continuous distribution to remain invariant within radial-quadrature and Monte Carlo
errors. This test should accompany the §2.3 correction.

**T3. Ormel–Cuzzi turbulent relative-velocity regime test (swarm).**
`_get_vrel_t` is currently never tested: the collision cases exercise only the
constant/additive/product kernels, and the physical (CUSTOM) kernel path is compiled but not
validated. Add a test that samples $(St_1,St_2,Re,\alpha)$ combinations spanning all six OC07
regimes (including regime boundaries) and compares the GPU value against an independent Python
transcription of OC07/DustPy. Also unit-test `_get_re_inv_sqrt` in the physical-unit build and
`_get_vrel_b` (Brownian).

**T4. Physical (CUSTOM) kernel collision test including the 2D vertical-overlap closure.**
The $\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)}$ factor and the well-mixed 2D Reynolds closure are only
reachable through `COAG_KERNEL=3`; add an analytical case with prescribed sizes/heights/velocities
that verifies the pair propensity including this closure.

**T5. Fragmentation-path event test.**
The coagulation branch (`vrel<=V_FRAG`) conserves mass by construction, but the fragmentation
branch (`size_k=max(INIT_SMIN,size_k\cdot U^2)`) is never executed by a test. Add a case forcing
`vrel>V_FRAG` and verify represented-mass conservation and the size/`par_numr` update.

**T6. Axisymmetric ($N_X=1$, $N_Z>1$) collision measure.**
`test_collision_1d` covers the 1D annular measure; the 3D axisymmetric revolution factor
$2\pi R$ in `_get_ball_measure` is untested. Add a meridian-plane case with an independent
revolved-measure reference.

**T7. `dyn_rate_calc` rate-by-rate and step-limit regression (swarm).**
The docs list this, but it deserves emphasis: none of the current tests verifies that an accepted
dynamics step actually respects every advertised limit (orbital phase, mesh crossing,
acceleration displacement, diffusion RMS/drift) for a worst-case particle. A direct kernel test
mirroring `test_diffusion_*` but for `dyn_rate_calc` would guard the timestep safety net.

**T8. Fluid angular-momentum conservation over long runs.**
The transport tests retain momentum norms but no test verifies the global invariants
$\sum_i\rho_i V_i$ and $\sum_i\rho_i\ell_{\phi,i}V_i$ over many FARGO steps and diffusion
substeps. A long-running 2D ring test checking these to roundoff would guard the FARGO shift +
residual + diffusion-momentum pipeline.

**T9. Fluid limiter / vacuum stress test.**
No test drives the invariant-domain limiter hard (head-on dust streams, caustic formation,
near-vacuum creation). Add a case where two opposing PPM-transported streams collide and verify
positive density, bounded velocity ratios, and that the §2.2 clip never fires.

**T10. 3D `VISC_FLOW` test.**
Only the 2D branch of `_get_visc_vel` is tested. Add a 3D case verifying the full
(radial + vertical-shear) Kanagawa expression at several heights.

**T11. Restart reproducibility for the fluid.**
`doc/testset_swarm.md` lists swarm restart checks, but the fluid restart path
(`fluid_runtime.cu` resume branch) has no regression: verify a resumed run reproduces an
uninterrupted run to roundoff, including optical-depth reconstruction.

**T12. Flag/operator-combination matrix (both branches).**
`HALFDISK+DIFFUSION+RADIATION`, `COLLISION+DIFFUSION+PR_EFFECT+MULTISIZE`,
`VISC_FLOW+IMPORTGAS` (rejected at compile time — good), `CONST_ST` combinations, and
collision-only vs transport+diffusion+collision vs all-on swarm runs are not exercised as a
matrix. At minimum a small smoke matrix that compiles and runs each legal flag combination for a
few steps (with finite-state checks) would catch `static_assert`/`if constexpr`-dependent
breakage that a single-configuration suite cannot.

**T13. Imported-gas temporal interpolation and trajectory coupling (swarm).**
`gas_lerp_calc` and the two-snapshot midpoint blending in `swarm_runtime.cu` are not tested.
Add a case with two prescribed gas frames that verifies the blended field and the resulting
one-step trajectory against a hand-computed value.

**T14. KNN-estimator convergence (swarm).**
The KNN suite validates search correctness but not the collision *estimator's* dependence on
`N_K`, `H_SEARCH`, and `N_P`. Add a statistical test showing the collision rate converges as
`N_K`/`N_P` increase for a known density field (e.g. uniform), which also guards the
$N_P/((N_K-1)M)$ normalization end-to-end.

---

## 5. Recommended priority order

1. **T2b** — verify the corrected physical contained mass of a finite 3D polydisperse domain.
2. **T3, T4, T5** — the physical collision kernel (the science payload of the code) is the least
   tested part of the swarm branch.
3. **T1, T2** — retain the now-verified midplane and exact-3D pressure-support formulas as
   independent regression tests.
4. **T8, T9** — fluid global conservation and limiter robustness.
5. **T7, T12, T13** — timestep safety, flag matrix, imported-gas coupling.
6. **T10, T11, T6, T14** — remaining coverage.
7. Regenerate the fluid 85-case table and the swarm runtime-collision comparisons on a target GPU
   (already recorded as required in the testset docs).

## 6. References used for the audit

- Colella & Woodward (1984) — PPM reconstruction and limiters.
- Harten, Lax & van Leer (1983) — HLL flux (pressureless branches checked against the exact
  pressureless Riemann solution).
- Masset (2000) — FARGO orbital advection.
- Crank & Nicolson (1947), Higueras & Roldán (2023) — CN positivity subcycling.
- Nakagawa, Sekiya & Hayashi (1986) — steady dust–gas drift.
- Kanagawa et al. (2017) — viscous disk radial velocity.
- Ormel & Cuzzi (2007) and the current DustPy implementation
  (`stammler/dustpy/dustpy/std/dust.f90`) — turbulent relative velocities and Brownian motion.
- Burns, Lamy & Soter (1979) — radiation pressure and Poynting–Robertson drag.
- Takeuchi & Lin (2002) — vertically structured disk velocities and dust drift.
