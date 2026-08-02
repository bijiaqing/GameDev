# Documentation audit disposition: fluid and swarm numerical guides

Original audit date: 2026-08-01  
Disposition review: 2026-08-02

The original third-party audit compared `numerics_fluid.md` and `numerics_swarm.md` with the
production source and did not report an incorrect implemented numerical formula. Its actionable
documentation findings have now been checked again against the current tree. Agreed changes were
applied to the canonical numerical guides; this file retains only the disposition record and the
items that still require a scientific or implementation decision.

The canonical descriptions remain:

- [`numerics_fluid.md`](numerics_fluid.md)
- [`numerics_swarm.md`](numerics_swarm.md)
- [`testset_fluid.md`](testset_fluid.md)
- [`testset_swarm.md`](testset_swarm.md)

## 1. Disposition of the audit findings

| Item | Verdict | Disposition |
|---|---|---|
| D1: project cylindrical diffusion rates into spherical mesh directions | agree | added $D_r,D_\theta,b_r,b_\theta$ explicitly to swarm timestep control |
| D2: collision-build radiation size bound | agree after clarification | documented the monodisperse, multisize non-collision, and multisize collision branches separately |
| D3: fluid primitive-state synchronization | agree | documented recovery after each advection launch and after each directional diffusion group |
| D4: fluid nonfinite CFL handling | agree | documented ring poisoning, diagnostic reporting, and immediate process termination |
| D5: thread-local storage units | agree | corrected 172 KiB to approximately 168 KiB or 172 kB |
| D6: diffusion-boundary upper-face clamps | agree | added the exact inward radial and polar clamp formulas and their cell-indexing purpose |
| D7: backend wedge-image thresholds | unresolved | retained for discussion in Section 3.1 rather than documenting two unexplained magic tolerances as a numerical contract |
| E1: diffusion-rate projection equations | agree | resolved together with D1 |
| E2: quantify the gas-column mismatch | agree after correcting the coefficient | added the exact finite-domain containment integral and the untruncated thin-disk expansion with coefficient $9/8$, not $3/2$ |
| E3: two-cell PPM boundary interpolation | agree | added the exact volume-coordinate linear weights |
| E4: one-ULP KNN culling expansion | agree after correction | documented `nextafterf` on $d_K^2$, which is the quantity actually stored and compared |
| E5: Sherman–Morrison identity | agree | added the rank-one inverse formula and the two tridiagonal solves it requires |
| S1: missing swarm parameters | agree with clarification | added output controls, `COAG_KERNEL`, `REYNOLDS_0`, `M_MOL`, and `X_SEC`; `COAG_KERNEL` also selects the physical kernel rather than only verification kernels |
| S2: missing fluid output interval | agree | added `DT_OUT` and `SAVE_MAX` to the fluid parameter table |
| S3: full-step pseudocode | agree | added compact production-step algorithms to both guides |
| B1: different initial vertical velocities | agree on behavior, rationale unresolved | documented the asymmetry without inventing a design motivation; retained for discussion in Section 3.3 |
| B2: swarm has no explicit azimuthal perturbation | agree | documented Monte Carlo sampling noise as the swarm seed fluctuation |
| B3: diffusion tensor bases have physical consequences | agree | stated in both comparison sections that spherical $r$ and cylindrical $R$ diffusion need not share a 3D radial equilibrium |
| B4: correctly one-sided material | agree | no artificial pairing was added for FARGO, vacuum regularization, KNN, Crank–Nicolson, or particle deposition |

## 2. Accepted corrections now represented canonically

### 2.1 Swarm dynamical-rate projection

The code forms cylindrical diffusivities and drifts, then projects them into the spherical mesh
directions used by the crossing limits:

$$
D_r=D_R\sin^2\theta+D_Z\cos^2\theta,
\qquad
D_\theta=D_R\cos^2\theta+D_Z\sin^2\theta,
$$

$$
b_r=b_R\sin\theta+b_Z\cos\theta,
\qquad
b_\theta=b_R\cos\theta-b_Z\sin\theta.
$$

These equations now appear next to the RMS and drift timestep rates in the swarm guide.

### 2.2 Finite-domain gas column

The proposed statement

$$
\int\rho_g\,dZ
=\Sigma_g\left[1+\frac32h_g^2+O(h_g^4)\right]
$$

was not adopted. Its coefficient was not derived for the configured finite spherical domain, and
the correction depends on the radial and polar boundaries. The guides instead give the exact
finite-domain relation

$$
\Sigma_{g,\mathcal D}(R)=\Sigma_g(R)\,\mathcal C_g(R),
$$

$$
\mathcal C_g(R)=\frac{1}{\sqrt{2\pi}H_g(R)}
\sum_k\int_{Z_{k,a}}^{Z_{k,b}}\mathcal G(R,Z)\,dZ.
$$

This makes the normalization convention quantitative. On an untruncated vertical line, setting
$x=Z/H_g$ gives

$$
\mathcal G=e^{-x^2/2}
\left[1+\frac{3}{8}h_g^2x^4+O(h_g^4)\right],
$$

and therefore

$$
\mathcal C_{g,\infty}=1+\frac{9}{8}h_g^2+O(h_g^4),
$$

because the standard-normal fourth moment is three. Thus the audit's proposed coefficient $3/2$
is not the asymptotic coefficient of the profile implemented by the code.

### 2.3 Exact squared-distance culling expansion

The KD-tree heap stores squared distances. The implemented expansion is therefore

$$
d_{\rm cull}^2
\leftarrow\mathtt{nextafterf}(d_K^2,+\infty),
$$

not an expansion applied directly to $d_K$.

### 2.4 Paired-guide clarifications

The guides now state the following cross-representation differences explicitly:

- fluid diffusion is diagonal in spherical coordinates while swarm diffusion is diagonal in
  cylindrical coordinates
- the fluid initializer adds a polar diffusive-balance velocity but no terminal-settling velocity
- the swarm initializer adds terminal settling but no deterministic diffusive-balance velocity
- the fluid applies an explicit azimuthal density perturbation while the swarm relies on finite
  representative sampling noise

These are documented behaviors, not claims that the branches share identical initial equilibria.

## 3. Items requiring discussion

### 3.1 Near-full-wedge periodic-image thresholds

The two collision backends currently use different tests for treating an azimuthal domain as a
full period:

$$
\Delta\phi<2\pi-10^{-12}
\quad\text{for KD-tree image allocation},
$$

$$
\Delta\phi<2\pi-10^{-6}
\quad\text{for Morton ghost allocation in single precision}.
$$

This difference is real, but calling it harmless is stronger than the present evidence. In the
narrow interval between these thresholds, one backend can create periodic images while the other
uses the direct Cartesian embedding. Before adding either threshold to the user guide, decide
whether to:

1. define one shared physical tolerance and use it in both implementations
2. retain precision-dependent tolerances but add exact-neighbor regression cases in the transition
   interval
3. formally define sufficiently near-full domains as full disks and quantify the permitted
   distance error

The first option gives the clearest backend contract, but it requires a source and KNN-test change
and was not performed during this documentation pass.

### 3.2 Optional finite-cutoff expansion of the gas column

The exact finite-domain factor $\mathcal C_g(R)$ and the untruncated $9h_g^2/8$ asymptotic are now
documented. If a paper also needs a compact asymptotic for a finite spherical cutoff, its boundary
terms must be derived for the stated domain and for the ordering of the cutoff relative to $H_g$.
Whether that additional specialized expression is useful remains open; it is not needed to define
the code's normalization convention.

### 3.3 Scientific intent of the initial vertical-velocity asymmetry

The source establishes what each branch initializes, but it does not encode why the closures were
chosen differently. The guides now report the difference without assigning motivation. A future
methods discussion should decide whether:

1. the asymmetry is intentional because the Eulerian and stochastic diffusion closures require
   different low-transient initial states
2. both branches should include terminal settling in addition to their diffusion treatment
3. one common vertical-equilibrium prescription should replace both current initializations

That decision affects the physical initial-value problem and should not be inferred from a
documentation audit alone.

## 4. Findings retained as verified background

The audit found the following representative components consistent with the inspected source:

- PPM reconstruction, HLL transport, FARGO shifting, invariant-domain limiting, SSPRK(3,3), and
  Crank–Nicolson diffusion with positivity subcycling
- exponential drag-weighted source integration and optical-depth construction
- the staggered swarm transport, radiation and Poynting–Robertson responses, and cylindrical
  Euler–Maruyama diffusion with velocity reprojection
- continuous finite-domain initialization and representative-mass normalization
- collision-rate construction, Ormel–Cuzzi relative velocities, and the exact KD-tree/Morton
  neighbor-selection contract

These review conclusions are not substitutes for the missing native tests listed in the two
`testset_*` documents.
