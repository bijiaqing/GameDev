# Fluid publication test suite

## 1. Purpose

The fluid validation suite supplies the numerical evidence needed to publish and use the Eulerian dust
model. Every retained case tests a complete physical operator, a geometric boundary treatment, or a
coupled evolution path against an independent reference. It deliberately omits micro-tests whose
only claim is that one local expression returns the value written in the source.

The canonical matrix is defined in `val/val_config.py`. With resolutions
$N=32,64,128,256$, it contains 19 models and 76 metric records. CUDA and ROCm use the same
backend-neutral model definitions and validators under `val/fluid/`.

This document explains what the cases establish. The fluid equations and production algorithms are
described in [fluid_numeric.md](fluid_numeric.md); archive commands are summarized in
[`val/README.md`](../val/README.md).

## 2. What constitutes publication evidence

A case is retained when it supports at least one of the following claims:

1. a complete evolution operator converges to a known continuum solution;
2. a curvilinear or periodic boundary preserves the correct physical solution;
3. a stiff or positivity-controlled method reproduces its exact discrete solution;
4. coupled production operators preserve a known equilibrium or mode;
5. a physical source prescription reproduces a closed-form response.

Compilation, finite-value guards, restart I/O, thread-versus-block equivalence, and performance are
important engineering concerns, but they are not separate scientific validation cases in this
suite.

## 3. Common numerical measurements

### 3.1 Finite-volume reference

The solver stores cell averages. Validators therefore compare a numerical cell average
$q_i^{\rm num}$ with an analytical cell average

$$
q_i^{\rm ref}=\frac{1}{V_i}\int_{V_i}q(\boldsymbol{x},t)\,dV,
$$

not with a point sample at the cell center. The volume $V_i$ is the exact measure for the tested
coordinate system. This distinction is essential on logarithmic radial and spherical-polar grids.

### 3.2 Error norms

For $e_i=q_i^{\rm num}-q_i^{\rm ref}$, the common geometry-weighted norms are

$$
L_1=\frac{\sum_i V_i|e_i|}{\sum_iV_i},\qquad
L_2=\left(\frac{\sum_iV_i e_i^2}{\sum_iV_i}\right)^{1/2},\qquad
L_\infty=\max_i|e_i|.
$$

Velocity errors exclude analytically empty cells, because dividing momentum by an arbitrarily small
density is not a meaningful velocity diagnostic.

For two resolutions $N_a<N_b$, the observed order is

$$
p=\frac{\log(E_{N_a}/E_{N_b})}{\log(N_b/N_a)}.
$$

The validators require finite output, model-specific final accuracy, and the stated final-order or
monotonic-sequence condition. Conservative periodic cases also require relative mass drift no larger
than $10^{-8}$. Open-boundary tests instead compare the remaining mass with the exact escaped mass;
they do not incorrectly demand mass conservation in the computational domain.

## 4. Retained matrix

| Group | Models | Main claim |
|---|---|---|
| equilibrium | `test_startup_3d` | initialized stratified disk is a discrete steady state |
| transport | `test_x_transport_2d`, `test_x_wedge_transport_2d`, `test_y_transport_cyl`, `test_y_transport_sph`, `test_y_outflow_2d`, `test_z_transport_3d`, `test_z_outflow_3d`, `test_z_reflect_3d` | conservative transport and physical boundaries in every active direction |
| diffusion | `test_x_diffusion_2d`, `test_x_wedge_diffusion_2d`, `test_y_diffusion_cyl`, `test_y_diffusion_sph`, `test_z_diffusion_3d`, `test_diffusion_poslimit` | Crank--Nicolson diffusion, metric factors, periodic seams, and positivity control |
| source | `test_source_drag` | stiff drag/source update matches a closed-form response |
| radiation | `test_optdepth`, `test_attenuation_2d` | optical-depth integration and attenuated radiation coupling |
| coupled | `test_ring_all_2d` | production transport, diffusion, source, and radiation composition |

Except for `test_source_drag`, which uses a fixed eight-cell parameter grid, these cases use the
requested resolution sequence. The publication matrix selects shift $3.25$ for periodic transport,
CFL $0.05$ for radial and polar transport, and power-law index $-1$ for both radiation cases.

## 5. Equilibrium initialization

### `test_startup_3d`

This case initializes the full three-dimensional dust density and momentum. Starting from two
identical copies of that state, it applies the production polar-advection operator to one copy and
the production polar-diffusion operator to the other. It then measures their summed finite-
difference tendency,

$$
\mathcal{R}=\frac{U^{n+1}-U^n}{\Delta t}.
$$

Each residual norm is divided by the combined magnitude of the two operator increments, so the test
measures cancellation of polar transport and diffusion rather than merely the smallness of an
unevolved state. It also checks the volume-integrated residual mass rate,

$$
\epsilon_M=
\frac{\left|\sum_iV_i\mathcal{R}_{\rho,i}\right|}
     {\sum_iV_i|\mathcal{R}_{\rho,i}^{\rm components}|},
$$

and requires $\epsilon_M<10^{-6}$. Resolution convergence distinguishes a genuinely balanced
discretization from an accidentally small coarse-grid residual.

The production initializer adds an azimuthal density perturbation drawn from the native vendor RNG.
cuRAND and hipRAND do not generate the same realization, so the absolute initialized mass is used
only as a positive-finite native check. Cross-backend validation compares the normalized residual,
mass-balance error, and convergence order; it does not require equal absolute initial mass.

## 6. Transport

All transport cases reduce the continuity equation to

$$
\frac{\partial\rho_d}{\partial t}+\nabla\!\cdot(\rho_d\boldsymbol{v})=0
$$

with a velocity field chosen to give an exact characteristic solution.

### 6.1 Periodic azimuthal transport

`test_x_transport_2d` advects a smooth Fourier mode through a noninteger FARGO shift of $3.25$
cells. For constant angular speed $\Omega$,

$$
\rho_d(x,t)=\rho_d(x-\Omega t,0).
$$

The reference integrates this shifted mode over every finite-volume cell. The case tests the FARGO
integer shift, residual transport, PPM reconstruction, momentum transport, and periodic seam in one
calculation.

`test_x_wedge_transport_2d` repeats the physical problem on a restricted periodic wedge. It is
retained separately because a wedge seam changes the minimum-image geometry and exercises a
publication-relevant domain configuration.

### 6.2 Radial transport

`test_y_transport_cyl` and `test_y_transport_sph` test the radial metric term. For constant outward
speed $u$ in effective radial dimension $d$,

$$
R_0=R-ut,\qquad
\rho_d(R,t)=\rho_d(R_0,0)\left(\frac{R_0}{R}\right)^{d-1}.
$$

The cylindrical and spherical cases use the corresponding exact shell measures. They therefore
test more than Cartesian advection on a relabeled coordinate.

`test_y_outflow_2d` lets part of a compact profile cross the outer radial boundary. Gaussian
quadrature constructs exact cell averages of the characteristic solution. In addition to density
and momentum errors, the validator compares

$$
M_{\rm remain}(t)=\int_{\Omega} \rho_d(R,t)\,dV
$$

with the numerical mass remaining in the domain and requires a nontrivial escaped fraction. This is
the direct validation of radial outflow rather than a zero-flux boundary check.

### 6.3 Polar transport and boundaries

`test_z_transport_3d` transports a smooth polar profile using the spherical-polar conservation law

$$
\frac{\partial\rho_d}{\partial t}
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}
 \left(\sin\theta\,\rho_d v_\theta\right)=0.
$$

The exact reference is integrated with the $\sin\theta$ cell measure and includes the corresponding
angular momentum.

`test_z_outflow_3d` uses a characteristic that leaves the polar domain and compares both the field
and escaped mass with the exact open-boundary solution. `test_z_reflect_3d` launches a profile toward
a reflecting boundary and compares with the exact reflected characteristic, including the sign
change of normal momentum.

## 7. Diffusion

The retained diffusion cases solve

$$
\frac{\partial\rho_d}{\partial t}=\nabla\!\cdot(D\nabla\rho_d)
$$

with constant $D$ and zero imposed bulk transport. Smooth eigenmodes decay as

$$
\rho_d(\boldsymbol{x},t)=\rho_0+\epsilon q(\boldsymbol{x})e^{-D\lambda t},
\qquad -\nabla^2q=\lambda q.
$$

`test_x_diffusion_2d` uses a periodic Fourier mode. `test_x_wedge_diffusion_2d` verifies the same
operator across a periodic wedge seam. `test_y_diffusion_cyl` and `test_y_diffusion_sph` use radial
Neumann eigenfunctions for the appropriate metric dimension; the validator integrates the radial
eigenvalue problem independently and then forms exact cell averages. `test_z_diffusion_3d` uses the
spherical-polar measure and corresponding angular eigenmode.

These cases jointly test Crank--Nicolson coefficients, cyclic or boundary closure, metric factors,
density fluxes, and the donor-momentum closure. Density and all momentum components are compared,
so passing density alone cannot hide an inconsistent diffusive momentum update.

### `test_diffusion_poslimit`

The standard-resolution branch starts from a high-contrast mode that activates positivity subcycling. Its exact
same-grid reference uses the eigenvalue of the discrete second-difference operator,

$$
\lambda_h=\frac{4}{\Delta x^2}\sin^2\!\left(\frac{k\Delta x}{2}\right),
$$

while a continuum reference is retained to measure spatial convergence. It demonstrates that the
positivity controller does not change the intended Crank--Nicolson solution or create negative
density.

Three fixed eight-cell directional variants additionally exercise the donor-outflow limiter in the
azimuthal, radial, and polar production kernels. Each starts from a $20{:}1$ density front and
nonuniform values of all three stored primitives. The timestep is chosen from the largest CN
coefficient, and the validator independently reconstructs the unlimited CN face flux and requires
at least one donor to export more old mass than is allowed by `POS_LIMIT`. The validator requires:

- finite, nonnegative accepted density and verified limiter activation;
- geometry-weighted conservation of mass and all three stored momenta;
- every recovered primitive to remain in the convex hull of the old donor values;
- nonincrease of the geometry-weighted convex quadratic $\sum_iM_iq_i^2$ for each primitive.

The three line geometries use $V_i\propto\Delta x$, $V_i\propto\Delta V_{y,i}$, and
$V_i\propto y\Delta V_{z,i}$ respectively. These variants test the nonlinear correction that the
smooth eigenmodes normally leave inactive; the smooth branches continue to establish the
unlimited scheme's convergence.

## 8. Drag and radiation

### 8.1 Exact drag/source response

`test_source_drag` verifies the stiff source equation

$$
\frac{d\boldsymbol{v}}{dt}
=-\frac{\boldsymbol{v}-\boldsymbol{v}_g}{t_s}+\boldsymbol{a}(t),
$$

for a force that varies linearly over the step. The validator evaluates the exponential integral in
closed form, including the small-argument series used to avoid cancellation. Eight cells span a wide
range of $\Delta t/t_s$, so the test covers both weak and stiff drag without treating spatial
resolution as a convergence parameter.

This case replaces `source_update` with a test-local implementation of the coefficient algebra.
It validates that stiff response, not the complete production source integration.

### 8.2 Optical depth

`test_optdepth` verifies the radial accumulation

$$
\tau(R)=\int_{R_{\min}}^R\kappa\rho_d(R')\,dR'.
$$

For the retained $\kappa\rho_d\propto R^{-1}$ profile,

$$
\tau(R)=C\ln\!\left(\frac{R}{R_{\min}}\right).
$$

The comparison uses exact shell-integrated increments and therefore checks both local extinction and
the cumulative radial scan.

### 8.3 Finite attenuation

`test_attenuation_2d` couples the computed optical depth to

$$
a_{\rm rad}(R)=\beta\frac{GM}{R^2}e^{-\tau(R)}.
$$

It compares optical depth, unchanged density, angular momentum relaxation, radial velocity, and the
time-centered source response with a same-grid exact reference. A separate continuum optical-depth
integral supplies a refinement-order check, preventing the test from validating only a duplicated
discrete prefix sum.

## 9. Coupled production composition

`test_ring_all_2d` evolves a radial eigenmode through the normal palindromic composition of
transport, diffusion, drag/source terms, optical depth, and radiation pressure. The reference
combines the expected diffusive decay with the exact equilibrium angular momentum and zero radial
motion. This case is intentionally retained even though its component operators are tested
separately: it detects ordering, buffer, and conserved-to-primitive conversion errors that isolated
operator tests cannot expose.

It is a short controlled coupled test, not a claim that an arbitrary nonlinear disk has a closed-form
solution.

The verification driver reproduces the operator composition while replacing the production runtime;
its native pass does not qualify every production-runtime branch or output/restart path.

## 10. Running and interpreting the suite

Run a native backend through `val/run_all.py` as shown in [`val/README.md`](../val/README.md).
The canonical fluid archive is below
`val/fluid/out/MODEL/BACKEND/SWEEP/`. Disposable validation executables and object files are isolated below
`val/fluid/obj/MODEL/BACKEND/`, never in the source-model directories. A complete default campaign contains 76 records and a passing
`val/fluid/out/_suite/BACKEND/SWEEP/manifest_all.json`.

The most important evidence is the combination of:

- agreement with independently constructed finite-volume reference fields;
- the expected resolution trend rather than a single permissive tolerance;
- conservation or exact escaped-mass accounting appropriate to the boundary;
- agreement of density and momentum, not density alone;
- matching CUDA and ROCm publication archives from one source fingerprint.

### Native archive assessment, 2026-09-05

The downloaded CUDA `sm_80` and ROCm `gfx942` archives each pass all 22 cases and 76 metric
records for both `thread` and `block`. Fresh archive checks and CUDA/ROCm metric comparisons
pass for each sweep, with zero fluid mismatches. The x/y/z limiter cases activate on all four
configurations; their largest normalized conservation residual is `5.150797131870754e-16`, with
zero primitive-range excess and zero quadratic growth.

Evidence is saved under `val/logs/qualification_20260905/` in `archive_{cuda,rocm}_{thread,block}.json`,
`comparison_{thread,block}.json`, and `assessment.json`. The thread comparison's fluid component
passes, although its overall result fails on swarm diagnostics described in `swarm_testset.md`.
Both saved thread campaigns have matching initial/final source SHA-256
`cf5db5ba782419cc09b3a92eeaaf812114f25a78be329c06d86b0b1b7ae3beca`.
The standalone block suites did not record a source fingerprint. Their native numerical passes
therefore do not establish matching-source provenance; the comparator's reported fingerprint comes
from earlier top-level campaign files, not the block runs. No GPU execution was performed during
this local archive assessment.

## 11. Deliberate limits

The suite does not claim exhaustive flag coverage, bitwise CUDA/ROCm identity, performance
superiority, restart correctness, or recovery after deliberately injected NaN/Inf values. It also
does not use a manufactured full-3D solution without a natural closed-form reference. Add a new case
only when it supports a distinct scientific or numerical claim that the retained matrix does not
already establish.
