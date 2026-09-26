# Fluid publication test suite

## 1. Purpose

The fluid validation suite supplies the numerical evidence needed to publish and use the Eulerian
dust model. Every retained case tests a complete physical operator, a geometric boundary treatment,
or a coupled evolution path against an independent reference. It deliberately omits micro-tests
whose only claim is that one local expression returns the value written in the source.

The canonical matrix is `FLUID_GROUPS` in `val/val_config.py`. It contains 19 models, which expand
to 22 cases because `test_diffusion_poslimit` adds three fixed-grid limiter variants. With the
standard resolutions $N=32,64,128,256$, the 18 resolution-swept cases write four metric records
each and the four fixed-grid cases one each, so a complete matrix has 76 metric records
(`EXPECTED_FLUID_METRICS`). CUDA and ROCm build the same backend-neutral model definitions under
`val/fluid/mod/` with the shared drivers and validators under `val/fluid/src/`. The matrix is run
once for each fluid line-kernel sweep (`thread` and `block`), and each sweep must pass on its own.

This document explains what the cases establish and how a result is judged. The fluid equations
and production algorithms are described in [fluid_numeric.md](fluid_numeric.md); commands, output
paths, and job layout are in [`val/README.md`](../val/README.md).

## 2. Evidence criterion

A case is retained when it supports at least one of the following claims:

1. a complete evolution operator converges to a known continuum solution;
2. a curvilinear, periodic, open, or reflecting boundary preserves the correct physical solution;
3. a stiff or positivity-controlled method reproduces its exact discrete solution;
4. coupled production operators preserve a known equilibrium or mode;
5. a physical source prescription reproduces a closed-form response.

Compilation, finite-value guards, restart I/O, thread-versus-block equivalence, and performance are
important engineering concerns, but they are not separate scientific validation cases in this
suite.

## 3. Common measurements

### 3.1 Finite-volume reference

The solver stores cell averages. Validators therefore compare a numerical cell average
$q_i^{\rm num}$ with an analytical cell average

$$
q_i^{\rm ref}=\frac{1}{V_i}\int_{V_i}q(\boldsymbol{x},t)\,dV,
$$

not with a point sample at the cell center. The volume $V_i$ is the exact measure for the tested
coordinate system, and the integrals use 16- or 32-point Gauss–Legendre quadrature per cell. This
distinction is essential on logarithmic radial and spherical-polar grids. The analytical
parameters are written into the validators rather than read from the simulation output, so an
incorrect run cannot redefine its own expected answer.

### 3.2 Error norms and observed order

For $e_i=q_i^{\rm num}-q_i^{\rm ref}$, the common geometry-weighted norms are

$$
L_1=\frac{\sum_i V_i|e_i|}{\sum_iV_i},\qquad
L_2=\left(\frac{\sum_iV_i e_i^2}{\sum_iV_i}\right)^{1/2},\qquad
L_\infty=\max_i|e_i|.
$$

Velocity errors exclude analytically empty cells: a cell enters only when both the analytical and
the numerical density exceed $10^{-12}$ times the largest analytical density. Dividing momentum by
an arbitrarily small density is not a meaningful velocity diagnostic.

For two resolutions $N_a\lt N_b$, the observed order is

$$
p=\frac{\log(E_{N_a}/E_{N_b})}{\log(N_b/N_a)}.
$$

With the factor-of-two sequence, acceptance uses the order $p$ between the two finest grids.

### 3.3 Acceptance gates

Every case passes through one of two gates in `val/fluid/src/run_model.py`.

The shared analytical gate applies to cases without a model-local validator. It requires

- every archived metric value to be finite;
- a relative change of the total mass no larger than $10^{-8}$, for cases that report it;
- a finest-grid $L_1$ error of the primary field (density, or optical depth for `test_optdepth`) no
  larger than $2\times10^{-2}$;
- a final observed order of that error of at least 1.5 (0.75 when fewer than four resolutions are
  requested);
- the same two conditions for each conserved momentum component `momx`, `momy`, and `momz`,
  except that a component whose finest $L_1$ error is at most $10^{-12}$ is the roundoff of an
  exactly vanishing solution and needs no observed order.

Exact-solution cases (`test_x_transport_2d` with an integer shift, `test_optdepth` with a zero
power) instead require every judged $L_1$ error to be at most $10^{-10}$, and `test_source_drag`
replaces the accuracy and order conditions by a maximum error of $10^{-12}$ over every compared
field. The velocity errors are archived with each record but do not enter the pass decision.

A validator-owned gate applies to cases whose model directory supplies `validate_case.py`. Each
metric record then carries its own `passed` flag, which combines an activation check (the tested
branch must actually be exercised) with case-specific tolerances. The runner additionally applies
the final-order requirement on the declared convergence field and every declared sequence
requirement: a final-error limit, a final-order limit, or strictly monotonic improvement with
resolution. The tolerances of both gates are listed with each case below.

Open-boundary tests compare the remaining mass with the exact remaining mass; they do not demand
mass conservation in the computational domain.

## 4. Retained matrix

| Group | Case | Grid $N_x\times N_y\times N_z$ | Suite arguments | Build flags | Main claim |
|---|---|---|---|---|---|
| equilibrium | `test_startup_3d` | $4\times N\times N$ | | `DIFFUSION` | polar advection–diffusion balance of the initialized disk |
| transport | `test_x_transport_2d` | $N\times4\times1$ | `--shift 3.25` | | FARGO azimuthal transport and full periodic seam |
| transport | `test_x_wedge_transport_2d` | $N\times4\times1$ | | | transport across a restricted periodic wedge seam |
| transport | `test_y_transport_cyl` | $4\times N\times1$ | `--cfl 0.05` | | radial transport with the cylindrical metric |
| transport | `test_y_transport_sph` | $4\times N\times4$ | `--cfl 0.05` | `DIFFUSION`, `CONST_NU` | radial transport with the spherical metric |
| transport | `test_y_outflow_2d` | $4\times N\times1$ | | | open outer radial boundary |
| transport | `test_z_outflow_3d` | $4\times4\times N$ | | `DIFFUSION`, `CONST_NU` | open outer polar boundary |
| transport | `test_z_reflect_3d` | $4\times4\times N$ | | `DIFFUSION`, `CONST_NU`, `HALF_DISK` | zero-flux midplane wall |
| transport | `test_z_transport_3d` | $4\times4\times N$ | `--cfl 0.05` | `DIFFUSION`, `CONST_NU` | polar transport with the $\sin\theta$ metric |
| diffusion | `test_x_diffusion_2d` | $N\times4\times1$ | | `DIFFUSION`, `CONST_NU` | azimuthal Crank–Nicolson diffusion |
| diffusion | `test_x_wedge_diffusion_2d` | $N\times4\times1$ | | `DIFFUSION`, `CONST_NU` | cyclic diffusion across a wedge seam |
| diffusion | `test_y_diffusion_cyl` | $4\times N\times1$ | | `DIFFUSION`, `CONST_NU` | radial diffusion, cylindrical metric |
| diffusion | `test_y_diffusion_sph` | $4\times N\times4$ | | `DIFFUSION`, `CONST_NU` | radial diffusion, spherical metric |
| diffusion | `test_z_diffusion_3d` | $4\times4\times N$ | | `DIFFUSION`, `CONST_NU`, `HALF_DISK` | polar diffusion on a hemisphere |
| diffusion | `test_diffusion_poslimit` | $N\times1\times1$ | | `DIFFUSION`, `CONST_NU` | positivity subcycling reproduces discrete CN |
| diffusion | `test_diffusion_poslimit` | $8\times1\times1$ | `--direction x --res 8` | `DIFFUSION`, `CONST_NU` | azimuthal donor-outflow limiter |
| diffusion | `test_diffusion_poslimit` | $4\times8\times1$ | `--direction y --res 8` | `DIFFUSION`, `CONST_NU` | radial donor-outflow limiter |
| diffusion | `test_diffusion_poslimit` | $4\times4\times8$ | `--direction z --res 8` | `DIFFUSION`, `CONST_NU` | polar donor-outflow limiter |
| source | `test_source_drag` | $8\times1\times1$ | `--res 8` | | stiff drag/source update matches a closed-form response |
| radiation | `test_optdepth` | $4\times N\times1$ | `--power -1.0` | `RADIATION` | radial optical-depth integration |
| radiation | `test_attenuation_2d` | $4\times N\times1$ | `--power -1.0` | `RADIATION` | attenuated radiation reaching the source update |
| coupled | `test_ring_all_2d` | $N\times N\times1$ | | `DIFFUSION`, `RADIATION`, `CONST_NU` | production transport, diffusion, source, and radiation composition |

Isolated-operator cases refine only the direction under test. Four cells in an inactive transverse
direction expose indexing mistakes without making every convergence run expensive; a single polar
cell denotes the two-dimensional midplane model. Unless a case states otherwise, the radial domain
is $0.5\le R\le2.5$ on a logarithmic grid, the full azimuthal domain is $0\le\phi\lt2\pi$, and the
resolved polar domain is $0.35\le\theta\le\pi-0.35$. `DIFFUSION` appears in several transport
builds only because every resolved-polar model requires it; those drivers advance transport alone.
Every build defines `DUST_REPR := fluid`, and each model's `const_defs.cuh` selects its
`VERIFY_*` branch of `val/fluid/src/const_defs.cuh`, which replaces the production physical setup
with test constants.

## 5. Equilibrium initialization

`test_startup_3d` initializes the three-dimensional dust density, its balancing polar velocity,
and the conserved momenta through the production initialization kernels. Starting from two
identical copies of that state, it applies the production polar-advection operator to one copy and
the production polar-diffusion operator to the other over a probe interval
$\Delta t=\Delta\theta/4$. It then measures their summed finite-difference tendency,

$$
\mathcal{R}=\frac{U^{n+1}-U^n}{\Delta t}.
$$

Each residual norm is divided by the corresponding norm of the combined magnitude of the two
operator increments, so the test measures cancellation of polar transport and diffusion rather
than merely the smallness of an unevolved state. It also checks the volume-integrated residual
mass rate,

$$
\epsilon_M=
\frac{\left|\sum_iV_i\mathcal{R}_{\rho,i}\right|}
     {\sum_iV_i|\mathcal{R}_{\rho,i}^{\rm components}|}.
$$

The model uses a narrow resolved disk ($0.8\le R\le1.2$, $\vert\theta-\pi/2\vert\le0.4$), gas
indices $p=3/2$ and $q=0$ that remove pressure-supported radial drift, and a finite Stokes number,
so the balance tested is the production vertical settling–diffusion equilibrium.

Acceptance (validator-owned): finite and nonnegative initialized, advected, and diffused states,
positive initialized mass, $\epsilon_M\lt10^{-6}$ at every resolution, and a final observed order
of the normalized $L_1$ residual of at least 1.5. Resolution convergence distinguishes a genuinely
balanced discretization from an accidentally small coarse-grid residual.

The production initializer multiplies the density by an azimuthal perturbation drawn from the
native vendor RNG (cuRAND or hipRAND), which do not generate the same realization. The absolute
initialized mass is therefore only a positive-finite native check, and cross-backend comparison
excludes it. The normalized residual, mass-balance error, and convergence order are compared.

## 6. Transport

All transport cases reduce the continuity equation to

$$
\frac{\partial\rho_d}{\partial t}+\nabla\!\cdot(\rho_d\boldsymbol{v})=0
$$

with a velocity field chosen to give an exact characteristic solution. Only the directional
production advection operator under test is advanced. The remaining momentum components are
initialized as fixed multiples of density (zero in some cases), so their exact values follow from
the exact density and test that momentum is carried consistently with mass.

### 6.1 Periodic azimuthal transport

`test_x_transport_2d` advects a smooth Fourier mode $\rho_d=1+0.1\,\overline{\sin2\phi}$ (the
overbar denotes the cell average) with specific angular momentum $R^2$, that is, unit angular
speed $\Omega$. For this constant angular speed,

$$
\rho_d(\phi,t)=\rho_d(\phi-\Omega t,0).
$$

Each step moves the profile by a prescribed noninteger FARGO shift of $3.25$ cells, until one full
orbit $t=2\pi$. The reference integrates the shifted mode over every finite-volume cell. The case
tests the FARGO integer shift, residual transport, PPM reconstruction, momentum transport, and
periodic seam in one calculation. It uses the shared gate.

`test_x_wedge_transport_2d` repeats the physical problem on the periodic wedge $-0.4\le\phi\lt0.8$:
a compact pulse on $0.45\le\phi\le0.75$ moves at angular speed $0.25$ until $t=1$, so its support
must cross the outer seam and reappear at the inner face. It is retained separately because a wedge
seam changes the periodic image geometry and exercises a publication-relevant domain configuration.

Acceptance (validator-owned): activation (the wrapped part of the pulse exceeds $0.05$ above the
background), finite state, density $\ge-2\times10^{-13}$, relative mass change
$\le2\times10^{-11}$; density $L_1\le5\times10^{-3}$ with final order at least 0.75 and monotonic
improvement; density $L_\infty\le6\times10^{-2}$ with monotonic improvement; and $L_1$ of the
momentum-vector error $\le5\times10^{-3}$ with final order at least 0.75 and monotonic
improvement.

### 6.2 Radial transport and outflow

`test_y_transport_cyl` and `test_y_transport_sph` test the radial metric term with a ballistic
homologous expansion. Every parcel keeps its initial radial speed $a R_0$ with $a=0.2$, so
$R=\lambda R_0$ with $\lambda=1+at$. In effective radial dimension $d$ (2 for the cylindrical
midplane, 3 for the spherical shell),

$$
\rho_d(R,t)=\lambda^{-d}\rho_d(R/\lambda,0),\qquad
v_R(R,t)=\frac{aR}{\lambda}.
$$

The initial profile is a smooth compact bump on $1.0\le R\le1.8$, advanced to $t=0.25$ with CFL
number $0.05$. The reference integrates density and radial momentum with the exact cylindrical or
spherical shell measures. These cases therefore test more than Cartesian advection on a relabeled
coordinate. They use the shared gate.

`test_y_outflow_2d` lets part of a compact profile on $1.7\le R\le2.5$ cross the outer radial
boundary at constant outward speed $u=0.2$ until $t=1$. For constant speed in effective radial
dimension $d$,

$$
R_0=R-ut,\qquad
\rho_d(R,t)=\rho_d(R_0,0)\left(\frac{R_0}{R}\right)^{d-1}.
$$

Gaussian quadrature constructs exact cell averages of this characteristic solution. In addition
to density errors, the validator compares

$$
M_{\rm remain}(t)=\int_{\Omega} \rho_d(R,t)\,dV
$$

with the numerical mass remaining in the domain. This is the direct validation of radial outflow
rather than a zero-flux boundary check.

Acceptance (validator-owned): activation (exact escaped fraction above 5 percent at $t=1$),
finite density $\ge-2\times10^{-13}$, radial momentum equal to $u\rho_d$ to $2\times10^{-11}$,
transverse momenta below $2\times10^{-13}$; density $L_1\le2\times10^{-3}$ and
$L_\infty\le3\times10^{-2}$, both improving monotonically, with final density order at least 0.75;
and relative remaining-mass error $\le5\times10^{-3}$ with final order at least 0.75 and monotonic
improvement.

### 6.3 Polar transport, outflow, and the midplane wall

`test_z_transport_3d` transports a smooth polar profile using the spherical-polar conservation law

$$
\frac{\partial\rho_d}{\partial t}
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}
 \left(\sin\theta\,\rho_d v_\theta\right)=0.
$$

With constant angular rate $\omega=v_\theta/r=0.15$, the polar line density $\rho_d\sin\theta$ is
translated rigidly, so

$$
\rho_d(\theta,t)=\rho_d(\theta-\omega t,0)\,\frac{\sin(\theta-\omega t)}{\sin\theta}.
$$

The initial profile is a compact bump in $\rho_d\sin\theta$ on $0.80\le\theta\le1.30$, advanced to
$t=0.3$ with CFL number $0.05$. The exact reference is integrated with the $\sin\theta$ cell
measure. The polar momentum of each radial shell is chosen so that the exact integrated
face-area-to-volume factor gives the same angular rate in every shell. The case uses the shared
gate.

`test_z_outflow_3d` translates a compact profile on $2.45\le\theta\le2.75$ at angular rate $0.15$
across the outer polar face $\theta=\pi-0.35$ until $t=1$. The production kernel permits outward
flux there and suppresses inflow. The validator compares the field and the remaining mass with the
exact open-boundary solution.

`test_z_reflect_3d` uses `HALF_DISK`, whose polar domain $0.35\le\theta\le\pi/2$ ends at the
midplane, where the production kernel imposes zero flux. The flow is a reflection-symmetric
compression toward the midplane, $v_\theta\propto(\pi/2-\theta)/(1-at)$ with $a=0.2$, applied to
a profile on $0.70\le\theta\le\pi-0.70$ that covers the midplane. The characteristics converge
linearly,

$$
\theta_0=\frac{\pi}{2}-\frac{\pi/2-\theta}{1-at},\qquad
\rho_d(\theta,t)\sin\theta=\frac{\rho_d(\theta_0,0)\sin\theta_0}{1-at}.
$$

The analytical polar speed and wall flux vanish at the midplane, so the exact solution conserves
mass and the case tests whether the discrete wall preserves the symmetric solution.

Acceptance for both polar-boundary cases (validator-owned): finite state, density
$\ge-2\times10^{-13}$, transverse momenta below $2\times10^{-13}$; density
$L_1\le4\times10^{-3}$ with final order at least 0.75 and monotonic improvement; density
$L_\infty\le5\times10^{-2}$ with monotonic improvement; polar-momentum $L_1$ with final order at
least 0.75 and monotonic improvement, limited to $4\times10^{-3}$ for outflow and $10^{-2}$ for the
wall. Activation requires, for outflow, that the pulse cross the outer face and the exact escaped
fraction exceed 5 percent, and for the wall, a midplane boundary, a compression factor
$1-at\lt0.9$, and an initial midplane profile value above 0.1. The remaining-mass error is limited
to $5\times10^{-3}$ with monotonic improvement for outflow and to $2\times10^{-11}$ for the wall.

## 7. Diffusion

### 7.1 Smooth eigenmodes

The retained diffusion cases solve

$$
\frac{\partial\rho_d}{\partial t}=\nabla\!\cdot(D\nabla\rho_d)
$$

with constant $D=5\times10^{-2}$ from production `CONST_NU` in the tracer limit (`STOKES_0 = 0`)
and zero imposed bulk transport. The Schmidt number is one in the tested direction and
$10^{300}$ in the others, which makes diffusion negligible there while keeping the same kernel
interface. These density-mode cases do not establish the concentration-mode equilibrium or general
variable-coefficient accuracy. For constant $D$, smooth eigenmodes decay as

$$
\rho_d(\boldsymbol{x},t)=\rho_0+\epsilon q(\boldsymbol{x})e^{-D\lambda t},
\qquad -\nabla^2q=\lambda q.
$$

`test_x_diffusion_2d` uses the periodic Fourier mode $\sin2\phi$, whose decay rate $4D/R^2$
depends on radius. `test_x_wedge_diffusion_2d` verifies the same operator with the fundamental
mode of the wedge $-0.4\le\phi\lt0.8$, so the cyclic Crank–Nicolson coupling closes across a seam
that is not $2\pi$. `test_y_diffusion_cyl` and `test_y_diffusion_sph` use radial Neumann
eigenfunctions for the appropriate metric dimension; the driver builds them from cylindrical or
spherical Bessel functions, while the validator integrates the radial eigenvalue problem
independently with a dense Runge–Kutta table and then forms exact cell averages.
`test_z_diffusion_3d` uses the Legendre mode $P_2(\cos\theta)$ on the hemisphere
$0\le\theta\le\pi/2$, whose natural zero-flux boundaries give the decay rate $6D/R^2$. All modes
start at amplitude $\epsilon=0.1$. The four full-domain cases advance to $t=0.5$ with a timestep of
one quarter of the smallest cell length, so temporal and spatial errors are refined together; the
wedge case advances to $t=0.2$.

The initial momenta are fixed multiples of density, so the exact momenta are the same multiples of
the exact density. These cases jointly exercise Crank–Nicolson coefficients, cyclic or boundary
closure, metric factors, density fluxes, and the donor-momentum closure. Density and all momentum
and velocity components are archived. `test_x_diffusion_2d`, `test_y_diffusion_cyl`,
`test_y_diffusion_sph`, and `test_z_diffusion_3d` use the shared gate, which judges density and
the three momentum components;
`test_x_wedge_diffusion_2d` uses the wedge validator.

Acceptance of `test_x_wedge_diffusion_2d` (validator-owned): activation (a period that differs
from $2\pi$ by more than one, a seam gradient above 0.05, and a radial decay contrast above 0.1),
finite state, density $\ge-2\times10^{-13}$, relative mass change $\le2\times10^{-11}$; density
$L_1\le2\times10^{-3}$ with final order at least 0.75 and monotonic improvement; density
$L_\infty\le2\times10^{-2}$ with monotonic improvement; and momentum-vector $L_1\le2\times10^{-3}$
with final order at least 0.75 and monotonic improvement.

### 7.2 Positivity-controlled diffusion

`test_diffusion_poslimit` has two branches, both built with `POS_LIMIT = 0.9`.

The resolution-swept branch starts from a high-contrast azimuthal mode,
$1+0.9\,\overline{\sin2\phi}$, that activates automatic positivity subcycling. Its exact same-grid
reference applies the Crank–Nicolson amplification factor of each substep with the eigenvalue of the
discrete second-difference operator,

$$
\lambda_h=\frac{4}{\Delta x^2}\sin^2\!\left(\frac{k\Delta x}{2}\right),
$$

and the driver also records the same substeps applied manually. A continuum reference measures
spatial convergence. The branch demonstrates that the positivity controller does not change the
intended Crank–Nicolson solution or create negative density.

Acceptance of the swept branch (validator-owned): activation (constant diffusivity, automatic
subcycling, more than one substep), finite output, agreement with the discrete amplification and
with the manual substep sequence to $5\times10^{-11}$, every substep minimum
$\ge-5\times10^{-14}$, an initial state equal to the prescribed mode to $5\times10^{-15}$, and a
final observed order of the continuum-reference $L_1$ error of at least 1.8.

Three fixed eight-cell directional variants additionally exercise the donor-outflow limiter in the
azimuthal, radial, and polar production kernels. Each starts from a $20{:}1$ density front and
nonuniform values of all three stored primitives. The timestep is chosen from the largest CN
coefficient. The validator independently reconstructs the unlimited CN trial and its face fluxes,
and requires at least one donor to export a larger fraction of its old mass than `POS_LIMIT`
allows. The validator requires:

- activation: a raw donor-outflow fraction above $0.9$ and a limited density that differs from the
  unlimited trial by more than $10^{-10}$;
- finite output and density $\ge-5\times10^{-13}$;
- geometry-weighted conservation of mass and all three stored momenta to $5\times10^{-12}$;
- every recovered primitive to remain in the range of the old donor values to $5\times10^{-12}$;
- no relative increase, beyond $5\times10^{-12}$, of the geometry-weighted convex quadratic
  $\sum_iM_iq_i^2$ for each primitive, where $M_i$ is the cell mass.

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

for a force that varies linearly over one step of length $\Delta t=1$. The eight cells are a
parameter index rather than a spatial grid: they carry $\Delta t/t_s=10^{-6}$, $10^{-3}$, $0.1$,
$1$, $10$, $10^2$, $10^4$, and $10^6$, so the test covers weak through strongly stiff drag
without treating spatial resolution as a convergence parameter. The validator evaluates the
closed-form exponential integral in 60-digit decimal arithmetic, independently of the device's
small-argument series, for all three velocity components.

This case replaces `source_update` with a test-local kernel that prescribes the gas velocity,
stopping times, and force endpoints, then calls the production `_get_drag_weights` quadrature. It
validates that stiff response, not the disk-dependent production source integration, which is
exercised by the attenuation and coupled ring cases.

Acceptance (shared gate): finite output, relative mass change $\le10^{-8}$, and maximum error
$\le10^{-12}$ over every compared field.

### 8.2 Optical depth

`test_optdepth` verifies the radial accumulation

$$
\tau(R)=\int_{R_{\min}}^R\kappa\rho_d(R')\,dR'.
$$

For the retained unit opacity and $\rho_d\propto R^{-1}$ profile,

$$
\tau(R)=\ln\!\left(\frac{R}{R_{\min}}\right).
$$

The comparison evaluates the production inclusive radial scan at each cell's outer face against
the exact integral to that face, so it checks both the local extinction increment and the
cumulative radial scan. It uses the shared gate with optical depth as the primary field.

### 8.3 Finite attenuation

`test_attenuation_2d` couples the computed optical depth to

$$
a_{\rm rad}(R)=\beta\frac{GM}{R^2}e^{-\tau(R)}
$$

for one source step with $\beta=1$ and unit opacity. Transport and diffusion are absent, so their
errors cannot obscure whether attenuation reaches `source_update` correctly. The validator
reconstructs the discrete inclusive scan, interpolates the cell-center optical depth that the
source update uses, and independently evaluates the exact frozen-coefficient drag weights. It
compares optical depth, unchanged density, angular-momentum relaxation, radial velocity, and the
time-centered source response with this same-grid exact reference. A separate continuum
optical-depth integral supplies a refinement-order check, preventing the test from validating only
a duplicated discrete prefix sum.

Acceptance (validator-owned): activation (well-mixed 2D model with unit radiation taper), finite
source response, density error below $5\times10^{-14}$, discrete optical-depth error below
$5\times10^{-12}$, source-response error below $2\times10^{-11}$, and a final observed order of
the continuum optical-depth $L_1$ error of at least 1.8.

## 9. Coupled production composition

`test_ring_all_2d` evolves an azimuthal Fourier mode on a power-law gas-density background through
the normal palindromic composition of diffusion, transport, optical depth, drag/source terms, and
radiation pressure, to $t=1$ with a timestep of one quarter of an azimuthal cell at the fastest
orbit. Radiation uses $\beta=0.2$ with zero opacity, so the force is known and unattenuated and
the analytical optical depth is zero. The mode rotates at the radiation-reduced Keplerian rate
$\Omega=\sqrt{(1-\beta)GM/R^3}$ and decays by azimuthal diffusion with the finite-Stokes
diffusivity $D/(1+{\rm St}^2)$. The reference combines this decay with the exact equilibrium
angular momentum $\sqrt{(1-\beta)GMR}$ and zero radial motion.

This case is intentionally retained even though its component operators are tested separately: it
detects ordering, buffer, and conserved-to-primitive conversion errors that isolated operator
tests cannot expose. It uses the shared gate on density and momentum; the velocity and
optical-depth errors are archived.

It is a short controlled coupled test, not a claim that an arbitrary nonlinear disk has a
closed-form solution. The verification driver reproduces the operator composition while replacing
the production runtime; its native pass does not qualify every production-runtime branch or
output/restart path.

## 10. Interpreting results and the archive contract

Commands, output directories, and the per-sweep archive layout are documented in
[`val/README.md`](../val/README.md). This section explains what the archived files mean.

### 10.1 Records and manifests

- A metric record (`metrics_N<res>.json`, with a variant suffix for parameter variants) is the
  validator output for one case at one resolution: error norms per field, activation and
  diagnostic values, the evidence tier, and, for validator-owned cases, the record's own `passed`
  flag.
- A model manifest (`manifest_<variant>.json`, or `manifest_default.json`) covers one case across
  its resolutions. It lists its metric files and environment record and stores the gate
  assessment: finiteness, observed orders, sequence checks, and the overall `passed`.
- The suite manifest (`manifest_all.json`) lists every case of the canonical matrix with its
  arguments and status, the effective resolutions, the sweep, and the metric-tier counts.

### 10.2 What a pass means

`val/check_archive.py` accepts a fluid sweep only when the suite manifest reports
`passed = true` for the complete canonical case list, every model manifest exists and passes with
its referenced metrics and environment record, all 76 metric records are present and finite, and
the tier counts partition the records. A passing archive therefore means that every case met the
gate listed above at the recorded resolutions, on one backend and one sweep.

The strongest evidence is the combination of

- agreement with independently constructed finite-volume reference fields;
- the expected resolution trend rather than a single permissive tolerance;
- conservation or exact escaped-mass accounting appropriate to the boundary;
- activation checks showing that the tested boundary, limiter, or seam branch was exercised;
- momentum as well as density within tolerance;
- consistent CUDA and ROCm archives from one source fingerprint.

### 10.3 Cross-backend comparison

`val/compare_backends.py` compares corresponding CUDA and ROCm fluid metric records field by field
with a default relative tolerance of $10^{-5}$ and absolute tolerance of $10^{-11}$, after
excluding the vendor-dependent initialized mass of `test_startup_3d`. It also requires both suite
manifests to pass with equal tier counts and, unless explicitly disabled, the two campaign records
to carry the same source fingerprint. The comparison does not require bitwise identity.

### 10.4 Evidence boundary

These are source-defined tests and acceptance criteria. Only a fresh native campaign whose source
fingerprint matches the cited source qualifies the CUDA or ROCm fluid solver. Source inspection,
build dry runs, and passing records produced from different source do not supply that numerical
evidence.

## 11. Deliberate limits

The suite does not claim exhaustive flag coverage, bitwise CUDA/ROCm identity, performance
superiority, restart correctness, or recovery after deliberately injected NaN/Inf values. The
isolated diffusion cases use constant $D$ in the tracer limit and do not test concentration-mode
diffusion or imported gas fields. The suite also does not use a manufactured full-3D solution
without a natural closed-form reference. Add a new case only when it supports a distinct
scientific or numerical claim that the retained matrix does not already establish.
