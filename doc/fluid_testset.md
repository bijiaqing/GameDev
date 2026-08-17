# Eulerian fluid verification

## Overview and scope

The shared definitions under `qav/comm/fluid/`, together with the drivers under `qav/cuda/fluid/` and
`qav/rocm/fluid/`, compare the production Eulerian dust-fluid implementation with
analytical finite-volume solutions or with an independently executed implementation of the same
discrete method. They are controlled verification problems, not production disk setups. The suite
tests the radial–azimuthal surface-density model, individual spherical 3D directional operators,
source integration, optical-depth construction, selected operator combinations, and both line-sweep
implementations.

Most cases call the production kernels directly from a backend test driver. The common test
`param_phys.cuh` imports the production physical helpers under private names and replaces only
`_get_nu` and `_get_alpha` when a constant analytical diffusivity is requested. In addition,
`test_source_drag` deliberately provides a model-local `source_update.cu`; it preserves the
production quadrature and interface but supplies closed-form gas velocities, stopping times, and
force endpoints. These substitutions isolate numerical operators whose production disk
coefficients would not yield simple closed-form solutions.

This document records the test design, mathematical references, acceptance semantics, execution
contract, and latest archived native results. The QAV implementation points back to this document rather
than maintaining a second copy of the case definitions. Generated JSON and binary field records
are ignored by Git below `qav/logs/`; the latest local archive was generated
from the source fingerprint and environments recorded in [`README.md`](README.md).

## Verification architecture and claim levels

The fluid tests provide three complementary kinds of evidence:

1. **Analytical finite-volume tests** compare isolated transport, diffusion, source, and
   optical-depth calculations with independently evaluated cell averages or closed-form states
2. **Coupled rotating-ring tests** compare the production symmetric operator composition with a
   smooth equilibrium whose rotation and optional diffusion have an analytical solution
3. **Thread/block differential tests** run the two production line-sweep implementations from
   byte-identical initial data and enforce finite-state, nonnegative-density, field-difference, and
   mass-difference tolerances

The first two levels answer whether the computed state approaches the mathematical solution. The
third answers whether two implementations intended to realize the same discrete algorithm remain
equivalent. Agreement between thread and block sweeps is not an independent proof that their shared
equations are correct; that conclusion comes from the analytical cases.

### Evidence tiers

The tier controls how evidence should be presented, not whether a test is retained:

- **Publication:** one fractional FARGO sequence (`shift=3.25`), the low-CFL cylindrical,
  spherical-radial, and polar transport convergence sequences, all four directional diffusion
  cases, source quadrature, logarithmic and generic optical-depth profiles, and all four coupled
  ring models
- **Release:** the integer and additional fractional FARGO shifts, operational-CFL radial and
  polar transport repetitions, and the constant-opacity exact optical-depth branch
- **Qualification:** matched thread/block sweeps, deliberate failure paths, large-LDS launches,
  and performance/profiling records

The four-resolution common archive deliberately retains both publication and release tiers: 57 of
its 85 metrics support the compact publication matrix and 28 are release-only regressions.
Per-case, per-model, and aggregate JSON manifests store these labels directly.

The analytical runner writes every metric, applies a deliberately broad regression gate, and records
the assessment in a variant-specific model manifest. Non-finite metrics, relative mass change above
$10^{-8}$, failure of an exact/source tolerance, a finest-grid primary $L_1$ error above $2\times
10^{-2}$, or loss of the established refinement trend makes the model and suite return nonzero. A
four-resolution non-degenerate sequence requires its final observed order to be at least $1.5$;
short workflow checks require $0.75$. Exact integer-shift and $p=0$ optical-depth cases use a
$10^{-10}$ finest-error gate rather than a meaningless roundoff order, while the source test uses a
$10^{-12}$ maximum error gate. These are regression limits around the substantially better archived
results, not claims that errors near the limits are scientifically adequate. The dedicated sweep
comparator continues to apply its tighter differential tolerances.

## Measurement and acceptance protocol

### Deterministic field errors

Analytical fields are averaged over the same finite-volume cells used by the production solver. For cell
error $e_i$ and exact cell measure $V_i$, the validator reports

$$
L_1=\frac{\sum_iV_i|e_i|}{\sum_iV_i},
\qquad
L_2=\left(\frac{\sum_iV_i e_i^2}{\sum_iV_i}\right)^{1/2},
\qquad
L_\infty=\max_i|e_i|.
$$

For a 2D radial–azimuthal cell and a 3D spherical cell, respectively,

$$
V_{ijk}^{(2D)}
=\Delta x\,\frac{y_{j+1}^2-y_j^2}{2},
$$

$$
V_{ijk}^{(3D)}
=\Delta x\,\frac{y_{j+1}^3-y_j^3}{3}
\left(\cos z_k-\cos z_{k+1}\right).
$$

The common azimuthal factor may be omitted from a normalized norm or relative mass change because
it cancels exactly. The total dust mass used by the validator is

$$
M_d=\sum_i\rho_{d,i}V_i,
\qquad
\epsilon_M=
\frac{|M_d(t)-M_d(0)|}{|M_d(0)|}.
$$

Each dynamical case compares density and the three conserved fields

$$
(\rho_d,\rho_d\ell_x,\rho_dv_y,\rho_d\ell_z),
$$

then separately compares the physical linear velocities written to file. Because $\ell_x$ and
$\ell_z$ are specific angular momenta in memory,

$$
v_x=\frac{\ell_x}{R},
\qquad
v_z=\frac{\ell_z}{y}.
$$

Velocity norms include only cells where both the exact and numerical densities exceed $10^{-12}$
of the exact peak density. This mask prevents the production vacuum convention from being reported
as transport error while retaining the conserved-field error in every cell.

### Convergence interpretation

The observed order between factor-of-two refinements is

$$
p=\log_2\left(\frac{E_h}{E_{h/2}}\right).
$$

For a smooth mode, a persistent positive $p$ near the formal order is evidence of convergence.
Orders computed from roundoff-level errors are not meaningful, and pre-asymptotic coarse grids may
temporarily produce lower or higher values. Spatial and temporal conclusions must also remain
separate: refinement at fixed CFL measures the combined space–time method, while a pure temporal
order requires a fixed spatial grid and independent timestep refinement.

Every retained refinement record should contain:

- density, all three conserved-momentum norms, and all three physical-velocity norms
- relative total-mass change and accepted step count
- the realized CFL number, FARGO shift, optical-depth power, and final time where applicable
- the selected line-sweep implementation, backend, compiler flags, GPU compiler, GPU, and driver

The thread/block comparator uses explicit automatic criteria:

$$
\frac{\|\rho_d^{\rm block}-\rho_d^{\rm thread}\|_2}
{\|\rho_d^{\rm thread}\|_2}\le10^{-7},
$$

$$
\frac{\|v_j^{\rm block}-v_j^{\rm thread}\|_2}
{\|v_j^{\rm thread}\|_2}\le10^{-5}
\quad (j=x,y,z),
$$

$$
\left|
\frac{M_d^{\rm block}-M_d^{\rm thread}}{M_d^{\rm thread}}
\right|\le10^{-10}.
$$

It additionally requires byte-identical initial fields, finite compared fields, and nonnegative
final density. Accepted-step counts and individual mass drifts are reported as diagnostics rather
than equality requirements because last-bit state differences may perturb later CFL choices.

### Independent-reference construction

The CUDA initializer uses eight-point Gauss–Legendre cell integration, whereas the Python validator
uses sixteen points. Radial diffusion references come from an independent fine-grid RK4 integration
of the Sturm–Liouville equation rather than from the production Crank–Nicolson matrix. The source
reference evaluates the closed-form exponential response independently and uses a cancellation-safe
series at small stiffness. Distinct transverse specific momenta $(0.7,-0.15,0.11)$ expose component
swaps or direction-specific momentum errors.

No numerical field from one fluid sweep is used as the analytical truth for another. The only
differential comparison is the explicitly labeled thread/block equivalence branch.

### Meaning of `--res`

The shared resolution argument refines different axes according to the case:

| Test family | Mesh generated from $N=$ `--res` | Meaning of refinement |
|---|---|---|
| azimuthal transport or diffusion | $(N_X,N_Y,N_Z)=(N,4,1)$ | refine the periodic $x$ operator |
| cylindrical radial transport, diffusion, or optical depth | $(4,N,1)$ | refine the 2D radial operator |
| spherical radial transport or diffusion | $(4,N,4)$ | refine the 3D radial operator while retaining transverse indexing |
| polar transport or diffusion | $(4,4,N)$ | refine the spherical polar operator |
| rotating-ring models | $(N,N,1)$ | refine both active 2D directions |
| source drag | $(8,1,1)$ | eight stiffness parameters; fixed `--res 8`, not a spatial grid |
| 2D thread/block sweep | $(N,N,1)$ | fixed-work implementation comparison |
| 3D thread/block sweep | $(N,N,N)$ | fixed-work implementation comparison with diffusion |

For the analytical matrix, `--quick` retains only the first two requested resolutions. Thus the
default full matrix performs 85 builds/runs, while the default quick matrix performs 43. Every
resolution and parameter variant is cleaned and rebuilt because the mesh and control parameters
are compile-time constants.

### Output artifact contract

| Artifact | Location and format | Purpose |
|---|---|---|
| raw analytical fields | `qav/logs/fluid/BACKEND/SWEEP/MODEL[/VARIANT]/*.dat` | headerless binary `real` arrays in x-fastest order |
| per-build metadata | `meta_N*.json` beside the raw fields | realized grid, time, step count, CFL, shift, and power |
| analytical metrics | `qav/logs/fluid/BACKEND/SWEEP/MODEL/metrics_*.json` | norms and mass change for one resolution and variant |
| environment record | `environment.json` in the model or variant directory | backend compiler, GPU, and driver diagnostics |
| sweep build and run records | `build.json` and `run.json` in each sweep-model directory | commands, return states, compiler or runtime output, accepted steps, and wall time |
| sweep timing | `timing.json` in each sweep-model directory | measured wall time for one implementation |
| model assessment | `qav/logs/fluid/BACKEND/SWEEP/MODEL/manifest_VARIANT.json` | metric inventory, regression assessment, and pass state |
| full suite assessment | `qav/logs/fluid/BACKEND/SWEEP/manifest_all.json` | complete common-matrix live state and final pass state |
| focused suite assessment | `qav/logs/fluid/BACKEND/SWEEP/groups/GROUP/manifest.json` | isolated focused-group state and final pass state |
| sweep assessment | `qav/logs/fluid/BACKEND/sweep_comparison_DIM_CONFIGURATION*.json` | automatic equivalence metrics and `passed: true` |

`SWEEP` is `thread` or `block`. Parameter variants are kept separate as `shift*`, `cfl*`, or `p*`
subdirectories so one run cannot overwrite another variant's raw data. The corresponding tag is
also included in the metric filename. Per-model manifests apply finite-value, mass, accuracy, and
convergence gates, while group-specific suite manifests retain the complete run inventory without
being overwritten by later focused groups.

Active QA runners create no persistent text result files. The retained clean archives therefore
need no legacy text migration step.

## Coverage matrix and case inventory

The current CUDA matrix covers the following combinations:

| Capability | 2D radial–azimuthal | directional 3D | fully coupled 3D |
|---|---:|---:|---:|
| conservative transport | yes, $x$ and cylindrical $y$ | yes, spherical $y$ and polar $z$ separately | no |
| density diffusion with momentum transport | yes, $x$ and cylindrical $y$ | yes, spherical $y$ and polar $z$ separately | differential sweep only |
| drag/source quadrature | dimension-independent algebraic case | no separate 3D spatial case | no coupled 3D case |
| optical-depth construction | yes | no analytical 3D attenuation case | no |
| radiation force | unattenuated equilibrium rings | no | no |
| transport–diffusion–radiation composition | four 2D ring combinations | no | no |
| thread/block implementation equivalence | yes | yes, production-like 3D state | yes as a differential test, not analytical truth |

Here, “directional 3D” means that the 3D metric and storage are active while only one directional
operator is evolved. It does not establish complete multidirectional coupling. The production fluid
branch targets radial–azimuthal 2D and full 3D models, so there is no separate 1D fluid-test column;
the cylindrical radial cases isolate a 2D directional operator rather than a standalone 1D model.

| Model | Production calculation exercised | Analytical solution |
|---|---|---|
| `test_x_transport_2d` | periodic FARGO, PPM/HLL, conservative limiter | rotating Fourier mode after one revolution |
| `test_y_transport_cyl` | radial PPM/HLL with $d=2$ geometry | compact homologous expansion |
| `test_y_transport_sph` | radial PPM/HLL with $d=3$ geometry | compact homologous expansion |
| `test_z_transport_3d` | polar PPM/HLL and $\sin z$ geometry | translation of $\rho\sin z$ at constant $\ell_\theta$ |
| `test_x_diffusion_2d` | cyclic Crank–Nicolson and diffusive momentum flux | Fourier decay on each ring |
| `test_y_diffusion_cyl` | $d=2$ radial Crank–Nicolson | Neumann shell eigenmode |
| `test_y_diffusion_sph` | $d=3$ radial Crank–Nicolson | Neumann shell eigenmode |
| `test_z_diffusion_3d` | polar Crank–Nicolson | $P_2(\cos z)$ decay |
| `test_source_drag` | exponential endpoint-force weights | linear-force drag relaxation over eight stiffness ratios |
| `test_optdepth` | optical-depth quadrature and radial prefix sum | radial power-law integral |
| `test_ring_transport_2d` | full transport/source composition | force-balanced Keplerian Fourier rings |
| `test_ring_diffusion_2d` | transport plus diffusion | rotating, exponentially damped rings |
| `test_ring_radiation_2d` | transport plus radiation | optically thin reduced-gravity rings |
| `test_ring_all_2d` | transport, diffusion, radiation, drag, gravity | damped reduced-gravity rings |
| `test_sweep_thread_2d`, `test_sweep_block_2d` | matched production-like 2D line sweeps | byte-identical initialization and tolerance-bounded differential state |
| `test_sweep_thread_3d`, `test_sweep_block_3d` | matched diffusion-enabled full-3D line sweeps | the same differential contract with exact spherical finite-volume mass |

The cylindrical and spherical radial tests isolate the Jacobian:

$$
dV_y=y\,dy\quad(d=2),
\qquad
dV_y=y^2\,dy\quad(d=3).
$$

The spherical radial model is an extruded 3D grid that evolves only the radial operator. It does
not validate multidirectional 3D coupling.

## Detailed test definitions

### Periodic FARGO transport

`test_x_transport_2d` isolates the production periodic FARGO, PPM, HLL, and conservative-limiter
path. Every radial ring begins with

$$
\rho_d(x,0)=1+0.1\cos(2x),
\qquad
\Omega=1,
$$

and therefore carries specific angular momentum

$$
\ell_x=R^2,
\qquad
v_y=\ell_z=0.
$$

The exact advected solution is

$$
\rho_d(x,t)=1+0.1\cos[2(x-t)].
$$

At $T=2\pi$ the exact solution equals the initial state. Integer and fractional ring shifts are
tested with nominal displacements of $3$, $3.25$, $3.5$, and $3.75$ cells per step. A purely integer
shift is an exact circular permutation, so its error is roundoff-level and a fitted convergence
order is meaningless. The fractional cases should show smooth second-order-or-better convergence,
retain uniform $\ell_x$, and conserve periodic mass to roundoff. This case tests one complete
azimuthal revolution but not radial shear between interacting rings, because each ring evolves
independently at the same angular speed.

### Homologous radial transport

`test_y_transport_cyl` and `test_y_transport_sph` isolate the nonuniform radial PPM/HLL sweep with
the two relevant Jacobians. Define

```math
B(y)=
\left\{\begin{array}{ll}
\exp\left[1-\dfrac{1}{1-u^2}\right],&|u|<1,\\
0,&|u|\ge1,
\end{array}\right.
\qquad
u=\frac{y-1.4}{0.4},
```

and initialize $v_y=ay$ with $a=0.2$. Before characteristic crossing,

$$
\lambda(t)=1+at,
\qquad
\rho_d(y,t)=\lambda^{-d}B\left(\frac{y}{\lambda}\right),
\qquad
v_r(y,t)=\frac{ay}{\lambda}.
$$

The dimension is $d=2$ for the vertically integrated cylindrical model and $d=3$ for the spherical
radial operator. Constant transverse specific momenta $\ell_x=0.7$ and $\ell_z=0.11$ ensure that all
conserved components follow the same mass flux. The final time is $T=0.25$, and the expanding
support remains away from both radial boundaries.

These cases test the exact radial volume, open-boundary implementation without actually crossing a
boundary, conservative momentum transport, SSPRK(3,3), CFL recomputation, and the invariant-domain
flux correction. The compact bump spans only a few cells at $N=32$, so the coarsest interval is not
expected to be asymptotic; the finer intervals supply the convergence evidence.

### Polar transport

`test_z_transport_3d` isolates the polar PPM/HLL operator on
$0.35\le z\le\pi-0.35$. Let $B(z)$ be the same smooth compact bump with support
$0.80<z<1.30$. For radial shell $j$, define

$$
G_j=\frac{\Delta A_{z,j}}{\Delta V_{y,j}}
=\frac{(y_{j+1/2}^2-y_{j-1/2}^2)/2}
{(y_{j+1/2}^3-y_{j-1/2}^3)/3}.
$$

The test initializes

$$
\rho_d(z,0)\sin z=B(z),
\qquad
\ell_{z,j}=\frac{\omega y_j}{G_j},
\qquad
\omega=0.15.
$$

The physical face speed represented by the shell is $v_{\theta,j}=\ell_{z,j}/y_j=\omega/G_j$.
Consequently, the exact finite-volume polar face-area-to-volume factor gives

$$
G_jv_{\theta,j}=\omega,
\qquad
\rho_d(z,t)=\frac{B(z-\omega t)}{\sin z}
$$

in every radial shell. The coarse fixed $N_Y=4$ mesh makes replacing $G_j$ by $1/y_j$ a visible
error rather than a negligible fine-grid approximation.

The transverse values $\ell_x=0.7$ and $v_y=0.05$ again expose component errors. The final time is
$T=0.30$, so the compact support remains away from both polar boundaries. The test therefore
validates the integrated radial polar-face measure, the $\sin z$ face measure, the
$d(-\cos z)$ cell volume, and conservative transport, but not an actual polar outflow or
midplane reflection.

### Diffusion eigenmodes

All four isolated diffusion cases use constant $D=0.05$ and final time $T=0.5$. Their timestep is
proportional to the smallest physical cell length, so refinement reduces both spatial and temporal
errors. The initialized specific momenta are uniform,

$$
(\ell_x,v_y,\ell_z)=(0.7,-0.15,0.11),
$$

which makes the exact conserved fields proportional to the diffusing density. Retention of these
ratios tests the time-centered diffusive mass flux and donor momentum closure in addition to the
density solve.

#### Azimuthal diffusion

`test_x_diffusion_2d` uses a periodic Fourier eigenmode. On each ring, the mode decays as

$$
\rho_d(x,t)=1+0.1e^{-4Dt/R^2}\cos(2x).
$$

The case exercises the cyclic Crank–Nicolson solve and its Sherman–Morrison closure. Each ring must
retain its mean density, and the spatially smooth mode should converge at second order.

#### Radial diffusion

`test_y_diffusion_cyl` and `test_y_diffusion_sph` use zero-flux eigenmodes of the dimension-dependent
radial Laplacian. They initialize $\rho_d=1+0.1Q_k(y)$, where $Q_k$ satisfies

$$
Q_k''+\frac{d-1}{y}Q_k'+k^2Q_k=0,
\qquad
Q_k'(Y_{\min})=Q_k'(Y_{\max})=0,
$$

with

$$
k_{d=2}=1.694299217770420,
\qquad
k_{d=3}=1.874562003084784.
$$

The exact solution is

$$
\rho_d(y,t)=1+0.1Q_k(y)e^{-Dk^2t}.
$$

The validator integrates the reference Sturm–Liouville equation independently rather than reusing
the CUDA matrix. These cases distinguish $y\,dy$ from $y^2\,dy$, exercise zero-flux radial
boundaries, and deliberately retain a nonuniform gas profile to detect accidental diffusion of
$\rho_d/\rho_g$ instead of $\rho_d$.

#### Polar diffusion

`test_z_diffusion_3d` uses the upper hemisphere with `HALF_DISK`. On every radial shell,

$$
\rho_d(z,t)=1+0.1P_2(\cos z)e^{-6Dt/y^2}
$$

is an exact eigenmode because the degree-two Legendre polynomial has angular eigenvalue $-6$.
Regularity at the axis and zero derivative at the reflecting midplane match the production
boundary conditions. The case tests $1/y^2$, $\sin z$, $d(-\cos z)$, and shell-dependent decay. It
does not force the positivity controller to take multiple Crank–Nicolson substeps; that remains a
separate required regression.

### Exponential source quadrature

`test_source_drag` treats its eight x cells as eight stiffness parameters rather than spatial
locations. Each component satisfies

$$
\frac{du}{dt}=-\frac{u-u_g}{t_s}+a_0+a_1t,
$$

the exact one-step result is

$$
u(\Delta t)=u_g+(u_0-u_g)e^{-h}
+a_0t_s(1-e^{-h})
+a_1\left[t_s\Delta t-t_s^2(1-e^{-h})\right],
\qquad
h=\frac{\Delta t}{t_s}.
$$

The eight values

$$
h\in\{10^{-6},10^{-3},0.1,1,10,10^2,10^4,10^6\}
$$

span the non-stiff, transitional, and strongly stiff limits. The three components use different
initial states, gas targets, and force endpoints so a component swap cannot pass. This test isolates
the endpoint-force exponential quadrature from gravity, disk geometry, and evolving gas fields.

### Optical-depth construction

`test_optdepth` performs no time integration. It initializes
$\Sigma_d=\sqrt{2\pi}H_g y^p$, so the well-mixed midplane extinction density is proportional to
$y^p$ and

```math
\tau(y)=
\left\{\begin{array}{ll}
\dfrac{y^{p+1}-Y_{\min}^{p+1}}{p+1},&p\ne-1,\\
\ln(y/Y_{\min}),&p=-1.
\end{array}\right.
```

at every radial outer face. The powers $p=0,-1,1$ exercise a constant integrand, the logarithmic
antiderivative, and an ordinary power law. The case validates cell optical-depth quadrature,
x-fastest/radial indexing, and the inclusive radial prefix sum. It does not apply the resulting
$e^{-\tau}$ attenuation to a radiation force.

### Coupled rotating rings

The four two-dimensional end-to-end models are `test_ring_transport_2d`,
`test_ring_diffusion_2d`, `test_ring_radiation_2d`, and `test_ring_all_2d`. They use

$$
\frac{\Sigma_d(R,x,0)}{\Sigma_g(R)}=1+0.1\cos(2x),
\qquad
\Omega_\beta(R)=\sqrt{\frac{1-\beta}{R^3}},
$$

with gas pressure support chosen so drag vanishes and radial forces balance. When azimuthal
diffusion is enabled,

$$
\frac{\Sigma_d(R,x,t)}{\Sigma_g(R)}
=1+0.1
\exp\left(-\frac{4D_xt}{R^2}\right)
\cos\left(2[x-\Omega_\beta(R)t]\right).
$$

Non-radiating models set $\beta=0$; radiating models set $\beta=0.2$, unit startup taper, and
$\kappa=0$. Diffusing models use $D_x=0.05$ and suppress cross-direction diffusion. The final time
is $T=1$.

The test gas profile is selected so its pressure-supported angular momentum equals that of the
dust. Drag therefore vanishes, and centrifugal acceleration balances stellar gravity plus the
configured radiation reduction. Any growing radial velocity reveals a source imbalance. Any
change of the ring-specific $\ell_x$ reveals inconsistent transport or diffusive momentum closure.

Together the four models cover transport only, transport plus diffusion, transport plus radiation,
and transport plus diffusion plus radiation. Because they are radial–azimuthal models, their
evolved density is the vertically integrated dust surface density. They validate unattenuated
radiation-force composition because $\kappa=0$; they do not test finite $e^{-\tau}$ coupling.

## Thread/block implementation comparison

The production fluid branch has two line-sweep implementations:

- `thread` assigns one GPU thread or work-item to one complete line and retains the reference arithmetic order
- `block` assigns one GPU block or workgroup to one line, stores explicit work arrays, and uses cooperative
  line operations to avoid very large thread-local arrays

The analytical matrix can be run independently with either implementation by setting
`FLUID_SWEEP=thread` or `FLUID_SWEEP=block`. The dedicated `sweep` group instead runs matched
production-like pairs and compares their saved states automatically. Its full configuration uses a
transport-only $1024^2$ model and a diffusion-enabled $128^3$ model, with radiation disabled in
both to avoid introducing a physical instability into an implementation comparison.

The comparator requires byte-identical frame-zero fields, finite outputs, nonnegative final
density, the relative field tolerances given above, and a finite-volume final-mass mismatch below
$10^{-10}$. It records each implementation's own mass drift, accepted step count, wall time, build
log, and environment. This branch tests discrete equivalence and performance on fixed work; it is
not a grid-convergence sequence and does not use one implementation as analytical truth.

## Latest archived native and cross-backend evidence

### Complete 2026-08-17 analytical archives

The complete thread-sweep matrix was regenerated on both backends under the source-matched campaign
recorded in [`README.md`](README.md), which also records the native compilers, targets, drivers,
and GPUs.

Both aggregate manifests report 22/22 parameter variants and 85/85 metrics passed, partitioned
into 57 publication and 28 release records. Independent local archive checks found no invalid
metrics or missing raw fields. The command on each native system was

```bash
python3 qav/tool/run_all.py --backend BACKEND --target TARGET --res 32 64 128 256
```

The following CUDA density $L_1$ orders use $N=32,64,128,256$ and list the three successive
refinement intervals. The ROCm archive independently passed the same per-model criteria, and the
cross-backend report accepted all corresponding metrics. Each model manifest retains the complete
error and order sequence.

| Case | Successive $L_1$ orders | Assessment |
|---|---|---|
| FARGO shift 3.25 | 2.3067, 2.4063, 2.4158 | pass |
| FARGO shift 3.50 | 2.3079, 2.3029, 2.4658 | pass |
| FARGO shift 3.75 | 2.0579, 2.4278, 2.4282 | pass |
| radial transport, $d=2$, CFL 0.05 | 1.6429, 2.2371, 2.6991 | pass; coarsest compact profile is under-resolved |
| radial transport, $d=2$, CFL 0.5 | 1.5555, 2.4004, 2.7963 | pass; same caveat |
| radial transport, $d=3$, CFL 0.05 | 1.5876, 2.2031, 2.7182 | pass; same caveat |
| radial transport, $d=3$, CFL 0.5 | 1.5235, 2.3568, 2.8012 | pass; same caveat |
| polar transport, CFL 0.05 | 1.1055, 2.2062, 2.3834 | pass; the coarse $N_Z=32$ profile is pre-asymptotic |
| polar transport, CFL 0.5 | 1.1642, 2.0916, 2.8261 | pass; same caveat |
| azimuthal diffusion | 1.9952, 1.9988, 1.9997 | pass |
| radial diffusion, $d=2$ | 1.9903, 1.9937, 1.9993 | pass |
| radial diffusion, $d=3$ | 1.9796, 1.9969, 1.9980 | pass |
| polar diffusion | 1.9974, 1.9997, 1.9999 | pass |
| optical depth, $p=-1$ | 2.0148, 2.0076, 2.0038 | pass |
| optical depth, $p=1$ | 2.0374, 2.0192, 2.0097 | pass |
| coupled transport | 2.1396, 2.2254, 2.2565 | pass |
| coupled transport + diffusion | 3.2074, 3.1870, 2.8243 | pass; pre-asymptotic superconvergence is not a third-order claim |
| coupled transport + radiation | 2.1483, 2.2076, 2.2699 | pass |
| coupled all physics | 3.1804, 3.1817, 2.6870 | pass; same caveat |

The low- and production-CFL transport step sequences are:

| Case | CFL | Steps at $N=32,64,128,256$ | Maximum recorded relative mass change |
|---|---:|---|---:|
| radial transport, $d=2$ | 0.05 | 21, 41, 81, 160 | $1.16\times10^{-12}$ |
| radial transport, $d=2$ | 0.5 | 3, 5, 9, 16 | $9.77\times10^{-14}$ |
| radial transport, $d=3$ | 0.05 | 22, 42, 81, 160 | $1.82\times10^{-12}$ |
| radial transport, $d=3$ | 0.5 | 3, 5, 9, 16 | $4.93\times10^{-16}$ |
| polar transport | 0.05 | 13, 25, 48, 95 | $3.99\times10^{-15}$ |
| polar transport | 0.5 | 2, 3, 5, 10 | $4.27\times10^{-16}$ |

The larger low-CFL mass roundoff in the two radial tests accumulates over roughly ten times as many
steps and remains below $2\times10^{-12}$.

The integer FARGO case remains near machine roundoff. Its negative reported orders are ratios of
roundoff noise, not a failed transport test. The $p=0$ optical-depth case can likewise be integrated
exactly or nearly exactly by the logarithmic midpoint rule, so its order is not meaningful.

The source reference uses the same cancellation-safe endpoint-weight series as the production GPU kernel for
$h<10^{-4}$. Reanalysis of the unchanged native state reduced the largest componentwise discrepancy
from $2.22\times10^{-5}$ to $2.22\times10^{-16}$; the former value was cancellation in the Python
reference, not a production-kernel error. Density and total mass are exact to the recorded precision.

### CUDA-versus-ROCm equivalence

The complete backend comparison found all 85 analytical metrics on both sides, matching tier
metadata, no missing records, and zero acceptance mismatches. Polar error norms are particularly
sensitive to vendor-dependent reductions, so the gate replaces a direct comparison of those
already reduced norms with an eight-record raw-field comparison of `test_z_transport_3d` at both
CFL values and all four resolutions.

Initial density was byte-identical in every raw-field record, and metadata and accepted-step counts
agreed. Among final density and conserved momenta, the largest relative differences were

$$
L_{2,\mathrm{rel}}=2.43\times10^{-6},
\qquad
L_{\infty,\mathrm{rel}}=3.97\times10^{-6},
$$

both below the $10^{-5}$ gate. The largest density-weighted velocity relative $L_2$ difference was
$4.71\times10^{-11}$. Unweighted velocity differences reached $2.28\times10^{-5}$ in relative
$L_2$ and $3.45\times10^{-4}$ in relative $L_\infty$ only in low-density tails. They remain
diagnostics because division by vanishing density is ill-conditioned; conserved fields and the
density-weighted velocity norm define acceptance.

The comparison report also verifies that the CUDA and ROCm campaign source fingerprints match.
Re-executing the archive checks and both comparison scripts locally from the downloaded tree
reproduced `PASS` without a GPU runtime.

### Qualification evidence requiring a separate refresh

The 2026-08-17 `run_all.py` campaign is the common publication-and-release gate. It deliberately
does not run the qualification-only thread/block sweep comparison, deliberate NaN/Inf injection,
large-LDS launch suite, or performance/profiling campaign. Earlier numerical and timing values for
those branches have therefore been removed from the current-result section rather than presented
as evidence for this source fingerprint.

Refresh those independent qualification branches when their claims are needed:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --sweep-dim all
python3 qav/rocm/fluid/test_common/run_suite.py --group sweep --sweep-dim all --target gfx942
python3 qav/rocm/fluid/test_common/run_suite.py --group failure --target gfx942
python3 qav/rocm/fluid/test_common/run_suite.py --group lds --target gfx942
```

The transferable release workflow, archive paths, and backend-neutral comparison commands are
specified in `qav/README.md`.

## Running and archiving the suite

Each native suite requires its backend toolchain, Python 3.9 or newer, and NumPy. GPU
diagnostic commands are recorded when available but are not numerical prerequisites.

From the repository root, use a two-resolution workflow check before a long run:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group all --quick
```

Run the complete reference-thread matrix with

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

The complete `--group all --res 32 64 128 256` matrix performs 85 builds/runs because several
models sweep CFL values, FARGO shifts, or optical-depth powers. Each resolution is cleaned and
rebuilt so model-local compile-time constants cannot reuse stale objects.

The analytical groups can be run separately:

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group transport --res 32 64 128 256
python3 qav/cuda/fluid/test_common/run_suite.py --group diffusion --res 32 64 128 256
python3 qav/cuda/fluid/test_common/run_suite.py --group source    --res 8
python3 qav/cuda/fluid/test_common/run_suite.py --group radiation --res 32 64 128 256
python3 qav/cuda/fluid/test_common/run_suite.py --group ring      --res 32 64 128 256
```

Here the `radiation` group currently means optical-depth construction; the radiation-force ring
models are in `ring`. To run the same analytical matrix through the block sweep, set the environment
selection explicitly:

```bash
FLUID_SWEEP=block python3 qav/cuda/fluid/test_common/run_suite.py \
    --group all --res 32 64 128 256
```

Run the automatic fixed-work differential branch with

```bash
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --quick
python3 qav/cuda/fluid/test_common/run_suite.py --group sweep --sweep-dim all
```

The full sweep defaults are $N=1024$ in 2D and $N=128$ in 3D. They can be changed with
`--sweep-res-2d` and `--sweep-res-3d`. A single model can be invoked through its local wrapper, for
example

```bash
python3 qav/cuda/fluid/test_x_transport_2d/run.py \
    --res 32 64 128 256 --shift 3.25
```

Keep the JSON metrics, `environment.json`, and any captured terminal-output JSON together when archiving a run.
For the sweep branch, also retain `build.json`,
`run.json`, `variables.json`, `timing.json`, and `sweep_comparison_*.json`. Analytical results are namespaced as
`qav/logs/fluid/cuda/SWEEP/groups/manual/MODEL/`, where `SWEEP` is `thread` or `block`.

The full analytical archive is complete when `manifest_all.json` reports `"passed": true`, all 85
metric files are present, and `qav/tool/check_archive.py` accepts the archive. Focused groups write
their models and manifest below `groups/GROUP/`, while direct wrappers use `groups/manual/`; neither
can overwrite any full-suite artifact. The sweep branch is
self-gating: a returned comparison JSON with `"passed": true` means all of its implemented
tolerances were satisfied.

## Coverage limits and verification still required

### Planned full-3D manufactured solution

No runnable model currently claims full coupled 3D verification. The planned `test_mms_3d` must
generate its density and all three stored-momentum residuals independently with SymPy, emit
test-only algebraic forcing expressions, insert density and momentum forcing symmetrically at the
required substep times, impose exact manufactured boundary values, and compare independent
high-order finite-volume averages of density, all momenta, physical velocities, and optical depth.
The reference must not call production geometry, gas, force, diffusion, or optical-depth helpers.

The harness must cover four configurations:

1. transport only with zero manufactured diffusivity and radiation
2. transport plus diffusion
3. transport plus radiation
4. transport plus diffusion and radiation

Because production requires `DIFFUSION` whenever `N_Z > 1`, the first and third configurations
must compile with that flag while returning exactly zero manufactured diffusivity. All dimensions
must refine together with the same prescribed timestep history at a given resolution. Before any
MMS order is interpreted, a source-only field with a known exact integral must validate the forcing
composition, the exact solution must stay above `RHO_VAC`, and limiter and positivity scaling must
remain inactive. Radiation cases retain the second-order floor of the production optical-depth
quadrature. Both full-disk and `HALF_DISK` polar boundaries should ultimately be exercised.

### Other missing verification

- Add production initialization tests for the convolved dust surface-density profile, exact
  hydrostatic gas stratification, size-dependent Stokes number, equilibrium angular momentum, and
  initial radial/polar drift in both 2D and 3D
- Add production-settling equilibrium tests that quantify the initial polar transient
- Add independent nonzero-$(p,q)$ tests of the midplane pressure-support target and the complete
  3D gas rotation law, evaluating the fixed-$Z$ pressure gradient and cylindrical stellar gravity
  separately before verifying their off-midplane cancellation
- Add radial and polar cases whose nonzero support actually crosses an outflow boundary, plus a
  `HALF_DISK` midplane-reflection case; current compact profiles remain away from those boundaries
- Add periodic-wedge seam cases for FARGO transport, PPM limiting, and cyclic azimuthal diffusion
  on domains with $X_{\max}-X_{\min}<2\pi$
- Run polar transport at fixed $N_Z$ while varying $N_Y$ independently, and verify both the exact
  $\Delta A_z/\Delta V_y$ radial metric and conservation under the complete 3D cell measure
- Add a long-running periodic FARGO-plus-diffusion ring test that retains total mass and all three
  globally integrated conserved momentum components to roundoff
- Add a limiter stress test with converging streams and near-vacuum cells, requiring nonnegative
  density, bounded momentum-to-density ratios, and no activation of the final vacuum fallback
- Add a resolved-vertical `VISC_FLOW` test at several heights against the complete Kanagawa target
- Add a vertically integrated `VISC_FLOW` test for the two-dimensional gas target and resulting
  steady dust drift
- Add restart-tolerance and legal flag-matrix regressions, including `HALF_DISK`, `DIFFUSION`,
  `RADIATION`, `VISC_FLOW`, and both fluid sweep implementations where compatible
- Add a CUDA diffusion case that forces `POS_LIMIT` to select more than one CN positivity substep;
  this path currently has only the supplementary CPU mock
- Add finite, spatially varying radiation attenuation; current ring radiation cases deliberately
  use zero opacity and therefore isolate force composition rather than $e^{-\tau}$ coupling
- Archive matched thread/block comparison JSON, build logs, `ptxas` resource reports, and profiler
  local-memory traffic for the production grids used in performance claims
- In long conservation tests, report mass and momentum changes caused by outflow and documented
  vacuum-state repair separately from conservative interior fluxes
- Repeat important publication runs without `--use_fast_math`, or document and measure its effect

The CUDA QA runner supports this comparison directly through `--math-mode precise`.  Precise-math
records are archived below the `groups/manual/` branch of
`qav/logs/fluid/cuda/thread_precise/` or `block_precise/`, so they cannot overwrite the default
fast-math baseline. Velocity norms exclude a cell unless both the exact and
numerical densities exceed $10^{-12}$ of the exact peak density; this prevents deliberately assigned
vacuum fallback velocities from masquerading as transport error while leaving all conserved-field
norms unmasked.

The directional 3D radial and polar tests do not establish complete 3D transport–diffusion–
radiation coupling; that claim requires the manufactured-solution suite above.

## References

- Salari & Knupp (2000), [verification by manufactured solutions](https://digital.library.unt.edu/ark:/67531/metadc702130/)
- Colella & Woodward (1984), [PPM](<https://doi.org/10.1016/0021-9991(84)90143-8>)
- Harten, Lax & van Leer (1983), [HLL-type Godunov schemes](https://doi.org/10.1137/1025002)
- Masset (2000), [FARGO](https://arxiv.org/abs/astro-ph/9910390)
- Gottlieb & Shu (1998), [strong-stability-preserving Runge–Kutta methods](https://doi.org/10.1090/S0025-5718-98-00913-2)
- Crank & Nicolson (1947), [implicit diffusion](https://doi.org/10.1017/S0305004100023197)
- Sherman & Morrison (1950), [cyclic rank-one inverse update](https://doi.org/10.1214/aoms/1177729893)
- Strang (1968), [symmetric operator splitting](https://doi.org/10.1137/0705041)
- Kanagawa et al. (2017), [viscous disk gas flow](https://arxiv.org/abs/1706.08975)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](<https://doi.org/10.1016/0019-1035(79)90050-2>)
