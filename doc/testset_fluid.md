# Eulerian fluid verification

## Purpose

The CUDA models under `qav/fluid/` compare production fluid kernels with analytical
finite-volume solutions. They are verification models, not production disk setups. Model-local
files replace a production component only when the test requires a prescribed state that the
production interface cannot express.

This document records the verification design and the retained historical CUDA baseline. Detailed
validator formulas remain in `qav/fluid/test_common/TEST_CASES.md`. The current local
`qav/fluid/out/` directory is empty, so the historical table below is not backed by machine-readable
artifacts in this checkout and must be regenerated before publication or release.

## Measurement protocol

Analytical fields are averaged over the same finite-volume cells used by the CUDA solver. For cell
error $e_i$ and cell measure $V_i$, the validator reports

$$
L_1=\frac{\sum_iV_i|e_i|}{\sum_iV_i},
\qquad
L_2=\left(\frac{\sum_iV_i e_i^2}{\sum_iV_i}\right)^{1/2},
\qquad
L_\infty=\max_i|e_i|.
$$

The observed order between resolutions $h$ and $h/2$ is

$$
p=\log_2\left(\frac{E_h}{E_{h/2}}\right).
$$

Every refinement sequence should retain:

- density and all three stored momentum norms
- physical linear-velocity norms away from analytical vacuum
- relative total-mass change
- accepted step count
- active model flags, compiler flags, CUDA compiler, and GPU

Spatial and temporal conclusions must not be mixed. A smooth spatial refinement at fixed CFL
measures the combined space–time method; a separate fixed-grid CFL refinement is needed for a pure
temporal-order claim.

The CUDA initializer uses eight-point Gauss–Legendre cell averages, while the Python validator
independently evaluates sixteen-point averages. Radial diffusion references come from an
independent RK4 integration of the Sturm–Liouville problem rather than the production CN matrix.
Distinct transverse specific momenta $(0.7,-0.15,0.11)$ make component swaps visible instead of
allowing every momentum field to share one indistinguishable value.

## Analytical cases

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

The cylindrical and spherical radial tests isolate the Jacobian:

$$
dV_y=y\,dy\quad(d=2),
\qquad
dV_y=y^2\,dy\quad(d=3).
$$

The spherical radial model is an extruded 3D grid that evolves only the radial operator. It does
not validate multidirectional 3D coupling.

## Analytical solution summary

### Periodic FARGO transport

For

$$
\rho_d(x,0)=1+0.1\cos(2x),
\qquad
\Omega=1,
$$

the solution is

$$
\rho_d(x,t)=1+0.1\cos[2(x-t)].
$$

At $T=2\pi$ the exact solution equals the initial state. Integer and fractional ring shifts are
tested separately. A purely integer shift is an exact circular permutation, so its error is
roundoff-level and a fitted convergence order is meaningless.

### Homologous radial transport

With compact profile $B(y)$ and $v_r=ay$,

$$
\lambda(t)=1+at,
\qquad
\rho_d(y,t)=\lambda^{-d}B\left(\frac{y}{\lambda}\right),
\qquad
v_r(y,t)=\frac{ay}{\lambda}.
$$

This tests radial volume, open boundary handling away from the support, momentum transport,
SSPRK(3,3), and the invariant-domain limiter.

### Polar transport

For constant $\ell_\theta=L$,

$$
c_z=\frac{L}{y^2},
\qquad
\rho_d(z,t)=\frac{B(z-c_zt)}{\sin z}.
$$

The compact support remains away from the boundaries.

### Diffusion

Azimuthal Fourier modes decay as

$$
\rho_d(x,t)=1+0.1e^{-4Dt/R^2}\cos(2x).
$$

Radial shell modes satisfy

$$
Q_k''+\frac{d-1}{y}Q_k'+k^2Q_k=0,
\qquad
Q_k'(Y_{\min})=Q_k'(Y_{\max})=0,
$$

and decay as $e^{-Dk^2t}$. The validator integrates the reference Sturm–Liouville equation
independently rather than reusing the CUDA matrix.

On a reflecting upper hemisphere, the polar mode

$$
\rho_d(z,t)=1+0.1P_2(\cos z)e^{-6Dt/y^2}
$$

tests the spherical polar metric.

### Source and optical depth

For

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

The optical-depth model initializes
$\Sigma_d=\sqrt{2\pi}H_g y^p$, so the well-mixed midplane extinction density is proportional to
$y^p$ and

$$
\tau(y)=
\begin{cases}
\dfrac{y^{p+1}-Y_{\min}^{p+1}}{p+1},&p\ne-1,\\[6pt]
\ln(y/Y_{\min}),&p=-1.
\end{cases}
$$

### Coupled rotating rings

The four two-dimensional end-to-end models use

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

These cases cover transport-only, transport plus diffusion, transport plus radiation, and all
three together. Because these are two-dimensional radial-azimuthal models, their stored density is
the vertically integrated dust surface density rather than a reconstructed midplane volume
density.

## Historical native CUDA baseline

### Complete merged-tree run of 2026-07-26

The complete 85-case thread-sweep run was recorded on 2026-07-26 with

```bash
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

The suite uses the reference thread sweep by default. Prefix the same command with
`FLUID_SWEEP=block` to compile and validate the block implementation through the identical
analytical cases. The Makefile stores the builds in separate object directories, and the runner
stores their fields, metrics, and environment records under `out/thread/` and `out/block/` so the
two evidence sets coexist.

For a direct matched-output and timing comparison, use the dedicated fixed-work branch:

```bash
python3 qav/fluid/test_common/run_suite.py --group sweep --quick
python3 qav/fluid/test_common/run_suite.py --group sweep --sweep-dim all
```

The full branch runs a transport-only `1024^2` pair and a diffusion-enabled `128^3` pair, with
radiation disabled in both. It requires byte-identical initial states, finite and nonnegative final
density, relative density and velocity tolerances, and a finite-volume mass mismatch below `1e-10`;
accepted-step counts are reported as a diagnostic. It is kept separate from `--group all` because
its purpose and runtime differ from convergence testing.

The development cross-comparisons motivating this branch found configuration-dependent performance.
At `1024^2`, the stable thread and block runs took `132.93 s` and `77.82 s`, respectively, so the
block method was approximately 1.71 times faster. In the vacuum-corrected `128^3` diffusion run,
frame 0 was byte-identical and frame 10 had density relative L2 difference `1.476936e-12`, velocity
relative L2 differences from `5.261644e-16` to `3.552797e-13`, and finite-volume mass mismatch
`1.88686857e-16`. The corresponding thread and block wall times were `346.65 s` and `1182.99 s`,
making the block method approximately 3.41 times slower. These results justify retaining both
implementations and benchmarking the intended grid rather than selecting one globally.

The build logs, profiler reports, and `sweep_comparison_*.json` summaries for those development
measurements are not present in the current repository. The numbers are therefore historical
evidence rather than the canonical benchmark baseline. Rerun the automated sweep branch on the
target GPU before making a production performance claim.

The historical run completed all 85 builds and simulations without a compilation failure, runtime
failure, Python traceback, or non-finite metric. The corrected output naming included CFL values in
radial and polar transport filenames, preventing the 12 low-CFL records from being overwritten.
The recorded environment was CUDA 12.1 with `nvcc` 12.1.105 and an NVIDIA A100-SXM4-40GB using
driver 580.159.04.

The following density $L_1$ orders use $N=32,64,128,256$ and list the three successive refinement
intervals.

| Case | Successive $L_1$ orders | Assessment |
|---|---|---|
| FARGO shift 3.25 | 2.3067, 2.4063, 2.4158 | pass |
| FARGO shift 3.50 | 2.3079, 2.3029, 2.4658 | pass |
| FARGO shift 3.75 | 2.0579, 2.4278, 2.4282 | pass |
| radial transport, $d=2$, CFL 0.05 | 1.6429, 2.2371, 2.6991 | pass; coarsest compact profile is under-resolved |
| radial transport, $d=2$, CFL 0.5 | 1.5555, 2.4004, 2.7963 | pass; same qualification |
| radial transport, $d=3$, CFL 0.05 | 1.5876, 2.2031, 2.7182 | pass; same qualification |
| radial transport, $d=3$, CFL 0.5 | 1.5235, 2.3568, 2.8012 | pass; same qualification |
| polar transport, CFL 0.05 | 1.6737, 2.3601, 2.4601 | pass |
| polar transport, CFL 0.5 | 1.6813, 2.3623, 2.4999 | pass |
| azimuthal diffusion | 1.9952, 1.9988, 1.9997 | pass |
| radial diffusion, $d=2$ | 1.9903, 1.9937, 1.9993 | pass |
| radial diffusion, $d=3$ | 1.9796, 1.9969, 1.9980 | pass |
| polar diffusion | 1.9974, 1.9997, 1.9999 | pass |
| optical depth, $p=-1$ | 2.0148, 2.0076, 2.0038 | pass |
| optical depth, $p=1$ | 2.0374, 2.0192, 2.0097 | pass |
| coupled transport | 2.1396, 2.2254, 2.2565 | pass |
| coupled transport + diffusion | 3.2074, 3.1870, 2.8243 | pass; pre-asymptotic superconvergence is not a third-order claim |
| coupled transport + radiation | 2.1483, 2.2076, 2.2699 | pass |
| coupled all physics | 3.1804, 3.1817, 2.6870 | pass; same qualification |

The low- and production-CFL transport step sequences are:

| Case | CFL | Steps at $N=32,64,128,256$ | Maximum recorded relative mass change |
|---|---:|---|---:|
| radial transport, $d=2$ | 0.05 | 21, 41, 81, 160 | $1.16\times10^{-12}$ |
| radial transport, $d=2$ | 0.5 | 3, 5, 9, 16 | $9.77\times10^{-14}$ |
| radial transport, $d=3$ | 0.05 | 22, 42, 81, 160 | $1.82\times10^{-12}$ |
| radial transport, $d=3$ | 0.5 | 3, 5, 9, 16 | $4.93\times10^{-16}$ |
| polar transport | 0.05 | 33, 65, 128, 254 | $1.08\times10^{-14}$ |
| polar transport | 0.5 | 4, 7, 13, 26 | $1.14\times10^{-15}$ |

The larger low-CFL mass roundoff in the two radial tests accumulates over roughly ten times as many
steps and remains below $2\times10^{-12}$.

The integer FARGO case remains near machine roundoff. Its negative reported orders are ratios of
roundoff noise, not a failed transport test. The $p=0$ optical-depth case can likewise be integrated
exactly or nearly exactly by the logarithmic midpoint rule, so its order is not meaningful.

The source reference uses the same cancellation-safe endpoint-weight series as the CUDA kernel for
$h<10^{-4}$. Reanalysis of the unchanged native state reduced the largest componentwise discrepancy
from $2.22\times10^{-5}$ to $2.22\times10^{-16}$; the former value was cancellation in the Python
reference, not a production-kernel error. Density and total mass are exact to the recorded precision.

## Running the suite

From the repository root:

```bash
python3 qav/fluid/test_common/run_suite.py --quick
python3 qav/fluid/test_common/run_suite.py --group transport --res 32 64 128 256
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

The complete `--group all --res 32 64 128 256` matrix performs 85 builds/runs because several
models sweep CFL values, FARGO shifts, or optical-depth powers. Each resolution is cleaned and
rebuilt so model-local compile-time constants cannot reuse stale objects.

Only NumPy is required by the Python validator. Keep the JSON metrics, `environment.json`, and full
terminal output together when archiving a run. Analytical results are namespaced as
`qav/fluid/out/SWEEP/MODEL/`, where `SWEEP` is `thread` or `block`.

## Verification still required

- Implement the full coupled 3D manufactured-solution harness specified in
  `qav/fluid/test_mms_3d/README.md`, including independent forcing and exact boundary data
- Add production-settling equilibrium tests that quantify the initial polar transient
- Add independent nonzero-$(p,q)$ tests of the midplane pressure-support target and the complete
  3D gas rotation law, evaluating the fixed-$Z$ pressure gradient and cylindrical stellar gravity
  separately before verifying their off-midplane cancellation
- Add radial and polar cases whose nonzero support actually crosses an outflow boundary, plus a
  `HALF_DISK` midplane-reflection case; current compact profiles remain away from those boundaries
- Add a long-running periodic FARGO-plus-diffusion ring test that retains total mass and all three
  globally integrated conserved momentum components to roundoff
- Add a limiter stress test with converging streams and near-vacuum cells, requiring nonnegative
  density, bounded momentum-to-density ratios, and no activation of the final vacuum fallback
- Add a resolved-vertical `VISC_FLOW` test at several heights against the complete Kanagawa target
- Add restart-tolerance and legal flag-matrix regressions, including `HALF_DISK`, `DIFFUSION`,
  `RADIATION`, `VISC_FLOW`, and both fluid sweep implementations where compatible
- Add a CUDA diffusion case that forces `POS_LIMIT` to select more than one CN positivity substep;
  this path currently has only the supplementary CPU mock
- Add finite, spatially varying radiation attenuation; current ring radiation cases deliberately
  use zero opacity and therefore isolate force composition rather than $e^{-\tau}$ coupling
- Archive matched thread/block comparison JSON, build logs, `ptxas` resource reports, and profiler
  local-memory traffic for the production grids used in performance claims
- Repeat important publication runs without `--use_fast_math`, or document and measure its effect
- Define automated pass thresholds only after identifying the asymptotic range on the target GPU

The directional 3D radial and polar tests do not establish complete 3D transport–diffusion–
radiation coupling; that claim requires the manufactured-solution suite above.

## References

- Salari & Knupp (2000), [verification by manufactured solutions](https://digital.library.unt.edu/ark:/67531/metadc702130/)
- Colella & Woodward (1984), [PPM](<https://doi.org/10.1016/0021-9991(84)90143-8>)
- Masset (2000), [FARGO](https://arxiv.org/abs/astro-ph/9910390)
- Crank & Nicolson (1947), [implicit diffusion](https://doi.org/10.1017/S0305004100023197)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](<https://doi.org/10.1016/0019-1035(79)90050-2>)
