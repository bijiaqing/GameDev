# Eulerian fluid verification

## Purpose

The CUDA models under `tst/fluid/` compare production fluid kernels with analytical
finite-volume solutions. They are verification models, not production disk setups. Model-local
files replace a production component only when the test requires a prescribed state that the
production interface cannot express.

This document records the current verification claim. Detailed formulas implemented by the
validator remain in `tst/fluid/verify_common/TEST_CASES.md`, and machine-readable results remain
under sweep-specific directories in `tst/fluid/out/`.

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

## Prepared analytical cases

| Model | Production calculation exercised | Analytical solution |
|---|---|---|
| `verify_x_transport_2d` | periodic FARGO, PPM/HLL, conservative limiter | rotating Fourier mode after one revolution |
| `verify_y_transport_cyl` | radial PPM/HLL with $d=2$ geometry | compact homologous expansion |
| `verify_y_transport_sph` | radial PPM/HLL with $d=3$ geometry | compact homologous expansion |
| `verify_z_transport_3d` | polar PPM/HLL and $\sin z$ geometry | translation of $\rho\sin z$ at constant $\ell_\theta$ |
| `verify_x_diffusion_2d` | cyclic Crank–Nicolson and diffusive momentum flux | Fourier decay on each ring |
| `verify_y_diffusion_cyl` | $d=2$ radial Crank–Nicolson | Neumann shell eigenmode |
| `verify_y_diffusion_sph` | $d=3$ radial Crank–Nicolson | Neumann shell eigenmode |
| `verify_z_diffusion_3d` | polar Crank–Nicolson | $P_2(\cos z)$ decay |
| `verify_source_drag` | exponential endpoint-force weights | linear-force drag relaxation over eight stiffness ratios |
| `verify_optdepth` | optical-depth quadrature and radial prefix sum | radial power-law integral |
| `verify_ring_transport_2d` | full transport/source composition | force-balanced Keplerian Fourier rings |
| `verify_ring_diffusion_2d` | transport plus diffusion | rotating, exponentially damped rings |
| `verify_ring_radiation_2d` | transport plus radiation | optically thin reduced-gravity rings |
| `verify_ring_all_2d` | transport, diffusion, radiation, drag, gravity | damped reduced-gravity rings |

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

## Recorded native CUDA results

### Complete merged-tree run of 2026-07-26

The current JSON files under `tst/fluid/out/thread/` contain all 85 records from

```bash
python3 tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

The suite uses the reference thread sweep by default. Prefix the same command with
`FLUID_SWEEP=block` to compile and validate the block implementation through the identical
analytical cases. The Makefile stores the builds in separate object directories, and the runner
stores their fields, metrics, and environment records under `out/thread/` and `out/block/` so the
two evidence sets coexist.

For a direct matched-output and timing comparison, use the dedicated fixed-work branch:

```bash
python3 tst/fluid/verify_common/run_suite.py --group sweep --quick
python3 tst/fluid/verify_common/run_suite.py --group sweep --sweep-dim all
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

The run completed all 85 builds and simulations without a compilation failure, runtime failure,
Python traceback, or non-finite JSON metric. Unlike the earlier archive, the current output layout
includes the CFL value in radial and polar transport filenames, so the 12 low-CFL records are no
longer overwritten. All 22 parameter variants have an `environment.txt` file recording CUDA 12.1
with `nvcc` 12.1.105 and an NVIDIA A100-SXM4-40GB using driver 580.159.04.

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

### Comparison with the previous `mod/rpi_fluid` results

All 73 records retained by the previous archive have a matching current record, and every matched
case uses the same accepted step count. Fractional FARGO transport, polar transport, every isolated
diffusion operator, optical depth, and the source update reproduce their earlier primary errors to
the displayed precision. Radial-transport density errors differ only in the pre-asymptotic coarse
meshes: the largest change is approximately 7.5 percent, while the $N=256$ density errors agree to
better than $3\times10^{-7}$ relatively and retain the same asymptotic orders.

Some coarse-grid radial-velocity norms change more strongly, by as much as approximately 50 percent
in one `vely` norm, but the $N=256$ values are unchanged to the displayed precision. This is retained
as a pre-asymptotic limiter sensitivity rather than interpreted as a converged-regime regression.

The ring tests are not directly comparable through absolute density error because the previous
models reconstructed a midplane volume density,

$$
\rho_g=\frac{\Sigma_g}{\sqrt{2\pi}h_gR},
$$

whereas the current two-dimensional models correctly evolve $\Sigma_g$. This changes both the
normalization and radial profile, making the current absolute ring $L_1$ errors approximately
1.9--2.2 times larger. After dividing each finest-grid $L_1$ error by the volume-weighted mean of
its own analytical density, the comparison is:

| Ring case | Previous relative $L_1$ | Current relative $L_1$ | Change |
|---|---:|---:|---:|
| transport | $1.3340\times10^{-5}$ | $1.2406\times10^{-5}$ | $-7.0\%$ |
| transport + diffusion | $1.2670\times10^{-6}$ | $1.1566\times10^{-6}$ | $-8.7\%$ |
| transport + radiation | $1.3375\times10^{-5}$ | $1.2103\times10^{-5}$ | $-9.5\%$ |
| all physics | $1.3781\times10^{-6}$ | $1.1999\times10^{-6}$ | $-12.9\%$ |

The changed ring normalization therefore does not indicate a regression; the normalized errors and
the observed orders are slightly improved. The current 85-record result set should replace the old
archive as the regression baseline.

The integer FARGO case remains near machine roundoff. Its negative reported orders are ratios of
roundoff noise, not a failed transport test. The $p=0$ optical-depth case can likewise be integrated
exactly or nearly exactly by the logarithmic midpoint rule, so its order is not meaningful.

The source reference now uses the same cancellation-safe endpoint-weight series as the CUDA kernel
for $h<10^{-4}$. Reanalysis of the unchanged native output reduces the largest componentwise error
from $2.22\times10^{-5}$ to $2.22\times10^{-16}$. Density and total mass are exact to the recorded
precision, so the source case passes.

## Running the suite

From the repository root:

```bash
python3 tst/fluid/verify_common/run_suite.py --quick
python3 tst/fluid/verify_common/run_suite.py --group transport --res 32 64 128 256
python3 tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

The complete `--group all --res 32 64 128 256` matrix performs 85 builds/runs because several
models sweep CFL values, FARGO shifts, or optical-depth powers. Each resolution is cleaned and
rebuilt so model-local compile-time constants cannot reuse stale objects.

Only NumPy is required by the Python validator. Keep the JSON metrics, `environment.txt`, and full
terminal output together when archiving a run. Analytical results are namespaced as
`tst/fluid/out/SWEEP/MODEL/`, where `SWEEP` is `thread` or `block`.

## Verification still required

- Implement the full coupled 3D manufactured-solution harness specified in
  `tst/fluid/verify_mms_3d/README.md`, including independent forcing and exact boundary data
- Add production-settling equilibrium tests that quantify the initial polar transient
- Add radial and polar cases whose nonzero support actually crosses an outflow boundary, plus a
  `HALFDISK` midplane-reflection case; current compact profiles remain away from those boundaries
- Add restart-tolerance and flag-matrix regressions
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
