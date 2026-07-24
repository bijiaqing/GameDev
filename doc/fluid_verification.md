# Eulerian fluid verification

## Purpose

The CUDA models under `new/tst/fluid/` compare production fluid kernels with analytical
finite-volume solutions. They are verification models, not production disk setups. Model-local
files replace a production component only when the test requires a prescribed state that the
production interface cannot express.

This document records the current verification claim. Detailed formulas implemented by the
validator remain in `new/tst/fluid/verify_common/TEST_CASES.md`, and machine-readable results remain
under `new/tst/fluid/out/`.

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
\frac{\rho_d(R,x,0)}{\rho_0(R)}=1+0.1\cos(2x),
\qquad
\Omega_\beta(R)=\sqrt{\frac{1-\beta}{R^3}},
$$

with gas pressure support chosen so drag vanishes and radial forces balance. When azimuthal
diffusion is enabled,

$$
\frac{\rho_d(R,x,t)}{\rho_0(R)}
=1+0.1
\exp\left(-\frac{4D_xt}{R^2}\right)
\cos\left(2[x-\Omega_\beta(R)t]\right).
$$

These cases cover transport-only, transport plus diffusion, transport plus radiation, and all
three together.

## Recorded native CUDA results

The current `new/tst/fluid/out/output1.txt` and JSON files contain the recorded native matrix.
There are 73 JSON metric records even though the complete run performed 85 builds: low- and
production-CFL Y/Z sweeps used the same metric filenames, so the later sweep overwrote 12 JSON
files. Their complete tables remain in `output1.txt`. Future reruns should include CFL in the
variant tag before relying on the JSON count alone.

The following density $L_1$ orders use $N=32,64,128,256$ and list the three successive refinement
intervals.

| Case | Successive $L_1$ orders | Assessment |
|---|---|---|
| FARGO shift 3.25 | 2.3067, 2.4063, 2.4158 | pass |
| FARGO shift 3.50 | 2.3079, 2.3029, 2.4658 | pass |
| FARGO shift 3.75 | 2.0579, 2.4278, 2.4282 | pass |
| radial transport, $d=2$ | 1.5915, 2.4017, 2.7952 | pass; coarsest compact profile is under-resolved |
| radial transport, $d=3$ | 1.5428, 2.3757, 2.8012 | pass; same qualification |
| polar transport | 1.6813, 2.3623, 2.4999 | pass |
| azimuthal diffusion | 1.9952, 1.9988, 1.9997 | pass |
| radial diffusion, $d=2$ | 1.9903, 1.9937, 1.9993 | pass |
| radial diffusion, $d=3$ | 1.9796, 1.9969, 1.9980 | pass |
| polar diffusion | 1.9974, 1.9997, 1.9999 | pass |
| optical depth, $p=-1$ | 2.0148, 2.0076, 2.0038 | pass |
| optical depth, $p=1$ | 2.0374, 2.0192, 2.0097 | pass |
| coupled transport | 2.1291, 2.2277, 2.2622 | pass |
| coupled transport + diffusion | 3.1899, 3.1469, 2.7400 | pass; pre-asymptotic superconvergence is not a third-order claim |
| coupled transport + radiation | 2.1363, 2.2206, 2.2732 | pass |
| coupled all physics | 3.1581, 3.0878, 2.5825 | pass; same qualification |

The final invariant-domain-limiter rerun restored the expected radial step sequences:

| Case | Steps at $N=32,64,128,256$ | Maximum recorded relative mass change |
|---|---|---:|
| radial transport, $d=2$ | 3, 5, 9, 16 | $6.57\times10^{-16}$ |
| radial transport, $d=3$ | 3, 5, 9, 16 | $8.63\times10^{-16}$ |
| polar transport | 4, 7, 13, 26 | $1.14\times10^{-15}$ |

The integer FARGO case remains near machine roundoff. Its negative reported orders are ratios of
roundoff noise, not a failed transport test. The $p=0$ optical-depth case can likewise be integrated
exactly or nearly exactly by the logarithmic midpoint rule, so its order is not meaningful.

The source metric currently records componentwise maximum differences up to
$2.22\times10^{-5}$. The discrepancy was traced to cancellation in the Python analytical
evaluator at $h=10^{-6}$, while the CUDA kernel uses a stable small-$h$ series. The reference
evaluator still needs an `expm1`/series correction and a rerun before the source case receives a
formal pass.

## Running the suite

From the repository root:

```bash
python3 new/tst/fluid/verify_common/run_suite.py --quick
python3 new/tst/fluid/verify_common/run_suite.py --group transport --res 32 64 128 256
python3 new/tst/fluid/verify_common/run_suite.py --group all --res 32 64 128 256
```

The complete `--group all --res 32 64 128 256` matrix performs 85 builds/runs because several
models sweep CFL values, FARGO shifts, or optical-depth powers. Each resolution is cleaned and
rebuilt so model-local compile-time constants cannot reuse stale objects.

Only NumPy is required by the Python validator. Keep the JSON metrics, `environment.txt`, and full
terminal output together when archiving a run.

## Verification still required

- Implement the full coupled 3D manufactured-solution harness specified in
  `new/tst/fluid/verify_mms_3d/README.md`, including independent forcing and exact boundary data
- Repair the cancellation-prone source reference and rerun it
- Add production-settling equilibrium tests that quantify the initial polar transient
- Add explicit boundary, restart-tolerance, and flag-matrix regressions
- Repeat important publication runs without `--use_fast_math`, or document and measure its effect
- Define automated pass thresholds only after identifying the asymptotic range on the target GPU

The directional 3D radial and polar tests do not establish complete 3D transport–diffusion–
radiation coupling. The radiation ring tests deliberately use zero opacity to isolate the force
composition; finite, spatially varying attenuation still belongs in the 3D manufactured suite.

## References

- Salari & Knupp (2000), [verification by manufactured solutions](https://digital.library.unt.edu/ark:/67531/metadc702130/)
- Colella & Woodward (1984), [PPM](<https://doi.org/10.1016/0021-9991(84)90143-8>)
- Masset (2000), [FARGO](https://arxiv.org/abs/astro-ph/9910390)
- Crank & Nicolson (1947), [implicit diffusion](https://doi.org/10.1017/S0305004100023197)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](<https://doi.org/10.1016/0019-1035(79)90050-2>)
