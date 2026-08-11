# Analytical definitions and expected results

This document defines exactly what each runnable `test_*` model computes. All evolved fields are
initialized as finite-volume averages. The validator compares density, stored momenta, and physical
linear velocities, together with optical depth for radiation cases.

The suffixes `cyl` and `sph` refer specifically to the radial Jacobian:

$$
d=2:\quad dV_y=y\,dy \qquad \text{(`cyl`, production }N_Z=1\text{)},
$$

$$
d=3:\quad dV_y=y^2\,dy \qquad \text{(`sph`, production }N_Z>1\text{)}.
$$

The spherical radial tests run an extruded three-dimensional grid but evolve only the Y operator.
They are not substitutes for the full multidirectional 3D MMS tests.

## Common numerical measurements

For each field error $e_i$, the validator reports

$$
L_1=\frac{\sum_iV_i|e_i|}{\sum_iV_i},\qquad
L_2=\left(\frac{\sum_iV_i e_i^2}{\sum_iV_i}\right)^{1/2},\qquad
L_\infty=\max_i|e_i|.
$$

It reports observed L1 order between successive resolutions and the relative total-mass change.
Velocity norms include only cells where both the exact and numerical densities exceed $10^{-12}$
of the exact peak, so the comparison does not mistake a production vacuum fallback velocity for
transport error.

Unless stated otherwise, the domain is

$$
0\le x<2\pi,\qquad 0.5\le y\le2.5.
$$

## `test_x_transport_2d`

Purpose: isolate the production periodic FARGO + PPM + HLL azimuthal sweep.

Initial state on every radial ring:

$$
\rho_d(x,0)=1+0.1\cos(2x),\qquad \Omega=1,
$$

$$
\ell_x=R^2\Omega=R^2,\qquad v_y=\ell_z=0.
$$

The exact solution is

$$
\rho_d(x,t)=1+0.1\cos[2(x-t)],\qquad
\rho_d\ell_x=R^2\rho_d.
$$

The final time is $T=2\pi$, so the exact final state equals the initial state. `--shift` controls
the nominal cells crossed per step. Run `3`, `3.25`, `3.5`, and `3.75` to exercise integer and
fractional FARGO shifts.

Expected: at least second-order convergence for the smooth mode; periodic mass conservation near
roundoff; uniform specific angular momentum retained. Error should not depend materially on the
integer part of the shift.

## `test_y_transport_cyl` and `test_y_transport_sph`

Purpose: isolate the production nonuniform radial PPM/HLL sweep and distinguish cylindrical from
spherical radial geometry.

Define the compact smooth profile

$$
B(y)=
\begin{cases}
\exp\!\left[1-\dfrac{1}{1-u^2}\right],&|u|<1,\\
0,&|u|\ge1,
\end{cases}
\qquad
u=\frac{y-1.4}{0.4}.
$$

Initialize

$$
\rho_d(y,0)=B(y),\qquad v_y(y,0)=a y,\qquad a=0.2.
$$

Constant transverse specific momenta $\ell_x=0.7$ and $\ell_z=0.11$ are included to verify that
every conservative component follows the same mass transport. Before characteristic crossing,

$$
\lambda(t)=1+a t,
$$

$$
\rho_d(y,t)=\lambda^{-d}B\!\left(\frac{y}{\lambda}\right),\qquad
v_y(y,t)=\frac{a y}{\lambda},
$$

with $d=2$ for `cyl` and $d=3$ for `sph`. The final time is $T=0.25$ and the compact support remains
away from the radial boundaries.

Expected with the present SSPRK(3,3) algorithm: mass conservation near roundoff and formally
second-order-or-better coupled convergence. The previous SSPRK2 CFL-0.5 global density orders at
$N=32,64,128,256$ are 1.7220, 2.3958, 1.6333 for `cyl` and 1.6393, 2.4543, 1.4472 for `sph`.
The native post-limiter SSPRK(3,3) orders are 1.5915, 2.4017, 2.7952 for `cyl` and
1.5428, 2.3757, 2.8012 for `sph`, confirming the predicted finer-grid behavior. The $N=32$ bump
spans only about five cells and is not expected to be asymptotic. The invariant-domain correction
also restores the expected step sequences 3, 5, 9, 16, so near-vacuum cells no longer collapse the
global CFL timestep.

## `test_z_transport_3d`

Purpose: isolate the production polar PPM/HLL sweep, including $\sin z$ face areas and
$d(-\cos z)$ cell volumes.

On $0.35\le z\le\pi-0.35$, define the same compact function with support $0.80<z<1.30$ and call it
$B(z)$. At each radius,

$$
\rho_d(z,0)\sin z=B(z),\qquad \ell_z=L=0.15.
$$

The angular characteristic speed is

$$
c_z=\frac{L}{y^2},
$$

and the exact solution is

$$
\rho_d(z,t)=\frac{B(z-c_zt)}{\sin z},\qquad
\rho_d\ell_z=L\rho_d.
$$

Constant transverse specific fields $\ell_x=0.7$ and $v_y=0.05$ test conservative transport of all
components. The final time is $T=0.30$ and the support remains separated from both boundaries.

Expected with SSPRK(3,3): mass conservation near roundoff and at least second-order convergence.
The native post-limiter CFL-0.5 density orders at $N=32,64,128,256$ are
1.6813, 2.3623, 2.4999. The step sequence is 4, 7, 13, 26.

## `test_x_diffusion_2d`

Purpose: test the production cyclic Sherman–Morrison CN solve, density diffusion, positivity
subcycling, and conservative momentum carried by the diffusive mass flux.

The test-only helper supplies constant $D=0.05$. On each ring,

$$
\rho_d(x,0)=1+0.1\cos(2x),
$$

$$
\rho_d(x,t)=1+0.1\exp\!\left(-\frac{4Dt}{R^2}\right)\cos(2x).
$$

All three stored specific momentum variables are initially uniform: $(0.7,-0.15,0.11)$.

Expected: second-order convergence, ring mass conservation near roundoff, and retention of all
three uniform specific momenta.

## `test_y_diffusion_cyl` and `test_y_diffusion_sph`

Purpose: test the production radial CN operator with the correct radial Jacobian, zero-flux
boundaries, density diffusion, and momentum coupling.

The test-only helper supplies $D=0.05$ while retaining the nonuniform production gas profile, so
the case distinguishes density diffusion from concentration diffusion. Initialize

$$
\rho_d(y,0)=1+0.1Q_k(y),
$$

where

$$
Q_k''+\frac{d-1}{y}Q_k'+k^2Q_k=0,
\qquad Q_k'(0.5)=Q_k'(2.5)=0.
$$

The first nonzero roots used are

$$
k_{d=2}=1.694299217770420,
\qquad
k_{d=3}=1.874562003084784.
$$

The exact decay is

$$
\rho_d(y,t)=1+0.1Q_k(y)e^{-Dk^2t}.
$$

The HIP/ROCm initializer evaluates the equivalent Bessel representation. Independently, the Python
validator integrates the Sturm–Liouville ODE with a fine-grid RK4 solve. Uniform stored specific
momenta $(0.7,-0.15,0.11)$ test the diffusive momentum flux.

Expected: second-order convergence, no-flux total-mass conservation near roundoff, and uniform
specific momentum retained.

## `test_z_diffusion_3d`

Purpose: test the production polar CN operator and its $1/y^2$, $\sin z$, and $d(-\cos z)$ metric
factors.

The domain is the upper hemisphere $0\le z\le\pi/2$ with the production `HALF_DISK` flag. The
nonuniform production gas profile is retained deliberately. With $D=0.05$ and
$P_2(\mu)=(3\mu^2-1)/2$,

$$
\rho_d(z,0)=1+0.1P_2(\cos z),
$$

$$
\rho_d(z,t)=1+0.1P_2(\cos z)
\exp\!\left(-\frac{6Dt}{y^2}\right).
$$

The mode is regular at the axis and has zero derivative at the reflecting midplane. The same
uniform stored specific momenta as the other diffusion tests are used.

Expected: second-order convergence independently on every radial shell, no-flux mass conservation
near roundoff, and uniform specific momentum retained.

## `test_source_drag`

Purpose: test the exponential endpoint-force quadrature independently of disk gravity and gas
profiles. This is the only runnable case that intentionally replaces a production kernel:
model-local `source_update.hip` keeps the production interface and quadrature formulas while
supplying closed-form gas velocities and force endpoints.

For each component,

$$
\frac{du}{dt}=-\frac{u-u_g}{t_s}+a_0+a_1t,
$$

with endpoint values $F_0=a_0$ and $F_1=a_0+a_1\Delta t$. The exact one-step value is

$$
u(\Delta t)=u_g+(u_0-u_g)e^{-h}
+F_0t_s(1-e^{-h})
+a_1\left[t_s\Delta t-t_s^2(1-e^{-h})\right],
\qquad h=\frac{\Delta t}{t_s}.
$$

Eight cells test $h=10^{-6},10^{-3},0.1,1,10,10^2,10^4,10^6$. The three components use different
initial values, gas velocities, and force endpoints.

Expected: agreement close to double-precision roundoff for all stiffness ratios, allowing for the
build's `--use_fast_math` transcendental accuracy.

## `test_optdepth`

Purpose: test the production cell optical-depth quadrature and radial cumulative-sum indexing
without dynamics.

Initialize the evolved surface density at logarithmic cell centers as

$$
\Sigma_d(y)=\sqrt{2\pi}H_g(y)y^p,\qquad \kappa=1.
$$

The production well-mixed 2D radiation closure reconstructs
\(\rho_{d,0}=\Sigma_d/(\sqrt{2\pi}H_g)=y^p\), so the continuum outward-face solution is

$$
\tau(y)=
\begin{cases}
\dfrac{y^{p+1}-0.5^{p+1}}{p+1},&p\ne-1,\\[6pt]
\ln(y/0.5),&p=-1.
\end{cases}
$$

Run `--power 0`, `--power -1`, and `--power 1`. The validator compares every cumulative outer-face
value.

Expected: second-order convergence for non-degenerate powers. Some powers can be integrated more
accurately or exactly by the logarithmic midpoint-times-width rule, so judge the family rather than
requiring the same nonzero error for every $p$.

## Four 2D rotating-ring models

The models are:

- `test_ring_transport_2d`;
- `test_ring_diffusion_2d`;
- `test_ring_radiation_2d`;
- `test_ring_all_2d`.

They exercise the symmetric production composition rather than one isolated kernel. At each
midplane radius,

$$
\frac{\rho_d(R,x,0)}{\rho_0(R)}=1+0.1\cos(2x),
$$

$$
\Omega_\beta(R)=\sqrt{\frac{1-\beta}{R^3}},\qquad
\ell_x=R^2\Omega_\beta=\sqrt{(1-\beta)R},
$$

with $v_y=\ell_z=0$. The verification gas uses constant aspect ratio $h=0.5$ and

$$
\mathrm{IDX\_P}=2-4\beta,
$$

which makes the production pressure-support parameter $\eta=\beta/2$. Therefore the production gas
angular momentum equals the dust value, drag vanishes, and centrifugal acceleration balances
gravity plus radiation. Radiation cases use $\beta=0.2$ and $\kappa=0$, with the model driver
setting the taper to one. Non-radiating cases use $\beta=0$.

Diffusion cases use the production constant-$\nu$ branch with $D_x=0.05$ and effectively zero
cross-direction diffusivities. Because $D_x$ is constant rather than proportional to $R^2$, the
exact damping rate differs by ring:

$$
\frac{\rho_d(R,x,t)}{\rho_0(R)}=1+0.1
\exp\!\left(-\frac{4D_xt}{R^2}\right)
\cos\!\left(2[x-\Omega_\beta(R)t]\right).
$$

Setting either $\beta$ or $D_x$ to zero yields the other three cases. The final time is $T=1$.

Expected: approximately second-order convergence of density, momenta, and velocities for
this smooth equilibrium family. Periodic total mass should remain near roundoff. Radiation-model
optical depth must be exactly zero apart from floating-point zeros because $\kappa=0$. Any growing
radial velocity indicates a force/drag imbalance; any change of uniform ring-specific angular
momentum indicates inconsistent advective or diffusive momentum transport.

## Full 3D coupled status

No runnable model in this document claims full coupled 3D verification. The required manufactured
forcing and exact boundary implementation are specified in `test_mms_3d/README.md`. Until that
harness exists, the spherical Y and polar Z cases verify their production directional operators,
not their complete source/diffusion/radiation coupling.
