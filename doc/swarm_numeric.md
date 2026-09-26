# Lagrangian swarm model: equations and numerical guide

## 1. Model overview

The Lagrangian branch represents dust with computational swarms, each carrying one phase-space
location and, in multisize models, one grain size and a represented physical grain count. Gas is
prescribed analytically or imported, dust does not back-react on it, and enabled operators advance
particle trajectories, stochastic spatial diffusion, radiation forces, and representative-particle
collisions. Shared headers, collision-search libraries, and application sources live under
`inc/swarm/` and `src/swarm/`; `inc/gpu.cuh` selects the CUDA or HIP runtime and
random-number APIs.

### 1.1 Supported configurations

| Geometry | Grid condition | Density interpretation | Dynamical closure |
|---|---|---|---|
| radial-only | `N_X == 1`, `N_Z == 1` | $\Sigma_d(R)$ | axisymmetric and vertically integrated, with midplane dynamics |
| radial–azimuthal | `N_X > 1`, `N_Z == 1` | $\Sigma_d(R,\phi)$ | vertically integrated, with midplane dynamics |
| radial–polar | `N_X == 1`, `N_Z > 1` | $\rho_d(r,\theta)$ | axisymmetric resolved vertical structure |
| full 3D | `N_X > 1`, `N_Z > 1` | $\rho_d(r,\theta,\phi)$ | resolved spherical dynamics |

`TRANSPORT` enables particle dynamics, `DIFFUSION` enables stochastic position diffusion,
`RADIATION` enables attenuated radiation pressure, `PR_EFFECT` adds Poynting–Robertson drag,
`COLLISION` enables representative-particle collisions, and `IMPORTGAS` selects gridded gas input.
`MULTISIZE` stores individual sizes and represented grain counts, while `SAVE_DENS` requests the
deposited mesh-density diagnostic. Section 2.5 lists every flag and the compile-time dependencies
between them. In particular, any model with `N_Z > 1` requires `DIFFUSION`, which prevents an
unsupported indefinitely settling dust layer.

The radial-only path with `N_X == 1 && N_Z == 1` is selected by the grid alone. The CUDA and ROCm
suite definitions and acceptance criteria, including the radial analytical and KNN cases, are
documented in [`swarm_testset.md`](swarm_testset.md).

### 1.2 Physical assumptions and relation to the fluid branch

Both dust branches use the same prescribed central-star and gas-disk model, but they approximate
the dust distribution differently. The swarm branch advances a weighted empirical phase-space
distribution, whereas the fluid branch advances cell-averaged density and momentum under a
single-valued velocity closure. Their common and distinct closures are

| Property | Eulerian fluid | Lagrangian swarm |
|---|---|---|
| dust representation | cell-averaged density and momentum | weighted computational representatives |
| grain sizes | one fixed Stokes species | monodisperse or multisize |
| velocity at one position | single valued | multiple representatives may cross with different velocities |
| gas response to dust | absent | absent |
| dust pressure and self-gravity | absent | absent |
| density or concentration diffusion | spherical finite-volume PDE | cylindrical Itô displacement |
| collisions | absent | optional coagulation and fragmentation |
| imported gas fields | absent | optional |
| radiation pressure | optional | optional, with optional Poynting–Robertson drag |

The diffusion-basis distinction is physical rather than merely an implementation detail. In a
resolved vertical domain, fluid $D_y$ diffuses along spherical radius $r$, whereas swarm $D_R$
diffuses along cylindrical radius $R$. Equal scalar coefficients therefore need not produce the
same radial equilibrium away from the midplane; the two closures coincide radially only where
$r=R$, including the vertically integrated model.

The mathematical bridge between the branches is obtained from moments of a dust mass distribution:

$$
\rho_d=\int f_d\,d^3v\,ds,
\qquad
\rho_d\boldsymbol u_d=\int\boldsymbol v f_d\,d^3v\,ds,
$$

$$
\int\boldsymbol v\boldsymbol v f_d\,d^3v\,ds
=\rho_d\boldsymbol u_d\boldsymbol u_d+\boldsymbol P_d.
$$

The fluid branch sets the velocity-dispersion tensor $\boldsymbol P_d$ to zero. The swarm instead
approximates the full distribution by the empirical measure

$$
f_d^{N_P}(\boldsymbol x,\boldsymbol v,s,t)
=\sum_{p=1}^{N_P}W_p
\,\delta(\boldsymbol x-\boldsymbol x_p)
\,\delta(\boldsymbol v-\boldsymbol v_p)
\,\delta(s-s_p),
$$

where $W_p$ is represented physical mass. Unlike the pressureless fluid closure, this empirical
measure permits several representatives with different velocities at the same position and can
therefore continue through multistreaming. Its finite-$N_P$ sampling noise and collision estimator
variance replace the fluid model's closure error.

The gas is one-way coupled in both branches. In the gas equations that are not evolved by this
code, the dust-induced contributions are set to

$$
\left.\frac{\partial\rho_g}{\partial t}\right|_{\rm dust}=0,
\qquad
\left.\frac{\partial(\rho_g\boldsymbol v_g)}{\partial t}\right|_{\rm dust}=0.
$$

Dust drag and collisions therefore do not alter the prescribed gas. Gas viscosity enters only
through the prescribed viscous target velocity, stochastic diffusivity, and turbulent collision
velocity; the code does not solve a gas viscous evolution equation.

## 2. Coordinates, geometry, and particle state

### 2.1 Coordinates and stored variables

The spherical computational coordinates are

$$
(x,y,z)=(\phi,r,\theta),
\qquad
R=y\sin z,
\qquad
Z=y\cos z.
$$

Each active representative stores

$$
\boldsymbol q=(\phi,r,\theta),
\qquad
\boldsymbol u=(\ell_\phi,v_r,\ell_\theta)
=(Rv_\phi,v_r,rv_\theta).
$$

When `N_Z == 1`, the code evaluates $R=y$ and $Z=0$ directly and keeps every particle at $z=\pi/2$
with $\ell_\theta=0$. When `N_X == 1`, one stored azimuthal cell represents the complete
axisymmetric ring and particles are centered exactly in the inactive coordinate. The radial-only
model retains $v_R$ and $\ell_\phi$ as dynamical variables even though azimuth is spatially
inactive.

The mesh measure uses $d=2$ for a vertically integrated disk and $d=3$ when the polar dimension is
active:

$$
\Delta V_y=\frac{y_{\rm out}^d-y_{\rm in}^d}{d},
\qquad
\Delta V_z=\cos z_{\rm in}-\cos z_{\rm out}.
$$

For an axisymmetric cell, deposition and collision normalization include the missing complete
azimuthal circumference or extent rather than treating the stored meridional section as a physical
volume.

More explicitly, let

$$
\Delta x=\frac{X_{\max}-X_{\min}}{N_X},
\qquad
a_y=\left(\frac{Y_{\max}}{Y_{\min}}\right)^{1/N_Y},
\qquad
\Delta z=\frac{Z_{\max}-Z_{\min}}{N_Z},
$$

$$
y_{j-1/2}=Y_{\min}a_y^j,
\qquad
z_{k-1/2}=Z_{\min}+k\Delta z.
$$

Then the exact cell measure used for deposition is

```math
V_{ijk}=\Delta\phi_{\rm cell}
\frac{y_{j+1/2}^{d}-y_{j-1/2}^{d}}{d}
\left\{\begin{array}{ll}
1,&N_Z=1,\\
\cos z_{k-1/2}-\cos z_{k+1/2},&N_Z>1,
\end{array}\right.
```

where

```math
\Delta\phi_{\rm cell}=
\left\{\begin{array}{ll}
\Delta x,&N_X>1,\\
2\pi,&N_X=1.
\end{array}\right.
```

The $2\pi$ branch is essential: an axisymmetric radial or radial–polar cell represents a complete
ring, not a wedge of numerical width $X_{\max}-X_{\min}$.

Complete grid cells use $(i,j,k)$ for $(x,y,z)=(\phi,r,\theta)$. Particle labels and the
initial/midpoint/final state labels used later are separate indices and do not alter this grid
convention.

Particle–mesh interpolation treats a radial cell value as located at its exact measure centroid

$$
\bar y_j
=\frac{d}{d+1}
\frac{y_{j+1/2}^{d+1}-y_{j-1/2}^{d+1}}
{y_{j+1/2}^{d}-y_{j-1/2}^{d}}.
$$

This differs slightly from the geometric midpoint of a logarithmic cell. Linear interpolation is
performed in physical $y$ between these centroids, while cell selection still uses the continuous
logarithmic coordinate $\xi_y=\ln(y/Y_{\min})/\ln a_y$.

### 2.2 Radial-only closure

The radial model is selected intrinsically by

$$
N_X=1,\qquad N_Y>1,\qquad N_Z=1,
$$

without a separate preprocessor flag. It represents an azimuthally symmetric, vertically
integrated disk: grid dust and gas densities are surface densities $\Sigma_d$ and $\Sigma_g$,
while gas drag, gravity, radiation, and particle dynamics use a midplane closure at $Z=0$.
Representative particles sample the radial mass distribution rather than acting as rigid rings.

The active particle state is

$$
R,\qquad v_R,\qquad \ell_\phi=Rv_\phi,
$$

with exact inactive assignments

$$
x=\frac{X_{\min}+X_{\max}}{2},\qquad
z=\frac{\pi}{2},\qquad
\ell_\theta=v_\theta=0.
$$

Azimuth is spatially inactive, but $\ell_\phi$ remains dynamical because it supplies centrifugal
support, azimuthal drag, collision velocity, and Poynting–Robertson damping. Section 4.2 gives the
corresponding radial density and characteristic equations.

The binary particle layout remains the common three-coordinate layout, with $v_\theta=0$ and the
inactive coordinates written at their exact values (Section 12). Grid density files contain $N_Y$
values of $\Sigma_d$, optical-depth files contain the radial cumulative outer-face field, and
`variables.txt` labels the geometry, density kinds, inactive coordinates, and midplane dynamical
closure.

### 2.3 Units, dimensions, and notation

The default dynamics uses

$$
G=M_\star=R_0=1,
$$

with orbital conversion scales

$$
t_0=\sqrt{\frac{R_0^3}{GM_\star}},
\qquad
v_0=\sqrt{\frac{GM_\star}{R_0}},
\qquad
\Omega_0=t_0^{-1}.
$$

The principal quantities have dimensions

| Quantity | Symbol | Dimension |
|---|---|---|
| represented swarm mass | $W_p$ | $M$ |
| physical grain count | $N_p$ | $1$ |
| grain diameter | $s$ | $L$ |
| compact-grain density | $\rho_0$ | $M L^{-3}$ |
| linear velocity | $v$ | $L T^{-1}$ |
| specific angular momentum | $\ell$ | $L^2T^{-1}$ |
| diffusivity | $D$ | $L^2T^{-1}$ |
| physical collision kernel | $K=\sigma\Delta v$ | $L^3T^{-1}$ |
| collision propensity | $\lambda$ | $T^{-1}$ |

The Stokes number, aspect ratio, radiation ratio, metallicity, viscosity parameter, and Schmidt
numbers are dimensionless:

$$
\mathrm{St}=\Omega_Kt_s,
\qquad
h_g=\frac{H_g}{R}=\frac{c_s}{R\Omega_K},
\qquad
\beta=\frac{|a_{\rm rad}|}{|a_{\rm grav}|},
$$

$$
Z_{\rm metal}=\frac{\Sigma_d}{\Sigma_g},
\qquad
\alpha=\frac{\nu}{H_g^2\Omega_K},
\qquad
\mathrm{Sc}_a=\frac{\nu}{D_a(1+\mathrm{St}^2)}.
$$

Here $r$ denotes spherical radius, $R=r\sin\theta$ cylindrical radius, and $Z=r\cos\theta$
cylindrical height. `CODE_UNIT` changes the auxiliary collision-microphysics calibration, not the
orbital equations by itself; a physical interpretation must convert all quantities entering one
dimensional formula consistently.

### 2.4 Principal code parameters

All parameters are `constexpr` constants in
[`inc/swarm/const_defs.cuh`](../inc/swarm/const_defs.cuh); a model-local `const_defs.cuh` replaces
the whole file (see [Configuring a model](../README.md#configuring-a-model)). The defaults below are
those of the shared header. A parameter marked with a flag exists only when that flag is defined.

| Code parameter | Default | Mathematical role |
|---|---|---|
| `G`, `M_S`, `R_0` | `1.0`, `1.0`, `1.0` | $G$, $M_\star$, and the reference radius $R_0$ |
| `S_0`, `RHO_0` | `1.0`, `1.0` | reference grain diameter $S_0$ and compact density $\rho_0$ |
| `N_P` | `10'000'000` | number of representatives $N_P$ |
| `N_X`, `X_MIN`, `X_MAX` | `100`, $-\pi$, $+\pi$ | azimuthal cells and domain $[X_{\min},X_{\max}]$ |
| `N_Y`, `Y_MIN`, `Y_MAX` | `100`, `0.5`, `1.5` | logarithmic radial cells and domain $[Y_{\min},Y_{\max}]$ |
| `N_Z`, `Z_MIN`, `Z_MAX` | `1`, $\pi/2$, $\pi/2$ | polar cells and domain $[Z_{\min},Z_{\max}]$ |
| `SIGMA_0`, `ASPR_0` | `1.0e-02`, `0.05` | $\Sigma_0$ and $h_0$ |
| `IDX_P`, `IDX_Q` | `-1.0`, `-0.4` | power-law indices $p$ and $q$ |
| `METAL_Z`, `STOKES_0` | `1.0e-02`, `1.0e-03` | $Z_{\rm metal}$ and reference $\mathrm{St}_0$ |
| `ALPHA` or `NU` (`DIFFUSION` or `COLLISION`) | `1.0e-04` or `1.0e-05` | turbulent $\alpha$, or constant $\nu$ with `CONST_NU` |
| `SCHMIDT_X`, `SCHMIDT_R` (`DIFFUSION`) | `1.0`, `1.0` | cylindrical Schmidt numbers $\mathrm{Sc}_\phi$ and $\mathrm{Sc}_R$ |
| `SCHMIDT_Z` (`DIFFUSION` or `COLLISION`) | `1.0` | vertical Schmidt number $\mathrm{Sc}_Z$ |
| `INIT_SMIN`, `INIT_SMAX` (`MULTISIZE`) | `1.0`, `1.0` | initialized grain-size interval $[s_{\min},s_{\max}]$ and fragment floor |
| `BETA_0`, `KAPPA_0`, `T_BETA` (`RADIATION`) | `1.0e+01`, `1.0`, $2\pi$ | radiation ratio $\beta_0$, opacity normalization $\kappa_0$, and ramp time $T_\beta$ |
| `C_LIGHT` (`PR_EFFECT`) | `1.0e+04` | light speed in units of $v_0$ |
| `CFL_DYN`, `DT_MAX` (`TRANSPORT`) | `0.45`, `0.1` | dynamics Courant factor and timestep ceiling |
| `DT_OUT`, `SAVE_MAX` | `1.0`, `100` | output-interval scale and final output index |
| `LOG_BASE` or `LIN_BASE` | `10` or `1` | logarithmic base (`LOGTIMING`, `LOGOUTPUT`) or linear particle-checkpoint stride |
| `COAG_KERNEL` (`COLLISION`) | `0` | constant (`0`), additive (`1`), product (`2`), or physical (`3`) kernel |
| `N_K`, `H_SEARCH` (`COLLISION`) | `200`, `1.0` | retained KNN count and local search cap in $H_g$ units |
| `V_FRAG` (`COLLISION`) | `1.0` | fragmentation threshold speed, in model velocity units |
| `REYNOLDS_0` (`COLLISION`, `CODE_UNIT`) | `1.0e+08` | code-unit turbulent Reynolds number at $R_0$ |
| `M_MOL`, `X_SEC` (`COLLISION`, no `CODE_UNIT`) | $2.3\times1.66054\times10^{-24}$, `2.0e-15` | gas molecular mass (g) and molecular cross section (cm²) |
| `COL_BATH_MAX`, `COL_BATH_EPS`, `COL_BATH_ALPHA` (`COLLISION`) | `0.05`, `0.02`, `1.0e-03` | frozen-bath duration cap, refresh tolerance, and audit confidence tail |
| `COL_BIN_X`, `COL_BIN_Y`, `COL_BIN_Z`, `COL_BIN_S` (`COLLISION`) | `8`, `4`, `2`, `8` | controller bins in azimuth, radius, polar angle, and log size |
| `COL_BIN_MIN` (`COLLISION`) | `64` | target minimum representatives per merged size bin |
| `COL_BATH_TPB`, `COL_EVENT_CAP` (`COLLISION`) | `64` (CUDA) or `128` (ROCm), `32` | threads per owner chain block and accepted events per continuation launch |
| `MORTON_TPB` (Morton search) | `64` | cooperative threads per Morton query |
| `MORTON_LEAF_TARGET`, `MORTON_MAX_LEVEL` (Morton search) | `128`, `20` | adaptive Morton leaf occupancy target and depth limit |
| `TPB` | `64` | threads per block of the one-thread-per-representative kernels |

Compile-time checks reject inconsistent values: `0 < COL_BATH_TPB <= 1024`, `COL_EVENT_CAP > 0`,
positive controller bin counts and `COL_BIN_MIN`, `COL_BATH_MAX > 0`, `0 < COL_BATH_EPS < 1`,
`0 < COL_BATH_ALPHA < 1`, a power-of-two `TPB` in collision builds, `N_K <= 4096` for the KD-tree
heap, and `N_P <= INT_MAX/3` (715 827 882) in collision builds, because each neighbor code packs a
particle index and an image index into one 32-bit integer (Section 9.5). The Morton builder
rejects `MORTON_MAX_LEVEL` outside 1 to 20 at run time. Static shared memory further bounds `N_K`
(Section 11.4).

### 2.5 Compile-time feature selection

The swarm executable is a compile-time specialization. Flags are added to `GPU_FLAGS` in the
model's `flags.mk` as `-D<FLAG>`; the collision search is a Makefile variable:

| Selection | Effect |
|---|---|
| `TRANSPORT` | integrate deterministic trajectories, drag, gravity, and enabled radiative forces |
| `DIFFUSION` | add cylindrical stochastic diffusion |
| `DIFFUSE_CONCENTRATION` | diffuse dust-to-gas concentration instead of density; requires `DIFFUSION` |
| `RADIATION` | deposit and accumulate optical depth and add radiation pressure |
| `PR_EFFECT` | add first-order Poynting–Robertson drag |
| `COLLISION` | evolve representative-particle coagulation and fragmentation with the frozen-bath chain |
| `MULTISIZE` | store sampled grain size and represented grain count for every swarm |
| `IMPORTGAS` | interpolate density and velocity from external gas snapshots |
| `VISC_FLOW` | prescribe viscous analytical gas motion |
| `CONST_ST` | hold the analytical-gas Stokes number fixed apart from its size factor |
| `CONST_NU` | use constant $\nu$ instead of constant $\alpha$ for diffusion or collision turbulence |
| `CODE_UNIT` | select the code-unit calibration of collision microphysics |
| `HALF_DISK` | reflect deterministic transport at the midplane |
| `SAVE_DENS` | write the particle-deposited dust-density mesh field |
| `COL_DIAGNOSTICS` | record frozen-bath controller schedules and event statistics in JSON diagnostics |
| `LOGTIMING` | use logarithmically spaced output times |
| `LOGOUTPUT` | retain linear output times but save particle checkpoints at logarithmic frame indices |
| `COLLISION_SEARCH := kdtree` or `morton` | select the exact KNN implementation used by collisions (Section 9.1) |

The headers reject the following combinations with a compile-time `#error`:

| Requirement or exclusion | Note |
|---|---|
| `DIFFUSION` and `RADIATION` require `TRANSPORT` | both operators run inside the dynamics step |
| `PR_EFFECT` requires `RADIATION` | the drag uses the attenuated radiation ratio $\beta$ |
| `VISC_FLOW` requires `DIFFUSION` and excludes `IMPORTGAS` | the target uses $\nu$ from `ALPHA` or `NU`; imported gas already prescribes velocity |
| `CONST_ST` excludes `IMPORTGAS` | imported density determines the Stokes number |
| `COLLISION` requires `MULTISIZE` | outcomes change grain size and represented number |
| `COLLISION` requires exactly one of `COLLISION_KDTREE` or `COLLISION_MORTON` | the Makefile defines one from `COLLISION_SEARCH` |
| `COL_CHAIN`, `BERNOULLI`, and `KNN_CACHE` are rejected | the frozen-bath chain with a cached neighbor list is the only collision integrator |
| at least one of `TRANSPORT` or `COLLISION` | an executable must evolve something |
| `LOGTIMING` excludes `LOGOUTPUT`, `TRANSPORT`, and `SAVE_DENS` | logarithmic timing is for collision-only runs |
| `N_Z > 1` requires `DIFFUSION` | a resolved vertical layer needs vertical mixing (a `static_assert` in `const_defs.cuh`) |

`DIFFUSE_CONCENTRATION` without `DIFFUSION` is rejected at compile time; `CONST_NU` and `CODE_UNIT`
have no effect when no operator uses them.
These selections specify which equations exist in an executable; they do not turn operators on or
off during a run.

## 3. Disk model, mass normalization, and initialization

### 3.1 Prescribed gas and stopping time

For analytical gas, the surface density and aspect ratio are

$$
\Sigma_g(R)=\Sigma_0\left(\frac{R}{R_0}\right)^p,
\qquad
h_g(R)=h_0\left(\frac{R}{R_0}\right)^{(q+1)/2}.
$$

Here $p=d\ln\Sigma_g/d\ln R$ and $q=d\ln T/d\ln R$. Vertical isothermality implies

$$
c_s\propto T^{1/2}\propto R^{q/2},
\qquad
h_g=\frac{c_s}{v_K}\propto R^{(q+1)/2}.
$$

The exact vertically isothermal point-mass stratification is

$$
\mathcal G(R,Z)
=\frac{\rho_g(R,Z)}{\rho_g(R,0)}
=\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g(R)^2}
\right],
$$

with

$$
\rho_g(R,Z)=\frac{\Sigma_g(R)}{\sqrt{2\pi}H_g(R)}\mathcal G(R,Z),
\qquad H_g=h_gR.
$$

The local orbital and sound-speed scales are

$$
\Omega_K(R)=\sqrt{\frac{GM_\star}{R^3}},
\qquad
v_K=R\Omega_K,
\qquad
c_s=h_gR\Omega_K.
$$

At the midplane, the usual radial pressure-support parameter for the vertically isothermal
pressure $P_g=\rho_gc_s^2$ is

$$
\eta_{\rm mid}=-\frac{1}{2\rho_gR\Omega_K^2}\frac{\partial P_g}{\partial R}\bigg|_{Z=0},
\qquad
\eta_{\rm mid}
=-\frac{h_g^2}{2}\left(p+\frac q2-\frac32\right).
$$

Away from the midplane, the code variable `eta` is the effective rotation-support parameter after
combining the cylindrical stellar gravity and pressure gradient, rather than the pressure term
alone. This distinction follows the vertically structured disk force balance discussed by
[Takeuchi & Lin (2002)](https://arxiv.org/abs/astro-ph/0208552). Define

$$
u=\frac{R}{\sqrt{R^2+Z^2}},
\qquad
A=p+\frac q2-\frac32.
$$

At fixed cylindrical height, the exact hydrostatic profile gives

$$
h_g^2\frac{\partial\ln P_g}{\partial\ln R}
=Ah_g^2+q(1-u)+(1-u^3).
$$

Because the cylindrical stellar gravity is $u^3$ times its midplane value, radial force balance is

$$
\frac{v_{\phi,g}^2}{v_K^2}
=u^3+h_g^2\frac{\partial\ln P_g}{\partial\ln R}
=1+Ah_g^2+q(1-u)
=1-2\eta(R,Z),
$$

where

$$
\eta(R,Z)=-\frac12\left[Ah_g^2+q(1-u)\right].
$$

The $(1-u^3)$ pressure term therefore cancels the off-midplane weakening of cylindrical gravity.
At $Z=0$, $u=1$ and the effective parameter reduces to the ordinary midplane pressure-support
definition. Section 3.6 uses this exact rotation target.

For constant `ALPHA`, the turbulent viscosity uses the $\alpha$ prescription of
[Shakura & Sunyaev (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S); the alternative
branch holds $\nu$ constant:

```math
\nu(R)=
\left\{\begin{array}{ll}
\alpha h_g^2R^2\Omega_K,&\text{constant }\mathtt{ALPHA},\\
\mathrm{NU},&\text{constant }\mathtt{NU}.
\end{array}\right.
```

For the constant-viscosity branch, the corresponding local turbulent parameter is

$$
\alpha(R)=\frac{\nu}{h_g^2R^2\Omega_K}.
$$

The Gaussian prefactor calibrates the reference midplane density. Since $\mathcal G$ is the exact
point-mass hydrostatic profile rather than its local Gaussian approximation, $\Sigma_g$ is not the
exact integral of this volume-density expression over $Z$. For a finite resolved domain whose
intersection at cylindrical radius $R$ consists of vertical intervals $[Z_{k,a},Z_{k,b}]$, the
implied column is

$$
\Sigma_{g,\mathcal D}(R)=\Sigma_g(R)\,\mathcal C_g(R),
$$

$$
\mathcal C_g(R)=\frac{1}{\sqrt{2\pi}H_g(R)}
\sum_k\int_{Z_{k,a}}^{Z_{k,b}}\mathcal G(R,Z)\,dZ.
$$

The factor $\mathcal C_g$ depends on the finite spherical radial and polar domain and need not
equal one. A vertically integrated model instead uses $\Sigma_g$ directly.

For comparison, on an untruncated vertical line let $x=Z/H_g$. Expanding the exact point-mass
profile gives

$$
\mathcal G(R,Z)
=e^{-x^2/2}
\left[1+\frac{3}{8}h_g^2x^4+O(h_g^4)\right],
$$

so the Gaussian fourth moment yields

$$
\mathcal C_{g,\infty}=1+\frac{9}{8}h_g^2+O(h_g^4).
$$

Finite spherical boundaries instead require the exact containment factor above and can dominate
the correction near a radial or polar edge.

Unless `CONST_ST` is selected, the local Stokes number uses the linear, subsonic Epstein-drag
scaling $t_s\propto s/(\rho_gc_s)$ of
[Epstein (1924)](https://doi.org/10.1103/PhysRev.23.710):

$$
\mathrm{St}(R,Z,s)
=\mathrm{St}_0\frac{s}{S_0}
\left(\frac{R}{R_0}\right)^{-p}
\mathcal G(R,Z)^{-1}.
$$

`CONST_ST` retains only the size factor. Imported-gas scaling is defined separately in Section 3.6.
For every branch, the drag time used by particle dynamics is

$$
t_s(R,Z,s)=\frac{\mathrm{St}(R,Z,s)}{\Omega_K(R)}.
$$

### 3.2 Dust surface-density profile and finite-domain containment

The physical dust profile starts from

$$
\Sigma_d(R)=Z_{\rm metal}\Sigma_g(R)
$$

before the Gaussian edge convolution

$$
\Sigma_{d,\rm conv}(R)
=\int_{Y_{\min}+2L_s}^{Y_{\max}-2L_s}
\Sigma_d(R')
\frac{\exp[-(R-R')^2/(2\sigma_s^2)]}{\sqrt{2\pi}\sigma_s}\,dR',
\qquad
L_s=0.05R_0,\qquad \sigma_s=\frac{L_s}{2}.
$$

The code evaluates this integral on a uniform cylindrical-radius axis with `N_Y + 1` points and
does not renormalize the truncated edge profile.

If $R_m=R_{\min,\rm init}+m\Delta R$ is the auxiliary axis, the convolution actually evaluated is

$$
\Sigma_{d,\rm conv}(R_m)
\approx\sum_{n\in\mathcal S}
\Sigma_d(R_n)
\frac{\exp[-(R_m-R_n)^2/(2\sigma_s^2)]}{\sqrt{2\pi}\sigma_s}
\Delta R,
$$

where $\mathcal S$ contains the source points inside
$[Y_{\min}+2L_s,Y_{\max}-2L_s]$. In a resolved vertical domain, the size-dependent dust scale
height uses the settling–diffusion balance discussed by
[Youdin & Lithwick (2007)](https://arxiv.org/abs/0707.2975):

$$
H_d(R,s)=H_g(R)
\sqrt{\frac{\alpha_Z}{\mathrm{St}_{\rm mid}(R,s)}},
\qquad
\alpha_Z=\frac{\alpha}{\mathrm{Sc}_Z}.
$$

This is the initializer implemented in `inc/swarm/swarm_host.cuh`. Unlike the fluid
initializer, it does not include finite-Stokes diffusivity suppression or switch with
`DIFFUSE_CONCENTRATION`; it is not an exact equilibrium of the evolved diffusion operator.

The spatial template for one size is

$$
\rho_d(R,Z\mid s)
=\frac{\Sigma_{d,\rm conv}(R)}{\sqrt{2\pi}H_d(R,s)}
\exp\left[-\frac{Z^2}{2H_d(R,s)^2}\right].
$$

This is a common full-column template whose size fraction is supplied later by $f_M(s)$; it is not
an additional copy of the total dust mass for every grain size.

For one grain size $s$, let

$$
I(s)=\int_{\mathcal D}\rho_d(\boldsymbol{x}\mid s)\,dV
$$

be the physical mass that the normalized surface profile would place inside the finite simulation
domain $\mathcal D$. The host initializer tabulates $I(s)$ on the same logarithmic size grid used
by the conditional spatial CDFs. At fixed cylindrical radius $R$, the spherical radial boundaries
and polar boundaries define at most two allowed vertical intervals. Their Gaussian containment is
integrated analytically with error functions, after which only the one-dimensional radial marginal

$$
\frac{dI}{dR}
=\Delta\phi\,R\Sigma_{d,\rm conv}(R)
\sum_k\left[
\Phi\left(\frac{Z_{k,\rm hi}}{H_d}\right)
-\Phi\left(\frac{Z_{k,\rm lo}}{H_d}\right)
\right]
$$

is numerically tabulated. Here $\Delta\phi=X_{\max}-X_{\min}$ for an active azimuth and $2\pi$
for an axisymmetric model. The auxiliary radial CDF has at least 2048 intervals, and the common
logarithmic grain-size axis has 128 entries whenever size-dependent 3D settling is active. Neither
resolution depends on the simulation polar grid, so a dust layer with $H_d\ll y\Delta z$ neither
loses mass nor collapses onto polar-cell centers during initialization.

The allowed vertical intervals follow directly from the spherical domain. At fixed cylindrical
radius $R$, the polar faces require

$$
R\cot Z_{\max}\le Z\le R\cot Z_{\min},
$$

and the outer spherical shell requires

$$
|Z|\le\sqrt{Y_{\max}^2-R^2}.
$$

If $R\lt Y_{\min}$, the inner shell removes

$$
|Z|<\sqrt{Y_{\min}^2-R^2},
$$

so the intersection can split into a lower and an upper interval. For any retained interval
$[Z_a,Z_b]$, its normalized Gaussian mass is

$$
\Delta\Phi(R,s;Z_a,Z_b)
=\Phi\left(\frac{Z_b}{H_d(R,s)}\right)
-\Phi\left(\frac{Z_a}{H_d(R,s)}\right),
$$

evaluated with `erf` or complementary `erfc` branches chosen to avoid subtracting two nearly equal
tail probabilities.

### 3.3 Physical dust mass in the modeled domain

The initialized physical number spectrum adopts the classical MRN power law of
[Mathis, Rumpl & Nordsieck (1977)](https://ui.adsabs.harvard.edu/abs/1977ApJ...217..425M):

$$
\frac{dN}{ds}\propto s^{-3.5},
$$

so, for a nondegenerate multisize interval $s_{\min}\lt s_{\max}$, compact grains with
$m_g(s)=\pi\rho_0s^3/6$ have the normalized mass spectrum

$$
f_M(s)=
\frac{s^{-1/2}}
{2\left(\sqrt{s_{\max}}-\sqrt{s_{\min}}\right)},
\qquad
\int_{s_{\min}}^{s_{\max}}f_M(s)\,ds=1.
$$

The physical dust mass represented inside the configured domain is therefore

$$
M_{\rm dust}
=\int_{s_{\min}}^{s_{\max}}f_M(s)I(s)\,ds
=\frac{1}{u_{\max}-u_{\min}}
\int_{u_{\min}}^{u_{\max}}I(u^2)\,du,
\qquad
u=\sqrt{s}.
$$

The code evaluates the last expression with 1024-interval Simpson quadrature. Positive entries of
the 128-knot domain-mass table are interpolated linearly in $(\log s,\log I)$; a nonpositive pair
falls back to linear interpolation in $I$. This interpolation is distinct from the arithmetic
interpolation of normalized radial CDF values used for position sampling.

The generic cases reduce as follows:

| Case | Contained mass used by the code |
|---|---|
| monodisperse | $M_{\rm dust}=I(S_0)$ |
| vertically integrated 1D or 2D | $I(s)=\Delta\phi\int R\Sigma_{d,\rm conv}(R)\,dR$, hence $M_{\rm dust}=I$ |
| finite 3D radial-polar domain | $M_{\rm dust}=\int f_M(s)I(s)\,ds$ with the analytic vertical containment above |
| vertically complete 3D domain | $I(s)$ approaches the common full-column value, so the size integral reduces to that value |
| imported gas | one analytical reference containment $I(S_0)$ is used for all sizes; imported density controls sampling shape, not absolute dust mass |

Representative grain counts, opacity, diagnostics, and dimensionless collision normalizations use
this runtime domain mass. There is no independent arbitrary dust-mass parameter.

In discrete form, with $N_q=1024$, $u_n=u_{\min}+n\Delta u$, and
$\Delta u=(u_{\max}-u_{\min})/N_q$, the implemented normalized Simpson rule is

```math
M_{\rm dust}\approx\frac{1}{u_{\max}-u_{\min}}
\frac{\Delta u}{3}
\left[
I(u_0^2)+I(u_{N_q}^2)
+4\sum_{\substack{n=1\\ n\ \mathrm{odd}}}^{N_q-1}I(u_n^2)
+2\sum_{\substack{n=2\\ n\ \mathrm{even}}}^{N_q-2}I(u_n^2)
\right].
```

Because $\Delta u/(u_{\max}-u_{\min})=1/N_q$, the implementation evaluates the weighted bracket
divided by $3N_q$.

### 3.4 Size proposals and finite representative weights

Let $q(s)$ be the probability density used to sample computational swarm sizes. Without radiation,

$$
q_{\rm mass}(s)=f_M(s),
\qquad
w(s)=\frac{f_M(s)}{q_{\rm mass}(s)}=1.
$$

With radiation, the proposal allocates equal full-column geometric area to each representative
before finite-domain containment:

$$
q_{\rm area}(s)
=\frac{s^{-3/2}}
{2\left(s_{\min}^{-1/2}-s_{\max}^{-1/2}\right)},
$$

and its exact importance weight is

$$
w_{\rm area}(s)
=\frac{f_M(s)}{q_{\rm area}(s)}
=s\,
\frac{s_{\min}^{-1/2}-s_{\max}^{-1/2}}
{\sqrt{s_{\max}}-\sqrt{s_{\min}}}.
$$

Because positions are sampled from
$p(\boldsymbol{x}\mid s,\boldsymbol{x}\in\mathcal D)=\rho_d/I(s)$, a finite realization uses

$$
\widetilde W_i=w(s_i)I(s_i),
\qquad
\mathcal N_M=
\frac{N_PM_{\rm dust}}{\sum_j\widetilde W_j},
$$

$$
W_i=\frac{\mathcal N_M}{N_P}\widetilde W_i
=M_{\rm dust}\frac{\widetilde W_i}{\sum_j\widetilde W_j},
\qquad
N_i=\frac{W_i}{m_g(s_i)}.
$$

Here $\mathcal N_M$ is the code variable `mass_norm`, $W_i$ is the physical mass represented by
swarm $i$, and $N_i$ is its stored `par_numr`. This construction enforces
$\sum_iW_i=M_{\rm dust}$ to roundoff. In a monodisperse model every swarm instead carries the
implicit equal weight $W_i=M_{\rm dust}/N_P$.

Both proposal laws are sampled by inverse transformation. For a power-law proposal
$q(s)\propto s^a$ with $a\ne-1$ and $U\sim\mathcal U(0,1)$,

$$
s=\left[s_{\min}^{a+1}
+U\left(s_{\max}^{a+1}-s_{\min}^{a+1}\right)
\right]^{1/(a+1)}.
$$

The mass proposal uses $a=-1/2$ and the equal-area radiation proposal uses $a=-3/2$.

### 3.5 Conditional spatial sampling

Monodisperse initialization samples cylindrical radius from this marginal and cylindrical height
from the exact truncated Gaussian on the selected vertical interval. The result is then converted
to stored spherical $(y,z)$ coordinates. For a multisize 3D disk, grain size is sampled first and
the radial CDF is interpolated on the common logarithmic size axis using the size-dependent dust
thickness defined in Section 3.2.

The conditional draw preserves the spherical shell and polar cutoffs without tying the physical
vertical distribution to the simulation cells. The cylindrical Jacobian $R$ appears once in the
radial marginal, while the normalized vertical Gaussian supplies the conditional height density.

For an intermediate size, the two adjacent normalized radial CDFs are interpolated arithmetically
on the logarithmic size axis. The exact truncated-Gaussian height is then sampled using that
particle's actual size rather than either neighboring knot size.

For one tabulated size, the normalized cylindrical-radius CDF is

$$
P_R(R\mid s)
=\frac{\displaystyle
\int_{R_{\min}}^R R'\Sigma_{d,\rm conv}(R')F(R',s)\,dR'}
{\displaystyle
\int_{R_{\min}}^{Y_{\max}}R'\Sigma_{d,\rm conv}(R')F(R',s)\,dR'},
$$

where $F(R,s)=\sum_k\Delta\Phi_k$ is the exact vertical containment. A uniform deviate $U_R$ is
inverted in a linearly interpolated radial-CDF bin. If $s$ lies between logarithmic knots $s_a$
and $s_b$, the CDF used for inversion is

$$
P_R(R\mid s)=(1-f_s)P_R(R\mid s_a)+f_sP_R(R\mid s_b),
\qquad
f_s=\frac{\ln s-\ln s_a}{\ln s_b-\ln s_a}.
$$

After drawing $R$, interval $k$ is selected with probability

$$
P(k\mid R,s)=\frac{\Delta\Phi_k(R,s)}{F(R,s)}.
$$

Within that interval, a second uniform deviate $U_Z$ gives the exact truncated-normal draw

$$
Z=H_d\Phi^{-1}\left[
\Phi\left(\frac{Z_{k,a}}{H_d}\right)
+U_Z\Delta\Phi_k
\right].
$$

The code evaluates $\Phi^{-1}$ with the piecewise rational approximation of Acklam (2000), while
the surrounding truncation and interval selection are specific to this initializer.

Finally,

$$
y=\sqrt{R^2+Z^2},
\qquad
z=\mathrm{atan2}(R,Z),
$$

while $x$ is uniform over the active azimuthal interval or exactly centered when azimuth is
inactive.

Unlike the fluid initializer, the swarm initializer applies no explicit azimuthal density
perturbation. Random spatial sampling supplies the finite-$N_P$ seed fluctuations of the empirical
particle distribution.

### 3.6 Initial velocities and imported gas

The initial velocity is the same no-backreaction steady drift used by the fluid branch and follows
the test-particle limit of
[Nakagawa, Sekiya & Hayashi (1986)](<https://doi.org/10.1016/0019-1035(86)90121-1>):

$$
\eta(R,Z)
=-\frac{1}{2}\left[
\left(p+\frac{q}{2}-\frac{3}{2}\right)h_g^2
+q\left(1-\frac{R}{\sqrt{R^2+Z^2}}\right)
\right],
\qquad
v_{\phi,g}=v_K\sqrt{\max(1-2\eta,0)}.
$$

$$
v_{R,d}
=\frac{v_{R,g}+2\mathrm{St}(v_{\phi,g}-v_K)}
{1+\mathrm{St}^2},
\qquad
v_{\phi,d}=v_{\phi,g}-\frac{\mathrm{St}}{2}v_{R,d}.
$$

An active vertical dimension also receives terminal settling
$v_Z=-\mathrm{St}\,\Omega_K Z$. `VISC_FLOW` replaces the zero gas radial velocity with the
analytic prescription of [Kanagawa et al. (2017)](https://arxiv.org/abs/1706.08975), requires
`DIFFUSION`, and cannot be combined with `IMPORTGAS` because imported gas velocities already
prescribe the gas flow.

This differs from the fluid initialization: the swarm adds terminal settling but no deterministic
velocity that balances its stochastic diffusive flux, whereas the fluid adds a polar
diffusive-balance velocity but no terminal-settling term. This asymmetry is intentional. Swarm
diffusion remains a separate stochastic positional operator and is not folded into the initialized
deterministic velocity. The two branches therefore should not be assumed to begin from an
identical vertical dynamical equilibrium.

The initialized cylindrical velocities are stored through

$$
\ell_\phi=Rv_{\phi,d},
\qquad
v_r=v_{R,d}\sin z+v_Z\cos z,
\qquad
\ell_\theta=y(v_{R,d}\cos z-v_Z\sin z).
$$

For a vertically integrated disk, that gas target is

```math
v_{R,g}=-\frac{3\nu}{R}
\left(g_\nu+p+\frac12\right),
\qquad
g_\nu=\frac{d\ln\nu}{d\ln R}
=\left\{\begin{array}{ll}
0,&\nu=\mathrm{constant},\\
q+3/2,&\alpha=\mathrm{constant}.
\end{array}\right.
```

For a resolved vertical domain, define

$$
\chi=\frac{R}{\sqrt{R^2+Z^2}},
\qquad
S=\frac{\chi-1}{h_g^2},
$$

$$
g_{\rho R}=p-\frac{q+3}{2}
+\frac{\chi(1-\chi^2)}{h_g^2}-(q+1)S,
\qquad
g_{\rho Z}=-\frac{\chi(1-\chi^2)}{h_g^2}.
$$

The implemented Kanagawa target is

$$
v_{R,g}=-\frac{\nu}{R}
\left[3\left(g_\nu+g_{\rho R}+\frac12\right)-q(1+g_{\rho Z})\right].
$$

It is projected into the spherical drag targets as

$$
v_{r,g}=v_{R,g}\sin z,
\qquad
\ell_{\theta,g}=yv_{R,g}\cos z.
$$

Imported gas initialization samples the imported gas density times $\epsilon$ using the same exact
cell-measure construction. Active azimuthal cells are sampled with `_get_dx()`; an inactive
azimuth is centered. Here $\epsilon$ controls the spatial shape of the dust sampling distribution;
the total represented dust mass is still normalized from `SIGMA_0`, `METAL_Z`, and the configured
domain rather than taken as an independent absolute mass from the imported field.

For cell $c$, with $\rho_g$ interpreted as surface density when `N_Z == 1`, the selection
probability is

$$
P_c=\frac{\epsilon_c\rho_{g,c}V_c}
{\sum_m\epsilon_m\rho_{g,m}V_m}.
$$

Every imported density and dust-to-gas ratio must be finite and nonnegative, and the denominator
must be finite and strictly positive. Initialization rejects the profile before normalizing or
sampling if any of these conditions fails. Individual zero-mass cells remain valid and simply have
zero selection probability.

After selecting a cell, the position is uniform in its exact volume coordinates

$$
s_y=\frac{y^d}{d},
\qquad
s_z=-\cos z,
$$

so, for independent $U_x,U_y,U_z\sim\mathcal U(0,1)$,

$$
x=x_{i-1/2}+U_x\Delta x,
\qquad
y=\left[y_{j-1/2}^d+U_y
\left(y_{j+1/2}^d-y_{j-1/2}^d\right)\right]^{1/d},
$$

$$
z=\cos^{-1}\left[
\cos z_{k-1/2}-U_z
\left(\cos z_{k-1/2}-\cos z_{k+1/2}\right)
\right].
$$

Inactive coordinates are assigned their exact centered values. When `N_Z == 1`, the imported
field is $\Sigma_g$ and

$$
\mathrm{St}(R,s)
=\mathrm{St}_0\frac{s}{S_0}\frac{\Sigma_0}{\Sigma_g(R)}.
$$

When `N_Z > 1`, the imported field is $\rho_g$ and `STOKES_0` is anchored to the analytical
reference midplane through

$$
\mathrm{St}(R,Z,s)
=\mathrm{St}_0\frac{s}{S_0}
\frac{\rho_{g,0}H_{g,0}}{\rho_g(R,Z)H_g(R)},
\qquad
\rho_{g,0}=\frac{\Sigma_0}{\sqrt{2\pi}H_{g,0}}.
$$

The imported density is assumed to have been calibrated from that $\Sigma_0,H_{g,0}$ reference
disk. A gap or other subsequent density depletion therefore increases the local Stokes number
above `STOKES_0` rather than resetting the depleted midplane to the reference value.
`CONST_ST` is therefore an analytic-gas-only option and is rejected when `IMPORTGAS` is enabled.

## 4. Governing kinetic, continuum, and particle equations

### 4.1 Phase-space equation, continuum moments, and characteristics

**Core swarm equations.** Let $f_d(\boldsymbol x,\boldsymbol v,s,t)$ be the mass-weighted dust
distribution introduced in Section 1.2. The continuum equation represented by the swarm is the
kinetic equation (written here for the default density-diffusion mode)

$$
\frac{\partial f_d}{\partial t}
+\nabla_{\boldsymbol x}\cdot(\boldsymbol v f_d)
+\nabla_{\boldsymbol v}\cdot(\boldsymbol a f_d)
=\nabla_{\boldsymbol x}\cdot
 (\boldsymbol D\nabla_{\boldsymbol x}f_d)
+\mathcal C_s[f_d],
$$

where $\boldsymbol a$ contains gravity, gas drag, radiation pressure, and Poynting–Robertson drag,
$\boldsymbol D$ is the cylindrical spatial-diffusion tensor, and $\mathcal C_s$ is the
representative-particle coagulation–fragmentation operator. Disabled physics removes its
corresponding term. In concentration mode replace $\boldsymbol D\nabla f_d$ by
$w\boldsymbol D\nabla(f_d/w)$, where $w$ is the prescribed gas density in the
appropriate dimension. Diffusivity depends on grain size through the local Stokes number.

Define the velocity-dispersion tensor and mass-weighted acceleration by

$$
\boldsymbol P_d
=\int(\boldsymbol v-\boldsymbol u_d)(\boldsymbol v-\boldsymbol u_d)f_d\,d^3v\,ds,
\qquad
\rho_d\overline{\boldsymbol a}
=\int\boldsymbol a f_d\,d^3v\,ds.
$$

For a monodisperse population in density mode, taking the zeroth and first velocity moments gives

$$
\frac{\partial\rho_d}{\partial t}
+\nabla\cdot(\rho_d\boldsymbol u_d)
=\nabla\cdot(\boldsymbol D\nabla\rho_d),
$$

$$
\frac{\partial(\rho_d\boldsymbol u_d)}{\partial t}
+\nabla\cdot
 (\rho_d\boldsymbol u_d\boldsymbol u_d+\boldsymbol P_d)
=\rho_d\overline{\boldsymbol a}
+\nabla\cdot
 [\boldsymbol D\nabla(\rho_d\boldsymbol u_d)].
$$

For multiple sizes, diffusivity cannot be taken outside the size integrals. The mass and
momentum diffusion terms are respectively $\nabla\cdot\int\boldsymbol D(s)\nabla f_d\,d^3v\,ds$
and $\nabla\cdot\int\boldsymbol v\boldsymbol D(s)\nabla f_d\,d^3v\,ds$, with the
concentration gradient substituted in concentration mode. The implemented collision event
preserves each representative's mass and velocity, so
$\mathcal C_s$ has zero mass and momentum moments even though it redistributes mass in grain-size
space. Drag and radiation depend on size, so $\overline{\boldsymbol a}$ generally contains
size–velocity correlations and cannot be reconstructed from $\rho_d$ and $\boldsymbol u_d$ alone.
Likewise, $\boldsymbol P_d$ need not vanish after trajectory crossing. The swarm solves the
particle characteristics and samples these unclosed moments rather than evolving the displayed
momentum PDE on a mesh.

As in the fluid branch, one coordinate-free system covers all supported geometries only after the
density measure and closure have been specified:

| Model | Density | Reduction of the master equations | Dynamical closure |
|---|---|---|---|
| full 3D | $\rho_d(r,\theta,\phi)$ | full spherical divergence | resolved vertical dynamics |
| radial–polar 2D | $\rho_d(r,\theta)$ | set $\partial_\phi=0$ but retain $u_\phi$ | axisymmetric resolved vertical dynamics |
| radial–azimuthal 2D | $\Sigma_d(R,\phi)$ | vertically integrate the 3D moments | midplane forces and a well-mixed unresolved column |
| radial-only 1D | $\Sigma_d(R)$ | vertically integrate and set $\partial_\phi=0$ | midplane radial dynamics while retaining $u_\phi$ |

The resolved-volume continuity equation is

$$
\frac{\partial\rho_d}{\partial t}
+\frac{1}{r^2}\frac{\partial}{\partial r}(r^2\rho_d u_r)
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}
 (\sin\theta\rho_d u_\theta)
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\phi}(\rho_d u_\phi)
=\mathcal D_{3D}[\rho_d],
$$

while vertical integration gives

$$
\frac{\partial\Sigma_d}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma_d u_R)
+\frac{1}{R}\frac{\partial}{\partial\phi}(\Sigma_d u_\phi)
=\mathcal D_{2D}[\Sigma_d].
$$

The radial–polar model drops the $\phi$ derivative from the first equation, and the radial-only
model drops it from the second. This is why separate full momentum systems are unnecessary, but
the dimensional reductions cannot be described as identical physics: vertical integration changes
the evolved density and requires a midplane closure.

Between stochastic diffusion displacements and collision events, every active swarm follows

$$
\frac{d\boldsymbol r}{dt}=\boldsymbol v,
$$

$$
\frac{d\boldsymbol v}{dt}
=-\frac{GM_\star}{r^2}\boldsymbol e_r
-\frac{\boldsymbol v-\boldsymbol v_g}{t_s}
+\boldsymbol a_{\rm rad+PR}.
$$

Without radiation, $\boldsymbol a_{\rm rad+PR}=0$. With radiation but without `PR_EFFECT`, it is
the attenuated outward radial acceleration $\beta GM_\star\boldsymbol e_r/r^2$. Section 6 gives the
complete first-order Poynting–Robertson form. The equations are integrated in spherical coordinates
with the stored variables $(\ell_\phi,v_r,\ell_\theta)$, so the equivalent radial centrifugal and
polar geometric terms appear explicitly in the component update.

The corresponding component equations are

$$
\frac{d\ell_\phi}{dt}
=-\frac{\ell_\phi-\ell_{\phi,g}}{t_s}-\gamma_{\rm PR}\ell_\phi,
$$

$$
\frac{d\ell_\theta}{dt}
=-\frac{\ell_\theta-\ell_{\theta,g}}{t_s}
+\frac{\ell_\phi^2\cos\theta}{R^2\sin\theta}
-\gamma_{\rm PR}\ell_\theta,
$$

$$
\frac{dv_r}{dt}
=-\frac{v_r-v_{r,g}}{t_s}
-(1-\beta)\frac{GM_\star}{r^2}
+\frac{\ell_\phi^2}{R^2r}
+\frac{\ell_\theta^2}{r^3}
-2\gamma_{\rm PR}v_r.
$$

Here $\gamma_{\rm PR}=0$ unless `PR_EFFECT` is enabled. For `N_Z == 1`,
$\ell_\theta=\ell_{\theta,g}=0$ identically and the polar equation is omitted.

Stochastic diffusion changes position while preserving the instantaneous Cartesian velocity, and
collisions change grain properties at fixed position. They are separate operators rather than
additional deterministic terms in the trajectory equation.

### 4.2 Radial-only reduction

The vertically integrated radial density obeys

$$
\frac{\partial\Sigma_d}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma_dv_R)
=\frac{1}{R}\frac{\partial}{\partial R}
\left(RD_R\frac{\partial\Sigma_d}{\partial R}\right),
$$

where transport follows particle characteristics and stochastic diffusion represents the
right-hand side. The midplane characteristic variables satisfy

$$
\frac{dR}{dt}=v_R,
$$

$$
\frac{d\ell_\phi}{dt}
=-\frac{\ell_\phi-\ell_{\phi,g}}{t_s}-\gamma_{\rm PR}\ell_\phi,
$$

$$
\frac{dv_R}{dt}
=\frac{\ell_\phi^2}{R^3}
-(1-\beta)\frac{GM_\star}{R^2}
-\frac{v_R-v_{R,g}}{t_s}
-2\gamma_{\rm PR}v_R.
$$

### 4.3 Ensemble measure and conserved represented mass

Integrating the empirical phase-space measure over velocity and size gives the represented spatial
density

$$
\rho_d^{N_P}(\boldsymbol x,t)
=\sum_{p=1}^{N_P}W_p\delta(\boldsymbol x-\boldsymbol x_p),
$$

or its vertically integrated analogue in 1D and 2D. The active represented mass is

$$
M_{d,\rm active}^{N_P}=\sum_{p\in\mathcal A}W_p,
$$

where $\mathcal A$ excludes absorbed representatives. Deterministic transport and stochastic
diffusion move positions without changing $W_p$. Coagulation and fragmentation update grain size
and represented number so that

$$
W_p=N_pm_g(s_p)=\text{constant}
$$

for every representative event. Consequently, these operators conserve total represented mass to
roundoff; only an absorbing physical boundary removes mass from the active domain. The collision
partner is a sample from the local bath and is not consumed, as required by the representative-
particle method rather than a literal pair-particle simulation.

Grid deposition is a diagnostic projection of this measure. If all stencil weights remain inside
the active domain and satisfy $\sum_cw_{pc}=1$, then

$$
\sum_c\rho_{d,c}V_c=\sum_{p\in\mathcal A}W_p.
$$

## 5. Semi-analytic trajectory integration

### 5.1 Staggered drag and force update

Particle trajectories use a staggered semi-analytic drag integrator. Coordinate drifts account for
the angular-momentum storage convention and the changing spherical radius. At the force stage,
frozen-coefficient drag is integrated exponentially, so a short stopping time is not an explicit
stability restriction.

The analytic force model includes:

- central gravity, optionally reduced by attenuated radiation pressure
- optional Poynting-Robertson damping when `PR_EFFECT` is enabled
- pressure-supported gas rotation
- optional viscous gas radial motion
- radial centrifugal acceleration
- polar geometric torque

When imported gas is enabled, the current and next snapshots are interpolated to each dynamics
step midpoint. This avoids holding the gas piecewise constant over a complete output interval.
If the two bracketing fields are $g_n$ and $g_{n+1}$ and the requested absolute frame fraction is
$f$, the target field is

$$
g(f)=(1-f)g_n+fg_{n+1}.
$$

The runtime stores a working field already advanced to $f_{\rm old}$, so the kernel uses the
incremental coefficient

$$
b=\frac{f-f_{\rm old}}{1-f_{\rm old}},
\qquad
g\leftarrow(1-b)g+b g_{n+1},
$$

which is algebraically identical to the direct interpolation above. The same operation is applied
to gas density and all three stored gas-velocity components.

For an initial state $i$, midpoint position $1$, and final state $j$, the first drift is

$$
y_1=y_i+\frac{\Delta t}{2}v_{r,i},
$$

$$
z_1=z_i+\frac{\Delta t}{2}
\frac{\ell_{\theta,i}}{y_iy_1},
\qquad
x_1=x_i+\frac{\Delta t}{2}
\frac{\ell_{\phi,i}}{y_iy_1\sin z_i\sin z_1}.
$$

For a vertically integrated model, $z_1=\pi/2$ and the azimuthal formula reduces to
$x_1=x_i+\ell_{\phi,i}\Delta t/(2y_iy_1)$; for a radial-only model $x_1$ is fixed as well.
The products of old and new metric factors are the discrete coordinate drift associated with the
stored angular variables.

At $(x_1,y_1,z_1)$, freeze $t_s$, the gas targets, and all position-dependent coefficients. Without
P-R drag, define

$$
\mathcal R(h)=1-e^{-h/t_s}.
$$

For any component with frozen nondrag force $F$, the exact response over $h$ is

$$
u(h)=u_i+\left(u_g+t_sF-u_i\right)\mathcal R(h).
$$

The code first evaluates forces $F_1$ from the initial angular momenta and constructs a midpoint
velocity with $h=\Delta t/2$. It then reevaluates the centrifugal and polar forces $F_2$ from
those midpoint angular momenta and computes the full velocity from the original state:

$$
u_j=u_i+\left(u_g+t_sF_2-u_i\right)\mathcal R(\Delta t).
$$

Azimuthal angular momentum has $F=0$. The final half drift is

$$
y_j=y_1+\frac{\Delta t}{2}v_{r,j},
$$

$$
z_j=z_1+\frac{\Delta t}{2}
\frac{\ell_{\theta,j}}{y_1y_j},
\qquad
x_j=x_1+\frac{\Delta t}{2}
\frac{\ell_{\phi,j}}{y_1y_j\sin z_1\sin z_j}.
$$

When radiation is active, the first drift is performed before opacity reconstruction so that
$\beta$ is evaluated at the same midpoint position as drag and gravity. When gas snapshots
$g^n,g^{n+1}$ bracket one output interval, the working field at the step midpoint $t_m$ is the
linear interpolation

$$
g(t_m)=(1-f_m)g^n+f_mg^{n+1},
\qquad
f_m=\frac{t_m-t_n}{t_{n+1}-t_n}.
$$

With smooth coefficients and no boundary event, the drift–centered-response–drift construction is
second-order in the deterministic timestep. The exponential drag factor is exact for frozen
$t_s$, gas target, and force; the remaining error comes from freezing their spatial dependence at
the midpoint and approximating the nonlinear force by the midpoint angular state.

### 5.2 Dynamical timestep

The runtime timestep is the inverse of the maximum active rate, capped by `DT_MAX`. The rate
calculation bounds orbital motion, azimuthal/radial/polar cell crossing, net-force displacement,
diffusion RMS displacement, deterministic diffusion drift, and radiation acceleration for the
smallest allowed grain where applicable. The production default `CFL_DYN = 0.45` keeps these
displacement bounds below their limiting value with a modest numerical safety margin.

The radial-only model retains the orbital rate because centrifugal and epicyclic dynamics remain
active. It also limits radial particle and gas-target crossing, force-driven displacement,
diffusion RMS displacement, deterministic diffusion drift, `DT_MAX`, and the remaining output
interval; azimuthal and polar crossing rates are inactive.

The kernel stores an inverse timestep rate. For particle $p$, let $j(p)$ be its radial cell and
define the local radial width

$$
\Delta r_p=y_{j(p)-1/2}(a_y-1),
$$

and the active physical azimuthal and polar lengths be
$L_x=R\Delta x$ and $L_z=y\Delta z$. The deterministic crossing candidates are

$$
\lambda_{\rm orb}=\frac{\Omega_K}{\mathrm{CFL\_DYN}},
\qquad
\lambda_x=\frac{|\ell_\phi|}{R^2\Delta x\,\mathrm{CFL\_DYN}},
$$

$$
\lambda_y=\frac{|v_r|}{\Delta r_p\,\mathrm{CFL\_DYN}},
\qquad
\lambda_z=\frac{|\ell_\theta|}{y^2\Delta z\,\mathrm{CFL\_DYN}}.
$$

When azimuth is active the code additionally includes
$\Omega_K/(\Delta x\,\mathrm{CFL\_DYN})$. Imported or viscous gas targets enter the same crossing
bounds, because stiff drag can transfer those velocities within one step.

For acceleration $a$ along a cell length $L$, requiring
$\tfrac12|a|\Delta t^2\le\mathrm{CFL\_DYN}L$ gives

$$
\lambda_a=\sqrt{\frac{|a|}{2\,\mathrm{CFL\_DYN}L}}.
$$

The radial candidate uses $L=\Delta r_p$ and the complete gravity-plus-centrifugal acceleration;
the polar candidate uses $L=y\Delta z$ and the polar geometric acceleration.

For the acceleration bound, radiation deliberately uses the unattenuated upper limit

$$
\beta_{\max}=\beta_0\frac{S_0}{s_{\rm bound}},
$$

where

```math
s_{\rm bound}=
\left\{\begin{array}{ll}
S_0,&\text{monodisperse},\\
s_i,&\text{multisize without collisions},\\
\min(s_i,s_{\min}),&\text{multisize with collisions}.
\end{array}\right.
```

Omitting attenuation and the time ramp makes the rate conservative before the local optical depth
or future collision product is known. For
diffusion, requiring the RMS displacement
$\sqrt{2D\Delta t}$ and deterministic drift $|b|\Delta t$ to remain within a fraction of $L$ gives

$$
\lambda_{D}=\frac{2D}{\mathrm{CFL\_DYN}^2L^2},
\qquad
\lambda_b=\frac{|b|}{\mathrm{CFL\_DYN}L}.
$$

The diffusion operator is defined in cylindrical $(R,Z)$ coordinates, while the mesh-crossing
limits are applied to spherical $(r,\theta)$ directions. The code therefore projects both the
variance and deterministic drift before evaluating these rates:

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

The radial rate uses $(D_r,b_r,L=\Delta r)$ and the polar rate uses
$(D_\theta,b_\theta,L=r\Delta\theta)$. Azimuth retains its cylindrical coefficient $D_\phi$ and
physical arc length $R\Delta\phi$.

After taking the maximum over all active candidates and particles,

$$
\Delta t_{\rm dyn}
=\min\left(
\frac{1}{\max_i\lambda_i},
\mathrm{DT\_MAX},
t_{\rm next\ output}-t
\right).
$$

## 6. Radiation pressure, optical depth, and Poynting–Robertson drag

### 6.1 Optical-depth construction and 2D closure

The swarm deposits extinction weight to the grid, converts it to a local radial optical-depth
increment, and performs a radial outer-face prefix sum. Particle forces interpolate that
cumulative field continuously at the particle radius.

In continuum notation,

$$
\tau(\phi,r,\theta)
=\int_{Y_{\min}}^r\rho_{\rm ext}(\phi,r',\theta)\,dr',
$$

where $\rho_{\rm ext}$ is extinction cross section per unit volume. For a mixture represented by
the swarms, its discrete measure is generated from the weights $Q_p$ below rather than from one
size-independent opacity.

Let the physical mass represented by swarm $p$ be

```math
W_p=
\left\{\begin{array}{ll}
m_g(s_p)N_p,&\text{multisize},\\
M_{\rm dust}/N_P,&\text{monodisperse}.
\end{array}\right.
```

Its extinction weight is

$$
Q_p=\kappa_0W_p\frac{S_0}{s_p}.
$$

In a vertically integrated model this becomes the midplane-equivalent weight

$$
Q_p^{\rm 2D}=\frac{Q_p}{\sqrt{2\pi}H_g(R_p)}.
$$

For continuous grid coordinates $(\xi_x,\xi_y,\xi_z)$, the particle is deposited on the adjacent
cell-centered stencil. If the one-dimensional neighbor fractions are $(f_x,f_y,f_z)$, its eight
weights are

$$
w_{abc}=f_x^a(1-f_x)^{1-a}
f_y^b(1-f_y)^{1-b}
f_z^c(1-f_z)^{1-c},
\qquad a,b,c\in\lbrace 0,1\rbrace,
\qquad \sum_{abc}w_{abc}=1.
$$

On the logarithmic radial mesh, $f_y$ is linear in physical radius between the relevant grid
locations rather than linear in $\log y$. Azimuth wraps periodically and an inactive dimension has
fraction zero. After atomic deposition, cell $c=(i,j,k)$ contains

$$
E_c=\sum_p w_{pc}Q_p^{(*)},
$$

where $Q_p^{(*)}=Q_p^{\rm 2D}$ in a vertically integrated model and $Q_p$ in 3D. The local radial
increment and cumulative outer-face optical depth are

$$
\Delta\tau_c=\frac{E_c}{V_c}\Delta r_j,
\qquad
\Delta r_j=y_{j-1/2}(a_y-1),
$$

$$
\tau_{i,j+1/2,k}=\sum_{m=0}^{j}\Delta\tau_{imk},
\qquad
\tau_{i,-1/2,k}=0.
$$

The force kernel interpolates this outer-face field back to the particle midpoint with the same
separable stencil, using zero optical depth at the inner radial boundary. Thus deposition and
interpolation remain continuous across cell centers while the logarithmic radial face centering is
handled explicitly.

In particular, if a particle has logarithmic coordinate $\xi_y=j+\delta$ with
$0\le\delta<1$, its physical-radius interpolation fraction toward the inner face is

$$
f_{\rm in}=\frac{a_y-a_y^\delta}{a_y-1},
$$

and

$$
\tau(y)=(1-f_{\rm in})\tau_{j+1/2}+f_{\rm in}\tau_{j-1/2}.
$$

For $j=0$, $\tau_{-1/2}=0$ is inserted explicitly.

The stored array ends at the outer face of the final radial cell. A particle in the outermost
half-cell therefore receives the last stored optical depth by radial stencil clamping rather than
an extrapolated value; this slightly overestimates attenuation there and is a documented boundary
approximation.

In a vertically integrated radial-only or radial–azimuthal disk, every dust species is assumed to
share the gas vertical profile. The midplane density closure is therefore

$$
\rho_{d,0}=\frac{\Sigma_d}{\sqrt{2\pi}H_g},
$$

independent of grain size and Stokes number. This well-mixed closure is used only when `N_Z == 1`;
radial–polar and full-3D models retain explicit vertical structure and settling.

### 6.2 Radiation-pressure ramp and attenuation

For geometric opacity $\kappa\propto s^{-1}$, the deposited extinction weight has the correct area
dimension. Radiation for each particle is

$$
\beta(s,t,\tau)
=\frac{\beta_0}{s/S_0}
f_\beta(t)e^{-\tau},
\qquad
f_\beta=s_t^2(3-2s_t),
\qquad
s_t=\mathrm{clip}(t/T_\beta,0,1).
$$

With radiation enabled, transport first drifts particles to midpoint positions, reconstructs
optical depth there, and then completes the force/drag step.

### 6.3 Poynting–Robertson response

When `PR_EFFECT` is enabled, the same attenuated and size-dependent $\beta$ supplies the
first-order Poynting–Robertson force of
[Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2), expressed here through
the rate

$$
\gamma_{\rm PR}=\frac{\beta GM_\star}{C_{\rm LIGHT}y^2}.
$$

The implemented first-order radiation acceleration is

$$
\boldsymbol a_{\rm grav+rad}
=-\frac{GM_\star}{y^2}\boldsymbol e_y
+\beta\frac{GM_\star}{y^2}
\left[
\left(1-\frac{v_y}{C_{\rm LIGHT}}\right)\boldsymbol e_y
-\frac{\boldsymbol v}{C_{\rm LIGHT}}
\right].
$$

Equivalently,

$$
\boldsymbol a_{\rm grav+rad}
=-(1-\beta)\frac{GM_\star}{y^2}\boldsymbol e_y
-\gamma_{\rm PR}
\left(
v_x\boldsymbol e_x+2v_y\boldsymbol e_y+v_z\boldsymbol e_z
\right).
$$

The stored tangential variables $\ell_\phi$ and $\ell_\theta$ are damped at
$\gamma_{\rm PR}$, while the spherical radial velocity is damped at
$2\gamma_{\rm PR}$. Gas and P-R drag are integrated together in the frozen-midpoint
exponential response:

$$
k_x=\frac{1}{t_s}+\gamma_{\rm PR},
\qquad
k_y=\frac{1}{t_s}+2\gamma_{\rm PR},
\qquad
k_z=\frac{1}{t_s}+\gamma_{\rm PR}.
$$

For any frozen component equation

$$
\frac{du}{dt}=-ku+\frac{u_g}{t_s}+F,
$$

the code evaluates

$$
u(h)
=e^{-kh}u(0)
+\frac{1-e^{-kh}}{k}
\left(
\frac{u_g}{t_s}+F
\right).
$$

The half response uses $h=\Delta t/2$ and forces evaluated from the initial angular momenta.
Those angular momenta are then used to reevaluate the centrifugal and polar terms, and the full
response uses $h=\Delta t$ from the original state with the updated midpoint forces. The
`expm1` form evaluates $1-e^{-kh}$ without small-argument cancellation.

The gas targets enter only through $u_g/t_s$, so P-R drag damps toward the stellar rest frame
rather than toward the gas. This integration is stable even when either damping rate is large,
although the trajectory timestep must still resolve spatial motion and coefficient variation.
No separate P-R stability term is added to `dyn_rate_calc` because the damping is analytic and
the P-R contribution is dissipative.

The dynamics always uses orbital code units, including builds without `CODE_UNIT`; that flag changes
collision microphysics rather than $G$, $M_\star$, $R_0$, position, or velocity units. `C_LIGHT` is
therefore the light speed in units of $v_0$ and must equal $c/v_0$ for the physical scales the model
represents. The default `C_LIGHT = 1.0e+04` is close to $c/v_0\approx1.007\times10^4$ for $R_0=1$ au
around one solar mass; a cgs value such as $c=2.99792458\times10^{10}$ would be inconsistent with
the orbital units.

## 7. Stochastic diffusion

### 7.1 Target diffusion equation and Itô process

`DIFFUSION` enables a Lagrangian stochastic representation of turbulent mixing in the spirit of
[Charnoz et al. (2011)](https://arxiv.org/abs/1105.3440). `DIFFUSE_CONCENTRATION` selects
concentration diffusion; without it, density diffuses. Both modes use the local Stokes number at
the current grain size:

$$
D_a=\frac{\nu}{\mathrm{Sc}_a(1+\mathrm{St}^2)},\qquad a=\phi,R,Z.
$$

The Schmidt numbers specify the gas-mixing coefficients before Stokes suppression.
Let $w=1$ for density diffusion and $w=\varrho_g$ for concentration diffusion,
where $\varrho$ denotes surface density in 2D and volume density in 3D. The target is

$$
\partial_t\varrho_d=\nabla\cdot\left[w\boldsymbol D\nabla(\varrho_d/w)\right].
$$

The diffusion tensor is diagonal in cylindrical coordinates. Independent Wiener
increments give the implemented Itô process:

$$
d\phi=\frac{\partial_\phi D_\phi+D_\phi\partial_\phi\ln w}{R^2}\,dt
 +\frac{\sqrt{2D_\phi}}{R}\,dW_\phi,
$$
$$
dR=\left(\partial_RD_R+D_R/R+D_R\partial_R\ln w\right)dt
 +\sqrt{2D_R}\,dW_R,
$$
$$
dZ=\left(\partial_ZD_Z+D_Z\partial_Z\ln w\right)dt
 +\sqrt{2D_Z}\,dW_Z.
$$

Inactive coordinates are omitted. The cylindrical volume-measure term $D_R/R$
applies in both modes. Derivatives hold grain size fixed and include
$\nabla\ln D_a=\nabla\ln\nu-2\mathrm{St}^2/(1+\mathrm{St}^2)\nabla\ln\mathrm{St}$.
Thus vertical diffusivity gradients can be nonzero even for viscosity depending only on $R$.
Analytic gas gradients follow the same spherical stratification used by drag.
With `IMPORTGAS`, gradients differentiate the actual periodic, boundary-clamped
trilinear gas-density interpolant; both diffusion half-steps use the working gas field.
`CONST_ST` freezes spatial Stokes variation only for analytic gas, as in the drag routine.

Euler–Maruyama uses independent standard normal draws, with physical displacement variance
exactly $2D_a\Delta t$ and no additional noise rescaling. The method has strong order $1/2$ and
weak order $1$ for regular spatially varying coefficients
([Kloeden & Platen 1992](https://doi.org/10.1007/978-3-662-12616-5)).
The dynamics rate includes these drift and RMS displacements; imported-gas rates
are evaluated at both supplied time endpoints. This endpoint policy is not a proof
of a bound throughout nonlinear temporal interpolation.

### 7.2 Boundaries and velocity reprojection

All supported geometries impose the same zero-flux radial diffusion boundary by repeatedly
reflecting the spherical radius into $[Y_{\min},Y_{\max}]$. In a vertically integrated model,
$y=R$, so this is also the cylindrical-radial boundary and does not change azimuth. In 3D, a
negative intermediate cylindrical $R$ is first mapped as
$(R,\phi)\mapsto(-R,\phi+\pi)$; this is a coordinate continuation across the cylindrical axis,
not a radial-boundary condition.

A random displacement is a spatial redistribution, not an impulse. The kernel reconstructs the
particle's Cartesian velocity before moving it and projects that unchanged velocity into the new
local spherical basis afterward. For an azimuthal wedge crossing, let $x_u$ be the unwrapped
post-diffusion angle and let the stored angle be

$$
x_w=x_u-m\Delta\phi_w,
\qquad m\in\mathbb Z.
$$

The velocity projection is evaluated at $x_u$, before the position is wrapped to $x_w$. This is
equivalent to rotating both the position and the Cartesian vector by the wedge identification
$-m\Delta\phi_w$; projecting the unchanged vector at $x_w$ would instead introduce an artificial
seam-local velocity kick. The same ordering is used after cylindrical-axis continuation in 3D.

This velocity treatment is internally consistent, but it is not the same momentum
closure as the fluid's donor-angular-momentum diffusion; density-only fluid–swarm diffusion
comparisons are valid until a common momentum equation is derived.

The reprojection is explicit. Before displacement,

$$
v_\phi=\frac{\ell_\phi}{R},
\qquad
v_\theta=\frac{\ell_\theta}{y},
$$

$$
v_R=v_r\sin z+v_\theta\cos z,
\qquad
v_Z=v_r\cos z-v_\theta\sin z,
$$

$$
v_X=v_R\cos x-v_\phi\sin x,
\qquad
v_Y=v_R\sin x+v_\phi\cos x.
$$

After moving to $(x',R',Z')$, the same fixed Cartesian components give

$$
v_R'=v_X\cos x'+v_Y\sin x',
\qquad
v_\phi'=v_Y\cos x'-v_X\sin x',
$$

$$
y'=\sqrt{R'^2+Z'^2},
\qquad
\sin z'=\frac{R'}{y'},
\qquad
\cos z'=\frac{Z'}{y'},
$$

$$
\ell_\phi'=R'v_\phi',
\qquad
v_r'=v_R'\sin z'+v_Z\cos z',
\qquad
\ell_\theta'=y'(v_R'\cos z'-v_Z\sin z').
$$

## 8. Representative-particle collisions

### 8.1 Representative particles and local event clocks

The collision update uses the representative-particle Monte Carlo interpretation of
[Zsom & Dullemond (2008)](https://arxiv.org/abs/0807.5052): a computational representative is
updated after sampling a physical partner from its local background population, while the sampled
partner representative is not consumed. Each event changes only the owner's grain size and
represented grain number, at fixed position and velocity.

The design follows from what a global collision step would require. A scheme that advanced all
representatives with one leap, in the spirit of explicit Poisson tau-leaping
([Cao, Gillespie & Petzold 2005](https://people.cs.vt.edu/~ycao/publication/JChemPhys_123_054104.pdf)),
with batch step

$$
\Delta t_{\mathrm{col}}=
\min\left(\frac{\mathrm{CFL}_{\mathrm{COL}}}{\max_i\lambda_i},
\Delta t_{\mathrm{remaining}}\right)
$$

would be controlled by the fastest representative. Raising the Courant factor does not solve this
bottleneck: if $\lambda_i\Delta t$ is large, collapsing several expected physical events into one
trial changes the stochastic process.

The code therefore implements a single collision integrator, the frozen-bath local continuous-time
event chain, which `COLLISION` always selects. Over a bath
interval $\tau_{\mathrm{bath}}$, each representative keeps a local clock, draws exact waiting times
with the direct method of [Gillespie (1977)](https://doi.org/10.1021/j100540a008),

$$
\delta t_i=-\frac{\ln U}{\lambda_i},
$$

and processes zero, one, or many events against immutable partner properties. After each event its
own rate and partner weights are recomputed. This removes the fastest-particle global microstep and
is exact conditional on the frozen bath; convergence under refinement of $\tau_{\mathrm{bath}}$
controls the bath-freezing error. Continuation launches bound divergent chain lengths without
truncating a stochastic path.

The remaining sections of this chapter define the pair rates (Sections 8.2 and 8.3), the outcome
law (Section 8.4), the chain itself (Section 8.5), the controller that chooses and audits bath
durations (Sections 8.6 and 8.7), its diagnostics (Section 8.8), and finite-bath convergence
(Section 8.9). The exact neighbor search that supplies each owner's partner
reservoir is described in Section 9.

A binned or otherwise modified representative-particle collision scheme is not merely a faster
implementation of the same stochastic process. It may change the estimator, population resolution,
and sampling variance. Any replacement or hybrid must therefore derive its conservation, bias,
variance, and resolution properties before runtime comparisons can justify adoption.

### 8.2 Pair propensities and neighborhood measure

For a pair $i,j$, the physical collision kernel is

$$
K_{ij}=\sigma_{ij}\Delta v_{ij},
\qquad
\sigma_{ij}=\frac{\pi}{4}(s_i+s_j)^2.
$$

In 3D, the pair propensity and owner rate are

$$
\lambda_{ij}=\frac{N_jK_{ij}}{V_{K,i}},
\qquad
\lambda_i=\sum_{j\in\mathcal N_i}\lambda_{ij},
$$

where $V_{K,i}$ is the accessible measure of the ball whose radius is the farthest retained KNN
distance. The retained set $\mathcal N_i$ includes the owner's own swarm, using $N_i-1\simeq N_i$
under the large-represented-number approximation. Every $j$ in these sums is an active particle. In
a vertically integrated model, the neighborhood measure is an area $A_{K,i}$ and the assumed
Gaussian vertical overlap gives

$$
\lambda_{ij}
=\frac{N_jK_{ij}}
{A_{K,i}\sqrt{2\pi(H_{g,i}^2+H_{g,j}^2)}}.
$$

This physical prescription is selected by `COAG_KERNEL = 3`. The other three values are
normalized synthetic kernel shapes intended for controlled coagulation studies:

$$
\kappa_0(m_i,m_j)=1,
\qquad
\kappa_1(m_i,m_j)=m_i+m_j,
\qquad
\kappa_2(m_i,m_j)=m_im_j.
$$

For `COAG_KERNEL = 0`, `1`, or `2`, the implemented pair propensity is

$$
\lambda_{ij}^{(q)}
=\frac{\lambda_0N_j\kappa_q(m_i,m_j)}{V_{K,i}},
\qquad
\lambda_0=\frac{N_P}{N_KM_{\rm dust}},
\qquad q\in\lbrace 0,1,2\rbrace,
$$

where $V_{K,i}$ denotes the active KNN measure; in a vertically integrated calculation it is the
KNN area. `variables.txt` records $\lambda_0$ as `LAMBDA_0`. These synthetic choices do not use
$\sigma_{ij}\Delta v_{ij}$ or the Gaussian vertical overlap factor and therefore must not be
interpreted as the physical collision prescription. Their relative speed is zero, so every
synthetic-kernel event is a sticking event (Section 8.4). The shared header defaults to
`COAG_KERNEL = 0`; a physical collision model must provide `COAG_KERNEL = 3` through its
model-specific `const_defs.cuh`.

The accessible measure corrects the neighborhood for physical boundaries. In the radial-only model,
both search backends search the collinear points $(R,0,0)$ and the exact accessible annular area is

$$
A_K=\pi\left[
\min(Y_{\max},R+d_K)^2-
\max(Y_{\min},R-d_K)^2
\right].
$$

Away from the exact radial-only case, the boundary correction treats each nearby radial or polar
face as locally planar. For a KNN radius $a$ and nonnegative distance $d\lt a$ to one face, the
excluded cap is

$$
C_2(a,d)=a^2\cos^{-1}\left(\frac{d}{a}\right)
-d\sqrt{a^2-d^2}
$$

in two search dimensions and

$$
C_3(a,d)=\frac{\pi}{3}(a-d)^2(2a+d)
$$

in three. Starting from $B_2=\pi a^2$ or $B_3=4\pi a^3/3$, every intersected face multiplies the
measure by

$$
1-\frac{C_D(a,d)}{B_D}.
$$

For an axisymmetric radial–polar search, the corrected two-dimensional meridional area is finally
multiplied by the revolution factor $2\pi R$. The product of individual cap fractions is a local
planar approximation when several curved boundaries overlap, not the exact multi-boundary
intersection.

The exact quantity is the measure of the Cartesian KNN ball intersected with the spherical disk
domain. An exact replacement would retain the 1D annular formula and evaluate boundary-overlapping
2D and 3D measures by deterministic quadrature of the exact intersection: solve the ball/domain
radial limits along each angular ray, integrate $R\,dR\,d\phi$ in a vertically integrated disk or
$r^2\sin z\,dr\,dz\,d\phi$ in 3D, and handle periodic wedge faces as identified copies rather than
physical cuts. Interior queries would keep the analytic disk or ball measure. A tabulated correction
in dimensionless boundary distances could amortize the quadrature, but it must be checked against
direct high-order quadrature before replacing the present cap formula.

### 8.3 Relative velocities

All physical collision paths use the same query-local closure, on CUDA and ROCm, with either
neighbor search and with analytic or imported gas. Both grain sizes use owner $i$'s position and gas
environment; partner position, image orientation and instantaneous particle velocities do not enter
the relative speed. Let $f_k=(1+\mathrm{St}_k^2)^{-1}$ and $v_n=-\eta R\Omega$ at the query
location. Then

$$
\Delta v_R=2v_n(\mathrm{St}_if_i-\mathrm{St}_jf_j),\qquad
\Delta v_\phi=v_n(f_i-f_j),
$$

$$
\Delta v_Z=Z\Omega[\min(\mathrm{St}_i,0.5)-\min(\mathrm{St}_j,0.5)],\qquad
\Delta v_{ij}^2=\Delta v_R^2+\Delta v_\phi^2+\Delta v_Z^2
+\Delta v_{B,ij}^2+\Delta v_{T,ij}^2.
$$

The drift closure assumes zero gas radial velocity and uses the analytic pressure-gradient and
temperature profiles, including with imported gas and with `VISC_FLOW`. Imported density is
sampled at the query for both Stokes numbers. The turbulent Reynolds normalization uses the query
surface density in 2D and the effective column $\sqrt{2\pi}\rho_g h_gR$ in 3D; analytic gas uses
$\Sigma_g(R)$ times the local stratification factor. `CONST_ST` retains its prescribed
size-to-Stokes relation. In a vertically integrated model the Gaussian overlap factor of Section 8.2
uses the partner's own cylindrical radius for $H_{g,j}$; this is the only place a partner position
enters the physical rate.

Brownian motion is included only in physical-unit builds, because code units provide no molecular
mass in simulation mass units:

$$
\Delta v_{B,ij}
=\min\left[
c_s,
\sqrt{\frac{8c_s^2M_{\rm mol}}{\pi}
\frac{m_i+m_j}{m_im_j}}
\right].
$$

Turbulent relative speeds follow the closed-form regimes of
[Ormel & Cuzzi (2007)](https://arxiv.org/abs/astro-ph/0702303). Let

$$
S=\max(\mathrm{St}_i,\mathrm{St}_j),
\qquad
s=\min(\mathrm{St}_i,\mathrm{St}_j),
\qquad
\epsilon=\frac{s}{S},
$$

$$
r_\eta=\mathrm{Re}^{-1/2},
\qquad
v_g^2=\frac32\alpha c_s^2,
\qquad
y_a=1.6,
$$

$$
y_s=1.6015125-0.63119577S+0.32938936S^2-0.29847604S^3.
$$

The turbulent Reynolds number is calibrated as

```math
\mathrm{Re}=
\left\{\begin{array}{ll}
\mathrm{Re}_0
\dfrac{\alpha}{\alpha_0}
\dfrac{\Sigma_g}{\Sigma_0},&\text{code-unit collision model},\\
\dfrac{\alpha\Sigma_gX_{\rm sec}}{2M_{\rm mol}},&\text{physical-unit collision model},
\end{array}\right.
```

where $\alpha_0$ is $\alpha$ evaluated at $R_0$. The implementation writes $\Delta v_T^2=v_g^2B$
with the six Ormel–Cuzzi regimes

```math
B=\left\{
\begin{array}{ll}
\dfrac{(S-s)^2}{r_\eta}, & S\lt0.2r_\eta,\\
\dfrac{S-s}{S+s}\left(\dfrac{S}{1+r_\eta/S}-\dfrac{s}{1+r_\eta/s}\right), & 0.2r_\eta\le S\lt r_\eta/y_a,\\
B_3, & r_\eta/y_a\le S\lt5r_\eta,\\
S\left[2y_a-(1+\epsilon)+\dfrac{2}{1+\epsilon}\left(\dfrac{1}{1+y_a}+\dfrac{\epsilon^3}{y_a+\epsilon}\right)\right], & 5r_\eta\le S\lt0.2,\\
S\left[2y_s-(1+\epsilon)+\dfrac{2}{1+\epsilon}\left(\dfrac{1}{1+y_s}+\dfrac{\epsilon^3}{y_s+\epsilon}\right)\right], & 0.2\le S\lt1,\\
\dfrac{1}{1+S}+\dfrac{1}{1+s}, & S\ge1,
\end{array}
\right.
```

where the transition coefficient is

$$
B_3=\frac{S-s}{S+s}\left(\frac{S}{1+y_a}-\frac{s^2}{s+y_aS}\right)+2(y_aS-r_\eta)+\frac{S}{1+y_a}-\frac{S^2}{S+r_\eta}+\frac{s^2}{y_aS+s}-\frac{s^2}{s+r_\eta}.
$$

A negative $B$ indicates a bug; the kernel prints an error and triggers a device assertion.

### 8.4 Collision outcomes

Let the projectile-to-target mass ratio of compact grains be

$$
q=\frac{m_j}{m_i}=\left(\frac{s_j}{s_i}\right)^3,
$$

and let the sticking packet size be

```math
G=\left\{\begin{array}{ll}
\max\left(1,\left\lfloor 10^{-4}/q\right\rfloor\right),&0\lt q\le10^{-6},\\
1,&\text{otherwise}.
\end{array}\right.
```

A packet groups $G$ identical tiny projectiles into one event, so each sticking event adds at most
about $10^{-4}$ of the target mass. Packets preserve the frozen-state mean mass growth but inflate
its variance. The outcome depends on whether $\Delta v_{ij}\ge V_{\rm frag}$ (`V_FRAG`, in the
model's velocity units) and on $q$. With a uniform deviate $U$ and the fragment floor
$s_{\rm floor}=$ `INIT_SMIN`, the outcome channels are

| Channel | Condition | Event rate | New owner diameter $s_i'$ |
|---|---|---|---|
| sticking | $\Delta v_{ij}\lt V_{\rm frag}$ | $\lambda_{ij}/G$ | $\left(s_i^3+Gs_j^3\right)^{1/3}$ |
| erosion remnant | $\Delta v_{ij}\ge V_{\rm frag}$, $q\le0.1$ | $\lambda_{ij}(1-q)/G$ | $s_i(1-Gq)^{1/3}$ |
| erosion debris | $\Delta v_{ij}\ge V_{\rm frag}$, $q\le0.1$ | $\lambda_{ij}q$ | $s_j$ |
| fragmentation | $\Delta v_{ij}\ge V_{\rm frag}$, $q\gt0.1$ | $\lambda_{ij}$ | $\left[\sqrt{s_{\rm floor}}+U\left(\sqrt{s_i}-\sqrt{s_{\rm floor}}\right)\right]^2$ |

Sticking produces target mass $m_i+Gm_j$ and an erosion remnant keeps $m_i-Gm_j$, while an erosion
debris event turns the owner into grains of the projectile mass $m_j$. Catastrophic fragmentation
draws a diameter whose upper limit is the old target diameter; the draw is clipped to
$[s_{\rm floor},s_i]$. The effective event rate that enters the chain is therefore
$\widetilde\lambda_{ij}=\varphi_{ij}\lambda_{ij}$ with $\varphi_{ij}=1/G$ for sticking,
$\varphi_{ij}=(1-q)/G+q$ for erosion, and $\varphi_{ij}=1$ for fragmentation. Given an erosion
event, the debris branch is selected with probability $q/\varphi_{ij}$.

Every event preserves represented mass by changing multiplicity,

$$
N_i'=N_i\frac{m_g(s_i)}{m_g(s_i')},
\qquad
W_i'=N_i'm_g(s_i')=W_i.
$$

The chain rejects an event whose new size or number is nonfinite or nonpositive, or whose
represented mass changes by more than $2\times10^{-12}$ relative (Section 11.6). The partner
representative is never modified.

For refresh control (Section 8.6), each channel also supplies conditional moments of the absolute
log-diameter jump $\left|\ln(s_i'/s_i)\right|$. Sticking has the deterministic jump
$\ln(1+Gq)/3$. Erosion mixes the remnant jump $j_r=-\ln(1-Gq)/3$ and the debris jump
$j_d=-\ln q/3$ with weights proportional to $(1-q)/G$ and $q$. For fragmentation, let
$L=\tfrac12\ln(s_i/s_{\rm floor})$; the first and second moments and the largest jump are

$$
\operatorname E\left|\ln\frac{s_i'}{s_i}\right|
=2\left(1-\frac{Le^{-L}}{1-e^{-L}}\right),
\qquad
\operatorname E\left(\ln\frac{s_i'}{s_i}\right)^2
=8-\frac{(4L^2+8L)e^{-L}}{1-e^{-L}},
\qquad
\max\left|\ln\frac{s_i'}{s_i}\right|=2L,
$$

with a series expansion for $L\lt10^{-3}$ that avoids cancellation near the fragment floor.

The fragmentation threshold is compared with $\Delta v_{ij}$ in the model's own units; the shared
header's `V_FRAG = 1.0` is not a physical calibration, and a physical model must set it together
with its disk parameters.

### 8.5 Frozen-bath continuous-time chain

The spatial index and each owner's physical top-$K$ neighbor identities and KNN measure are fixed
over one collision operator because positions do not change. At a local group refresh, that group
publishes its current partner sizes and represented numbers $(s_j^{(b)},N_j^{(b)})$; other groups
retain their last published values. Each owner chain reads these immutable reservoir arrays until
the current launch and audit finish.

Within a bath, owner $i$ evolves by the Gillespie direct method against that immutable reservoir:

$$
\lambda_i(s_i)=\sum_{j\in\mathcal N_i}\widetilde\lambda_{ij}(s_i;s_j^{(b)},N_j^{(b)}),
\qquad
\delta t_i=-\frac{\ln U_1}{\lambda_i},
$$

and, when the sampled clock remains inside the bath, the partner is drawn from

$$
P(j\mid i,s_i)=\frac{\widetilde\lambda_{ij}(s_i;s_j^{(b)},N_j^{(b)})}{\lambda_i(s_i)}.
$$

Partner selection walks the cumulative pair rates in neighbor-slot order and falls back to the last
positive slot if roundoff leaves the target uncovered. Uniform deviates are clipped strictly below
one so that $-\ln U$ and the inverse-CDF selection stay finite.

The owner size and represented number are updated after every event, so all owner-dependent pair
rates are recomputed before its next clock. This permits zero, one, or many events per owner without
using the fastest particle to impose a global collision microstep. The event cap `COL_EVENT_CAP`
bounds one kernel launch but not the stochastic path: local time, event count, accumulated
compensators (Section 8.7), and RNG state persist across continuation launches, and the cap is
checked before drawing another clock. Continuation queues contain only unfinished owners.

Two shortcuts avoid repeated neighbor passes without changing the process. At each refresh, a
cooperative kernel computes every owner's bath-start total rate and jump-rate moments and caches
them. A one-thread-per-owner screen then draws the first waiting time from the cached total; owners
whose first waiting time spans the whole interval complete immediately with their exact
compensators, and only the remaining owners enter the cooperative chain kernel. For an owner whose
first event falls inside the interval, the trial draw is not committed, so the full chain repeats
the same first draw; post-event and continuation states never use the cached totals. Owners that are
inactive or have a zero neighborhood measure complete without events.

Analytic physical-unit configurations without `CONST_ST` additionally cache the owner's gas
coefficients (height, Keplerian frequency, drift speed, Stokes scalings, sound speed, Reynolds and
turbulent scales) once per collision operator. Imported-gas, code-unit, and constant-Stokes paths
evaluate the same query-local closure without this cache.

The chain kernel assigns one block of `COL_BATH_TPB` threads to each queued owner. The threads
evaluate the $N_K$ pair rates and jump moments in parallel and reduce them; one thread advances the
owner's clock and applies the sampled event. On MI300A (`gfx942`) with `N_K = 256` and
`COL_BATH_TPB = 128`, the rate and chain kernels use a wavefront-shuffle reduction instead of the
general shared-memory fold.

Collisions share the per-representative RNG stream with diffusion (`RNG_STREAM_POLICY =
shared_per_particle` in `variables.txt`). The stream is persistent, so continuation launches,
reordered queues, and restarts do not change which random numbers an owner consumes.

### 8.6 Controller groups, size bins, and refresh durations

The spatial controller groups are the `COL_BIN_X`×`COL_BIN_Y`×`COL_BIN_Z` bins, uniform in azimuth,
spherical radius, and polar angle over the domain, with inactive azimuthal or polar dimensions
collapsed to one bin. They are assigned once per geometry epoch (Section 11.3). Owners are
dispatched only when their spatial group is due, and a refresh wave republishes, rerates, and rebins
only the due groups.

Inside every spatial group, grain sizes are binned into `COL_BIN_S` logarithmic size bins whose
bounds adapt independently per group. Immediately before rebuilding a refreshed group's bins, GPU
reductions find its current minimum and maximum active grain diameters. The bin range becomes
$[0.5s_{\min},8s_{\max}]$ and stays fixed through that collision interval and its audit; sizes
outside the range are clamped into the end bins. Groups not being refreshed retain their bounds.
This adds three GPU launches per refresh and no device-to-host copies, and it does not invalidate
the search index or the neighbor cache. `variables.txt` records `COL_SIZE_BIN_POLICY =
moving_per_group` and `COL_SIZE_RANGE_FACTORS = 0.5 8`. Adjacent size bins inside one spatial group
are then merged, from small to large sizes, until each merged bin holds at least `COL_BIN_MIN`
representatives; a sparse tail joins the preceding merged bin.

For each merged size bin $q$ with represented mass $M_q=\sum_{i\in q}w_i$, where $w_i=N_im_g(s_i)$
is the owner's represented mass at bath start, define the mass-weighted first and second absolute
log-diameter jump rates

$$
A_q=\frac{1}{M_q}\sum_{i\in q}w_i\sum_{j\in\mathcal N_i}\widetilde\lambda_{ij}
\operatorname E\left|\ln\frac{s_i'}{s_i}\right|,
\qquad
B_q=\frac{1}{M_q}\sum_{i\in q}w_i\sum_{j\in\mathcal N_i}\widetilde\lambda_{ij}
\operatorname E\left(\ln\frac{s_i'}{s_i}\right)^2,
$$

evaluated at bath start. Frozen rates give a mean accumulated absolute change $hA_q$ and a
compound-Poisson fluctuation scale $\sqrt{hB_q}$, and the requested duration of the group bounds
both by the tolerance:

$$
h_{\rm requested}=\min\left(
h_{\rm op},\ \mathtt{COL\_BATH\_MAX},\
\min_q\frac{\varepsilon}{A_q},\ \min_q\frac{\varepsilon^2}{B_q}
\right),
\qquad
\varepsilon=\mathtt{COL\_BATH\_EPS}\,\ell_c,
$$

where $h_{\rm op}$ is the collision operator horizon, terms with vanishing denominators are
omitted, and $\ell_c\in[0.25,1]$ is the group's adaptive safety factor (Section 8.7). This is a
refresh control, not a guaranteed numerical-error bound.

The scheduler avoids floating-point stagnation by using integer endpoints on a lattice of $2^{52}$
ticks per operator horizon. A group at level $l$ advances by $h_{\rm op}/2^l$, with $l$ the coarsest
level not exceeding the requested duration. Two groups are linked when any owner of one retains a
neighbor in the other, and linked groups are limited to a 16:1 duration ratio (four levels). Pending
endpoints are immutable: when a due group adapts, its new level is capped so that no in-flight
neighbor endpoint must move, and it must align with the current tick. A group whose last audit
passed may coarsen directly to its requested level, subject to these constraints; a group whose
audit failed may refine but not coarsen. A requested duration below $h_{\rm op}/2^{52}$ stops the
run.

### 8.7 Post-bath audit and controller adaptation

For the actual selected duration $\tau_b$, the predicted mass-weighted touched fraction and event
activity are

$$
F_q=\frac{1}{M_q}\sum_{i\in q}w_i
\left[1-\exp(-\lambda_i^{(b)}\tau_b)\right],
\qquad
E_q=\frac{1}{M_q}\sum_{i\in q}w_iH_i,
$$

where $\lambda_i^{(b)}$ is the bath-start rate and $H_i$ is the path-integrated hazard defined
below. The first-event probability depends only on the bath-start state because the owner cannot
change before that event; the expected total activity must instead follow the complete owner path.

Bath-start log-size jump moments select the requested duration, and the bath-start total rate is
retained for the first waiting time and touched-fraction prediction. The post-bath audit does not
assume that this rate remains fixed after the owner changes size. Along the piecewise-constant owner
path, the chain accumulates the exact predictable compensators

$$
H_i=\int_0^{\tau_b}\lambda_i[s_i(t)]\,dt,
$$

$$
J_{1,i}=\int_0^{\tau_b}\sum_j\widetilde\lambda_{ij}[s_i(t)]
\operatorname E\left(\left|\ln\frac{s_i'}{s_i}\right|\right)dt,
\qquad
J_{2,i}=\int_0^{\tau_b}\sum_j\widetilde\lambda_{ij}[s_i(t)]
\operatorname E\left[\left(\ln\frac{s_i'}{s_i}\right)^2\right]dt.
$$

Event activity is compared with $H_i$, while absolute logarithmic size change is compared with
$J_{1,i}$ and its second-moment compensator $J_{2,i}$. Because these compensators follow the
realized owner path, the event and size-change audits carry no bath-start-rate bias for kernels
whose rate depends on the owner size.

Writing $n_i$ for the realized number of events and $s_i^{(0)},s_i^{(1)}$ for the bath endpoints,
the corresponding realized summaries are

$$
\widehat F_q=\frac{1}{M_q}\sum_{i\in q}w_i\mathbf 1_{n_i\gt0},
\qquad
\widehat E_q=\frac{1}{M_q}\sum_{i\in q}w_in_i,
$$

$$
\widehat G_q=\frac{1}{M_q}\sum_{i\in q}w_i
\left|\ln\frac{s_i^{(1)}}{s_i^{(0)}}\right|.
$$

With $p_i=1-\exp(-\lambda_i^{(b)}\tau_b)$, the predictor for logarithmic activity is
$G_q^{\rm pred}=M_q^{-1}\sum_{i\in q}w_iJ_{1,i}$. The controller constructs Bernstein envelopes for
$F_q$, $E_q$, and $G_q^{\rm pred}$. For $Q$ occupied merged bins of the group and
$L=\ln(2Q/\mathtt{COL\_BATH\_ALPHA})$, each envelope has the form

$$
U_{X,q}=X_q^{\rm pred}+\sqrt{2V_{X,q}L}+\frac{b_{X,q}L}{3}.
$$

Here the mass-normalized variances are assembled from $w_i^2p_i(1-p_i)$ for touched owners,
$w_i^2H_i$ for event activity, and $w_i^2J_{2,i}$ for logarithmic activity. The bounded-increment
terms use the largest $w_i/M_q$ for $F$ and $E$, and the largest mass-weighted single-event
logarithmic jump for $G$. The touched fraction is additionally bounded above by one.

To measure how much represented mass moved between the joint spatial-size bins held fixed during
that interval, the code also records

$$
D_{\rm bath}=\frac{1}{2M_{\rm tot}}
\sum_q\left|M_q^{(1)}-M_q^{(0)}\right|.
$$

Owners retain their bath-start bin for the $F/E/G$ summaries, whereas the end state is re-binned on
the same fixed edges for $D_{\rm bath}$. A bath has an activity overshoot when realized touched or
event activity exceeds its confidence envelope, and a distribution overshoot when $\widehat G_q$
exceeds both `COL_BATH_EPS` and its envelope or when $D_{\rm bath}$ exceeds `COL_BATH_EPS`.

The group's safety factor $\ell_c$ starts at one. It is halved, down to the floor $0.25$, after a
distribution overshoot or after two consecutive overshooting baths of either kind, and it grows by a
factor 1.25, up to one, after three consecutive quiet baths. Two consecutive overshoots at the floor
are recorded as a persistent overshoot
(`COL_CONTROLLER_FAILURE = two_consecutive_minimum_scale_overshoots`). Completed baths are never
rejected and replayed, because conditioning acceptance on a random post-bath fluctuation would bias
the stochastic process, and persistent overshoots do not abort the run. Evolution can therefore
continue outside the requested audit tolerance; only invalid states, rates, and clocks stop it
(Section 11.6).

Controller memory persists across the split collision operators within one output interval and
resets at every output boundary, so particle and RNG checkpoints are sufficient to restart the
controller exactly.

### 8.8 Collision diagnostics

Production builds omit collision diagnostic files and event-category bookkeeping. The validation
runtimes enable `COL_DIAGNOSTICS`, which writes the per-group schedule of each output interval to
the schema-2 file `collision_chain_<frame>.json` and interval timing and event statistics of every
collision operator to `collision_local_<timestamp>.jsonl`. The former uses the existing controller
audit transfers; the latter adds per-event memory traffic, a GPU reduction, and a small
device-to-host copy per collision operator. Disabling diagnostics removes these extra costs,
history accumulation, and file writing, but retains the rate and audit transfers needed for
duration decisions and completion and error checks.

In `collision_chain_<frame>.json`, `bath_count` counts group records, `wave_count` counts refresh
waves, and `continuation_launches` counts all chain launches. Each bath record contains its
duration, the safety factor before and after the audit, the largest predicted and realized
summaries, $D_{\rm bath}$, and the three overshoot flags. The event statistics in the JSONL record
count events in seven categories: sticking with $q\le10^{-6}$, $q\le10^{-4}$, $q\le10^{-2}$, and
larger $q$; fragmentation; erosion remnant; and erosion debris, each with its summed logarithmic
mass change. Each JSONL group record also reports the group's level and duration, its initial,
final, minimum, and maximum requested durations, the constraint that bound the initial request
(horizon, mean change, or fluctuation), counts of coarsened, refined, and neighbor-constrained
updates, the largest ratio of actual to requested duration, and the largest age of a neighboring
group's published reservoir at the start of an update.

### 8.9 Finite-bath convergence

The frozen bath is exact only conditional on the frozen reservoir; finite-bath convergence is a
separate scientific requirement. The default `COL_BATH_EPS = 0.02` does not define a universal
tolerance. It is a model parameter that must be calibrated together with `N_K`, `H_SEARCH`, and
`N_P` before a scientific production campaign. A study that relies on the collisional size
distribution should compare at least two successively smaller bath tolerances across independent
seeds, using distributional observables at equal physical time. This qualification belongs to that
physical model rather than to a permanent matrix of synthetic validation cases; the retained chain
tests and their evidence boundary are described in [`swarm_testset.md`](swarm_testset.md).

## 9. Exact nearest-neighbor search

### 9.1 Backend selection and role in the estimator

The search backend is part of the collision estimator because it determines the selected local set
and the farthest-neighbor measure. In the physical kernel, represented counts carry the absolute
number normalization. Collision neighborhoods use one exact backend selected by the Makefile
variable `COLLISION_SEARCH`, set in the model's `flags.mk` or on the command line:

```make
COLLISION_SEARCH := kdtree
```

or

```make
COLLISION_SEARCH := morton
```

When the variable is unset, the default is `kdtree` for `GPU_BACKEND=cuda` and `morton` for
`GPU_BACKEND=rocm`; any other value stops the build. The Makefile translates the selection into the
internal macro `COLLISION_KDTREE` or `COLLISION_MORTON`, and both GPU backends accept either search.
Backend-specific objects are stored under separate search directories, so switching between KD-tree
and Morton cannot silently reuse objects compiled for the other backend. `variables.txt` records the
selection as `COLLISION_SEARCH`, together with `MORTON_LEAF_TARGET` and `MORTON_MAX_LEVEL` for the
Morton search.

The KD-tree follows the multidimensional search-tree concept introduced by
[Bentley (1975)](https://doi.org/10.1145/361002.361007), while the spatial ordering of the second
backend follows [Morton (1966)](https://dominoweb.draco.res.ibm.com/0dabf9473b9c86d48525779800566a39.html).
The KD-tree is an independent mature reference. The Morton backend uses an adaptive pointer-free
hierarchy, cooperative top-$K$ selection, and boundary-only periodic ghosts.

Both search backends populate the same full physical-neighbor cache on CUDA and ROCm: for every
owner, $N_K$ packed neighbor codes (Section 9.5) and one double-precision accessible measure. The
cache is built once per geometry epoch (Section 11.3) and reused by every bath of that epoch.
Subsequent bath selection, local chains, controller audits, and RNG handling are independent of the
search backend. No per-neighbor pair-rate array is retained, because rates become stale when the
owner changes size.

### 9.2 Search contract

For particle $i$, the location-dependent search cap is

$$
q_i=H_{\mathrm{SEARCH}}h_g(R_i)R_i
=H_{\mathrm{SEARCH}}H_g(R_i).
$$

Both backends must return the same original particle indices, squared distances, valid count, and
farthest valid distance under the lexicographic $(d^2,\mathrm{id})$ selection rule. They retain the
same $N_K$, $q_i$, accessible-ball normalization, pair physics, event probability, partner sampling,
and coagulation or fragmentation update. Candidates farther than $q_i$ are never retained, so an
owner in a sparse region may have fewer than $N_K$ valid neighbors; its measure then uses the
farthest valid distance. Approximate neighbors or a fixed-radius population estimator would be a
different numerical model and require a new derivation.

The searched Cartesian point is

$$
\boldsymbol X_i=(R_i\cos\phi_i,R_i\sin\phi_i,Z_i),
\qquad
d_{ij}^2=|\boldsymbol X_i-\boldsymbol X_j|^2.
$$

Candidate $a$ precedes candidate $b$ exactly when

$$
d_a^2\lt d_b^2
\quad\text{or}\quad
\left(d_a^2=d_b^2\ \text{and}\ c_a\lt c_b\right).
$$

For production collision queries, $c=3i+a$ packs the stable physical index $i$ and selected image
$a$ as defined in Section 9.5. Equal-distance ties therefore order by physical index, then image.
Duplicate-image removal compares decoded physical indices and retains the nearest code. Generic
Morton queries use ordinary identifiers; their active filtering uses the explicit identifier stride.

For equivalent candidate geometries, the contract fixes the selected set, not its storage order.
Morton returns a sorted list, whereas the KD-tree retains that set in heap order. Ordered rate sums
are reproducible within each backend, but an identical random target can therefore select a
different partner after switching backends. Backend trajectories and RNG states need not remain
byte-equal; mass conservation and analytical or statistical acceptance are the cross-backend
invariants. Neighbor sets and rates should agree where the admitted candidate geometries are
equivalent.

At the beginning of each geometry epoch, the code records one active byte per physical
representative. Absorbed representatives may remain in the immutable search hierarchy, but both
searches reject their stable identifiers before insertion. They therefore cannot occupy one of the
retained $N_K$ slots or reduce the active-neighbor radius. This avoids rebuilding a compacted index
while making absorbed particles invisible to collision selection.

### 9.3 KD-tree search

KD-tree construction and traversal use the bundled pointer-free tree implementation in
`inc/swarm/kdtree/`. The tree holds one record per physical representative or, in a partial
azimuthal wedge, three records per representative (the physical point and both adjacent periodic
images), and one query runs for each physical tree record.

The candidate heap uses the packed neighbor code as its equal-distance tie breaker and expands its
squared culling distance by one floating-point unit,

$$
d_{\rm cull}^2
\leftarrow\mathtt{nextafterf}(d_K^2,+\infty).
$$

The heap starts from the cap $q_i^2$, and each 8-byte heap slot stores only $(d^2,c)$; the shuffled
tree slot is not retained. Collision physics decodes the physical index for state access. This
prevents mutable tree slots from changing which member of an exact-distance tie is retained while
minimizing thread-local storage. For partial wedges, the heap checks for repeated physical
identifiers only when the minimum separation between adjacent periodic images is no larger than
twice the current query radius. Disjoint-image queries use ordinary $O(\log N_K)$ heap insertion
without an $O(N_K)$ duplicate scan.

On CUDA, the heap stores one independent strided shared-memory column per query thread. Launches use
`kdtree_heap::threads` queries per block: 16 through `N_K = 256`, then 8, 4, 2, and 1 through 512,
1024, 2048, and 4096, which keeps heap storage at or below 32 KiB per block. On ROCm, each query
keeps a private heap and a block holds 64 queries, a pairing tuned for MI300A. These limits apply to
the heap only; other collision buffers can impose tighter resource limits (Section 11.4).

### 9.4 Adaptive Morton hierarchy

The shared implementation lives in `inc/swarm/morton/` and supports CUDA and ROCm. The production
default `MORTON_TPB = 64` assigns 64 threads to each query block. Candidate compaction uses the
hardware `warpSize` (32 or 64 lanes), with stable packing across all warps in the block. Block
sizes divisible by 32, up to the device limit and at most 1024, use this path; other block sizes
retain the general cooperative path. Shared-memory requirements can limit usable block sizes
further. Top-$K$ merging reuses the sorted retained prefix, sorts the candidate half, retains the
lower half, and skips batches that cannot improve the cutoff. Periodic-image deduplication and
distance/identifier ordering are preserved. Ordinary merge padding is the next power of two
covering both `2*N_K` and `N_K+MORTON_TPB`; the periodic workspace covers `3*N_K+MORTON_TPB`.
Non-power-of-two neighbor counts are supported.

The Morton builder maps Cartesian coordinates to integer cells, interleaves their bits into 64-bit
keys, stable-sorts the records, and refines cells above `MORTON_LEAF_TARGET` on the GPU. At each
level, independent threads partition sorted-key ranges by child code; a device scan assigns child
slots and a second kernel fills the next level. Compact record and node arrays contain no pointers.
Keys and nodes stay on the device, including during node-buffer growth. The host reads only scalar
frontier counts, to allocate storage and launch the next level, and the final leaf count.
Leaf-occupancy statistics are copied to the host only when a benchmark requests them.

For root origin $\boldsymbol o$, width $W$, and maximum level $L$, each Cartesian coordinate is
quantized as

$$
I_\alpha=\mathrm{clip}\left(
\left\lfloor\frac{2^L(x_\alpha-o_\alpha)}{W}\right\rfloor,
0,2^L-1
\right).
$$

If $b_n(I)$ is bit $n$ of integer $I$, the three-dimensional key is

$$
M=\sum_{n=0}^{L-1}
\left[b_n(I_x)2^{3n}+b_n(I_y)2^{3n+1}+b_n(I_z)2^{3n+2}\right].
$$

For a two-dimensional search $I_z=0$. Equal keys retain their input order because the GPU sort is
stable. Recursive key prefixes then define the adaptive cells without storing parent or child
pointers in each particle record. The root cube has width $2.0002\,Y_{\max}$ centered on the star.

For node bounds $[\boldsymbol b_{\min},\boldsymbol b_{\max}]$, traversal uses the conservative
lower bound

$$
d_{\min}^2=\sum_\alpha
\left[\max(b_{\min,\alpha}-x_\alpha,0,x_\alpha-b_{\max,\alpha})\right]^2.
$$

A level-aware single-precision padding

$$
p=2(L_{\max}+2)\epsilon_{\rm float}
\max\left(1,|\boldsymbol b_{\min}|_\infty,|\boldsymbol b_{\max}|_\infty\right)
$$

prevents accumulated recursive-subdivision roundoff from making the pruning bound too large. One
GPU block evaluates candidate tiles cooperatively and retains the nearest $N_K$ pairs in shared
memory. The traversal stack holds 256 node indices; an overflow is reported and stops the run rather
than silently accepting an incomplete result.

The repeated block-parallel bitonic merge, which is the only Morton top-$K$ implementation,
retains the nearest pairs in lexicographic $(d^2,\mathrm{id})$ order. Distances remain available
because traversal pruning, closest-ghost deduplication, and the collision-volume radius require
them even though collision physics consumes the original particle identifiers. When periodic images
overlap, the query temporarily retains $3N_K$ records, deduplicates physical identifiers, and
performs a final ordered selection.

All threads in a query block consume a shared traversal-node index. A block barrier precedes every
replacement of that index, while the existing following barrier publishes the replacement. Both
halves are required: a following barrier alone allows a fast warp to begin the next iteration and
overwrite the index while a slower warp is still reading the preceding value.

The Morton candidate tiles use the same active-byte array as the KD heap. An absorbed record is
replaced by the empty sentinel before each shared-memory merge, so it cannot enter the retained
top-$K$ list. The hierarchy may still contain its spatial record, which costs traversal work but
does not affect neighbor identity or collision normalization.

### 9.5 Periodic boundary images

Both collision backends use the shared float-scale tolerance

$$
\epsilon_{\rm period}=10^{-6}.
$$

An active azimuthal domain is treated as a partial periodic wedge only when

$$
\Delta\phi_w\lt2\pi-\epsilon_{\rm period};
$$

otherwise it is treated as a complete period and no image records are constructed. This convention
matches the single-precision Cartesian search coordinates. It affects only deliberately configured
near-full wedges whose missing angle is at most $10^{-6}$ radians; exact $2\pi$ domains and ordinary
partial wedges are unaffected. If the omitted angle is $\delta\phi$, the largest seam displacement
introduced by this convention is

$$
2R\sin\left(\frac{\delta\phi}{2}\right)
\le R\epsilon_{\rm period}.
$$

Consequently, a model whose KNN cutoff or relevant neighbor separation is comparable to
$10^{-6}R$ should use an exactly full domain or a clearly partial wedge instead of relying on the
tolerance transition.

For wedge width $\Delta\phi_w=X_{\max}-X_{\min}$, the two possible image maps are

```math
\begin{pmatrix}X'\\ Y'\end{pmatrix}
=\begin{pmatrix}
\cos\Delta\phi_w&\mp\sin\Delta\phi_w\\
\pm\sin\Delta\phi_w&\cos\Delta\phi_w
\end{pmatrix}
\begin{pmatrix}X\\ Y\end{pmatrix},
\qquad
Z'=Z.
```

The sign is chosen so a source adjacent to one seam is copied across the opposite seam. Image
construction uses GPU `sincosf`, so the periodic isometry follows the same single-precision
arithmetic as the search records.

The selected periodic image is part of the collision neighbor, not merely a search-side copy of
its position. A packed 32-bit code stores

$$
c=3i+a,
\qquad
a=0,1,2,
$$

where $i$ is the physical representative and $a$ denotes the original, $-\Delta\phi_w$, or
$+\Delta\phi_w$ image. Decoding uses $i=\lfloor c/3\rfloor$ and $a=c\bmod3$. The invalid sentinel
remains negative. KD-tree heap entries and Morton record identifiers retain this code through
top-$K$ selection; duplicate removal compares physical indices while keeping the nearest selected
image. The neighbor cache stores the same code in its four-byte entry, so retaining image
orientation adds no cache array or per-neighbor memory. The packing is the reason collision builds
require $N_P\le\lfloor(2^{31}-1)/3\rfloor$. The physical relative-speed closure uses only the query
environment and the two grain sizes (Section 8.3), so it does not reconstruct or subtract partner
image velocities.

The KD-tree stores both adjacent images for every particle. The Morton owner instead copies only
source records whose Cartesian distance to either azimuthal face is no larger than

$$
q_{\max}=\max_i q_i.
$$

Lower-face sources are rotated by one positive wedge width and upper-face sources by one negative
wedge width. The original and compact ghost records are indexed together, while every record keeps
the original index of its physical representative. The global maximum is conservative for
position-dependent cutoffs: a query still uses its own $q_i$, while the larger construction halo
guarantees that no eligible source was omitted. When multiple images of one representative can
enter the same query ball, the query temporarily retains up to $3N_K$ records, deduplicates by
original particle index, and then selects the exact nearest $N_K$ physical particles. Ordinary disk
wedges use the cheaper disjoint-image path.

The two search representations therefore need not admit identical image sets. In a nearly
full-period wedge, unrestricted minimization over $0,\pm\Delta\phi_w$ can select an image that
Morton never searched. Spatial neighbor validation must therefore use the retained image, not
reconstruct the nearest image from positions. Each search is validated against its own candidate
set; cross-search equality is required only where those sets are equivalent.

When $H_gR$ varies strongly, $q_{\max}$ could be replaced by certified radial halo bins to reduce
ghost memory. For source and query radial bins $s$ and $b$, define

$$
D_{sb}=\max(0,R_s-R_{b+1},R_b-R_{s+1}),
$$

and use for source bin $s$

$$
h_s=\max_{\lbrace b:D_{sb}\le c_b\rbrace}c_b,
\qquad
c_b=\max_{i:R_i\in b}q_i.
$$

A source would be ghosted when its seam distance is at most $h_s$ plus roundoff padding. Such a
refinement must reproduce the global-halo neighbor sets exactly before it can replace the global
halo; it is not implemented.

### 9.6 Backend characteristics

The Morton representation offers contiguous pointer-free storage, bounded cooperative scratch,
boundary-only periodic records, explicit workload diagnostics, and natural Morton-range ownership
for multiple GPUs. Its hierarchy alone is compact, but the production owner also retains unsorted
query coordinates, azimuths, per-particle cutoffs, and overflow flags. Consequently, full-disk total
search storage can exceed the single-array KD-tree even when the Morton hierarchy is smaller;
partial wedges favor Morton more strongly because the KD-tree stores three complete copies
(Section 11.4). The KD-tree remains an independent mature reference and a strong single-GPU option.
Search selection preserves the physical pair-rate formula, but the candidate-set distinction above
can change the retained neighborhood in a wide wedge. Timing and memory must be measured for the
intended model, compiler, backend, and GPU; the correctness criteria are documented in
[`swarm_testset.md`](swarm_testset.md).

For broader context on cooperative GPU similarity search, see
[Johnson, Douze & Jégou (2017)](https://arxiv.org/abs/1702.08734); the exact top-$K$ and periodic
deduplication procedures used here are code-specific.

Following the domain-decomposition and halo-exchange pattern exemplified by
[García et al. (2012)](https://arxiv.org/abs/1210.1017), a distributed Morton search could let each
GPU own contiguous coarse cells or key ranges and import read-only halo records. A local result
would be globally certified when every intersecting remote cell has been examined and either
$d_{K,i}$ or $q_i$ is below the conservative distance to all remaining remote cells. Particle
migration, halo exchange, and collision-property synchronization are not implemented.

## 10. Operator composition, timestep hierarchy, and boundaries

### 10.1 Symmetric operator composition

For one dynamics interval, the code applies the symmetric splitting of
[Strang (1968)](https://doi.org/10.1137/0705041) in the form

$$
C^{1/2}D^{1/2}T D^{1/2}C^{1/2},
$$

where $C$ is collision, $D$ positional diffusion, and $T$ transport. This removes first-order Lie
splitting between enabled modules, but it does not make the stochastic diffusion or the
collision chain deterministically second order.

One dynamics step is executed as

```text
choose dt_dyn from all active particle rates, DT_MAX, and the remaining output interval
interpolate imported gas fields to the dynamics-step midpoint when required
evolve the frozen-bath collision chain over dt_dyn/2
apply stochastic positional diffusion over dt_dyn/2
if radiation is active, drift to midpoint positions, reconstruct optical depth, and finish transport
otherwise apply the complete semi-analytic transport update
apply stochastic positional diffusion over dt_dyn/2
evolve the frozen-bath collision chain over dt_dyn/2
advance the synchronized dynamics and output clocks
```

Disabled operators are skipped. A collision-only executable instead evolves one collision operator
over the complete remaining output interval, with imported gas interpolated to its midpoint.

Dynamics substeps exactly tile each output interval,

$$
\sum_n\Delta t_{{\rm dyn},n}=\Delta t_{\rm out},
$$

and the power-of-two bath durations of every controller group independently tile each opening or
closing half interval,

$$
\sum_m\Delta t_{{\rm col},m}=\frac12\Delta t_{{\rm dyn},n}.
$$

Thus all clocks synchronize at the end of every dynamics step even though collision groups and
dynamics use different internal timesteps. In a collision-only build, the bath durations instead
tile the complete output interval directly.

### 10.2 Boundary policies

Boundary policies are operator-specific:

- transport is periodic in azimuth
- transport absorbs radial exits and full-disk polar exits
- `HALF_DISK` reflects transport at the midplane and absorbs at the other polar edge
- diffusion is periodic in azimuth and reflecting at finite radial and polar boundaries in every
  supported geometry

Periodic wrapping applies

$$
x\leftarrow X_{\min}+\mathrm{mod}(x-X_{\min},X_{\max}-X_{\min}).
$$

For diffusion, a lower or upper radial overshoot is repeatedly folded by

$$
y\leftarrow2Y_{\min}-y
\quad\text{or}\quad
y\leftarrow2Y_{\max}-y,
$$

and the polar coordinate is folded analogously about $Z_{\min}$ or $Z_{\max}$. Repetition handles
an increment wider than one domain. If roundoff leaves a folded coordinate exactly on its upper
face, the implementation applies the inward clamps

$$
y\leftarrow Y_{\max}-10^{-12}(Y_{\max}-Y_{\min}),
\qquad
z\leftarrow Z_{\max}-10^{-12}(Z_{\max}-Z_{\min}),
$$

with the polar expression used only when that dimension is active. This keeps subsequent
half-open cell indexing inside the domain. Before spherical reconstruction, a negative cylindrical
radius is continued through the axis by

$$
(R,x)\leftarrow(-R,x+\pi).
$$

For transport in a half disk, crossing the midplane uses

$$
z\leftarrow\pi-z,
\qquad
\ell_\theta\leftarrow-\ell_\theta.
$$

An absorbing exit instead maps the representative to the inactive sentinel state

$$
y=0,
\qquad
z=\frac{\pi}{2},
\qquad
(\ell_\phi,v_r,\ell_\theta)=(0,0,0).
$$

An active `HALF_DISK` configuration is accepted only when $Z_{\max}=\pi/2$; the swarm runtime checks
this before allocating or evolving particle state. An absorbed representative is skipped by
subsequent transport, diffusion, deposition, opacity, timestep, and collision calculations; in a
multisize build it keeps its last grain size and represented number.

The production helpers apply boundary conditions to the completed operator endpoint. Repeated
folding makes diffusion positions lie inside the reflecting interval and is consistent with the
zero-flux process in the small-step limit, but it is not an exact finite-step transition for
spatially varying drift or diffusivity. Likewise, endpoint absorption cannot detect a trajectory
that crosses an absorbing face and returns inside during one transport step. A rigorous
boundary-event implementation would do the following:

1. construct a dense trajectory consistent with the staggered transport step
2. solve for the earliest radial or polar face crossing within the step
3. terminate an absorbing trajectory at that event, or reflect the normal velocity and reintegrate
   the unused part of the step for a reflecting face
4. treat stochastic crossings with a Brownian-bridge or reflected-transition construction and
   verify weak convergence under diffusion-step refinement

The CFL controls make missed transport crossings unlikely in ordinary runs, but do not constitute a
mathematical event detector. The collision neighborhood measure has its own locally planar boundary
approximation (Section 8.2).

### 10.3 Formal accuracy and statistical error

For smooth coefficients and no boundary event, deterministic trajectory error has the form

$$
\|\boldsymbol q_{\Delta t}-\boldsymbol q\|
\lesssim C_t\Delta t^2,
$$

because the semi-analytic response is centered at the midpoint. Other operators have different
notions of convergence:

| Component | Convergence property | Leading nonideal effect |
|---|---|---|
| deterministic transport and drag | second order for smooth frozen-coefficient variation | endpoint boundary handling and coefficient variation |
| imported-gas interpolation | second-order linear interpolation between bracketing snapshots | accuracy of the supplied gas cadence |
| Euler–Maruyama diffusion | strong order $1/2$, weak order $1$ | finite diffusion step and boundary folding |
| symmetric operator composition | second order for deterministic smooth operators | does not raise the intrinsic stochastic or collision-chain order |
| frozen-bath continuous-time chain | exact local Gillespie path conditional on the frozen partner reservoir | finite bath duration, sticking packets, and the fixed top-$K$ reservoir |
| exact KNN selection | no neighbor-approximation error under the stated search contract | finite-$N_P$ sampling and approximate boundary volume |
| initialization and deposition | deterministic quadrature plus Monte Carlo sampling | quadrature error and sampling noise |

For an ensemble observable $A$, independent-sampling noise scales asymptotically as

$$
\mathrm{SE}(\bar A)\simeq
\sqrt{\frac{\mathrm{Var}(A)}{N_{\rm eff}}},
$$

where $N_{\rm eff}$ can be smaller than $N_P$ for unequal importance weights, clustering, or
correlated collision histories. A useful weight-only estimate is

$$
N_{\rm eff}=\frac{\left(\sum_pW_p\right)^2}{\sum_pW_p^2}.
$$

Consequently, backend comparisons after stochastic divergence should use conserved quantities,
rate probes, distributional distances, and ensemble convergence rather than byte equality of
individual trajectories.

## 11. GPU implementation and computational characteristics

### 11.1 Source map and parallel mapping

The main numerical components map to the production source as follows:

| Scientific operation | Principal implementation |
|---|---|
| mass bank, size sampling, spatial CDFs, file I/O, and `variables.txt` | `inc/swarm/swarm_host.cuh` |
| grid indexing, interpolation, and deposition stencils | `inc/swarm/param_grid.cuh`, `inc/swarm/swarm_grid.cuh` |
| gas, Stokes-number, and grain-mass helpers | `inc/swarm/param_phys.cuh` |
| particle state initialization | `src/swarm/particle_init.cu` |
| semi-analytic dynamics and transport boundaries | `inc/swarm/_transport.cuh`, `src/swarm/ssa_substep_1.cu`, `src/swarm/ssa_substep_2.cu`, `src/swarm/ssa_transport.cu` |
| dynamical timestep rates | `src/swarm/dyn_rate_calc.cu` |
| stochastic diffusion | `inc/swarm/_diffusion.cuh`, `src/swarm/diffusion_pos.cu` |
| density and opacity deposition | `src/swarm/dustdens_*.cu`, `src/swarm/optdepth_*.cu` |
| imported-gas time interpolation | `src/swarm/gas_lerp_calc.cu` |
| pair physics, relative velocities, and KNN measure | `inc/swarm/_collision.cuh` |
| records shared by chain kernels and host controller | `inc/swarm/_col_types.cuh` |
| device pair rates and gas-environment cache | `inc/swarm/_col_rates.cuh` |
| outcome sampling and jump moments | `inc/swarm/_col_event.cuh` |
| moving size-bin bounds | `inc/swarm/_col_sizes.cuh` |
| host bath controller, audit, and JSON archive | `inc/swarm/_col_bound.cuh` |
| power-of-two group scheduler | `inc/swarm/_col_sched.cuh` |
| chain workspace and `evolve_local_collisions()` | `inc/swarm/_col_chain.cuh` |
| packed neighbor codes and cache slot offsets | `inc/swarm/_col_image.cuh` |
| KNN neighbor-cache kernel | `inc/swarm/_col_cache.cuh` |
| search records and nonfinite-state screen | `src/swarm/col_site_init.cu` |
| chain kernels, one per file | `src/swarm/col_*.cu` |
| KD-tree and Morton search | `inc/swarm/kdtree/`, `inc/swarm/morton/` |
| operator driver and output clock | `src/swarm/swarm_runtime.cu` |

Most swarm operators assign one GPU thread to one representative:

$$
p=\mathtt{threadIdx.x}+\mathtt{blockDim.x}\,\mathtt{blockIdx.x}.
$$

Transport, diffusion, initialization, timestep rates, and particle-to-grid deposition use this
mapping with `TPB` threads per block; the global dynamics timestep is a Thrust maximum reduction of
the per-particle rates. Deposition is a scatter operation and uses atomic additions,

$$
E_c\leftarrow E_c+w_{pc}Q_p,
$$

whereas gas and optical-depth interpolation are gather operations. Radial optical-depth prefix
sums assign one thread to each independent $(x,z)$ ray. The collision chain instead assigns one
block of `COL_BATH_TPB` threads to each owner (Section 8.5), the KD-tree search one thread to each
query (Section 9.3), and the Morton search one block of `MORTON_TPB` threads to each query
(Section 9.4).

The chain driver `evolve_local_collisions()` runs on the host. Per collision operator it caches gas
environments, rebuilds owner lists and the group dependency graph once per geometry epoch,
publishes the reservoir and computes bath-start rates and merged size bins, and then repeatedly
advances the earliest due groups: republish and rerate, adapt their levels, screen no-event owners,
run chain continuations until none remain, and audit the completed baths. Per refresh wave, the
host copies the due owner list, the group durations, and the size-bin map to the device and reads
back bin counts, merged rate and audit records, the queue length after each continuation launch,
and an error flag.

### 11.2 Precision and backends

The particle state is stored in double precision, while both collision-search backends use
single-precision Cartesian search coordinates and squared distances. The physical collision rate,
grain properties, dynamics, and accumulated diagnostics remain in `real` (double) precision.
Therefore the exact KNN contract means exact agreement under the backend's floating-point search
metric and lexicographic tie rule, not exact real-arithmetic geometry.

The ordinary workgroup width `TPB` is 64 on both GPU backends. CUDA uses the CUDA Runtime, cuRAND,
CUB, and Thrust; ROCm uses HIP Runtime, hipRAND, hipCUB, and rocThrust. Both backends compile the
same `.cu` files, with ROCm using `hipcc -x hip`, and `inc/gpu.cuh` maps runtime allocation, copy,
error, Thrust-policy, and random-number APIs to CUDA/cuRAND or HIP/hipRAND. Builds do not invoke
HIPIFY or generate HIP sources, and model overrides retain precedence on both backends. Both
backends compile with `-O2` and without blanket fast-math options, so particle and collision
nonfinite checks retain their specified semantics. Native accuracy validation on both backends is
described in [`val/README.md`](../val/README.md).

Explicit backend branches remain where hardware or toolchain behavior differs: shared (CUDA) versus
register (ROCm) storage of the collision RNG state in the chain kernel, the MI300A wavefront
reductions in `col_bath_rate` and `col_chain_run`, the private-heap KD-tree layout and query width,
ballot intrinsics in the Morton tile packing, the default `COL_BATH_TPB`, and Thrust execution
policies. Launch widths are compile-time constants; there is no automatic performance tuning.

### 11.3 Collision geometry epochs

A collision **geometry epoch** is the maximal interval during which particle positions, active
identities, and the mapping between particle identity and array index remain unchanged. Search
coordinates, the hierarchy, periodic ghosts, position-dependent cutoffs, physical top-$K$
identities, accessible measures, controller spatial bins, owner lists, and the group dependency
graph are invariant inside such an epoch. Sizes, represented numbers, pair rates, gas-dependent
microphysics, random clocks, size bins, and controller state are not geometry and are refreshed at
their normal numerical frequency.

Initialization, checkpoint loading, transport, diffusion, and any other operation that moves,
inserts, removes, reorders, or migrates representatives invalidate the epoch. Collision events,
rate calculations, gas interpolation, deposition, checkpoint writing, and output-boundary
controller reset do not. A resumed process therefore begins invalid and rebuilds once, whereas an
uninterrupted output boundary does not force a rebuild. Every collision operator still screens the
particle state for nonfinite values, including when it reuses the geometry.

This lifetime rule preserves the two separate collision half-operators in the Strang sequence; it
reuses only their unchanged geometry. For $M$ transported dynamics steps, the production reuse path
has

$$
N_{\rm call}=2M,
\qquad
N_{\rm build}=M+1,
\qquad
N_{\rm reuse}=M-1,
\qquad
N_{\rm invalidate}=M.
$$

Thus hierarchy construction approaches a factor-two reduction for long transported runs, but this
is not a factor-two reduction in total collision time. A collision-only run builds the geometry once
per process.

### 11.4 Memory footprint

Device memory is dominated by per-representative arrays. Ignoring allocator alignment, the
persistent allocations of one swarm executable are:

| Component | Present when | Device bytes per representative |
|---|---|---|
| particle state `swarm` (a pinned host copy of the same size also exists) | always | 48 monodisperse, 64 multisize |
| dynamics rate | `TRANSPORT` | 8 |
| RNG state | `DIFFUSION` or `COLLISION` | `sizeof(curandState)` = 48 on CUDA; `sizeof(hiprandState)` on ROCm |
| owner state (active byte, frozen size, number and rate, clocks, compensators, counters, flags) | `COLLISION` | 78 |
| chain workspace (cached moments 32, owner list and two queues 12, jump rates 16) | `COLLISION` | 60 |
| neighbor cache (packed codes and accessible measure) | `COLLISION` | $4N_K+8$ |
| gas-environment cache | `COLLISION`, analytic physical-unit gas without `CONST_ST` | 64 |
| per-owner event statistics | `COLLISION`, `COL_DIAGNOSTICS` | 112 |
| KD-tree records (24 bytes each) | `COLLISION_SEARCH=kdtree` | 24 for a full period or `N_X == 1`, 72 for a partial wedge |
| Morton query arrays (point, azimuth, cutoff, overflow flag) | `COLLISION_SEARCH=morton` | 24 |
| Morton hierarchy records (16 bytes each) | `COLLISION_SEARCH=morton` | 16 per physical or ghost record |
| Morton nodes (60 bytes each) | `COLLISION_SEARCH=morton` | node count scales as records per leaf occupancy; capacity is at most twice the node count |

Per-cell fields add `8*N_G` bytes each for `SAVE_DENS` and `RADIATION`, and `64*N_G` bytes for the
current and next imported gas density and velocity with `IMPORTGAS`; the controller bins are
negligible. The collision contribution is therefore about $146+4N_K$ bytes per representative plus
the search index and optional caches, so the neighbor cache dominates for ordinary $N_K$.

For example, a transported multisize physical-kernel model with `DIFFUSION`, analytic gas, the
default `N_K = 200`, and the KD-tree over a full period needs about

$$
64+8+48+(146+4\cdot200)+64+24\approx1150\ \text{bytes}
$$

per representative on CUDA: about 1.15 GB for $N_P=10^6$ and 11.5 GB for the default
$N_P=10^7$, before transient buffers. The neighbor cache alone, $4N_PN_K+8N_P$ bytes, accounts for
0.808 GB and 8.08 GB of these totals. Memory therefore scales linearly with both $N_P$ and $N_K$,
and doubling `N_K` nearly doubles the collision footprint.

Transient allocations raise the peak above this persistent sum:

- a fresh start holds the sampled positions (and sizes in a multisize build), 24 or 32 bytes per
  representative on the device and on the pinned host, while all persistent arrays are already
  allocated
- every geometry rebuild runs the Thrust-based KD-tree builder or rebuilds the Morton hierarchy,
  whose 8-byte sort keys, sort buffers, and, for a partial wedge, temporary ghost records exist only
  during construction
- Thrust reductions allocate small temporary buffers

Allocator alignment, the compiler's layout of the vendor RNG state, and the wedge-dependent Morton
ghost count change these numbers slightly, so the allocation of an intended production build should
be measured on its target GPU before a long run.

On the host, the pinned particle array, the fresh-start sample arrays, and a few 4-byte
per-representative vectors used to rebuild controller owner lists are the main costs. A failed
device allocation stops the run with the `Error: ...` message of `GPU_CHECK`, or with an exception
from the Morton builder (Section 11.6).

Static shared memory also bounds `N_K`. Per block, `col_chain_run` holds about $44N_K$ bytes of
pair-rate, jump-moment, and reduction arrays and `col_bath_rate` about $33N_K$ bytes. Under the
48 KiB static shared-memory limit per block of CUDA devices, the chain kernel limits `N_K` to
roughly 1100; AMD devices provide 64 KiB of LDS per workgroup. The Morton query holds its padded
merge workspace of 8 bytes per slot plus a 1 KiB traversal stack, and the CUDA KD-tree heap at most
32 KiB (Section 9.3).

### 11.5 Random streams and reproducibility

Initialization uses a fixed host random seed and stochastic evolution uses one persistent backend
RNG stream per representative, initialized with seed 1 and subsequence equal to the particle
index. Saving and restoring those states preserves an interrupted run for the same executable and
backend RNG-state layout. Changing the KNN backend can change partner ordering and RNG consumption
even when the neighbor set is identical, so cross-backend trajectories are not expected to remain
bytewise synchronized.

The CUDA stream is an opaque `curandState` and the ROCm stream an opaque `hiprandState`. Neither
vendor state is a portable file format: CUDA RNG checkpoints must be resumed with CUDA and HIP RNG
checkpoints with ROCm. Search implementations likewise use vendor-specific construction
primitives, but both must satisfy the same exact top-$K$ identifier, distance, active-mask, and
periodic-image contract.

### 11.6 Failure behavior

Inconsistent flags and constants fail at compile time (Sections 2.4 and 2.5). At run time the swarm
driver stops rather than continuing from an invalid state:

| Condition | Behavior |
|---|---|
| failed GPU runtime call or kernel launch | `GPU_CHECK` or `GPU_KERNEL_CHECK` prints `Error: ...` with the call and location and exits with `EXIT_FAILURE`; with `CUDA_SYNC_TRACE` or `HIP_SYNC_TRACE`, every kernel check also synchronizes, so asynchronous kernel faults are attributed to the failing kernel |
| missing input file, or a file whose size differs from the expected array | `Error: Failed to load file: ...` or `Error: Failed to load gas data files for frame ...`, exit with `EXIT_FAILURE`; the size check also catches restarts with a different `N_P`, `MULTISIZE` setting, or RNG-state layout |
| failed output write | `Error: Failed to save file: ...`, exit with `EXIT_FAILURE` |
| invalid resume argument | `Error: Invalid resume file number: ...`, exit with `EXIT_FAILURE` |
| `HALF_DISK` with $Z_{\max}\ne\pi/2$, zero or nonfinite initialized dust mass, or invalid imported density or dust-to-gas ratio | `std::runtime_error` before evolution starts |
| nonfinite particle state before a collision search | `Error: non-finite particle state before collision search at particle ...`, exit with `EXIT_FAILURE` |
| Morton traversal-stack overflow or Morton allocation failure | `std::runtime_error` |
| invalid chain state | `std::runtime_error` with `local collision chain error N`: `1` invalid pair rate, `2` nonfinite or negative total rate, `3` invalid waiting time, `4` non-advancing clock, `5` no positive partner slot, `6` invalid post-event size or number, or represented-mass drift above $2\times10^{-12}$ relative |
| more than $10^6$ continuation launches in one wave, invalid controller rate or audit moments, or a requested duration finer than the $2^{-52}$ scheduler resolution | `std::runtime_error` |

The driver does not catch these exceptions, so the process terminates through the C++ runtime
rather than with `EXIT_FAILURE`; files already written remain valid checkpoints. Controller audit
overshoots, including persistent ones, are recorded but never stop a run (Section 8.7).

## 12. Output and restart semantics

This section defines what the stored fields mean numerically. File names, cadences, the restart
command, and reading the binary layout are described in [Output files](../README.md#output-files),
[Restarting a simulation](../README.md#restarting-a-simulation), and
[Reading output with Python](../README.md#reading-output-with-python).

A multisize particle checkpoint holds

$$
(x,y,z,v_\phi,v_r,v_\theta,s,N_p)
$$

for every representative; a monodisperse checkpoint omits $(s,N_p)$. The runtime arrays store
$(\ell_\phi,v_r,\ell_\theta)$ internally, and the velocity conversion is

$$
v_\phi=\frac{\ell_\phi}{R},
\qquad
v_\theta=\frac{\ell_\theta}{y}
$$

when saving, and

$$
\ell_\phi=Rv_\phi,
\qquad
\ell_\theta=yv_\theta
$$

when loading. Saving and loading both reassert every inactive coordinate, so a radial restart has
exact $x=(X_{\min}+X_{\max})/2$, $z=\pi/2$, and $\ell_\theta=0$ before evolution resumes. Absorbed
representatives are stored in the sentinel state of Section 10.2, with zero velocity. Because the
checkpoint stores linear velocity, $R(\ell_\phi/R)$ and $y(\ell_\theta/y)$ can differ from their
original values by a floating-point ulp, so a resumed run with transport or diffusion is not
bitwise identical to an uninterrupted one even when the random-state file is byte-identical.

Collision or diffusion builds save one raw backend RNG state per representative beside every
particle checkpoint and restore it on resume, so a resumed stochastic run continues the same random
streams as an uninterrupted run built with the same backend and RNG-state layout. The collision
controller resets at every output boundary (Section 8.7) and the collision geometry is rebuilt from
the loaded positions (Section 11.3), so no collision state beyond the particle and RNG checkpoints
is needed. The resumed clock is reconstructed from the output schedule; `variables.txt` is written
only by a fresh start.

Mesh fields and particle checkpoints have distinct cadences. Grid density and optical depth, when
enabled, are reconstructed at every output index. The physical frame spacing is either linear,

$$
t_n=n\,\mathrm{DT\_OUT},
$$

or, with `LOGTIMING`, logarithmic,

$$
t_0=0,
\qquad
t_n=\mathrm{DT\_OUT}\,\mathrm{LOG\_BASE}^{\,n}
\quad(n\ge1).
$$

Particle checkpoints are written at frame 0 and then at every frame with `LOGTIMING`, at frame
indices that are integer powers of `LOG_BASE` (including 1) with `LOGOUTPUT`, or at every
`LIN_BASE`-th frame otherwise. A skipped particle checkpoint does not skip the evolution or the mesh
diagnostic at that output time.

A deposited dust-density diagnostic uses the same stencil as opacity:

$$
\rho_{d,c}=\frac{1}{V_c}\sum_p w_{pc}W_p,
$$

with $\rho_{d,c}$ interpreted as $\Sigma_{d,c}$ when `N_Z == 1`. Because $\sum_cw_{pc}=1$ for an
active interior particle, the grid integral recovers the total active represented mass up to
atomic-addition roundoff and boundary-stencil conventions. The written optical depth is the
cumulative outer-face field of Section 6.1 reconstructed from the end-of-interval positions; it is
not the midpoint field used by the last force update.

## 13. Known limitations

Verification definitions, evidence, and untested regimes are maintained in
[`swarm_testset.md`](swarm_testset.md), and limitations shared with the fluid branch are listed in
[Current limitations](../README.md#current-limitations). The limitations below concern the swarm
model and its numerics.

- The swarm diffusion-momentum treatment (velocity reprojection) differs from the fluid's
  donor-momentum closure, so the two branches should not be compared as the same velocity equation.
- `N_K == 1` and a polar domain reaching a coordinate singularity are not scientifically meaningful
  configurations and are not fully guarded. Analytical and imported-gas initialization do reject
  zero, negative, or nonfinite integrated dust mass before normalizing their sampling CDFs.
- The initialized vertical profile is not an equilibrium of the evolved diffusion operator, and the
  initial velocity contains no diffusive-balance drift (Sections 3.2 and 3.6).
- Boundary conditions act on operator endpoints; missed absorbing crossings within one transport
  step and finite-step reflection errors are controlled only by the timestep (Section 10.2).
- The optical depth in the outermost half-cell is clamped to the last stored face value
  (Section 6.1).
- The locally planar KNN boundary-cap correction is asymptotically consistent, not an exact
  curved-boundary intersection (Section 8.2).
- The collision relative-speed closure is query-local and analytic: it ignores the particles'
  actual velocities and assumes zero gas radial velocity even with `VISC_FLOW` or imported gas
  (Section 8.3).
- Sticking packets for $q\le10^{-6}$ preserve the mean mass growth but inflate its variance, and
  synthetic kernels never fragment (Section 8.4).
- The frozen-bath chain is exact only conditional on its frozen reservoir. Bath tolerance, neighbor
  count, search radius, and representative count are model-dependent controls that each scientific
  use must converge (Section 8.9); audit overshoots are recorded but do not stop a run.
- Collision memory grows as $4N_K$ bytes per representative for the neighbor cache alone, and
  collision builds require $N_P\le715\,827\,882$ (Section 11.4).
- A partial wedge whose missing angle is below $10^{-6}$ is treated as a full period
  (Section 9.5).
- The Morton search uses one global ghost halo; radial halo bins and multi-GPU ownership are not
  implemented (Sections 9.5 and 9.6).

## 14. References

- Mathis, Rumpl & Nordsieck (1977), [MRN grain-size distribution](https://ui.adsabs.harvard.edu/abs/1977ApJ...217..425M)
- Charnoz et al. (2011), [Lagrangian turbulent diffusion](https://arxiv.org/abs/1105.3440)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Ormel & Cuzzi (2007), [turbulent relative velocities](https://arxiv.org/abs/astro-ph/0702303)
- Zsom & Dullemond (2008), [representative-particle coagulation](https://arxiv.org/abs/0807.5052)
- Gillespie (1977), [stochastic reaction simulation](https://doi.org/10.1021/j100540a008)
- Kloeden & Platen (1992), [numerical stochastic differential equations](https://doi.org/10.1007/978-3-662-12616-5)
- Strang (1968), [symmetric operator splitting](https://doi.org/10.1137/0705041)
- Bentley (1975), [multidimensional binary search trees](https://doi.org/10.1145/361002.361007)
- Morton (1966), [geodetic database and file sequencing](https://dominoweb.draco.res.ibm.com/0dabf9473b9c86d48525779800566a39.html)
- Johnson, Douze & Jégou (2017), [GPU similarity search](https://arxiv.org/abs/1702.08734)
- García et al. (2012), [multi-GPU spatial decomposition and halos](https://arxiv.org/abs/1210.1017)
- Cao, Gillespie & Petzold (2005), [explicit Poisson tau-leaping](https://people.cs.vt.edu/~ycao/publication/JChemPhys_123_054104.pdf)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Takeuchi & Lin (2002), [vertically structured gas and dust drift](https://arxiv.org/abs/astro-ph/0208552)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](https://doi.org/10.1016/0019-1035(79)90050-2)
- Shakura & Sunyaev (1973), [$\alpha$ viscosity](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S)
- Epstein (1924), [drag on small spheres in a dilute gas](https://doi.org/10.1103/PhysRev.23.710)
- Acklam (2000), *An algorithm for computing the inverse normal cumulative distribution function*
