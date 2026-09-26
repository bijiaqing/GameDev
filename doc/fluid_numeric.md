# Eulerian dust-fluid model: equations and numerical guide

## 1. Model overview

The Eulerian branch represents monodisperse, pressureless dust as a finite-volume continuum on the
disk grid. Gas is prescribed analytically, dust does not back-react on it, and the solver combines
conservative transport with local forces and optional density or concentration diffusion. Shared
headers and application sources live under `inc/fluid/` and `src/fluid/`; `inc/gpu.cuh` selects
the CUDA or HIP runtime and random-number APIs. Building, configuring, running, and reading output
are described in the [user guide](../README.md); this document states the equations, their
discretization, and the numerical and implementation properties that determine what a run means.

### 1.1 Supported configurations

The branch supports:

- a two-dimensional radial–azimuthal disk with `N_Z == 1`
- a full three-dimensional spherical grid with `N_Z > 1`
- pressureless dust transport
- gas drag, gravity, spherical geometric forces, and optional radiation pressure
- optional turbulent diffusion of dust density or dust-to-gas concentration
- optional viscous gas accretion when diffusion is enabled

| Geometry | Grid condition | Evolved density | Interpretation |
|---|---|---|---|
| radial–azimuthal | `N_X > 1`, `N_Z == 1` | $\Sigma_d$ | vertically integrated disk |
| full 3D | `N_X > 1`, `N_Z > 1` | $\rho_d$ | spherical radial–polar volume |

Radial–polar two-dimensional models are not supported: the azimuthal dimension is always active.
Every 3D model requires `DIFFUSION` to support the dust layer vertically. Section 2.5 lists the
feature flags and the compile-time checks that enforce these conditions.

### 1.2 Physical assumptions and relation to the swarm branch

Both dust branches use the same prescribed central-star and gas-disk model, but they approximate
the dust distribution differently. The fluid branch advances low-order moments of a single-valued
velocity field, whereas the swarm branch advances a finite empirical phase-space distribution.
Their common and distinct closures are

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

The mathematical bridge between the branches is obtained from moments of a dust mass distribution
$f_d(\boldsymbol x,\boldsymbol v,s,t)$:

$$
\rho_d=\int f_d\,d^3v\,ds,
\qquad
\rho_d\boldsymbol u_d=\int\boldsymbol v f_d\,d^3v\,ds,
$$

$$
\int\boldsymbol v\boldsymbol v f_d\,d^3v\,ds
=\rho_d\boldsymbol u_d\boldsymbol u_d+\boldsymbol P_d.
$$

The pressureless fluid branch closes this hierarchy with $\boldsymbol P_d=0$, so its momentum flux
is

$$
\boldsymbol\Pi_d=\rho_d\boldsymbol u_d\boldsymbol u_d
$$

and contains no velocity-dispersion tensor. It is appropriate while the dust velocity remains
approximately single valued. After trajectory crossing, a kinetic or swarm description can retain
several velocities at the same location, whereas the fluid branch cannot represent the resulting
multistream distribution without an additional closure.

The gas is one-way coupled in both branches. In the gas equations that are not evolved by this
code, the dust-induced contributions are set to

$$
\left.\frac{\partial\rho_g}{\partial t}\right|_{\rm dust}=0,
\qquad
\left.\frac{\partial(\rho_g\boldsymbol v_g)}{\partial t}\right|_{\rm dust}=0.
$$

Thus dust drag changes dust momentum but never applies the equal-and-opposite backreaction to the
prescribed gas. Gas viscosity enters only through the prescribed viscous target velocity and dust
diffusivity; the code does not solve a gas viscous evolution equation.

## 2. Coordinates, grid, and evolved state

### 2.1 Coordinates and finite-volume measures

The computational coordinates are

$$
x=\phi,\qquad y=r,\qquad z=\theta,
$$

and

$$
R=y\sin z,\qquad Z=y\cos z.
$$

The mesh is uniform in $x$ and $z$ and logarithmic in $y$:

$$
\Delta x=\frac{X_{\max}-X_{\min}}{N_X},
\qquad
a_y=\left(\frac{Y_{\max}}{Y_{\min}}\right)^{1/N_Y},
\qquad
\Delta z=\frac{Z_{\max}-Z_{\min}}{N_Z}.
$$

Its faces and coordinate centers are

$$
y_{j-1/2}=Y_{\min}a_y^j,
\qquad
y_j=Y_{\min}a_y^{j+1/2},
$$

$$
z_{k-1/2}=Z_{\min}+k\Delta z,
\qquad
z_k=Z_{\min}+\left(k+\frac12\right)\Delta z.
$$

Complete grid cells use $(i,j,k)$ for the $(x,y,z)=(\phi,r,\theta)$ directions, and arrays are
stored with $i$ varying fastest. A symbol $i$ used later inside a purely one-dimensional
tridiagonal or reconstruction derivation denotes a generic line index and is not a second
radial-index convention.

The radial finite-volume measure is

```math
\Delta V_y=\frac{y_{\rm out}^d-y_{\rm in}^d}{d},
\qquad
d=
\left\lbrace\begin{array}{ll}
2,&N_Z=1,\\
3,&N_Z>1.
\end{array}\right.
```

The polar measure is $\Delta V_z=\cos z_{\rm in}-\cos z_{\rm out}$ when the polar dimension is
active. Consequently, `N_Z == 1` represents a vertically integrated disk and evolves dust surface
density $\Sigma_d$; `N_Z > 1` evolves volume density $\rho_d$.

The complete cell measure can be written

```math
V_{ijk}=\Delta x\,
\frac{y_{j+1/2}^{d}-y_{j-1/2}^{d}}{d}
\left\lbrace\begin{array}{ll}
1,&N_Z=1,\\
\cos z_{k-1/2}-\cos z_{k+1/2},&N_Z>1.
\end{array}\right.
```

Radial fluxes use $A_{y,j+1/2}=y_{j+1/2}^{d-1}$. In 3D, the radial contribution to a polar face
area is

$$
\Delta A_{z,j}
=\int_{y_{j-1/2}}^{y_{j+1/2}}y\,dy
=\frac{y_{j+1/2}^2-y_{j-1/2}^2}{2},
$$

so the separated polar face factor is $A_{z,j,k+1/2}=\Delta A_{z,j}\sin z_{k+1/2}$.

Every azimuthal domain is periodic. A range shorter than $2\pi$ is therefore a periodic wedge, not
an open sector of an otherwise complete disk: transport, FARGO shifts, invariant-domain bounds, and
azimuthal diffusion all identify its two azimuthal faces. Non-axisymmetric structures repeat with
period $X_{\max}-X_{\min}$, which must be treated as part of the model definition.

### 2.2 Primitive and conserved variables

The primitive arrays named `dustvelx/y/z` contain

$$
(\ell_\phi,v_r,\ell_\theta)
=(Rv_\phi,v_r,rv_\theta),
$$

not three linear Cartesian or spherical velocities. The conserved arrays contain

$$
(\rho_d,\rho_d\ell_\phi,\rho_dv_r,\rho_d\ell_\theta).
$$

For either density interpretation, define

```math
\boldsymbol U=
\begin{pmatrix}
\varrho_d\\ m_x\\ m_y\\ m_z
\end{pmatrix}
=\begin{pmatrix}
\varrho_d\\
\varrho_d\ell_\phi\\
\varrho_dv_r\\
\varrho_d\ell_\theta
\end{pmatrix},
\qquad
(\ell_\phi,v_r,\ell_\theta)
=\frac{(m_x,m_y,m_z)}{\varrho_d}.
```

When $\varrho_d\lt\rho_{\rm vac}$, this division is replaced by

$$
\ell_\phi=\sqrt{GM_\star R},
\qquad
v_r=0,
\qquad
\ell_\theta=0,
\qquad
m_a=\varrho_du_a.
$$

This regularized vacuum state keeps all later arithmetic finite without assigning scientific
meaning to the velocity of an almost empty cell. The same reset is applied whenever primitives are
recovered from conserved momenta, whenever conserved momenta are rebuilt from primitives, and at
the start of the source update.

For a 2D disk, $\rho_d$ in these expressions means the evolved surface density. File output
converts $\ell_\phi$ and $\ell_\theta$ to $v_\phi$ and $v_\theta$ without changing the device state
(Section 10).

### 2.3 Units, dimensions, and notation

The default constants set

$$
G=M_\star=R_0=1.
$$

The corresponding orbital scales are

$$
t_0=\sqrt{\frac{R_0^3}{GM_\star}},
\qquad
v_0=\sqrt{\frac{GM_\star}{R_0}},
\qquad
\Omega_0=t_0^{-1}.
$$

These equations define the conversion to a chosen physical $M_\star$ and $R_0$; setting the
constants to one does not make dimensional consistency optional. With the defaults, one orbital
period at $R_0$ is $2\pi$ code time units. The principal dimensions are

| Quantity | Symbol | Dimension |
|---|---|---|
| surface density | $\Sigma$ | $M L^{-2}$ |
| volume density | $\rho$ | $M L^{-3}$ |
| linear velocity | $v$ | $L T^{-1}$ |
| specific angular momentum | $\ell$ | $L^2T^{-1}$ |
| kinematic viscosity or diffusivity | $\nu,D$ | $L^2T^{-1}$ |
| opacity | $\kappa$ | $L^2M^{-1}$ |
| stopping time | $t_s$ | $T$ |

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

Throughout this guide, $r$ denotes spherical radius, $R=r\sin\theta$ cylindrical radius, and
$Z=r\cos\theta$ cylindrical height. A generic $\varrho_d$ denotes whichever density the active
geometry evolves: $\Sigma_d$ in a vertically integrated model and $\rho_d$ in 3D.

### 2.4 Code parameters

All parameters are compile-time constants in
[`inc/fluid/const_defs.cuh`](../inc/fluid/const_defs.cuh); a model replaces the complete file with
its own `const_defs.cuh` ([user guide](../README.md#constants)). The type `real` is `double`.
Lengths are in units of $R_0$, times in units of $t_0$, and angles in radians. Parameters marked
with a flag exist only when that flag is defined.

| Constant | Default | Mathematical role | Defined when |
|---|---|---|---|
| `G`, `M_S`, `R_0` | 1, 1, 1 | $G$, $M_\star$, and the reference radius $R_0$ | always |
| `N_X`, `X_MIN`, `X_MAX` | 1024, 0, $2\pi$ | azimuthal cell count and periodic range | always |
| `N_Y`, `Y_MIN`, `Y_MAX` | 1024, 0.5, 2.5 | radial cell count and spherical-radius range | always |
| `N_Z`, `Z_MIN`, `Z_MAX` | 1, $\pi/2$, $\pi/2$ | polar cell count and colatitude range | always |
| `SIGMA_0` | $10^{-2}$ | gas surface density $\Sigma_0$ at $R_0$ | always |
| `ASPR_0` | 0.05 | gas aspect ratio $h_0$ at $R_0$ | always |
| `IDX_P`, `IDX_Q` | $-1$, $-0.4$ | power-law indices $p$ of $\Sigma_g$ and $q$ of $T$ | always |
| `METAL_Z` | $10^{-2}$ | initial metallicity $Z_{\rm metal}$ before the edge taper | always |
| `STOKES_0` | $10^{-3}$ | reference midplane Stokes number $\mathrm{St}_0$ at $R_0$ | always |
| `ALPHA` | $10^{-3}$ | turbulent $\alpha$ | `DIFFUSION` without `CONST_NU` |
| `NU` | $10^{-5}$ | constant kinematic viscosity $\nu$ | `DIFFUSION` and `CONST_NU` |
| `SCHMIDT_X`, `SCHMIDT_Y`, `SCHMIDT_Z` | 1, 1, 1 | directional Schmidt numbers $\mathrm{Sc}_{x,y,z}$ | `DIFFUSION` |
| `POS_LIMIT` | 0.9 | CN explicit-side coefficient-sum bound and exported old-donor mass fraction per diffusion substep | `DIFFUSION` |
| `BETA_0` | 10 | unattenuated radiation ratio $\beta_0$ | `RADIATION` |
| `KAPPA_0` | $5\times10^{4}$ | opacity $\kappa_0$ | `RADIATION` |
| `T_BETA` | $2\pi$ | radiation ramp time $T_\beta$; a value $\le0$ disables the ramp | `RADIATION` |
| `CFL_DYN` | 0.45 | explicit transport Courant factor, $0\lt$ `CFL_DYN` $\le0.5$ | always |
| `DT_MAX` | 0.1 | global and substep timestep ceiling | always |
| `DT_OUT`, `SAVE_MAX` | $2\pi$, 100 | interval between saved frames and final saved-frame index | always |
| `RHO_VAC` | $10^{-30}$ | density below which the regularized vacuum state is used | always |
| `TPB` | 64 | threads per block for elementwise and thread-line kernels | always |

The default Schmidt numbers make the azimuthal and radial diffusivities negligible, so a default
`DIFFUSION` build diffuses effectively only in the polar direction. The derived launch constants
`N_G = N_X*N_Y*N_Z` and `NB_G`, `NB_X`, `NB_Y`, `NB_Z` are computed from these values and are not
independent parameters.

### 2.5 Compile-time features and configuration checks

The production fluid branch is specialized at compilation rather than switched at runtime:

| Selection | Effect | Requirement or interaction |
|---|---|---|
| `DIFFUSION` | add the three directional diffusion operators and their momentum closure; use the diffusion balance in the initial polar velocity | required for `N_Z > 1` |
| `DIFFUSE_CONCENTRATION` | diffuse dust-to-gas concentration; omit for density diffusion | requires `DIFFUSION` (`#error` in `fluid_kern.cuh`) |
| `CONST_NU` | use constant $\nu$ instead of constant $\alpha$ wherever viscosity is required | acts only with `DIFFUSION`; otherwise ignored |
| `VISC_FLOW` | use the viscous gas radial target velocity in the source update, initialization, and CFL rate | requires `DIFFUSION` |
| `RADIATION` | construct optical depth, add attenuated radiation pressure, and write `optdepth` frames | none |
| `HALF_DISK` | make the polar boundary at `Z_MAX` a reflecting midplane | with `N_Z > 1`, requires `Z_MAX` $=\pi/2$; no effect when `N_Z == 1` |
| `FLUID_SWEEP=thread` | assign one GPU thread to each complete directional line | Makefile default on CUDA |
| `FLUID_SWEEP=block` | assign one cooperative GPU block to each directional line (defines `FLUID_BLOCK_SWEEP`) | Makefile default on ROCm |

Transport and the local source update have no feature flag and are always compiled. The two sweep
choices implement the same discrete operators and differ only in work decomposition and temporary
storage (Section 9). `FLUID_SWEEP` is a Makefile variable; the other selections are preprocessor
definitions added to `GPU_FLAGS` in the model's `flags.mk`.

The headers reject unsupported configurations at compile time:

- `N_X > 1` and `N_Y > 1`; `N_Z >= 1`
- `X_MAX > X_MIN` and $0\lt$ `Y_MIN` $\lt$ `Y_MAX`
- `Z_MIN > 0` and `Z_MAX` $\lt\pi$
- `N_Z == 1` requires `Z_MIN == Z_MAX` $=\pi/2$; `N_Z > 1` requires `Z_MAX > Z_MIN`
- with `N_Z > 1`, a full disk must strictly contain the midplane, `Z_MIN` $\lt\pi/2\lt$ `Z_MAX`,
  while `HALF_DISK` requires `Z_MAX` $=\pi/2$
- `N_Z > 1` requires `DIFFUSION`
- $0\lt$ `CFL_DYN` $\le0.5$
- `VISC_FLOW` requires `DIFFUSION` (a preprocessor `#error` in `inc/fluid/fluid_kern.cuh`)

There is no upper bound on `X_MAX - X_MIN`; a range other than a divisor of $2\pi$ defines a
periodic domain with no physical counterpart.

## 3. Prescribed disk and initialization

### 3.1 Gas model

The analytic gas surface density and aspect ratio are

$$
\Sigma_g(R)=\Sigma_0\left(\frac{R}{R_0}\right)^p,
\qquad
h_g(R)=h_0\left(\frac{R}{R_0}\right)^{(q+1)/2}.
$$

Here $p=d\ln\Sigma_g/d\ln R$ and $q=d\ln T/d\ln R$. Vertical isothermality gives

$$
c_s\propto T^{1/2}\propto R^{q/2},
\qquad
h_g=\frac{c_s}{v_K}\propto R^{(q+1)/2}.
$$

The vertically isothermal point-mass hydrostatic stratification is evaluated exactly as

$$
\frac{\rho_g(R,Z)}{\rho_g(R,0)}
=\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g^2}
\right].
$$

The gas volume density used by the code is

$$
\rho_g(R,Z)
=\frac{\Sigma_g(R)}{\sqrt{2\pi}H_g(R)}
\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g(R)^2}
\right],
\qquad H_g=h_gR.
$$

Thus $\Sigma_g/(\sqrt{2\pi}H_g)$ fixes the reference midplane density; because the vertical
profile uses the exact point-mass potential rather than a Gaussian approximation, $\Sigma_g$ is
not recovered by exactly integrating this expression over $Z$.

For a finite resolved domain whose intersection at cylindrical radius $R$ consists of vertical
intervals $[Z_{k,a},Z_{k,b}]$, the column implied by the prescribed volume profile is

$$
\Sigma_{g,\mathcal D}(R)=\Sigma_g(R)\,\mathcal C_g(R),
$$

$$
\mathcal C_g(R)=\frac{1}{\sqrt{2\pi}H_g(R)}
\sum_k\int_{Z_{k,a}}^{Z_{k,b}}
\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g(R)^2}
\right]dZ.
$$

Thus $\mathcal C_g$ depends on the finite spherical radial and polar domain and need not equal one.
In a vertically integrated model, $\Sigma_g$ is used directly and this 3D containment factor is not
introduced.

For comparison, on an untruncated vertical line let $\zeta=Z/H_g$. The thin-disk expansion is

$$
\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g^2}
\right]
=e^{-\zeta^2/2}
\left[1+\frac{3}{8}h_g^2\zeta^4+O(h_g^4)\right],
$$

and the Gaussian fourth moment gives

$$
\mathcal C_{g,\infty}=1+\frac{9}{8}h_g^2+O(h_g^4).
$$

Finite spherical boundaries replace this asymptotic normalization by the exact containment factor
above and can be especially important near a radial or polar edge.

The remaining local disk scales are

$$
\Omega_K(R)=\sqrt{\frac{GM_\star}{R^3}},
\qquad
v_K=R\Omega_K,
\qquad
c_s=h_gR\Omega_K.
$$

At the midplane, the usual radial pressure-support parameter for the vertically isothermal gas
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

At fixed cylindrical height, differentiating the exact hydrostatic profile gives

$$
h_g^2\frac{\partial\ln P_g}{\partial\ln R}
=Ah_g^2+q(1-u)+(1-u^3).
$$

The cylindrical component of stellar gravity is $u^3$ times its midplane value. Radial force
balance therefore becomes

$$
\frac{v_{\phi,g}^2}{v_K^2}
=u^3+h_g^2\frac{\partial\ln P_g}{\partial\ln R}
=1+Ah_g^2+q(1-u)
=1-2\eta(R,Z),
$$

with

$$
\eta(R,Z)=-\frac12\left[Ah_g^2+q(1-u)\right].
$$

Thus the $(1-u^3)$ pressure term exactly cancels the off-midplane weakening of cylindrical
gravity. At $Z=0$, $u=1$ and this effective parameter reduces to the ordinary midplane
pressure-support definition above. The gas azimuthal target used by initialization and by the
source update is

$$
v_{\phi,g}=v_K\sqrt{\max(1-2\eta,0)},
\qquad
\ell_{\phi,g}=Rv_{\phi,g}.
$$

When `DIFFUSION` is enabled, the turbulent viscosity uses the $\alpha$ prescription of
[Shakura & Sunyaev (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S). It is
either

$$
\nu(R)=\alpha h_g^2R^2\Omega_K
$$

for constant `ALPHA`, or $\nu(R)=\mathrm{NU}$ for constant `NU`. In the latter case the local
equivalent turbulent parameter is

$$
\alpha(R)=\frac{\nu}{h_g^2R^2\Omega_K}.
$$

Without `DIFFUSION` no viscosity is defined, and `VISC_FLOW` is rejected.

The monodisperse Stokes number assumes the linear, subsonic Epstein-drag scaling
$t_s\propto(\rho_gc_s)^{-1}$ of
[Epstein (1924)](https://doi.org/10.1103/PhysRev.23.710), and therefore follows

$$
\mathrm{St}(R,Z)
=\mathrm{St}_0
\left(\frac{R}{R_0}\right)^{-p}
\left[\frac{\rho_g(R,Z)}{\rho_g(R,0)}\right]^{-1}.
$$

The dimensional stopping time used by the source solver is

$$
t_s(R,Z)=\frac{\mathrm{St}(R,Z)}{\Omega_K(R)}.
$$

### 3.2 Dust density profile

The pre-convolution dust surface profile is

$$
\Sigma_d(R)=Z_{\rm metal}\Sigma_g(R).
$$

The initialized edge taper is the Gaussian convolution

$$
\Sigma_{d,\rm conv}(R)
=\int_{Y_{\min}+2L_s}^{Y_{\max}-2L_s}
\Sigma_d(R')
\frac{\exp[-(R-R')^2/(2\sigma_s^2)]}{\sqrt{2\pi}\sigma_s}\,dR',
\qquad
L_s=0.05R_0,\qquad \sigma_s=\frac{L_s}{2}.
$$

The convolution is evaluated on a temporary uniform cylindrical-radius axis. In 3D its lower bound
is the smallest cylindrical radius covered by the spherical domain,

$$
R_{\min,\mathrm{init}}
=Y_{\min}\min[\sin(Z_{\min}),\sin(Z_{\max})],
$$

rather than $Y_{\min}$. This prevents high-latitude cells with $R\lt Y_{\min}$ from being clipped
to an unrelated radial profile value. In 2D, $R_{\min,\mathrm{init}}=Y_{\min}$.

The axis has `N_Y + 1` points $R_m=R_{\min,\rm init}+m\Delta R$ with
$\Delta R=(Y_{\max}-R_{\min,\rm init})/N_Y$, used as both source and destination points. The
implemented rectangular sum is

$$
\Sigma_{d,\rm conv}(R_m)
\approx\sum_{n\in\mathcal S}
\Sigma_d(R_n)
\frac{\exp[-(R_m-R_n)^2/(2\sigma_s^2)]}{\sqrt{2\pi}\sigma_s}
\Delta R,
$$

where $\mathcal S$ contains only source points in $[Y_{\min}+2L_s,Y_{\max}-2L_s]$. Each grid cell
then takes the linear interpolation of this tabulated profile at its center cylindrical radius
$R=y_j\sin z_k$; the profile is zero outside $[R_{\min,\rm init},Y_{\max}]$.

The edge convolution is not renormalized. In 2D the convolved surface density is evolved directly
and its unresolved vertical profile is assumed to be well mixed with the gas. In 3D it is embedded
in a Gaussian dust layer using the settling–diffusion balance discussed by
[Youdin & Lithwick (2007)](https://arxiv.org/abs/0707.2975),

```math
H_d=H_g\sqrt{\frac{\alpha_{z,d}}{\mathrm{St}_{\rm mid}+\chi_c\alpha_{z,d}}},
\qquad
\alpha_{z,d}=\frac{\alpha}{\mathrm{Sc}_z(1+\mathrm{St}_{\rm mid}^2)},
\qquad
\chi_c=\left\lbrace\begin{array}{ll}
0,&\text{density diffusion},\\
1,&\text{concentration diffusion}.
\end{array}\right.
```

Specifically,

$$
\rho_d(R,Z)
=\frac{\Sigma_{d,\rm conv}(R)}{\sqrt{2\pi}H_d(R)}
\exp\left[-\frac{Z^2}{2H_d(R)^2}\right].
$$

This is the midplane, small-height settling approximation for the selected flux mode, not an
exact equilibrium of the height-dependent diffusivity. Initialization then applies

$$
\varrho_d(x_i,y_j,z_k)
\leftarrow
\varrho_d(x_i,y_j,z_k)\max(1+0.1\xi_i,0),
\qquad
\xi_i\sim\mathcal N(0,1),
$$

using one reproducible deviate for each azimuthal column, so every radial and polar cell at fixed
$x_i$ receives the same perturbation. The deviate comes from the vendor generator seeded by the
azimuthal index, so CUDA and ROCm builds initialize different, individually reproducible noise
realizations.

Because the edge convolution and azimuthal perturbation are not renormalized, the initialized mass
is the finite-volume integral of the resulting field rather than a separately imposed parameter:

```math
M_d=
\left\lbrace\begin{array}{ll}
\displaystyle\int\Sigma_d(R,\phi)R\,dR\,d\phi,&N_Z=1,\\
\displaystyle\int\rho_d(r,\theta,\phi)r^2\sin\theta\,dr\,d\theta\,d\phi,&N_Z>1.
\end{array}\right.
```

### 3.3 Initial velocity

The initial velocity is the local no-backreaction drift relative to the pressure-supported gas,
using the steady test-particle limit of
[Nakagawa, Sekiya & Hayashi (1986)](<https://doi.org/10.1016/0019-1035(86)90121-1>). With the
effective rotation-support parameter $\eta(R,Z)$ and gas azimuthal target $v_{\phi,g}$ of
Section 3.1, the initialized cylindrical drift is

$$
v_{R,d}
=\frac{v_{R,g}+2\mathrm{St}(v_{\phi,g}-v_K)}{1+\mathrm{St}^2},
\qquad
v_{\phi,d}=v_{\phi,g}-\frac{\mathrm{St}}{2}v_{R,d}.
$$

Without `VISC_FLOW`, $v_{R,g}=0$. With `VISC_FLOW`, the prescribed cylindrical gas radial velocity
follows the steady viscous expression of [Kanagawa et al. (2017)](https://arxiv.org/abs/1706.08975).
In the vertically integrated case it reduces to

```math
v_{R,g}
=-\frac{3\nu}{R}\left(g_\nu+p+\frac12\right),
\qquad
g_\nu=\frac{d\ln\nu}{d\ln R}
=\left\lbrace\begin{array}{ll}
0,&\nu=\mathrm{constant},\\
q+\frac32,&\alpha=\mathrm{constant}.
\end{array}\right.
```

For the resolved 3D expression, with $u=R/\sqrt{R^2+Z^2}$ as in Section 3.1, define

$$
S=\frac{u-1}{h_g^2},
$$

$$
g_{\rho R}=p-\frac{q+3}{2}
+\frac{u(1-u^2)}{h_g^2}-(q+1)S,
\qquad
g_{\rho Z}=-\frac{u(1-u^2)}{h_g^2}.
$$

The implemented target is

$$
v_{R,g}
=-\frac{\nu}{R}
\left[3\left(g_\nu+g_{\rho R}+\frac12\right)-q(1+g_{\rho Z})\right].
$$

In 3D, the polar primitive also includes the velocity needed to approximately balance the
initialized polar diffusive flux,

$$
v_{\theta,\mathrm{diff}}
=\frac{D_z w}{r\rho_d}\frac{\partial(\rho_d/w)}{\partial\theta},
$$

where $w=1$ for density diffusion and $w=\rho_g$ for concentration diffusion, evaluated with
one-sided boundary differences and centered interior differences.

This initial vertical closure is not the same as the swarm initialization. The fluid adds this
polar diffusive-balance velocity but does not add the swarm's terminal-settling velocity
$v_Z=-\mathrm{St}\,\Omega_KZ$. This asymmetry is intentional: the swarm evolves diffusion as a
separate stochastic positional operator and does not encode that operator in its initialized
deterministic velocity. The two branches therefore should not be assumed to begin from an
identical vertical dynamical equilibrium.

The cylindrical drift and polar balance are finally stored as

$$
v_r=v_{R,d}\sin z,
\qquad
\ell_\theta=y\left(v_{R,d}\cos z+v_{\theta,\rm diff}\right),
\qquad
\ell_\phi=Rv_{\phi,d},
$$

with $\ell_\theta=0$ when `N_Z == 1`. Conserved momenta are then built as $m_a=\varrho_du_a$, with
the vacuum reset of Section 2.2.

## 4. Governing continuum equations

### 4.1 Continuum equations and component source terms

Let $\varrho_d$ denote the evolved dust density: $\varrho_d=\Sigma_d$ in 2D and
$\varrho_d=\rho_d$ in 3D.

**Core fluid equations.** In coordinate-independent conservation form, the default density mode
advances

$$
\frac{\partial\varrho_d}{\partial t}
+\nabla\cdot(\varrho_d\boldsymbol v_d)
=\nabla\cdot(\boldsymbol D\nabla\varrho_d),
$$

and

$$
\frac{\partial(\varrho_d\boldsymbol v_d)}{\partial t}
+\nabla\cdot(\varrho_d\boldsymbol v_d\boldsymbol v_d)
=\varrho_d\left[
-\frac{GM_\star}{r^2}\boldsymbol e_r
+\boldsymbol a_{\rm rad}
-\frac{\boldsymbol v_d-\boldsymbol v_g}{t_s}
\right]
+\boldsymbol S_{m,D}.
$$

The pressure tensor is zero: the advective momentum flux is
$\varrho_d\boldsymbol v_d\boldsymbol v_d$ rather than
$\varrho_d\boldsymbol v_d\boldsymbol v_d+\boldsymbol P_d$. The term $\boldsymbol S_{m,D}$ denotes
the conservative donor-momentum flux paired with diffusive mass transport and vanishes when
`DIFFUSION` is disabled; Section 7.4 gives its discrete definition and limitations. In
concentration mode replace $\boldsymbol D\nabla\varrho_d$ by $w\boldsymbol D\nabla(\varrho_d/w)$
throughout the mass-diffusion operator, with the corresponding donor-momentum flux. Disabling
`RADIATION` sets $\boldsymbol a_{\rm rad}=0$.

The coordinate-free equations are common to both supported geometries, but their differential
operators and physical closures are not identical. Their relationship is

| Model | Density | Meaning of $\nabla\cdot$ | Additional closure |
|---|---|---|---|
| radial–azimuthal 2D | $\varrho_d=\Sigma_d(R,\phi)$ | divergence in the disk plane | vertical integration and a well-mixed unresolved column |
| full 3D | $\varrho_d=\rho_d(r,\theta,\phi)$ | three-dimensional spherical divergence | explicitly resolved polar structure |

The two continuity equations are therefore

$$
\frac{\partial\Sigma_d}{\partial t}
+\frac{1}{R}\frac{\partial}{\partial R}(R\Sigma_dv_R)
+\frac{1}{R}\frac{\partial}{\partial\phi}(\Sigma_dv_\phi)
=\mathcal D_{2D}[\Sigma_d],
$$

and

$$
\frac{\partial\rho_d}{\partial t}
+\frac{1}{r^2}\frac{\partial}{\partial r}(r^2\rho_dv_r)
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}
 (\sin\theta\rho_dv_\theta)
+\frac{1}{r\sin\theta}\frac{\partial}{\partial\phi}(\rho_dv_\phi)
=\mathcal D_{3D}[\rho_d],
$$

where Section 7 expands $\mathcal D_{2D}$ and $\mathcal D_{3D}$. There is no separate production
1D fluid model. Writing every momentum component again for 2D and 3D would duplicate the same
covariant conservation law; only the metric terms and inactive components differ.

The code evaluates the common momentum equation in spherical finite-volume coordinates using
$(\ell_\phi,v_r,\ell_\theta)$. The centrifugal and polar connection terms introduced by that basis
are included in the source operator. At a fixed cell, the source part of the stored variables is

$$
\frac{d\ell_\phi}{dt}
=-\frac{\ell_\phi-\ell_{\phi,g}}{t_s},
$$

$$
\frac{d\ell_\theta}{dt}
=-\frac{\ell_\theta-\ell_{\theta,g}}{t_s}
+\frac{\ell_\phi^2\cos\theta}{R^2\sin\theta},
$$

$$
\frac{dv_r}{dt}
=-\frac{v_r-v_{r,g}}{t_s}
-(1-\beta)\frac{GM_\star}{r^2}
+\frac{\ell_\phi^2}{R^2r}
+\frac{\ell_\theta^2}{r^3},
\qquad
t_s=\frac{\mathrm{St}}{\Omega_K}.
$$

Here $v_{r,g}=v_{R,g}\sin\theta$ and $\ell_{\theta,g}=rv_{R,g}\cos\theta$ are the spherical
projections of the gas radial target, which vanish without `VISC_FLOW`. The polar torque and
$\ell_\theta$ are absent when `N_Z == 1`.

### 4.2 Discrete conservation and physical scope

For cell measure $V_{ijk}$, the discrete dust mass and stored momenta are

$$
M_d^h=\sum_{ijk}\varrho_{d,ijk}V_{ijk},
\qquad
Q_a^h=\sum_{ijk}m_{a,ijk}V_{ijk},
\quad
a\in\lbrace x,y,z\rbrace.
$$

Every internal transport or diffusive face contributes equal and opposite fluxes to its two
adjacent cells. Therefore, for periodic or zero-flux boundaries,

$$
\Delta M_d^h=0,
\qquad
\Delta Q_a^h=0
$$

up to floating-point summation error during those operators. Outflow boundaries change these sums
by the explicitly computed boundary flux. The source operator preserves density exactly but changes
the stored momenta through drag, gravity, radiation, and spherical geometry. The vacuum reset of
Section 2.2 and the transport vacuum fallback of Section 5.4 can change the stored momenta of
near-empty cells.

This conservation statement concerns the variables actually stored. In particular,
$Q_x^h=\int\rho_d\ell_\phi\,dV$ is axial angular momentum, while $Q_y^h$ is radial linear
momentum. Their physical conservation can be broken by the prescribed external gas and stellar
forces even when their numerical transport is conservative.

## 5. Conservative pressureless transport

### 5.1 Directional finite-volume update

Density and all three momenta are transported conservatively, one direction at a time. Each
directional operator uses:

1. geometry-aware PPM reconstruction (Section 5.2)
2. a pressureless HLL interface flux (Section 5.3)
3. a first-order cell-centered HLL flux as a robust invariant-domain base
4. one conservative face coefficient that limits the high-minus-low correction (Section 5.4)

For a directional finite-volume coordinate with cell measure $V_i$ and face factor $A_{i+1/2}$,
every forward-Euler transport evaluation has the conservative form

$$
\boldsymbol U_i^{\rm FE}
=\boldsymbol U_i^n
-\frac{\Delta t}{V_i}
\left(A_{i+1/2}\boldsymbol F_{i+1/2}
-A_{i-1/2}\boldsymbol F_{i-1/2}\right).
$$

For azimuth, $V_i=\Delta x$ and $A=1$. For radial transport, $V_i=\Delta V_{y,i}$ and
$A_{i+1/2}=y_{i+1/2}^{d-1}$. For polar transport through radial cell $j$, the exact finite-volume
update is

$$
\boldsymbol U_{j,k}^{\rm FE}
=\boldsymbol U_{j,k}^n
-\Delta t\frac{\Delta A_{z,j}}{\Delta V_{y,j}\Delta V_{z,k}}
\left(\sin z_{k+1/2}\boldsymbol F_{k+1/2}
-\sin z_{k-1/2}\boldsymbol F_{k-1/2}\right).
$$

Here $\Delta A_{z,j}=(y_{j+1/2}^2-y_{j-1/2}^2)/2$ and
$\Delta V_{y,j}=(y_{j+1/2}^3-y_{j-1/2}^3)/3$. Their ratio is the exact radial
face-area-to-volume factor; it is not approximated by $1/y_j$.

The normal transport speeds are the residual angular speed $(\ell_\phi-\ell_{\rm frame})/R^2$ in
azimuth (Section 5.5), $v_r$ radially, and $\ell_\theta/r$ in the polar direction. The reconstructed
quantities are density and the three primitives $(\ell_\phi,v_r,\ell_\theta)$.

### 5.2 PPM reconstruction

The reconstruction is the piecewise parabolic method of
[Colella & Woodward (1984)](<https://doi.org/10.1016/0021-9991(84)90143-8>). The uniform azimuthal
mesh uses its bounded four-cell face value. On the logarithmic radial mesh and spherical polar
mesh, face weights are precomputed on the host from exact cubic moment constraints in the
finite-volume coordinates

$$
s_y=\frac{y^d}{d},
\qquad
s_z=-\cos z.
$$

On the uniform azimuthal mesh, the unlimited four-cell face estimate is

$$
q_{i+1/2}^{*}
=\frac{7(q_i+q_{i+1})-(q_{i-1}+q_{i+2})}{12}.
$$

On nonuniform radial and polar meshes, the four weights $w_m$ solve cubic moment-exactness
conditions in $s_y$ or $s_z$ and give

$$
q_{i+1/2}^{*}=\sum_{m=-1}^{2}w_m\bar q_{i+m}.
$$

If $s_f$ is the target face, $L$ is a local stencil scale, and $t=(s-s_f)/L$, those weights are
defined by

$$
\sum_{m=-1}^{2}w_m
\frac{1}{t_{m,+}-t_{m,-}}
\int_{t_{m,-}}^{t_{m,+}}t^n\,dt
=\delta_{n0},
\qquad n=0,1,2,3.
$$

The weighted cell averages therefore reproduce the value at $t=0$ for every polynomial through
cubic degree.

Boundary-adjacent internal faces use a two-cell linear interpolation where a complete four-cell
stencil is unavailable. If $s_L$ and $s_R$ are the arithmetic centers of the adjacent cells in the
appropriate volume coordinate and $s_f$ is their common face, then

$$
\omega=\frac{s_f-s_L}{s_R-s_L},
\qquad
q_f=(1-\omega)\bar q_L+\omega\bar q_R.
$$

The physical boundary faces take the value of their single adjacent cell. Every face value, on
every mesh, is clipped to $[\min(q_i,q_{i+1}),\max(q_i,q_{i+1})]$.

Inside one cell, PPM represents the reconstructed parabola by left and right faces $q_L,q_R$ and

$$
\Delta q=q_R-q_L,
\qquad
q_6=6\bar q-3(q_L+q_R).
$$

The monotonicity correction is

$$
(q_R-\bar q)(\bar q-q_L)\le0
\quad\Longrightarrow\quad
q_L=q_R=\bar q.
$$

Otherwise,

$$
\Delta q\,q_6>(\Delta q)^2
\quad\Longrightarrow\quad
q_L=3\bar q-2q_R,
$$

$$
-\Delta q\,q_6>(\Delta q)^2
\quad\Longrightarrow\quad
q_R=3\bar q-2q_L.
$$

The code recomputes $\Delta q$ and $q_6$ after either correction. These conditions remove a newly
created internal extremum while retaining the parabolic profile where it is locally monotone.

After these corrections, integrating a fraction $c\in[0,1]$ next to the right or left face gives
the time-averaged upwind states

$$
q_R^{\rm tr}=q_R-\frac{c}{2}
\left[\Delta q-\left(1-\frac{2c}{3}\right)q_6\right],
$$

$$
q_L^{\rm tr}=q_L+\frac{c}{2}
\left[\Delta q+\left(1-\frac{2c}{3}\right)q_6\right].
$$

Azimuthal FARGO transport uses the nonzero tracing fraction $c$ because one conservative orbital
advection update spans its requested substep. Radial and polar transport pass $c=0$ to these
formulas and obtain their temporal order from the three SSPRK flux evaluations in Section 5.6.
Applying both characteristic tracing and SSPRK time centering in those directions would duplicate
the time evolution. Reconstructed interface densities are clipped at zero before the flux is
evaluated.

### 5.3 Pressureless wave structure and HLL flux

In one spatial direction, the density and normal momentum subsystem is

```math
\frac{\partial}{\partial t}
\begin{pmatrix}\rho\\ \rho u\end{pmatrix}
+\frac{\partial}{\partial x}
\begin{pmatrix}\rho u\\ \rho u^2\end{pmatrix}=0.
```

Its flux Jacobian has the repeated eigenvalue

$$
\lambda_1=\lambda_2=u,
$$

so pressureless Euler is only weakly hyperbolic. Separating states can create vacuum, while
converging characteristics can form a singular concentration in the ideal pressureless Riemann
problem. The finite-volume code does not represent an exact delta shock. HLL replaces the local
interaction by a bounded two-wave numerical fan; this regularization is robust but introduces
numerical diffusion. The invariant-domain correction of Section 5.4 prevents that regularized flux
from creating negative density or unbounded transported primitive ratios.

The pressureless flux uses the two-wave construction of
[Harten, Lax & van Leer (1983)](https://doi.org/10.1137/1025002) and retains its proper
left-going, right-going, and two-wave branches. For a complete pressureless state $\boldsymbol U$
transported at normal speed $a$, the physical flux is $\boldsymbol F=a\boldsymbol U$. With

$$
s_L=\min(a_L,a_R),
\qquad
s_R=\max(a_L,a_R),
$$

the two-wave HLL branch is

$$
\boldsymbol F_{\rm HLL}
=\frac{s_R\boldsymbol F_L-s_L\boldsymbol F_R
+s_Ls_R(\boldsymbol U_R-\boldsymbol U_L)}{s_R-s_L}.
$$

If both interface speeds are nonnegative, the code uses $\boldsymbol F_L$; if both are
nonpositive, it uses $\boldsymbol F_R$. The high-order flux $\boldsymbol F^H$ evaluates this
formula with the PPM interface states and their normal speeds; the low-order flux
$\boldsymbol F^L$ evaluates it with the two adjacent cell-centered states.

### 5.4 Invariant-domain correction

The limiter enforces nonnegative density and local bounds on every momentum-to-density ratio. The
same limited face flux enters neighboring cells with opposite signs, so the normal update remains
conservative.

Each forward-Euler evaluation first applies the complete low-order update with $\boldsymbol F^L$.
Any cell whose low-order density is negative is converted to an exact vacuum by setting its density
and all three momentum components to zero. This fallback should remain inactive when the CFL and
invariant-domain assumptions hold; its purpose is to prevent a residual negative density from
producing an undefined vacuum velocity, not to replace the conservative limiter.

Let $\delta\boldsymbol F=\boldsymbol F^H-\boldsymbol F^L$ be the PPM antidiffusive correction at
an internal face. Proceeding face by face in increasing index order, each face applies

$$
\boldsymbol U_i\leftarrow
\boldsymbol U_i-\lambda_i\alpha\,\delta\boldsymbol F,
\qquad
\boldsymbol U_{i+1}\leftarrow
\boldsymbol U_{i+1}+\lambda_{i+1}\alpha\,\delta\boldsymbol F
$$

to the current state, where $\lambda_i=\Delta t A_{i+1/2}/V_i$ and one shared $0\le\alpha\le1$ is
the largest accepted scale satisfying

$$
\varrho_d\ge0,
\qquad
u_{a,\min}\varrho_d\le m_a\le u_{a,\max}\varrho_d
$$

in both adjacent cells for $u_a\in\lbrace\ell_\phi,v_r,\ell_\theta\rbrace$. The bounds
$u_{a,\min}$ and $u_{a,\max}$ are the minimum and maximum of $u_a$ over the cell and its immediate
neighbors along the line (periodic in azimuth, truncated at radial and polar edges), taken from the
primitive state before the update and widened by $10^{-12}\max(1,|u_{a,\min}|,|u_{a,\max}|)$.
Boundary faces carry no correction.

Every listed condition is an affine inequality $g(\boldsymbol U)\ge0$. If the proposed face
correction changes it by $\Delta g$, the admissible coefficient is reduced only when $\Delta g\lt0$:

$$
\alpha\leftarrow
\min\left[
\alpha,
(1-10^{-12})\frac{g(\boldsymbol U)}{-\Delta g}
\right],
$$

and $\alpha=0$ when $g(\boldsymbol U)\le0$ and $\Delta g\lt0$. The final face coefficient is the
minimum over density and the lower and upper bounds of all three primitive ratios in both adjacent
cells, clipped to $[0,1]$. Because a face sees the corrections already applied at earlier faces, the
result depends on the face order; both GPU implementations use the same serial order.

### 5.5 FARGO azimuthal transport

Azimuthal transport uses the FARGO orbital-advection decomposition of
[Masset (2000)](https://arxiv.org/abs/astro-ph/9910390). For each ring, the arithmetic mean
$\ell_\phi$ defines a nearest-integer periodic shift. PPM transports the residual, including the
fractional part of the ring-mean displacement. The CFL calculation bounds each cell's displacement
relative to the ring mean by `CFL_DYN`. The nearest-integer shift frame can differ from that mean by
at most half a cell per step, so the actual PPM tracing fraction is bounded by
`CFL_DYN + 0.5 <= 1` rather than by `CFL_DYN` alone.

For a ring of $N_X$ cells at fixed $(r,\theta)$,

$$
\bar\ell_\phi=\frac{1}{N_X}\sum_{i=0}^{N_X-1}\ell_{\phi,i},
\qquad
\delta n=\frac{\bar\ell_\phi\Delta t}{R^2\Delta x},
\qquad
n=\mathrm{round}(\delta n).
$$

The conserved arrays are first shifted periodically by $n$ cells. The angular speed of that integer
frame and the residual speed are

$$
\Omega_{\rm frame}=\frac{n\Delta x}{\Delta t},
\qquad
\ell_{\rm frame}=R^2\Omega_{\rm frame},
\qquad
\Omega_i^{\rm res}=\frac{\ell_{\phi,i}-\ell_{\rm frame}}{R^2}.
$$

PPM traces with

$$
c_i=\frac{|\Omega_i^{\rm res}|\Delta t}{\Delta x}
$$

and the HLL normal speeds are the reconstructed residual angular speeds. This decomposition is
algebraically a periodic integer translation followed by conservative residual transport, applied
as one forward-Euler update with the correction of Section 5.4.

The allowed endpoint `CFL_DYN = 0.5` can make $c_i=1$ exactly when the half-cell frame offset and
the bounded residual displacement align. This remains within the PPM tracing domain, but it leaves
no roundoff margin at the one-cell limit. The production default is therefore `CFL_DYN = 0.45`;
this safety margin is not a correction to the method, and the limiting value 0.5 remains a
supported edge case.

### 5.6 Radial and polar time integration and boundaries

The radial and polar method-of-lines operators use the three-stage TVD Runge–Kutta construction of
[Shu & Osher (1988)](https://doi.org/10.1016/0021-9991(88)90177-5), commonly denoted SSPRK(3,3). PPM
provides spatial reconstruction; SSPRK supplies temporal integration. If $L(\boldsymbol U)$ is one
limited spatial flux-divergence evaluation, the stages are

$$
\boldsymbol U^{(1)}
=\boldsymbol U^n+\Delta tL(\boldsymbol U^n),
$$

$$
\boldsymbol U^{(2)}
=\frac34\boldsymbol U^n
+\frac14\left[\boldsymbol U^{(1)}+\Delta tL(\boldsymbol U^{(1)})\right],
$$

$$
\boldsymbol U^{n+1}
=\frac13\boldsymbol U^n
+\frac23\left[\boldsymbol U^{(2)}+\Delta tL(\boldsymbol U^{(2)})\right].
$$

Each bracket is a complete low-order update with invariant-domain correction; the convex
combinations preserve nonnegative density and the primitive bounds of their components.

The radial boundaries are outflow-only. A full polar disk is outflow-only at both polar edges;
`HALF_DISK` reflects at the midplane `Z_MAX` and remains outflow-only at `Z_MIN`. At an
outflow-only boundary with outward normal speed $a_n$ of the adjacent cell-centered state, the
boundary mass flux is

```math
F_\varrho=
\left\lbrace\begin{array}{ll}
a_n\max(\varrho_d,0),&a_n\text{ points out of the domain},\\
0,&a_n\text{ points into the domain},
\end{array}\right.
\qquad
F_{m_a}=F_\varrho u_a.
```

This boundary flux is first order: it uses the adjacent cell average rather than a reconstructed
face state.

At a reflecting `HALF_DISK` midplane, all normal flux components are set to zero. Here
“reflecting” means an impermeable finite-volume symmetry wall; the fluid operator does not move a
parcel through the face and then reverse its polar momentum. Reflection-symmetric continuum states
instead have zero normal velocity at the midplane, which is the interpretation exercised by the
validation suite.

PPM is formally high order on smooth fields, but Strang composition limits the multidimensional
transport claim to second order. SSPRK(3,3) does not make the full scheme third order in time. The
complete solver's accuracy also depends on boundary fluxes, limiter activation, and the old-donor
diffusion-momentum closure described in Section 7.4.

## 6. Drag, gravity, geometry, and radiation

### 6.1 Exponential drag-weighted source update

At a fixed cell, the dust primitive satisfies a drag-plus-force equation of the form

$$
\frac{d\boldsymbol u}{dt}
=-\frac{\boldsymbol u-\boldsymbol u_g}{t_s}
+\boldsymbol F(\boldsymbol u,t).
$$

The source kernel evaluates the exponential drag relaxation exactly for frozen $t_s$ and uses
drag-weighted old/new force coefficients. A cancellation-safe series is used for small
$\Delta t/t_s$. Forces include central gravity, reduced by radiation when enabled, radial
centrifugal acceleration, and polar geometric torque. The update is sequential in $\ell_\phi$,
$\ell_\theta$, and $v_r$, with force re-evaluation after the angular updates. Cells below
$\rho_{\rm vac}$ are set to the vacuum state of Section 2.2 and skip the update.

More precisely, set

$$
\tau=\frac{\Delta t}{t_s},
\qquad
E=e^{-\tau},
\qquad
Q=1-E.
$$

If a component obeys

$$
\frac{du}{dt}=-\frac{u-u_g}{t_s}+F(t),
$$

and the nondrag force is interpolated linearly from $F^n$ to $F^{n+1}$ during the step, its
implemented update is

$$
u^{n+1}=Eu^n+Qu_g+w_nF^n+w_{n+1}F^{n+1},
$$

with

$$
w_{n+1}=t_s\frac{\tau-Q}{\tau},
\qquad
w_n=t_sQ-w_{n+1}.
$$

These weights recover trapezoidal force integration as $\tau\rightarrow0$ and suppress an old
force exponentially when drag is stiff. Direct evaluation subtracts nearly equal numbers for
small $\tau$, so the code instead uses

$$
w_n=\Delta t\left(
\frac12-\frac{\tau}{3}+\frac{\tau^2}{8}-\frac{\tau^3}{30}
\right)+O(\tau^4),
$$

$$
w_{n+1}=\Delta t\left(
\frac12-\frac{\tau}{6}+\frac{\tau^2}{24}-\frac{\tau^3}{120}
\right)+O(\tau^4)
$$

when $\tau\lt10^{-4}$; $Q$ itself is evaluated as $-\mathrm{expm1}(-\tau)$. Azimuthal angular
momentum has no nondrag force, so

$$
\ell_\phi^{n+1}=E\ell_\phi^n+Q\ell_{\phi,g}.
$$

The code then evaluates the old and new polar torques using $\ell_\phi^n$ and $\ell_\phi^{n+1}$,
advances $\ell_\theta$, reevaluates the centrifugal force with both updated angular momenta, and
finally advances $v_r$. This ordering is what gives the symbols $F^n$ and $F^{n+1}$ their precise
meaning here; they are sequential endpoint approximations inside one cell-local source solve, not
forces from two separate hydrodynamic states.

### 6.2 Radiation and optical depth

The radiation-pressure prescription is the radial geometric-optics term reviewed by
[Burns, Lamy & Soter (1979)](https://doi.org/10.1016/0019-1035(79)90050-2), with the code adding
time-dependent startup and optical attenuation:

$$
\beta(t,\tau)
=\beta_0 f_\beta(t)e^{-\tau},
\qquad
f_\beta=s^2(3-2s),
\qquad
s=\mathrm{clip}(t/T_\beta,0,1).
$$

The ramp is evaluated at the midpoint time $t^n+\Delta t/2$ of the global step, and $f_\beta=1$
when $T_\beta\le0$. The corresponding radial acceleration is

$$
\boldsymbol a_{\rm rad}
=\beta(t,\tau)\frac{GM_\star}{r^2}\boldsymbol e_r,
$$

so gravity and radiation combine to

$$
\boldsymbol a_{\rm grav+rad}
=-(1-\beta)\frac{GM_\star}{r^2}\boldsymbol e_r.
$$

In continuum notation the optical depth from the inner radial boundary is

$$
\tau(\phi,r,\theta)
=\int_{Y_{\min}}^r\kappa_0\rho_{\rm ext}(\phi,r',\theta)\,dr'.
$$

The local radial optical-depth increment in cell $(i,j,k)$ is

$$
\Delta\tau_{ijk}=\kappa_0\rho_{{\rm ext},ijk}\Delta r_j,
\qquad
\Delta r_j=y_{j-1/2}(a_y-1),
$$

where

```math
\rho_{\rm ext}=
\left\lbrace\begin{array}{ll}
\Sigma_d/(\sqrt{2\pi}H_g),&N_Z=1,\\
\rho_d,&N_Z>1.
\end{array}\right.
```

In 2D, this well-mixed closure converts surface density to midplane extinction density using the
gas scale height at the cell center, without a Stokes-dependent dust scale height. The stored
outer-face value is the inclusive radial prefix sum

$$
\tau_{i,j+1/2,k}=\sum_{m=0}^{j}\Delta\tau_{imk},
\qquad
\tau_{i,-1/2,k}=0.
$$

Cumulative values therefore live at radial outer faces and are interpolated to logarithmic cell
centers with

$$
\tau_c=\tau_{\rm in}+\frac{\tau_{\rm out}-\tau_{\rm in}}{\sqrt{a_y}+1},
$$

which is exact when optical depth is linear in physical radius within the logarithmic cell. The
source cell sees attenuation $e^{-\tau_c}$ constructed from the inner and outer face values of its
own radial cell. Optical depth is recomputed from the density at the midpoint of the symmetric
global step, so radiation uses the centered mass distribution rather than the state at only the
beginning or end of the step.

## 7. Diffusion and momentum consistency

### 7.1 Target diffusion equation and coordinate basis

The default equation is

$$
\frac{\partial\rho_d}{\partial t}
=\nabla\cdot(\boldsymbol D\nabla\rho_d),
$$

with $\rho_d$ interpreted as $\Sigma_d$ in 2D. Defining `DIFFUSE_CONCENTRATION` instead selects

$$
\frac{\partial\rho_d}{\partial t}
=\nabla\cdot\left[\rho_g\boldsymbol D\nabla\left(\frac{\rho_d}{\rho_g}\right)\right],
$$

whose discretization is described in Section 7.3. The gas weight is the analytic surface density
in 2D and volume density in 3D.

In a vertically integrated radial–azimuthal disk, the scalar density operator is

$$
\frac{\partial\Sigma_d}{\partial t}
=\frac{1}{R^2}\frac{\partial}{\partial\phi}
\left(D_x\frac{\partial\Sigma_d}{\partial\phi}\right)
+\frac{1}{R}\frac{\partial}{\partial R}
\left(RD_y\frac{\partial\Sigma_d}{\partial R}\right).
$$

In 3D spherical coordinates it is

$$
\frac{\partial\rho_d}{\partial t}
=\frac{1}{r^2\sin^2\theta}\frac{\partial}{\partial\phi}
\left(D_x\frac{\partial\rho_d}{\partial\phi}\right)
+\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2D_y\frac{\partial\rho_d}{\partial r}\right)
+\frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}
\left(\sin\theta D_z\frac{\partial\rho_d}{\partial\theta}\right).
$$

The fluid diffusion tensor is diagonal in the spherical $x/y/z$ basis:

$$
D_x=\frac{\nu}{\mathrm{Sc}_x(1+\mathrm{St}^2)},
\qquad
D_y=\frac{\nu}{\mathrm{Sc}_y(1+\mathrm{St}^2)},
\qquad
D_z=\frac{\nu}{\mathrm{Sc}_z(1+\mathrm{St}^2)}.
$$

Both modes use the local Stokes number. The Schmidt numbers describe gas mixing before Stokes
suppression. The disk profiles used to evaluate $\nu$ depend on cylindrical $R$; this does not
change the coordinate basis of the differential operator.

### 7.2 Crank–Nicolson finite-volume solve

Each direction uses the second-order implicit trapezoidal method of
[Crank & Nicolson (1947)](https://doi.org/10.1017/S0305004100023197) in finite-volume form. The
periodic azimuthal system is reduced with the rank-one inverse update of
[Sherman & Morrison (1950)](https://doi.org/10.1214/aoms/1177729893); radial and polar systems use
zero diffusive flux at their physical boundaries. Crank–Nicolson is linearly stable but not
unconditionally positive, as emphasized for diffusion discretizations by
[Higueras & Roldán (2023)](https://arxiv.org/abs/2301.01066), so each kernel subcycles according
to `POS_LIMIT`.

For a generic one-dimensional finite-volume line, define the outward diffusive face flux

$$
\mathcal F_{i+1/2}=-A_{i+1/2}D_{i+1/2}
\frac{\rho_{i+1}-\rho_i}{\delta l_{i+1/2}}
$$

and the discrete operator

$$
(L\rho)_i=-\frac{\mathcal F_{i+1/2}-\mathcal F_{i-1/2}}{V_i}.
$$

One Crank–Nicolson substep of length $\delta t$ is

$$
\left(I-\frac{\delta t}{2}L\right)\rho^{n+1}
=\left(I+\frac{\delta t}{2}L\right)\rho^n.
$$

Writing

$$
c_i^-=\frac{\delta t}{2}
\frac{A_{i-1/2}D_{i-1/2}}{V_i\delta l_{i-1/2}},
\qquad
c_i^+=\frac{\delta t}{2}
\frac{A_{i+1/2}D_{i+1/2}}{V_i\delta l_{i+1/2}},
$$

the directional geometry substituted into these coefficients is

| Direction | Crank–Nicolson face coupling |
|---|---|
| azimuthal | $c^-_i=c^+_i=\delta tD_x/[2(R\Delta x)^2]$, with $D_x$ at the ring center |
| radial | $A_{i+1/2}=y_{i+1/2}^{d-1}$, $V_i=\Delta V_{y,i}$, $D$ at the radial face, and $\delta l$ the distance between neighboring radial centers |
| polar | $A_{k+1/2}=\sin z_{k+1/2}$, $V_k=y\Delta V_{z,k}$, $D$ at the polar face, and $\delta l=y\Delta z$ |

The row solved by the radial and polar Thomas algorithm is

$$
-c_i^-\rho_{i-1}^{n+1}
+(1+c_i^-+c_i^+)\rho_i^{n+1}
-c_i^+\rho_{i+1}^{n+1}
=c_i^-\rho_{i-1}^{n}
+(1-c_i^- - c_i^+)\rho_i^{n}
+c_i^+\rho_{i+1}^{n}.
$$

At a physical boundary the missing coefficient is set to zero, which is the discrete
$\mathcal F=0$ condition. In azimuth,

$$
c=\frac{\delta t}{2}\frac{D_x}{(R\Delta x)^2},
$$

and the same equation is cyclic, with the first and last rows coupled. Sherman–Morrison reduces
that cyclic system to two ordinary tridiagonal solves without changing the matrix being solved.
For the rank-one representation $A+\boldsymbol u\boldsymbol v^T$, the identity used is

$$
(A+\boldsymbol u\boldsymbol v^T)^{-1}\boldsymbol b
=A^{-1}\boldsymbol b
-\frac{A^{-1}\boldsymbol u\,\boldsymbol v^TA^{-1}\boldsymbol b}
{1+\boldsymbol v^TA^{-1}\boldsymbol u}.
$$

The two tridiagonal solves evaluate $A^{-1}\boldsymbol b$ and $A^{-1}\boldsymbol u$; the remaining
factor is a scalar correction.

For a requested interval $\Delta t$, the kernels first form the full-step coefficients and choose

$$
N_{\rm sub}
=\max\left(1,
\left\lceil\frac{\max_i(c_i^-+c_i^+)}{\mathrm{POS\_LIMIT}}\right\rceil
\right),
\qquad
\delta t=\frac{\Delta t}{N_{\rm sub}},
$$

separately for every line. For the periodic azimuthal line this is equivalently based on
$2c=\Delta tD_x/(R\Delta x)^2$. The subdivision controls the sign of the explicit Crank–Nicolson
right-hand side; unconditional linear stability by itself would not prevent negative density
oscillations. Each substep solves density, reconstructs and limits the face mass fluxes, and
transports momentum as described in Section 7.4 before the next substep begins.

### 7.3 Concentration diffusion

In concentration mode the radial and polar solves use $q_i=\rho_{d,i}/w_i$ as the unknown, with
$w_i=\rho_{g,i}$ (arbitrary common normalization):

$$
\mathcal F_{i+1/2}=-A_{i+1/2}D_{i+1/2}w_{i+1/2}
\frac{q_{i+1}-q_i}{\delta l_{i+1/2}},\qquad
c_i^\pm=\frac{\delta t}{2}\frac{A_{i\pm1/2}D_{i\pm1/2}w_{i\pm1/2}}
{V_i\delta l_{i\pm1/2}w_i}.
$$

The tridiagonal row of Section 7.2 then applies to $q$. Gas weights and diffusivities are
evaluated at the corresponding centers and faces, and the positivity subcycling criterion uses
these weighted coefficients. Final mass and donor momentum updates use the conservative face flux,
preserving a constant-concentration equilibrium. Axisymmetric gas makes the azimuthal concentration
operator equal to its density form. Initialization uses the selected diffusion flux in the polar
velocity balance (Section 3.3), and its approximate small-height dust scale height includes the
midplane Stokes suppression and gas weighting (Section 3.2).

### 7.4 Conservative donor-momentum closure

The minimum conservative momentum correction transports donor values of
$(\ell_\phi,v_r,\ell_\theta)$ with the diffusive mass flux. It conserves the corresponding stored
momenta across internal faces and avoids changing density while leaving momentum stale. This is an
intentional generalized-property closure, not the spherical-coordinate expansion of a complete
Reynolds-averaged momentum tensor.

After solving density, the code reconstructs the time-centered integrated mass flux

$$
\mathcal F_{\rho,i+1/2}^{n+1/2}
=-\frac{A_{i+1/2}D_{i+1/2}}{2\delta l_{i+1/2}}
\left[(\rho_{i+1}^{n}-\rho_i^{n})
+(\rho_{i+1}^{n+1}-\rho_i^{n+1})\right].
$$

Crank–Nicolson positivity of the trial density does not by itself guarantee that the old-state
donor rule below forms a nonnegative mixture: opposing face transfers can each exceed the donor's
old mass while leaving a positive net density. The code therefore limits the reconstructed face
transfers before either density or momentum is accepted. Let

$$
M_i^n=V_i\max(\rho_i^n,0),
\qquad
O_i=\delta t\left[
\max(\mathcal F_{\rho,i+1/2},0)
+\max(-\mathcal F_{\rho,i-1/2},0)
\right]
$$

be the old cell mass and its total raw outward transfer. The donor factor is

```math
\theta_i=
\begin{cases}
1, & O_i=0,\\[3pt]
\min\!\left(1,
\dfrac{\mathrm{POS\_LIMIT}\,M_i^n}{O_i}\right), & O_i>0.
\end{cases}
```

Each face is scaled exactly once by the cell that supplies its mass:

```math
\widehat{\mathcal F}_{\rho,i+1/2}=
\begin{cases}
\theta_i\mathcal F_{\rho,i+1/2},
&\mathcal F_{\rho,i+1/2}\ge0,\\[3pt]
\theta_{i+1}\mathcal F_{\rho,i+1/2},
&\mathcal F_{\rho,i+1/2}<0.
\end{cases}
```

All donor factors are evaluated from the unchanged raw-flux array before any face is scaled. The
accepted density is then reconstructed conservatively,

$$
M_i^{n+1}=M_i^n-\delta t
\left(\widehat{\mathcal F}_{\rho,i+1/2}
-\widehat{\mathcal F}_{\rho,i-1/2}\right).
$$

Because at most `POS_LIMIT` of each donor's old mass can leave during one substep,

$$
M_i^n-\theta_iO_i
\ge (1-\mathrm{POS\_LIMIT})M_i^n\ge0.
$$

The statement includes an exactly empty cell: every outward transfer from a zero-mass donor is
suppressed. Internal faces remain conservative because the same accepted face value enters its two
adjacent cells with opposite signs. When every $\theta_i=1$, the accepted density is the original
Crank–Nicolson solution. Where the limiter activates, the update is a nonlinear conservative
correction and is not exactly time-centered CN locally.

Reducing the timestep alone cannot guarantee termination of a strict old-donor check: an empty
cell can receive and pass an implicit CN transfer at every finite trial step. The face limiter
enforces the bound directly, including zero old mass.

For each stored primitive $u_a\in\lbrace\ell_\phi,v_r,\ell_\theta\rbrace$, the associated momentum
flux is

```math
\mathcal F_{m_a,i+1/2}
=\widehat{\mathcal F}_{\rho,i+1/2}u_{a,\rm donor},
\qquad
u_{a,\rm donor}=
\left\lbrace\begin{array}{ll}
u_{a,i},&\mathcal F_{\rho,i+1/2}\ge0,\\
u_{a,i+1},&\mathcal F_{\rho,i+1/2}<0,
\end{array}\right.
```

followed by

$$
m_{a,i}^{n+1}=m_{a,i}^n
-\frac{\delta t}{V_i}
\left(\mathcal F_{m_a,i+1/2}-\mathcal F_{m_a,i-1/2}\right).
$$

The donor primitive is taken from the old substep state, with the vacuum values of Section 2.2
for a donor below $\rho_{\rm vac}$. The same accepted face flux enters its two cells with opposite
signs, so internal diffusive transfers conserve both mass and every stored momentum exactly up to
roundoff. More strongly, the update can be written as

$$
(M q)_i^{n+1}
=(M_i^n-\theta_iO_i)q_i^n
+\sum_j\widehat I_{ji}q_j^n,
$$

where every retained and incoming mass weight is nonnegative and their sum is $M_i^{n+1}$. Hence a
nonempty cell's new primitive lies in the convex hull of its old primitive and the old primitives
of its face donors. For any convex function $\eta(q)$, in particular $q^2$, this mixture also obeys
the global discrete inequality

$$
\sum_iM_i^{n+1}\eta(q_i^{n+1})
\le\sum_iM_i^n\eta(q_i^n)
$$

under periodic or zero-flux boundaries.

These invariants do not establish second-order momentum accuracy for spatially varying primitives.
The mass flux is time-centered when unlimited, but the transported primitive is the old upwind
value. Constant-primitive eigenmode tests exercise CN density accuracy and consistent momentum
scaling; the nonuniform-front tests establish bounds and conservation, not general momentum order.

### 7.5 Physical meaning and the rejected Reynolds alternative

The default density-diffusion flux is

$$
\boldsymbol J=-\boldsymbol D\cdot\boldsymbol\nabla\varrho_d,
$$

where $\varrho_d$ denotes $\Sigma_d$ in the vertically integrated model and $\rho_d$ in 3D.
Concentration mode instead uses $\boldsymbol J=-w\boldsymbol D\cdot\nabla(\varrho_d/w)$; the same
donor-momentum closure applies to either flux. The mass equation fixes the total transport flux
$\varrho_d\boldsymbol v_d+\boldsymbol J$, but it does not uniquely determine which momentum an
unresolved diffusive exchange carries. GameDev closes that ambiguity by treating each stored
generalized velocity

$$
q_a\in\lbrace\ell_\phi,v_r,\ell_\theta\rbrace
$$

as a parcel property transported by the net diffusive mass flux:

$$
\frac{\partial(\varrho_d q_a)}{\partial t}
+\boldsymbol\nabla\cdot(\boldsymbol Jq_a)=0
\qquad\text{during an isolated diffusion step}.
$$

Consequently, a spatially uniform $q_a$ remains uniform while density diffuses. Conversely, if
$\boldsymbol J=0$, this closure produces no momentum mixing even when $q_a$ varies in space. That
is a deliberate pressureless-model statement: unresolved counter-streaming exchanges with zero net
mass flux are omitted.

A Reynolds-averaged interpretation is a different continuum model. With unresolved stress
$\mathcal R_{ij}$, its conserved momentum and flux would be

$$
P_j=\varrho_dv_j+J_j,
\qquad
T_{ij}=\varrho_dv_iv_j+J_iv_j+v_iJ_j+\mathcal R_{ij}.
$$

Adapting this structure to density diffusion would require the corrected momentum state
$\varrho_d\boldsymbol v_d+\boldsymbol J$, both cross fluxes, all cylindrical or spherical
tensor-divergence terms, transverse components of $\boldsymbol J$ at each face, and consistent
primitive recovery, initialization, source, boundary, and restart semantics. Adding only $v_iJ_j$
to the present diffusion kernel would omit $\partial_tJ_j$ and create an inconsistent hybrid. The
donor and complete Reynolds closures can both be Galilean invariant when their states and fluxes
are transformed consistently, but they conserve different definitions of momentum and need not
agree at finite diffusivity.

The unresolved tensor $\mathcal R_{ij}$ is independent of whether density or concentration is
diffused. Setting it to zero removes turbulent dust pressure and shear stress; it is a closure
choice, not a mathematical consequence of Fickian diffusion. GameDev retains the generalized donor
model with $\mathcal R_{ij}=0$. A Reynolds density-diffusion model should be reconsidered only as a
separate physical model and must be implemented and verified as a complete system. Relevant
derivations of conservative mean dust momentum include
[Huang & Bai (2022)](https://arxiv.org/abs/2206.01023), whose published model diffuses dust
concentration; GameDev supports that mass flux but retains its distinct donor-momentum closure.

## 8. Global integrator and timestep control

### 8.1 Palindromic operator composition

One accepted step uses the symmetric second-order composition introduced by
[Strang (1968)](https://doi.org/10.1137/0705041), here written as

$$
D_y^{1/2}D_x^{1/2}D_z^{1/2}
A_x^{1/2}A_y^{1/2}A_z^{1/2}
S
A_z^{1/2}A_y^{1/2}A_x^{1/2}
D_z^{1/2}D_x^{1/2}D_y^{1/2}.
$$

One accepted production step can be summarized as

```text
choose dt from the current CFL state, DT_MAX, and the next output boundary
apply Dy(dt/2), Dx(dt/2), Dz(dt/2); recover primitive velocity
advance Ax(dt/2), Ay(dt/2), Az(dt/2), subcycling and recovering after every launch
if radiation is active, reconstruct cumulative optical depth at this midpoint state
apply the centered source update S(dt); rebuild conserved momenta
advance Az(dt/2), Ay(dt/2), Ax(dt/2), again with fresh CFL substeps
apply Dz(dt/2), Dx(dt/2), Dy(dt/2); recover primitive velocity
validate the evolved state, advance clocks, and write any due output
```

Inactive diffusion, polar, and radiation operations are omitted without changing the relative
order of the remaining operators. Primitive velocity is recovered after every advection launch and
after each three-direction diffusion group; the directional diffusion solves operate directly on
density and conserved momenta and do not consume primitive velocity between their launches. Each
diffusion half-step is one launch per direction with its own positivity subcycling
(Section 7.2).

Each directional advection half-interval is independently subcycled. Before every advection
launch, the global CFL rate is recomputed from the current synchronized state, so acceleration or
an earlier advection sweep cannot leave a later sweep using stale velocities. For a requested
directional interval $h$, the driver repeatedly chooses

$$
\delta t_m=\min\left[
h-\sum_{n\lt m}\delta t_n,
\frac{\mathrm{CFL\_DYN}}{\max_{ijk}\lambda_{ijk}},
\mathrm{DT\_MAX}
\right]
$$

and advances that direction until $\sum_m\delta t_m=h$. The state is recovered and
$\lambda_{ijk}$ is recomputed between launches. Different directional operators may therefore use
different internal substep partitions, but they all end at the same composition time before the
next operator begins.

### 8.2 CFL rates and step limits

The transport rate includes:

$$
\frac{|\Omega-\bar\Omega|}{\Delta x},
\qquad
|v_r|\frac{A_y}{V_y},
\qquad
|v_\theta|\frac{\Delta A_z}{\Delta V_y}
\frac{\max(\sin z_i,\sin z_o)}{\Delta(-\cos z)}.
$$

In a cell, the precise maximum is

$$
\lambda_{ijk}=\max\left[
\frac{|\ell_\phi-\bar\ell_\phi|}{R^2\Delta x},
|v_r|\frac{A_{y,j+1/2}}{\Delta V_{y,j}},
\left|\frac{\ell_\theta}{r}\right|
\frac{\Delta A_{z,j}}{\Delta V_{y,j}}
\frac{\max_{z\in[z_{k-1/2},z_{k+1/2}]}\sin z}
{\Delta V_{z,k}}
\right].
$$

The maximum sine is set to one if the polar cell straddles the midplane. With `VISC_FLOW`, the
same radial and polar geometrical rates are also evaluated for the gas target velocity before stiff
drag can transfer that velocity to the dust. Vacuum cells contribute zero. For finite states, the
host reduction uses

$$
\Delta t=\min\left(
\frac{\mathrm{CFL\_DYN}}{\max_{ijk}\lambda_{ijk}},
\mathrm{DT\_MAX}
\right),
$$

and $\Delta t=\mathrm{DT\_MAX}$ when every rate is zero. The global step $\Delta t$ is computed
from the state at the beginning of the step and shortened further whenever necessary to land
exactly on an output time. The diffusion and source operators use this global step; only the
advection half-intervals are subdivided.

`CFL_DYN <= 0.5` is required by the nearest-integer FARGO shift and enforced at compile time.
`DT_MAX` supplies an independent ceiling. Diffusion is not placed in this explicit CFL bound
because it is solved implicitly with positivity subcycling, and the source update has no step
restriction of its own because drag relaxation is integrated exponentially; `DT_MAX` is the only
limit on the source step when transport rates are small.

### 8.3 Consolidated boundary conditions

The continuum boundary conditions paired with the operators are

| Boundary | Transport | Diffusion |
|---|---|---|
| azimuthal | periodic | periodic |
| inner and outer radial | outflow only | zero normal flux |
| full-disk polar | outflow only | zero normal flux |
| `HALF_DISK` midplane | reflecting | zero normal flux |
| `HALF_DISK` high-latitude edge | outflow only | zero normal flux |

Periodicity identifies

$$
\boldsymbol U(X_{\min})=\boldsymbol U(X_{\max}).
$$

The zero-flux diffusion condition is

$$
\boldsymbol n\cdot\boldsymbol D\nabla\varrho_d=0,
$$

while the outflow transport condition retains only a normal characteristic directed out of the
domain, as written in Section 5.6. These choices intentionally differ: an advected representative
of the continuum may leave the modeled disk, whereas turbulent diffusion is confined by a
reflecting numerical wall. Optical depth additionally assumes no unresolved material interior to
the radial domain,

$$
\tau(Y_{\min})=0.
$$

### 8.4 Finite-state checks and failure behavior

The driver does not retry or roll back a step. A failed read or write of a frame file, and two
state checks, abort the run with an `Error: ...` message on standard error and exit status
`EXIT_FAILURE`. The state checks are:

- every CFL evaluation, before each global step and before every advection launch, marks a complete
  azimuthal ring with infinite rate if any of its densities, conserved momenta, or primitives is
  nonfinite; the host then reports the limiting cell index $(i,j,k)$
- a full-grid finite check of density, conserved momenta, primitives, and, with `RADIATION`,
  optical depth runs after initialization or restart, after the midpoint optical-depth rebuild,
  at the end of every accepted step, and after the optical-depth rebuild for output; it reports one
  offending cell

Negative densities are not treated as failures: transport converts a negative low-order density to
exact vacuum (Section 5.4), and diffusion cannot produce one (Section 7.4). The most recent saved
frame is the restart point after an abort.

### 8.5 Formal accuracy and error sources

For a smooth transport/source problem whose component updates are second order, a mesh scale $h$,
and timestep $\Delta t$, the intended deterministic truncation form is

$$
\|e\|\lesssim C_xh^p+C_t\Delta t^2+C_{\rm split}\Delta t^2,
$$

where PPM is nominally third order in smooth one-dimensional regions and Strang splitting limits
the composed time order to second order. This estimate is not established for general
variable-primitive momentum diffusion. The principal operator properties are

| Operator | Smooth-region property | Important reduction |
|---|---|---|
| azimuthal FARGO–PPM | exact integer shift plus high-order residual reconstruction | PPM limiting reduces order near extrema or sharp fronts |
| radial/polar PPM + SSPRK(3,3) | third-order method-of-lines time integrator for an isolated sweep | multidimensional Strang composition limits the global claim to second order |
| drag/source response | exact frozen linear drag and second-order endpoint force weighting | coefficient freezing and sequential nonlinear force evaluation supply the remaining error |
| Crank–Nicolson density diffusion | second order in time and centered finite-volume space where the donor limiter is inactive | positivity subcycling preserves the CN operator; donor limiting changes it locally |
| diffusive momentum transport | conservative old-donor mixing with primitive bounds and convex-quadratic nonincrease | constant-primitive tests do not establish general second-order accuracy |
| optical-depth quadrature | exact cell integral for cellwise constant extinction and linear face interpolation | density discretization and radial interpolation set the error |

At a limiter activation, vacuum reset, or outflow boundary, the local order can fall to first order.
After pressureless caustic formation, refinement need not converge to a single-valued continuum
solution because the physical closure itself has failed; that is distinct from a discretization
error.

## 9. GPU implementation and computational characteristics

### 9.1 Source map

The main numerical components map to the production source as follows:

| Scientific operation | Principal implementation |
|---|---|
| grid measures and disk profiles | `inc/fluid/param_grid.cuh`, `inc/fluid/param_phys.cuh` |
| convolved profile, PPM geometry setup, CFL reduction, and file conversion | `inc/fluid/fluid_host.cuh` |
| PPM, HLL, invariant-domain, and vacuum-recovery helpers | `inc/fluid/_transport.cuh` |
| drag weights, flag checks, and kernel declarations | `inc/fluid/fluid_kern.cuh` |
| density and velocity initialization | `src/fluid/init_rho_calc.cu`, `src/fluid/init_vel_calc.cu` |
| conservative transport | `src/fluid/advection_[xyz]{th,bl}.cu` |
| transport rate | `src/fluid/cfl_rate_calc.cu` |
| drag, gravity, geometry, and radiation | `src/fluid/source_update.cu` |
| density and donor-momentum diffusion | `src/fluid/diffusion_[xyz]{th,bl}.cu` |
| optical-depth increment and prefix sum | `src/fluid/optdepth_calc.cu`, `src/fluid/optdepth_csum.cu` |
| primitive/conserved conversion | `src/fluid/momentum_getv.cu`, `src/fluid/momentum_setv.cu` |
| nonfinite-state check | `src/fluid/inf_cell_flag.cu` |
| operator driver and output clock | `src/fluid/fluid_runtime.cu` |

Each production model is a compile-time specialization. A model-local `const_defs.cuh`, when
present, supplies its grid and physical constants; `flags.mk` selects optional operators and may
set the sweep. These are not runtime parameters: changing one requires recompilation. The
[user guide](../README.md#configuring-a-model) describes the model files and build variables.

### 9.2 Thread and block sweeps

The same PPM/HLL and Crank–Nicolson discretizations have two compile-time GPU implementations of
the six directional kernels. All other kernels are shared.

| Property | `FLUID_SWEEP=thread` | `FLUID_SWEEP=block` |
|---|---|---|
| kernels | `advection_[xyz]th`, `diffusion_[xyz]th` | `advection_[xyz]bl`, `diffusion_[xyz]bl` |
| work unit | one thread per complete line | one block of `TPB_BLOCK` threads per line |
| launch | `NB_X`, `NB_Y`, or `NB_Z` blocks of `TPB` threads | $N_{\rm line}$ blocks of `TPB_BLOCK` threads |
| line storage | thread-local arrays of line length | 12 full-grid workspace fields for advection; dynamic shared memory for diffusion |
| cooperative steps | none | loads, face reconstruction, HLL fluxes, low-order update, SSPRK combinations, donor factors, face scaling, momentum fluxes, stores |
| serial steps per line | all | FARGO ring mean, invariant-domain correction pass, subcycle count, Thomas or Sherman–Morrison recurrence |

For directional sweeps, the independent line counts are

$$
N_{{\rm line},x}=N_YN_Z,
\qquad
N_{{\rm line},y}=N_XN_Z,
\qquad
N_{{\rm line},z}=N_XN_Y.
$$

The thread implementation is the reference transcription. In the block implementation, eleven
advection workspace fields hold the conserved state, recovered primitives, and antidiffusive
fluxes; the twelfth stages the low-order density update so cells can advance cooperatively without
reading density already overwritten by another thread. The face-ordered invariant-domain
correction and the Thomas or Sherman–Morrison recurrence remain serial within each line, executed
by one thread of the block, to preserve the discrete numerical ordering of Sections 5.4 and 7.2.
Both implementations keep the old density until all three momentum components have consumed it,
and the block kernels synchronize factor construction, face scaling, and workspace reuse; the
donor factors of Section 7.4 reuse existing line storage. This is an implementation and
memory-layout choice, not a different numerical method; the validation campaign runs the complete
fluid matrix with both sweeps ([`val/README.md`](../val/README.md)).

### 9.3 Memory footprint

Every build allocates eight full-grid `real` device arrays: density, three primitives, three
conserved momenta, and the CFL rate. `RADIATION` adds one device array for optical depth. Four
pinned host arrays (five with `RADIATION`) stage output and restart transfers, and the radial and
polar PPM weights add $4(N_Y+1)+4(N_Z+1)$ values. The block sweep adds its persistent advection
workspace

$$
M_{\rm adv,work}=12N_G\,\mathrm{sizeof}(\mathtt{real}).
$$

In double precision one full-grid array is 8 MiB at $1024^2$ and 16 MiB at $128^3$, so

| Grid | Base device arrays | `RADIATION` | Block workspace | Pinned host |
|---|---|---|---|---|
| $1024\times1024\times1$ | 64 MiB | +8 MiB | +96 MiB | 32 MiB (+8 MiB) |
| $128^3$ | 128 MiB | +16 MiB | +192 MiB | 64 MiB (+16 MiB) |

The dynamic shared-memory requirements per diffusion block are

$$
M_{{\rm sh},x}=4N_X\,\mathrm{sizeof}(\mathtt{real}),
\qquad
M_{{\rm sh},y}=6N_Y\,\mathrm{sizeof}(\mathtt{real}),
\qquad
M_{{\rm sh},z}=6N_Z\,\mathrm{sizeof}(\mathtt{real}).
$$

On CUDA the host opts each block diffusion kernel in to this dynamic size at startup; on ROCm it
checks the device LDS capacity and the kernel's static-plus-dynamic allocation before any launch.
A request beyond the device limit aborts at startup. In double precision a six-array line of
length 1366 already exceeds a 64-KiB limit.

The thread kernels declare line-length arrays per thread:

| Kernel | Declared `real` values per thread | At line length 1024 |
|---|---|---|
| `advection_xth` | $21N_X$ | 168 KiB |
| `advection_yth`, `advection_zth` | $15N+4$ | 120 KiB |
| `diffusion_xth` | $6N_X$ | 48 KiB |
| `diffusion_yth`, `diffusion_zth` | $9N$ | 72 KiB |

These are source-level footprints if all arrays remain distinct. Compiler lifetime reuse can
reduce them, so compiler resource reports (for example `-Xptxas -v` through `CUDA_FLAGS`, or
`RESOURCE_REPORT` on ROCm) and profiler local-memory traffic are the authoritative measures.

### 9.4 Choosing a sweep

When neither the command line nor the model's `flags.mk` sets `FLUID_SWEEP`, the Makefile uses
`thread` on CUDA and `block` on ROCm. The choice changes only work decomposition, memory, and
speed; the two sweeps agree to rounding error, and each is validated against the same acceptance
criteria. The practical trade-offs are:

- **Exposed parallelism.** A thread-sweep directional launch runs only $N_{\rm line}$ threads,
  1024 for a $1024^2$ disk, which is a small fraction of a modern GPU's resident-thread capacity.
  The block sweep runs `TPB_BLOCK` times as many threads (32 on CUDA, 64 on ROCm) and reads
  contiguous azimuthal cells in its azimuthal kernels, whereas neighboring thread-sweep azimuthal
  threads read addresses $N_X$ cells apart.
- **Serial fraction.** In the block sweep one thread per line performs the invariant-domain pass
  and the tridiagonal recurrence while the other lanes wait. The thread sweep performs every step
  serially but keeps all lanes busy on different lines.
- **Per-thread memory.** The thread sweep's local arrays grow with line length (Section 9.3). On
  gfx942 the thread-sweep azimuthal advection exceeds a linker limit at `N_X = 1024`, which is why
  ROCm defaults to the block sweep ([user guide](../README.md)).
- **Global and shared memory.** The block sweep needs $12N_G$ additional device values and
  dynamic shared memory that limits the diffusion line length to the device's per-block maximum.

Use the block sweep on ROCm and whenever the thread sweep's per-thread arrays exceed the
compiler's or device's limits. Otherwise, performance is grid and machine dependent: benchmark both
sweeps on the intended production grid and GPU before fixing one for a campaign, and record the
selection with the run. Sweep-performance evidence is not part of the retained
publication-verification suite.

### 9.5 Backend differences and diagnostics

Both backends compile the same `.cu` files, with ROCm using `hipcc -x hip`. `inc/gpu.cuh` maps
runtime allocation, copy, error, Thrust-policy, and random-number APIs to CUDA/cuRAND or
HIP/hipRAND; builds do not invoke HIPIFY or generate HIP sources. Explicit backend branches remain
only where the hardware differs: the block-sweep width `TPB_BLOCK` (32 threads on CUDA, one 64-lane
wavefront on ROCm), and the dynamic shared-memory opt-in on CUDA versus the LDS capacity check on
ROCm. Ordinary elementwise and thread-line kernels use `TPB = 64` on both backends.

The backend does not change the discrete equations. Both backends compile with `-O2` and omit
blanket fast-math flags; finite-only assumptions could invalidate the nonfinite-state checks of
Section 8.4. The azimuthal density perturbation uses the vendor generator, so CUDA and ROCm
initialize different noise realizations. Cross-backend validation therefore compares convergence,
conserved fields, and density-weighted velocity norms rather than requiring byte equality.

The global CFL reduction is quiet by default. It retains the nonfinite-state check but skips the
three diagnostic velocity copies, ring-velocity reduction, and limiting-cell printout, which are
enabled with `CUDA_SYNC_TRACE` or `HIP_SYNC_TRACE`. Those trace builds also synchronize after every
kernel launch and print the CFL-limiting cell after each opening diffusion direction and before
every advection launch.

## 10. Output and restart semantics

Frame $n$ holds the state at

$$
t_n=n\,\mathrm{DT\_OUT}.
$$

The driver shortens the last accepted step in each interval so $t=t_n$ to floating-point
precision; output interpolation is therefore unnecessary. File names, layout, and reading are
described in the [user guide](../README.md#output-files); the numerical meaning of the saved fields
is:

| Field | Saved quantity |
|---|---|
| `dustdens` | evolved density $\varrho_d$: $\Sigma_d$ when `N_Z == 1`, $\rho_d$ when `N_Z > 1` |
| `dustvelx` | $v_\phi=\ell_\phi/R$ at the cell center |
| `dustvely` | $v_r$ |
| `dustvelz` | $v_\theta=\ell_\theta/r$ at the cell center, zero when `N_Z == 1` |
| `optdepth` | cumulative outer-face optical depth $\tau_{i,j+1/2,k}$, not the center value $\tau_c$ |

The velocity conversion is performed in host memory, so diagnostic I/O never temporarily overwrites
the angular variables being evolved on the GPU. Vacuum cells carry the regularized velocity of
Section 2.2. The saved optical depth is recomputed from the end-of-step density just before the
frame is written; it is not the midpoint optical depth used by the preceding source update.

On restart from frame $n$, the model time is reset to $n\,\mathrm{DT\_OUT}$, the saved linear
velocities are converted back with

$$
\ell_\phi=Rv_\phi,
\qquad
\ell_\theta=rv_\theta,
$$

and $m_a=\varrho_du_a$ reconstructs all conserved arrays, with the vacuum reset of Section 2.2,
before the next operator is launched. Optical depth is reconstructed when radiation is active, and
the restored state passes the finite check of Section 8.4. Frames contain no conserved momenta and
no random state; the fluid evolution itself is deterministic. Because floating-point evaluation of
$R(\ell_\phi/R)$ and $r(\ell_\theta/r)$ need not reproduce the original angular momentum bit for
bit, restart validation must use explicit state and conservation tolerances rather than require
byte-identical evolved fields.

## 11. Known limitations

Verification definitions, evidence, and untested regimes are maintained in
[`fluid_testset.md`](fluid_testset.md); limitations shared with the swarm branch are listed in the
[user guide](../README.md#current-limitations). The limitations below concern the fluid physical or
numerical model itself rather than current test coverage.

- The fluid is monodisperse with one Stokes-number profile and pressureless. It cannot represent
  multistreaming after caustic formation; a swarm or another kinetic representation is required in
  that regime.
- The diffusion-momentum closure is an intentional coordinate-dependent parcel model: net
  diffusive mass carries the donor values of $(\ell_\phi,v_r,\ell_\theta)$, while unresolved
  momentum exchange at zero net mass flux is omitted. It must not be described as a complete
  Reynolds-averaged momentum tensor. A Reynolds option would be a different physical model, not a
  correction term that can be added independently to the present kernels.
- The 3D initializer balances polar advection and the selected diffusion flux only to discretization
  error. The normalized instantaneous mismatch and its polar-resolution convergence are measured by
  `test_startup_3d`; later momentum relaxation can still produce a physical startup transient and is
  not assumed to vanish under mesh refinement.
- Transport boundaries are outflow-only, while diffusion boundaries are zero-flux walls, and
  optical depth assumes no material inside `Y_MIN`. Mass and radiation shielding near the inner
  edge therefore depend on the chosen radial domain.
- The azimuthal domain is always periodic; a range shorter than $2\pi$ imposes azimuthal
  periodicity on every non-axisymmetric structure.
- The block diffusion kernels execute each Thomas or Sherman–Morrison recurrence serially within
  its line, and the block advection kernels retain a serial face-ordered correction pass. A more
  parallel replacement would have to preserve the conservation and positivity properties of
  Sections 5.4 and 7.4, for example through a face-budget limiter and a batched tridiagonal solve,
  and would require the full analytical suite and profiler evidence.

## 12. References

- Colella & Woodward (1984), [PPM](<https://doi.org/10.1016/0021-9991(84)90143-8>)
- Harten, Lax & van Leer (1983), [HLL flux](https://doi.org/10.1137/1025002)
- Masset (2000), [FARGO](https://arxiv.org/abs/astro-ph/9910390)
- Shu & Osher (1988), [TVD Runge–Kutta time integration](https://doi.org/10.1016/0021-9991(88)90177-5)
- Crank & Nicolson (1947), [implicit trapezoidal diffusion](https://doi.org/10.1017/S0305004100023197)
- Sherman & Morrison (1950), [rank-one inverse update](https://doi.org/10.1214/aoms/1177729893)
- Higueras & Roldán (2023), [Crank–Nicolson positivity](https://arxiv.org/abs/2301.01066)
- Strang (1968), [symmetric operator splitting](https://doi.org/10.1137/0705041)
- Huang & Bai (2022), [multifluid dust algorithms](https://arxiv.org/abs/2206.01023)
- Youdin & Lithwick (2007), [particle stirring and settling](https://arxiv.org/abs/0707.2975)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Takeuchi & Lin (2002), [vertically structured gas and dust drift](https://arxiv.org/abs/astro-ph/0208552)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
- Burns, Lamy & Soter (1979), [radiation forces on small particles](https://doi.org/10.1016/0019-1035(79)90050-2)
- Shakura & Sunyaev (1973), [$\alpha$ viscosity](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S)
- Epstein (1924), [drag on small spheres in a dilute gas](https://doi.org/10.1103/PhysRev.23.710)
