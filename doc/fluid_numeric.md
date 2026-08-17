# Eulerian dust-fluid model: equations and numerical guide

## 1. Model overview

The Eulerian branch represents monodisperse, pressureless dust as a finite-volume continuum on the
disk grid. Gas is prescribed analytically, dust does not back-react on it, and the solver combines
conservative transport with local forces and optional density diffusion. Backend-neutral complete
files live under `inc/comm/fluid/` and `src/comm/fluid/`; runtime, random-number, declaration, and
launch-policy files live under the corresponding `cuda/fluid/` or `rocm/fluid/` branch.

### 1.1 Supported configurations

The branch supports:

- a two-dimensional radial–azimuthal disk with `N_Z == 1`
- a full three-dimensional spherical grid with `N_Z > 1`
- pressureless dust transport
- gas drag, gravity, spherical geometric forces, and optional radiation pressure
- optional turbulent diffusion of dust density
- optional viscous gas accretion when diffusion is enabled

Radial–polar two-dimensional models are not supported. The azimuthal dimension is always active.
The fluid is monodisperse and does not back-react on the prescribed gas.

| Geometry | Grid condition | Evolved density | Interpretation |
|---|---|---|---|
| radial–azimuthal | `N_X > 1`, `N_Z == 1` | $\Sigma_d$ | vertically integrated disk |
| full 3D | `N_X > 1`, `N_Z > 1` | $\rho_d$ | spherical radial–polar volume |

Transport and source evolution are always active. `DIFFUSION` enables density diffusion,
`RADIATION` enables attenuated radiation pressure, `VISC_FLOW` replaces the static gas radial
target with the viscous prescription, and `HALF_DISK` selects a reflecting upper midplane in 3D.
`VISC_FLOW` requires `DIFFUSION`, and every 3D fluid model requires `DIFFUSION` to support the dust
layer vertically.

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
| density diffusion | spherical finite-volume PDE | cylindrical Itô displacement |
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

### 1.3 Guide structure

The remainder of this chapter defines the grid and evolved state, constructs the prescribed disk
and initial condition, states the continuum equations, describes each numerical operator, and then
documents their composition, CUDA implementations, output semantics, validation boundary, and
known limitations. Sections 1–7 intentionally parallel the corresponding physical topics in the
swarm guide; the later organization differs only where the swarm requires a separate collision
chapter.

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

Every active azimuthal domain is periodic. A range shorter than $2\pi$ is therefore a periodic
wedge, not an open sector of an otherwise complete disk: transport, FARGO shifts, invariant-domain
bounds, and azimuthal diffusion all identify its two azimuthal faces. Non-axisymmetric structures
repeat with period $X_{\max}-X_{\min}$, which must be treated as part of the model definition.

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

Complete grid cells use $(i,j,k)$ for the $(x,y,z)=(\phi,r,\theta)$ directions. A symbol $i$
used later inside a purely one-dimensional tridiagonal or reconstruction derivation denotes a
generic line index and is not a second radial-index convention.

The radial finite-volume measure is

```math
\Delta V_y=\frac{y_{\rm out}^d-y_{\rm in}^d}{d},
\qquad
d=
\left\{\begin{array}{ll}
2,&N_Z=1,\\
3,&N_Z>1.
\end{array}\right.
```

The polar measure is $\Delta V_z=\cos z_{\rm in}-\cos z_{\rm out}$ when the polar dimension is active.
Consequently, `N_Z == 1` represents a vertically integrated disk and evolves dust surface density
$\Sigma_d$; `N_Z > 1` evolves volume density $\rho_d$.

The complete cell measure can be written

```math
V_{ijk}=\Delta x\,
\frac{y_{j+1/2}^{d}-y_{j-1/2}^{d}}{d}
\left\{\begin{array}{ll}
1,&N_Z=1,\\
\cos z_{k-1/2}-\cos z_{k+1/2},&N_Z>1.
\end{array}\right.
```

Radial fluxes use $A_{y,j+1/2}=y_{j+1/2}^{d-1}$. In 3D, the radial contribution
to a polar face area is

$$
\Delta A_{z,j}
=\int_{y_{j-1/2}}^{y_{j+1/2}}y\,dy
=\frac{y_{j+1/2}^2-y_{j-1/2}^2}{2},
$$

so the separated polar face factor is
$A_{z,j,k+1/2}=\Delta A_{z,j}\sin z_{k+1/2}$.

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

When $\varrho_d<\rho_{\rm vac}$, this division is replaced by

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
meaning to the velocity of an almost empty cell.

For a 2D disk, $\rho_d$ in these expressions means the evolved surface density. File output
converts $\ell_\phi$ and $\ell_\theta$ to $v_\phi$ and $v_\theta$ without changing the device state.

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
constants to one does not make dimensional consistency optional. The principal dimensions are

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
\mathrm{Sc}_a=\frac{\nu}{D_a}.
$$

Throughout this guide, $r$ denotes spherical radius, $R=r\sin\theta$ cylindrical radius, and
$Z=r\cos\theta$ cylindrical height. A generic $\varrho_d$ denotes whichever density the active
geometry evolves: $\Sigma_d$ in a vertically integrated model and $\rho_d$ in 3D.

### 2.4 Principal code parameters

| Code parameter | Mathematical role |
|---|---|
| `G`, `M_S`, `R_0` | $G$, $M_\star$, and the reference radius $R_0$ |
| `SIGMA_0`, `ASPR_0` | $\Sigma_0$ and $h_0$ |
| `IDX_P`, `IDX_Q` | power-law indices $p$ and $q$ |
| `METAL_Z` | initial metallicity $Z_{\rm metal}$ |
| `STOKES_0` | reference midplane Stokes number $\mathrm{St}_0$ |
| `ALPHA` or `NU` | turbulent $\alpha$ or constant kinematic viscosity $\nu$ |
| `SCHMIDT_X/Y/Z` | directional Schmidt numbers $\mathrm{Sc}_{x,y,z}$ |
| `BETA_0`, `KAPPA_0`, `T_BETA` | radiation ratio $\beta_0$, opacity $\kappa_0$, and ramp time $T_\beta$ |
| `CFL_DYN`, `DT_MAX` | explicit Courant factor and global timestep ceiling |
| `DT_OUT`, `SAVE_MAX` | interval between saved frames and final saved-frame index |
| `POS_LIMIT` | upper bound controlling the Crank–Nicolson explicit-side coefficient sum |
| `RHO_VAC` | density below which primitive velocity uses the regularized vacuum state |

### 2.5 Compile-time feature selection

The production fluid branch is specialized at compilation rather than switched at runtime:

| Selection | Effect |
|---|---|
| `DIFFUSION` | add the three directional density-diffusion operators and their momentum closure |
| `RADIATION` | construct optical depth and add attenuated radiation pressure |
| `VISC_FLOW` | use the viscous gas target velocity; requires `DIFFUSION` |
| `CONST_NU` | use constant $\nu$ instead of constant $\alpha$ wherever viscosity is required |
| `HALF_DISK` | reflect the active polar boundary at the midplane |
| `FLUID_SWEEP=thread` | assign one CUDA thread to each complete directional line |
| `FLUID_SWEEP=block` | assign one cooperative GPU block to each directional line |

Transport and the local source update have no feature flag and are always compiled. The two sweep
choices implement the same mathematical operators and differ only in work decomposition and
temporary storage.

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
pressure-support definition above. Section 3.3 uses this exact rotation target.

The turbulent branch uses the $\alpha$ prescription of
[Shakura & Sunyaev (1973)](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S). The viscosity
is selected as either

$$
\nu(R)=\alpha h_g^2R^2\Omega_K
$$

for constant `ALPHA`, or $\nu(R)=\mathrm{NU}$ for constant `NU`. In the latter case the local
equivalent turbulent parameter is

$$
\alpha(R)=\frac{\nu}{h_g^2R^2\Omega_K}.
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

For comparison, on an untruncated vertical line let $x=Z/H_g$. The thin-disk expansion is

$$
\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g^2}
\right]
=e^{-x^2/2}
\left[1+\frac{3}{8}h_g^2x^4+O(h_g^4)\right],
$$

and the Gaussian fourth moment gives

$$
\mathcal C_{g,\infty}=1+\frac{9}{8}h_g^2+O(h_g^4).
$$

Finite spherical boundaries replace this asymptotic normalization by the exact containment factor
above and can be especially important near a radial or polar edge.

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
L_s=0.05R_0,\qquad \sigma_s=\frac{L_s}{2},
$$

evaluated by a uniform-$R$ quadrature with `N_Y + 1` source and destination points.

If $R_m=R_{\min,\rm init}+m\Delta R$ is that auxiliary axis, the implemented rectangular sum is

$$
\Sigma_{d,\rm conv}(R_m)
\approx\sum_{n\in\mathcal S}
\Sigma_d(R_n)
\frac{\exp[-(R_m-R_n)^2/(2\sigma_s^2)]}{\sqrt{2\pi}\sigma_s}
\Delta R,
$$

where $\mathcal S$ contains only source points in
$[Y_{\min}+2L_s,Y_{\max}-2L_s]$.

The convolution is evaluated on a temporary uniform cylindrical-radius axis. In 3D its lower bound
is the smallest cylindrical radius covered by the spherical domain,

$$
R_{\min,\mathrm{init}}
=Y_{\min}\min[\sin(Z_{\min}),\sin(Z_{\max})],
$$

rather than $Y_{\min}$. This prevents high-latitude cells with $R<Y_{\min}$ from being clipped
to an unrelated radial profile value. In 2D, $R_{\min,\mathrm{init}}=Y_{\min}$.

The edge convolution is not renormalized. In 2D the convolved surface density is evolved directly
and its unresolved vertical profile is assumed to be well mixed with the gas. In 3D it is embedded
in a Gaussian dust layer using the settling–diffusion balance discussed by
[Youdin & Lithwick (2007)](https://arxiv.org/abs/0707.2975),

$$
H_d=H_g\sqrt{\frac{\alpha_z}{\mathrm{St}_{\rm mid}}},
\qquad
\alpha_z=\frac{\alpha}{\mathrm{Sc}_z}.
$$

Specifically,

$$
\rho_d(R,Z)
=\frac{\Sigma_{d,\rm conv}(R)}{\sqrt{2\pi}H_d(R)}
\exp\left[-\frac{Z^2}{2H_d(R)^2}\right].
$$

This is the equilibrium implied by the selected internal density-diffusion equation, not the
standard gas-concentration diffusion closure. Initialization then applies

$$
\varrho_d(x_i,y_j,z_k)
\leftarrow
\varrho_d(x_i,y_j,z_k)\max(1+0.1\xi_i,0),
\qquad
\xi_i\sim\mathcal N(0,1),
$$

using one reproducible deviate for each azimuthal column, so every radial and polar cell at fixed
$x_i$ receives the same perturbation.

Because the edge convolution and azimuthal perturbation are not renormalized, the initialized mass
is the finite-volume integral of the resulting field rather than a separately imposed parameter:

```math
M_d=
\left\{\begin{array}{ll}
\displaystyle\int\Sigma_d(R,\phi)R\,dR\,d\phi,&N_Z=1,\\
\displaystyle\int\rho_d(r,\theta,\phi)r^2\sin\theta\,dr\,d\theta\,d\phi,&N_Z>1.
\end{array}\right.
```

### 3.3 Initial velocity

The initial velocity is the local no-backreaction drift relative to the pressure-supported gas,
using the steady test-particle limit of
[Nakagawa, Sekiya & Hayashi (1986)](<https://doi.org/10.1016/0019-1035(86)90121-1>).
The effective rotation-support parameter derived in Section 3.1 and the gas azimuthal target are

$$
\eta(R,Z)
=-\frac{1}{2}\left[
\left(p+\frac{q}{2}-\frac{3}{2}\right)h_g^2
+q\left(1-\frac{R}{\sqrt{R^2+Z^2}}\right)
\right],
\qquad
v_{\phi,g}=v_K\sqrt{\max(1-2\eta,0)}.
$$

The initialized cylindrical drift is

$$
v_{R,d}
=\frac{v_{R,g}+2\mathrm{St}(v_{\phi,g}-v_K)}{1+\mathrm{St}^2},
\qquad
v_{\phi,d}=v_{\phi,g}-\frac{\mathrm{St}}{2}v_{R,d}.
$$

If `VISC_FLOW` is enabled, the prescribed cylindrical gas radial velocity follows the steady
viscous expression of [Kanagawa et al. (2017)](https://arxiv.org/abs/1706.08975). In the vertically
integrated case it reduces to

```math
v_{R,g}
=-\frac{3\nu}{R}\left(g_\nu+p+\frac12\right),
\qquad
g_\nu=\frac{d\ln\nu}{d\ln R}
=\left\{\begin{array}{ll}
0,&\nu=\mathrm{constant},\\
q+\frac32,&\alpha=\mathrm{constant}.
\end{array}\right.
```

For the resolved 3D expression, define

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

The implemented target is

$$
v_{R,g}
=-\frac{\nu}{R}
\left[3\left(g_\nu+g_{\rho R}+\frac12\right)-q(1+g_{\rho Z})\right].
$$

In 3D, the polar primitive also includes the velocity
needed to approximately balance the initialized polar diffusive flux,

$$
v_{\theta,\mathrm{diff}}
=\frac{D_z}{r\rho_d}\frac{\partial\rho_d}{\partial\theta},
$$

evaluated with one-sided boundary differences and centered interior differences.

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
\ell_\phi=Rv_{\phi,d}.
$$

## 4. Governing continuum equations

### 4.1 Continuum equations and component source terms

Let $\varrho_d$ denote the evolved dust density: $\varrho_d=\Sigma_d$ in 2D and
$\varrho_d=\rho_d$ in 3D.

**Core fluid equations.** In coordinate-independent conservation form, the model advances

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
$\varrho_d\boldsymbol v_d\boldsymbol v_d+\boldsymbol P_d$. The term
$\boldsymbol S_{m,D}$ denotes the conservative donor-momentum flux paired with diffusive mass
transport and vanishes when `DIFFUSION` is disabled; Section 7 gives its discrete definition and
limitations. Disabling `RADIATION` sets $\boldsymbol a_{\rm rad}=0$.

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

The polar torque and $\ell_\theta$ are absent when `N_Z == 1`.

### 4.2 Discrete conservation and physical scope

For cell measure $V_{ijk}$, the discrete dust mass and stored momenta are

$$
M_d^h=\sum_{ijk}\varrho_{d,ijk}V_{ijk},
\qquad
Q_a^h=\sum_{ijk}m_{a,ijk}V_{ijk},
\quad
a\in\{x,y,z\}.
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
the stored momenta through drag, gravity, radiation, and spherical geometry.

This conservation statement concerns the variables actually stored. In particular,
$Q_x^h=\int\rho_d\ell_\phi\,dV$ is axial angular momentum, while $Q_y^h$ is radial linear
momentum. Their physical conservation can be broken by the prescribed external gas and stellar
forces even when their numerical transport is conservative.

## 5. Conservative pressureless transport

### 5.1 Reconstruction, flux, and invariant-domain limiting

Density and all three momenta are transported conservatively. Each directional operator uses:

1. geometry-aware PPM reconstruction
2. a pressureless HLL interface flux
3. a first-order cell-centred HLL flux as a robust invariant-domain base
4. one conservative face coefficient that limits the high-minus-low correction

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
$A_{i+1/2}=y_{i+1/2}^{d-1}$. For polar transport through radial cell $j$, the exact
finite-volume update is

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

The reconstruction is the piecewise parabolic method of
[Colella & Woodward (1984)](<https://doi.org/10.1016/0021-9991(84)90143-8>). The uniform azimuthal
mesh uses its bounded four-cell face value. On the logarithmic radial mesh and spherical polar
mesh, face weights are precomputed from exact cubic moment constraints in the finite-volume
coordinates

$$
s_y=\frac{y^d}{d},
\qquad
s_z=-\cos z.
$$

Boundary-adjacent internal faces use a two-cell linear interpolation where a complete four-cell
stencil is unavailable. If $s_L$ and $s_R$ are the arithmetic centers of the adjacent cells in
the appropriate volume coordinate and $s_f$ is their common face, then

$$
\omega=\frac{s_f-s_L}{s_R-s_L},
\qquad
q_f=(1-\omega)\bar q_L+\omega\bar q_R.
$$

The pressureless flux uses the two-wave construction of
[Harten, Lax & van Leer (1983)](https://doi.org/10.1137/1025002) and retains its proper
left-going, right-going, and two-wave branches.

On the uniform azimuthal mesh, the unlimited four-cell face estimate is

$$
q_{i+1/2}^{*}
=\frac{7(q_i+q_{i+1})-(q_{i-1}+q_{i+2})}{12},
$$

and the stored face is clipped to
$[\min(q_i,q_{i+1}),\max(q_i,q_{i+1})]$. On nonuniform radial and polar meshes, the four weights
$w_m$ solve cubic moment-exactness conditions in $s_y$ or $s_z$ and give

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

Inside one cell, PPM represents the reconstructed parabola by left and right faces $q_L,q_R$ and

$$
\Delta q=q_R-q_L,
\qquad
q_6=6\bar q-3(q_L+q_R).
$$

After the standard monotonicity corrections, integrating a fraction $c\in[0,1]$ next to the right
or left face gives the time-averaged upwind states

$$
q_R^{\rm tr}=q_R-\frac{c}{2}
\left[\Delta q-\left(1-\frac{2c}{3}\right)q_6\right],
$$

$$
q_L^{\rm tr}=q_L+\frac{c}{2}
\left[\Delta q+\left(1-\frac{2c}{3}\right)q_6\right].
$$

The monotonicity correction itself is

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

Azimuthal FARGO transport uses the nonzero tracing fraction $c$ because one conservative orbital
advection update spans its requested substep. Radial and polar transport pass $c=0$ to these
formulas and obtain their temporal order from the three SSPRK flux evaluations in Section 5.3.
Applying both characteristic tracing and SSPRK time centering in those directions would duplicate
the time evolution.

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
numerical diffusion. The invariant-domain correction prevents that regularized flux from creating
negative density or unbounded transported primitive ratios.

For a complete pressureless state $\boldsymbol U$ transported at normal speed $a$, the physical
flux is $\boldsymbol F=a\boldsymbol U$. With

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
nonpositive, it uses $\boldsymbol F_R$.

The limiter enforces nonnegative density and local bounds on every momentum-to-density ratio. The
same limited face flux enters neighboring cells with opposite signs, so the normal update remains
conservative. As a final floating-point safety fallback, any cell whose updated density is still
negative is converted to an exact vacuum by setting its density and all three momentum components
to zero. This fallback should remain inactive when the CFL and invariant-domain assumptions hold;
its purpose is to prevent a residual negative density from producing an undefined vacuum velocity,
not to replace the conservative limiter.

More explicitly, let $\boldsymbol F^L$ be the cell-centred low-order HLL flux and
$\delta\boldsymbol F=\boldsymbol F^H-\boldsymbol F^L$ the PPM antidiffusive correction. A face
applies

$$
\boldsymbol U_i\leftarrow
\boldsymbol U_i-\lambda_i\alpha\,\delta\boldsymbol F,
\qquad
\boldsymbol U_{i+1}\leftarrow
\boldsymbol U_{i+1}+\lambda_{i+1}\alpha\,\delta\boldsymbol F,
$$

where $\lambda_i=\Delta t A_{i+1/2}/V_i$ and one shared $0\le\alpha\le1$ is the largest accepted
scale satisfying

$$
\varrho_d\ge0,
\qquad
u_{a,\min}\varrho_d\le m_a\le u_{a,\max}\varrho_d
$$

in both adjacent cells for $u_a\in\{\ell_\phi,v_r,\ell_\theta\}$.

Every listed condition is an affine inequality $g(\boldsymbol U)\ge0$. If the proposed face
correction changes it by $\Delta g$, the admissible coefficient is reduced only when
$\Delta g<0$:

$$
\alpha\leftarrow
\min\left[
\alpha,
(1-10^{-12})\frac{g(\boldsymbol U)}{-\Delta g}
\right].
$$

The final face coefficient is the minimum over density and the lower and upper bounds of all three
primitive ratios in both adjacent cells, clipped to $[0,1]$.

### 5.2 FARGO azimuthal transport

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
algebraically a periodic integer translation followed by conservative residual transport.

The allowed endpoint `CFL_DYN = 0.5` can make $c_i=1$ exactly when the half-cell frame offset and
the bounded residual displacement align. This remains within the PPM tracing domain, but it leaves
no roundoff margin at the one-cell limit. The production default is therefore `CFL_DYN = 0.45`;
this safety margin is not a correction to the method, and the limiting value 0.5 remains a
supported edge case.

### 5.3 Radial and polar integration and boundaries

The radial and polar method-of-lines operators use the three-stage TVD Runge–Kutta construction
of [Shu & Osher (1988)](https://doi.org/10.1016/0021-9991(88)90177-5), commonly denoted
SSPRK(3,3). PPM provides spatial reconstruction; SSPRK supplies temporal integration. The radial boundaries
are outflow-only. A full polar disk is outflow-only at both polar edges; `HALF_DISK` reflects at the
midplane and remains outflow-only at its other polar edge.

If $L(\boldsymbol U)$ is one spatial flux-divergence evaluation, the stages are

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

At an outflow-only boundary with outward normal speed $a_n$, the boundary mass flux is

```math
F_\varrho=
\left\{\begin{array}{ll}
a_n\max(\varrho_d,0),&a_n\text{ points out of the domain},\\
0,&a_n\text{ points into the domain},
\end{array}\right.
\qquad
F_{m_a}=F_\varrho u_a.
```

At a reflecting `HALF_DISK` midplane, all normal flux components are set to zero.

PPM is formally high order on smooth fields, but the complete multidimensional solver should be
described as second-order accurate: Strang composition, Crank–Nicolson diffusion, boundary fluxes,
and limiter activation set the global claim. SSPRK(3,3) should not be used to claim that the full
scheme is third-order in time.

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
centrifugal acceleration, and polar geometric torque. The update is sequential in
$\ell_\phi$, $\ell_\theta$, and $v_r$, with force re-evaluation after the angular updates.

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

when $\tau<10^{-4}$. Azimuthal angular momentum has no nondrag force, so

$$
\ell_\phi^{n+1}=E\ell_\phi^n+Q\ell_{\phi,g}.
$$

The code then evaluates the old and new polar torques using $\ell_\phi^n$ and
$\ell_\phi^{n+1}$, advances $\ell_\theta$, reevaluates the centrifugal force with both updated
angular momenta, and finally advances $v_r$. This ordering is what gives the symbols $F^n$ and
$F^{n+1}$ their precise meaning here; they are sequential endpoint approximations inside one
cell-local source solve, not forces from two separate hydrodynamic states.

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

The corresponding radial acceleration is

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

Optical depth is reconstructed at the source midpoint. Cumulative values live at radial outer
faces and are interpolated to logarithmic cell centers with

$$
\tau_c=\tau_{\rm in}+\frac{\tau_{\rm out}-\tau_{\rm in}}{\sqrt{a_y}+1},
$$

which is exact when optical depth is linear in physical radius within the logarithmic cell. In 2D,
the well-mixed closure converts surface density to midplane extinction density using
$\sqrt{2\pi}H_g$, without a Stokes-dependent scale height.

The local radial optical-depth increment in cell $(i,j,k)$ is

$$
\Delta\tau_{ijk}=\kappa_0\rho_{{\rm ext},ijk}\Delta r_j,
\qquad
\Delta r_j=y_{j-1/2}(a_y-1),
$$

where

```math
\rho_{\rm ext}=
\left\{\begin{array}{ll}
\Sigma_d/(\sqrt{2\pi}H_g),&N_Z=1,\\
\rho_d,&N_Z>1.
\end{array}\right.
```

The stored outer-face value is the inclusive radial prefix sum

$$
\tau_{i,j+1/2,k}=\sum_{m=0}^{j}\Delta\tau_{imk},
\qquad
\tau_{i,-1/2,k}=0.
$$

Thus the source cell sees attenuation $e^{-\tau_c}$ constructed from the inner and outer face
values of its own radial cell. Optical depth is recomputed from the density at the midpoint of the
symmetric global step, so radiation uses the centered mass distribution rather than the state at
only the beginning or end of the step.

## 7. Density diffusion and momentum consistency

### 7.1 Target diffusion equation and coordinate basis

The selected equation is

$$
\frac{\partial\rho_d}{\partial t}
=\nabla\cdot(\boldsymbol D\nabla\rho_d),
$$

with $\rho_d$ interpreted as $\Sigma_d$ in 2D. The code deliberately does not diffuse
$\rho_d/\rho_g$.

In a vertically integrated radial–azimuthal disk, the scalar operator is

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
D_x=\frac{\nu}{\mathrm{Sc}_x},
\qquad
D_y=\frac{\nu}{\mathrm{Sc}_y},
\qquad
D_z=\frac{\nu}{\mathrm{Sc}_z}.
$$

The disk profiles used to evaluate $\nu$ depend on cylindrical $R$; this does not change the
coordinate basis of the differential operator.

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
| azimuthal | $c^-_i=c^+_i=\delta tD_x/[2(R\Delta x)^2]$ |
| radial | $A_{i+1/2}=y_{i+1/2}^{d-1}$, $V_i=\Delta V_{y,i}$, and $\delta l$ is the distance between neighboring radial centers |
| polar | $A_{k+1/2}=\sin z_{k+1/2}$, $V_k=y\Delta V_{z,k}$, and $\delta l=y\Delta z$ |

the row solved by the radial and polar Thomas algorithm is

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
\delta t=\frac{\Delta t}{N_{\rm sub}}.
$$

For the periodic azimuthal line this is equivalently based on
$2c=\Delta tD_x/(R\Delta x)^2$. The subdivision controls the sign of the explicit
Crank–Nicolson right-hand side; unconditional linear stability by itself would not prevent
negative density oscillations.

### 7.3 Conservative donor-momentum closure

The current minimum conservative momentum correction transports donor values of
$(\ell_\phi,v_r,\ell_\theta)$ with the diffusive mass flux. It conserves the corresponding stored
momenta across internal faces and avoids changing density while leaving momentum stale. It is not
yet a complete physical derivation of turbulent dust momentum diffusion; see Section 11.

After solving density, the code reconstructs the time-centered integrated mass flux

$$
\mathcal F_{\rho,i+1/2}^{n+1/2}
=-\frac{A_{i+1/2}D_{i+1/2}}{2\delta l_{i+1/2}}
\left[(\rho_{i+1}^{n}-\rho_i^{n})
+(\rho_{i+1}^{n+1}-\rho_i^{n+1})\right].
$$

For each stored primitive $u_a\in\{\ell_\phi,v_r,\ell_\theta\}$, the associated momentum flux is

```math
\mathcal F_{m_a,i+1/2}
=\mathcal F_{\rho,i+1/2}u_{a,\rm donor},
\qquad
u_{a,\rm donor}=
\left\{\begin{array}{ll}
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

The same face flux enters its two cells with opposite signs, so internal diffusive transfers
conserve both mass and every stored momentum exactly up to roundoff.

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
order of the remaining operators.

The inactive polar operators are skipped. Primitive velocity is recovered after every advection
launch and after each three-direction diffusion group; the directional diffusion solves operate
directly on density and conserved momenta and do not consume primitive velocity between their
launches. Each directional advection half-interval is independently subcycled. Before every
advection launch, the global CFL rate is recomputed from the current synchronized state, so
acceleration or an earlier advection sweep cannot leave a later sweep using stale velocities.

For a requested directional interval $h$, the driver repeatedly chooses

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
drag can transfer that velocity to the dust. Vacuum cells contribute zero, while any nonfinite
state makes its complete azimuthal ring contribute an infinite rate; the host reduction reports
the first such cell and aborts the run before another operator is launched. For finite states, the
host reduction uses

$$
\Delta t=\min\left(
\frac{\mathrm{CFL\_DYN}}{\max_{ijk}\lambda_{ijk}},
\mathrm{DT\_MAX}
\right),
$$

with the interval shortened further whenever necessary to land exactly on an output time.

`CFL_DYN <= 0.5` is required by the nearest-integer FARGO shift and enforced at compile time.
`DT_MAX` supplies an independent ceiling. Diffusion is not placed in this explicit CFL bound
because it is solved implicitly with positivity subcycling.

### 8.3 Consolidated boundary conditions

The continuum boundary conditions paired with the operators are

| Boundary | Transport | Diffusion |
|---|---|---|
| azimuthal | periodic | periodic |
| inner and outer radial | outflow only | zero normal flux |
| full-disk polar | outflow only | zero normal flux |
| `HALF_DISK` midplane | reflecting | zero normal flux |

Periodicity identifies

$$
\boldsymbol U(X_{\min})=\boldsymbol U(X_{\max}).
$$

The zero-flux diffusion condition is

$$
\boldsymbol n\cdot\boldsymbol D\nabla\varrho_d=0,
$$

while the outflow transport condition retains only a normal characteristic directed out of the
domain, as written in Section 5.3. These choices intentionally differ: an advected representative
of the continuum may leave the modeled disk, whereas turbulent diffusion is confined by a
reflecting numerical wall. Optical depth additionally assumes no unresolved material interior to
the radial domain,

$$
\tau(Y_{\min})=0.
$$

### 8.4 Formal accuracy and error sources

For a smooth solution, a mesh scale $h$, and timestep $\Delta t$, the intended deterministic
truncation form is

$$
\|e\|\lesssim C_xh^p+C_t\Delta t^2+C_{\rm split}\Delta t^2,
$$

where PPM is nominally third order in smooth one-dimensional regions but the complete split solver
is conservatively described as second order. The principal operator properties are

| Operator | Smooth-region property | Important reduction |
|---|---|---|
| azimuthal FARGO–PPM | exact integer shift plus high-order residual reconstruction | PPM limiting reduces order near extrema or sharp fronts |
| radial/polar PPM + SSPRK(3,3) | third-order method-of-lines time integrator for an isolated sweep | multidimensional Strang composition limits the global claim to second order |
| drag/source response | exact frozen linear drag and second-order endpoint force weighting | coefficient freezing and sequential nonlinear force evaluation supply the remaining error |
| Crank–Nicolson diffusion | second order in time and centered finite-volume space | positivity subcycling changes step partition but not the solved operator |
| optical-depth quadrature | exact cell integral for cellwise constant extinction and linear face interpolation | density discretization and radial interpolation set the error |

At a limiter activation, vacuum reset, or outflow boundary, the local order can fall to first order.
After pressureless caustic formation, refinement need not converge to a single-valued continuum
solution because the physical closure itself has failed; that is distinct from a discretization
error.

## 9. GPU implementation and computational characteristics

The main numerical components map to the production source as follows:

| Scientific operation | Principal implementation |
|---|---|
| convolved profile and PPM geometry setup | `inc/{cuda,rocm}/fluid/fluid_host.cuh` |
| density and velocity initialization | `src/{cuda,rocm}/fluid/init_rho_calc.*`, `src/comm/fluid/init_vel_calc.cu` |
| conservative transport | `src/comm/fluid/advection_[xyz]{th,bl}.cu` |
| drag, gravity, geometry, and radiation | `src/comm/fluid/source_update.cu` |
| density and donor-momentum diffusion | `src/comm/fluid/diffusion_[xyz]{th,bl}.cu` |
| optical-depth increment and prefix sum | `src/comm/fluid/optdepth_calc.cu`, `src/comm/fluid/optdepth_csum.cu` |
| primitive/conserved conversion | `src/comm/fluid/momentum_getv.cu`, `src/comm/fluid/momentum_setv.cu` |
| operator driver and output clock | `src/{cuda,rocm}/fluid/fluid_runtime.*` |

The same PPM/HLL and Crank–Nicolson discretizations have two compile-time GPU implementations:

- `FLUID_SWEEP := thread` selects `advection_[xyz]th` and `diffusion_[xyz]th`, assigning one GPU
  thread to each complete directional line
- `FLUID_SWEEP := block` selects `advection_[xyz]bl` and `diffusion_[xyz]bl`, assigning one GPU
  block to each line and distributing cellwise work among its threads

The thread implementation is the reference transcription and stores its line work in thread-local
arrays. In particular, `advection_xth` contains 21 arrays of length `N_X`, a source-level footprint
of approximately 168 KiB (172 kB) per thread at `N_X = 1024` if all arrays remain distinct.
Compiler lifetime reuse can reduce that footprint, so `ptxas` resource reports and profiler
local-memory traffic are the authoritative measures.

The block implementation replaces those advection arrays with 12 persistent full-grid workspace
fields and places four azimuthal or six radial/polar diffusion work arrays in dynamic shared
memory. Eleven advection fields hold the conserved state, recovered primitives, and antidiffusive
fluxes; the twelfth stages the low-order density update so cells can advance cooperatively without
reading density already overwritten by another thread. Cellwise reconstruction, low-order update,
and SSPRK combinations are cooperative, while the face-ordered invariant-domain correction and
Thomas or Sherman–Morrison recurrence remain serial within each line to preserve the verified
numerical ordering. This is an implementation and memory-layout choice, not a different numerical
method.

The backend does not change the discrete equations. Ordinary elementwise and thread-line kernels
use `TPB = 64` on both backends, while block-line kernels use 32 threads on CUDA and one 64-lane
wavefront on ROCm. CUDA defaults to `--use_fast_math` and offers `CUDA_MATH=precise`; ROCm omits
blanket `-ffast-math` because its finite-only assumptions can invalidate nonfinite-state checks.
Cross-backend validation therefore compares convergence, conserved fields, and density-weighted
velocity norms rather than requiring byte equality.

On ROCm, dynamic shared memory is AMD LDS. The host checks both the device LDS capacity and the
kernel's static-plus-dynamic allocation before launching a block diffusion line. For double
precision the directional dynamic requests are the values below; a six-array line of length 1366
already exceeds a 64-KiB limit. The thread-line azimuthal advection stack is also too large for the
validated gfx942 linker at `N_X = 1024`, making the block sweep the supported large-2D ROCm path.
These are hardware resource constraints, not changes to the numerical operator.

Performance is grid dependent, so the intended production grid should be benchmarked before
selecting a default; neither implementation is universally preferred. The matched comparison
procedure and the status of its archived evidence are recorded in `fluid_testset.md`.

For directional sweeps, the independent line counts are

$$
N_{{\rm line},x}=N_YN_Z,
\qquad
N_{{\rm line},y}=N_XN_Z,
\qquad
N_{{\rm line},z}=N_XN_Y.
$$

The thread implementation assigns one thread to each line, whereas the block implementation
assigns one block to each line. Its persistent advection workspace is

$$
M_{\rm adv,work}=12N_G\,\mathrm{sizeof}(\mathtt{real}).
$$

For double precision this is 96 MiB at $1024^2$ and 192 MiB at $128^3$. The extra staged-density
field costs one additional full-grid array relative to the former serial low-order block update.

The dynamic shared-memory requirements per diffusion block are

$$
M_{{\rm sh},x}=4N_X\,\mathrm{sizeof}(\mathtt{real}),
\qquad
M_{{\rm sh},y}=6N_Y\,\mathrm{sizeof}(\mathtt{real}),
\qquad
M_{{\rm sh},z}=6N_Z\,\mathrm{sizeof}(\mathtt{real}).
$$

The density, three primitive fields, three conserved momenta, and CFL-rate field use eight
full-grid `real` arrays before optional optical depth and block workspace are counted. Nonuniform
PPM weights are computed once on the host and uploaded, so geometry setup is not repeated inside
every transport step.

Each production model is a compile-time specialization. A model-local `const_defs.cuh`, when
present, supplies its grid and physical constants; `flags.mk` selects optional operators, and the
Makefile chooses the thread or block sweep. These are not runtime parameters: changing one requires
recompilation, while `variables.txt` records the resulting configuration for analysis.

## 10. Output and restart semantics

The driver clips accepted steps to land exactly on each `DT_OUT` boundary. It writes density and
physical linear velocity, leaving the internal angular-momentum primitives unchanged on the GPU.
On restart, the saved linear velocities are converted back to $(\ell_\phi,v_r,\ell_\theta)$,
conserved momenta are rebuilt from the loaded density, and optical depth is reconstructed when
radiation is active. Nonfinite density, momentum, primitive, and optical-depth values are checked
at the documented synchronization points.

The file conversion is exactly

$$
v_\phi=\frac{\ell_\phi}{R},
\qquad
v_\theta=\frac{\ell_\theta}{r},
$$

on output and

$$
\ell_\phi=Rv_\phi,
\qquad
\ell_\theta=rv_\theta
$$

on input. These transformations are performed in host memory, so diagnostic I/O never temporarily
overwrites the angular variables being evolved on the GPU. After loading,
$m_a=\varrho_du_a$ reconstructs all conserved arrays before the next operator is launched.
Because floating-point evaluation of $R(\ell_\phi/R)$ and $r(\ell_\theta/r)$ need not reproduce
the original angular momentum bit for bit, restart validation must use explicit state and
conservation tolerances rather than require byte-identical evolved fields.

At frame $n$, the physical output time is

$$
t_n=n\,\mathrm{DT\_OUT}.
$$

The driver shortens the last accepted step in each interval so $t=t_n$ to floating-point
precision; output interpolation is therefore unnecessary. Every frame contains `dustdens`,
`dustvelx`, `dustvely`, and `dustvelz`, each with $N_G$ double-precision values; `optdepth` is added
when radiation is enabled. `variables.txt` records the active grid, physical parameters, and
timestep controls needed to interpret those arrays.

## 11. Known limitations

Verification definitions, evidence, and untested regimes are maintained in
[`fluid_testset.md`](fluid_testset.md). The limitations below concern the physical or numerical
model itself rather than current test coverage.

- The diffusion-momentum closure is provisional. A complete density-diffusion momentum equation
  should be derived in spherical coordinates, including its tensor and geometric terms, before
  clumping claims rely on momentum transport by diffusion. The formulation of
  [Huang & Bai (2022)](https://arxiv.org/abs/2206.01023) provides relevant conservative structure
  but cannot be copied directly because it diffuses concentration rather than the selected density.
- The 3D initializer balances diffusion only to discretization error and can produce a small
  initial polar transient.
- The block diffusion kernels still execute each Thomas or Sherman–Morrison recurrence serially
  within its line, and the block advection kernels retain a serial face-ordered correction pass.
  Further parallelization should use a conservation- and positivity-preserving face-budget limiter
  and a batched Thomas/PCR-style tridiagonal solve, with the full analytical suite and profiler
  evidence required after each change; an FFT is not required.
- Pressureless dust cannot represent multistreaming after caustic formation. A swarm or another
  kinetic representation is required in that regime.
- `--use_fast_math` trades correctly rounded division/square root and subnormal handling for speed.
  Verification tolerances and reproducibility claims must reflect that build choice.

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
