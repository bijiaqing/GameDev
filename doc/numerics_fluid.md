# Eulerian dust-fluid numerics

## Scope

The current Eulerian implementation is under `inc/fluid/` and `src/fluid/`. It supports:

- a two-dimensional radial–azimuthal disk with `N_Z == 1`
- a full three-dimensional spherical grid with `N_Z > 1`
- pressureless dust transport
- gas drag, gravity, spherical geometric forces, and optional radiation pressure
- optional turbulent diffusion of dust density
- optional viscous gas accretion when diffusion is enabled

Radial–polar two-dimensional models are not supported. The azimuthal dimension is always active.
The fluid is monodisperse and does not back-react on the prescribed gas.

## Coordinates, grid, and state

The computational coordinates are

$$
x=\phi,\qquad y=r,\qquad z=\theta,
$$

and

$$
R=y\sin z,\qquad Z=y\cos z.
$$

The mesh is uniform in $x$ and $z$ and logarithmic in $y$. Its radial finite-volume measure is

$$
\Delta V_y=\frac{y_o^d-y_i^d}{d},
\qquad
d=
\begin{cases}
2,&N_Z=1,\\
3,&N_Z>1.
\end{cases}
$$

The polar measure is $\Delta V_z=\cos z_i-\cos z_o$ when the polar dimension is active.
Consequently, `N_Z == 1` represents a vertically integrated disk and evolves dust surface density
$\Sigma_d$; `N_Z > 1` evolves volume density $\rho_d$.

The primitive arrays named `dustvelx/y/z` contain

$$
(\ell_\phi,v_r,\ell_\theta)
=
(Rv_\phi,v_r,rv_\theta),
$$

not three linear Cartesian or spherical velocities. The conserved arrays contain

$$
(\rho_d,\rho_d\ell_\phi,\rho_dv_r,\rho_d\ell_\theta).
$$

For a 2D disk, $\rho_d$ in these expressions means the evolved surface density. File output
converts $\ell_\phi$ and $\ell_\theta$ to $v_\phi$ and $v_\theta$ without changing the device state.

## Prescribed gas and initialization

The analytic gas surface density and aspect ratio are

$$
\Sigma_g(R)=\Sigma_0\left(\frac{R}{R_0}\right)^p,
\qquad
h_g(R)=h_0\left(\frac{R}{R_0}\right)^{(q+1)/2}.
$$

The vertically isothermal point-mass hydrostatic stratification is evaluated exactly as

$$
\frac{\rho_g(R,Z)}{\rho_g(R,0)}
=\exp\left[
\frac{R/\sqrt{R^2+Z^2}-1}{h_g^2}
\right].
$$

The monodisperse Stokes number follows

$$
\mathrm{St}(R,Z)
=\mathrm{St}_0
\left(\frac{R}{R_0}\right)^{-p}
\left[\frac{\rho_g(R,Z)}{\rho_g(R,0)}\right]^{-1}.
$$

The pre-convolution dust surface profile is

$$
\Sigma_d(R)=Z_{\rm metal}\Sigma_g(R).
$$

The convolution is evaluated on a temporary uniform cylindrical-radius axis. In 3D its lower bound
is the smallest cylindrical radius covered by the spherical domain,

$$
R_{\min,\mathrm{init}}
=
Y_{\min}\min[\sin(Z_{\min}),\sin(Z_{\max})],
$$

rather than $Y_{\min}$. This prevents high-latitude cells with $R<Y_{\min}$ from being clipped
to an unrelated radial profile value. In 2D, $R_{\min,\mathrm{init}}=Y_{\min}$.

The edge convolution is not renormalized. In 2D the convolved surface density is evolved directly
and its unresolved vertical profile is assumed to be well mixed with the gas. In 3D it is embedded
in a Gaussian dust layer using the density-diffusion scale height

$$
H_d=H_g\sqrt{\frac{\alpha_z}{\mathrm{St}_{\rm mid}}},
\qquad
\alpha_z=\frac{\alpha}{\mathrm{Sc}_z}.
$$

This is the equilibrium implied by the selected internal density-diffusion equation, not the
standard gas-concentration diffusion closure. Initialization adds one reproducible Gaussian
amplitude per azimuthal column, so the same perturbation multiplies every radial and polar cell at
fixed $x$.

The initial velocity is the local no-backreaction drift relative to the pressure-supported gas.
If `VISC_FLOW` is enabled, the prescribed cylindrical gas radial velocity follows the
Kanagawa et al. steady viscous expression. In 3D, the polar primitive also includes the velocity
needed to approximately balance the initialized polar diffusive flux.

## Pressureless finite-volume transport

Density and all three momenta are transported conservatively. Each directional operator uses:

1. geometry-aware PPM reconstruction
2. a pressureless HLL interface flux
3. a first-order cell-centred HLL flux as a robust invariant-domain base
4. one conservative face coefficient that limits the high-minus-low correction

The uniform azimuthal reconstruction uses the bounded four-cell Colella–Woodward face value. On
the logarithmic radial mesh and spherical polar mesh, face weights are precomputed from exact cubic
moment constraints in the finite-volume coordinates

$$
s_y=\frac{y^d}{d},
\qquad
s_z=-\cos z.
$$

Boundary-adjacent internal faces use a two-cell linear interpolation where a complete four-cell
stencil is unavailable. The pressureless HLL flux retains its proper left-going, right-going, and
two-wave branches.

The limiter enforces nonnegative density and local bounds on every momentum-to-density ratio. The
same limited face flux enters neighboring cells with opposite signs, so the normal update remains
conservative. As a final floating-point safety fallback, any cell whose updated density is still
negative is converted to an exact vacuum by setting its density and all three momentum components
to zero. This fallback should remain inactive when the CFL and invariant-domain assumptions hold;
its purpose is to prevent a residual negative density from producing an undefined vacuum velocity,
not to replace the conservative limiter.

Azimuthal transport uses FARGO orbital advection. For each ring, the arithmetic mean
$\ell_\phi$ defines a nearest-integer periodic shift. PPM transports the residual, including the
fractional part of the ring-mean displacement. The CFL calculation bounds each cell's displacement
relative to the ring mean by `CFL_DYN`. The nearest-integer shift frame can differ from that mean by
at most half a cell per step, so the actual PPM tracing fraction is bounded by
`CFL_DYN + 0.5 <= 1` rather than by `CFL_DYN` alone.

The radial and polar method-of-lines operators use the three-stage Shu–Osher SSPRK(3,3) method.
PPM provides spatial reconstruction; SSPRK supplies temporal integration. The radial boundaries
are outflow-only. A full polar disk is outflow-only at both polar edges; `HALFDISK` reflects at the
midplane and remains outflow-only at its other polar edge.

PPM is formally high order on smooth fields, but the complete multidimensional solver should be
described as second-order accurate: Strang composition, Crank–Nicolson diffusion, boundary fluxes,
and limiter activation set the global claim. SSPRK(3,3) should not be used to claim that the full
scheme is third-order in time.

## Source operator

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

Radiation uses

$$
\beta(t,\tau)
=\beta_0 f_\beta(t)e^{-\tau},
\qquad
f_\beta=s^2(3-2s),
\qquad
s=\operatorname{clip}(t/T_\beta,0,1).
$$

Optical depth is reconstructed at the source midpoint. Cumulative values live at radial outer
faces and are interpolated to logarithmic cell centers with

$$
\tau_c=\tau_i+\frac{\tau_o-\tau_i}{\sqrt{\Delta y}+1},
$$

which is exact when optical depth is linear in physical radius within the logarithmic cell. In 2D,
the well-mixed closure converts surface density to midplane extinction density using
$\sqrt{2\pi}H_g$, without a Stokes-dependent scale height.

## Density diffusion

The selected equation is

$$
\frac{\partial\rho_d}{\partial t}
=\nabla\cdot(\boldsymbol D\nabla\rho_d),
$$

with $\rho_d$ interpreted as $\Sigma_d$ in 2D. The code deliberately does not diffuse
$\rho_d/\rho_g$.

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

Each direction uses a second-order finite-volume Crank–Nicolson line solve. The periodic
azimuthal system is reduced with Sherman–Morrison; radial and polar systems use zero diffusive
flux at their physical boundaries. Crank–Nicolson is linearly stable but not unconditionally
positive, so each kernel subcycles according to `POS_LIMIT`.

The current minimum conservative momentum correction transports donor values of
$(\ell_\phi,v_r,\ell_\theta)$ with the diffusive mass flux. It conserves the corresponding stored
momenta across internal faces and avoids changing density while leaving momentum stale. It is not
yet a complete physical derivation of turbulent dust momentum diffusion; see “Open numerical
work” below.

## Global composition and timestep

One accepted step is the palindromic composition

$$
D_y^{1/2}D_x^{1/2}D_z^{1/2}
A_x^{1/2}A_y^{1/2}A_z^{1/2}
S
A_z^{1/2}A_y^{1/2}A_x^{1/2}
D_z^{1/2}D_x^{1/2}D_y^{1/2}.
$$

The inactive polar operators are skipped. Velocity is recovered after every conservative operator.
Each directional advection half-interval is independently subcycled. Before every launch, the
global CFL rate is recomputed from the current state, so acceleration or an earlier sweep cannot
leave a later sweep using stale velocities.

The transport rate includes:

$$
\frac{|\Omega-\bar\Omega|}{\Delta x},
\qquad
|v_r|\frac{A_y}{V_y},
\qquad
\frac{|v_\theta|}{r}\frac{\max(\sin z_i,\sin z_o)}{\Delta(-\cos z)}.
$$

`CFL_DYN <= 0.5` is required by the nearest-integer FARGO shift and enforced at compile time.
`DT_MAX` supplies an independent ceiling. Diffusion is not placed in this explicit CFL bound
because it is solved implicitly with positivity subcycling.

## Directional CUDA implementations

The same PPM/HLL and Crank–Nicolson discretizations have two compile-time CUDA implementations:

- `FLUID_SWEEP := thread` selects `advection_[xyz]th` and `diffusion_[xyz]th`, assigning one CUDA
  thread to each complete directional line
- `FLUID_SWEEP := block` selects `advection_[xyz]bl` and `diffusion_[xyz]bl`, assigning one CUDA
  block to each line and distributing cellwise work among its threads

The thread implementation is the reference transcription and stores its line work in thread-local
arrays. In particular, `advection_xth` contains 21 arrays of length `N_X`, a source-level footprint
of approximately 172 KiB per thread at `N_X = 1024` if all arrays remain distinct. Compiler lifetime
reuse can reduce that footprint, so `ptxas` resource reports and profiler local-memory traffic are
the authoritative measures.

The block implementation replaces those advection arrays with 11 persistent full-grid workspace
fields and places four azimuthal or six radial/polar diffusion work arrays in dynamic shared
memory. Cellwise reconstruction and flux work are cooperative, while the face-ordered
invariant-domain correction and Thomas or Sherman–Morrison recurrence remain serial within each
line to preserve the verified numerical ordering. This is an implementation and memory-layout
choice, not a different numerical method.

Performance is grid dependent. Historical development comparisons found the block method faster for a
`1024^2` transport model but slower for a `128^3` diffusion model. The intended production grid
should therefore be benchmarked before selecting a default; neither implementation is universally
preferred. The matched comparison procedure and the status of its archived evidence are recorded
in `testset_fluid.md`.

## Output and restart state

The driver clips accepted steps to land exactly on each `DT_OUT` boundary. It writes density and
physical linear velocity, leaving the internal angular-momentum primitives unchanged on the GPU.
On restart, the saved linear velocities are converted back to $(\ell_\phi,v_r,\ell_\theta)$,
conserved momenta are rebuilt from the loaded density, and optical depth is reconstructed when
radiation is active. Nonfinite density, momentum, primitive, and optical-depth values are checked
at the documented synchronization points.

## Open numerical work

- The diffusion-momentum closure is provisional. A complete density-diffusion momentum equation
  should be derived in spherical coordinates, including its tensor and geometric terms, before
  clumping claims rely on momentum transport by diffusion. The Huang–Bai formulation is relevant
  structure but cannot be copied directly because it diffuses concentration rather than the
  selected density.
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

## References

- Colella & Woodward (1984), [PPM](<https://doi.org/10.1016/0021-9991(84)90143-8>)
- Harten, Lax & van Leer (1983), [HLL flux](https://doi.org/10.1137/1025002)
- Masset (2000), [FARGO](https://arxiv.org/abs/astro-ph/9910390)
- Crank & Nicolson (1947), [implicit trapezoidal diffusion](https://doi.org/10.1017/S0305004100023197)
- Higueras & Roldán (2023), [Crank–Nicolson positivity](https://arxiv.org/abs/2301.01066)
- Strang (1968), [symmetric operator splitting](https://doi.org/10.1137/0705041)
- Huang & Bai (2022), [multifluid dust algorithms](https://arxiv.org/abs/2206.01023)
- Nakagawa, Sekiya & Hayashi (1986), [steady dust–gas drift](<https://doi.org/10.1016/0019-1035(86)90121-1>)
- Kanagawa et al. (2017), [viscous disk velocity](https://arxiv.org/abs/1706.08975)
