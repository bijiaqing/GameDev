# Full-3D MMS preparation status

This directory is intentionally **not a buildable MODEL yet**. It records the remaining work needed
for the four full-3D manufactured-solution cases in `.github/rpi_fluid_convergence_tests.md`.
There is no `flags.mk`, so the Makefile will reject accidental use rather than run a non-MMS disk
and label it as verification.

## Required generated artifacts

- `derive_mms.py`: a SymPy derivation of the selected rho, q, v_y, v_z, and v_x fields; spherical
  continuity residual; all three stored-momentum residuals; diffusion fluxes; drag, gravity,
  centrifugal, and polar-torque terms; and finite-opacity radiation.
- `mms_residuals.cuh`: generated algebraic expressions only. It must not call production geometry,
  gas, force, or optical-depth helpers when constructing the reference.
- `f_mms_source.cu`: test-only density and momentum forcing evaluated at the required substep times.
- `fluid_main.cu`: model-local symmetric forcing composition and exact manufactured boundary
  updates. A forward-Euler density source is not acceptable.
- `validate.py`: independent high-order finite-volume averages, L1/L2/Linf errors for density,
  ratio, all momenta and physical velocities, and the exact continuum optical depth.

## Four required flag configurations

1. transport only: diffusion coefficients zero, beta zero;
2. transport plus diffusion: nonzero tensor D, beta zero;
3. transport plus radiation: D zero, beta nonzero;
4. all physics: nonzero D and beta.

Because production requires `DIFFUSION` whenever `N_Z>1`, configurations 1 and 3 must compile with
`DIFFUSION` enabled while the manufactured diffusivities are exactly zero. All four must refine
N_X, N_Y, and N_Z together and use the same prescribed timestep history at a given resolution.

## Acceptance prerequisites

Before interpreting an MMS rate, independently verify the forcing harness on a source-only field
whose exact integral is known. The exact solution must remain above `RHO_VAC`, and the limiter and
positivity scaling must remain inactive. Radiation rows have a second-order floor from the
production optical-depth quadrature. Both `HALFDISK` and full-disk polar boundary variants should
eventually be exercised.
