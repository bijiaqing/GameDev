# Particle radial-drift models

These 35 standalone models comprise the 28-point test matrix and seven fine-step
reference runs based on the parameter choices of Fung & Muley
(2019), Section 3.3 / Figure 4: https://arxiv.org/pdf/1909.02006.
They measure this code's accuracy, not agreement with digitized published results.

- Stokes numbers: 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000.
- Test timesteps: 1, 0.1, 0.01, 0.001 in inverse initial Keplerian frequency.
- Reference timestep: 1e-5 for each of the seven Stokes numbers.
- One particle per model; GM = initial radius = initial Keplerian frequency = 1.
- End time: 10 max(1, St). Ten equally spaced output intervals, plus the initial state.
- Every output interval is an integer multiple of the nominal timestep. The runtime
  may take a tiny final remainder step because it accumulates time in floating point.
- No collisions, diffusion, radiation, imported gas, or viscous radial flow.

## Run

From the repository root, for example:

```sh
make -C val/paper/swarm_drift MODEL=swarm_drift_st1em3_dt1e0 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/swarm_drift/obj/swarm_drift_st1em3_dt1e0/rocm/gamedev
```

The example uses ROCm gfx942; select the target for your GPU.
Names encode the Stokes number and timestep; `1em3` means 1e-3, `1e3` means 1e3.
Each model writes to its own `val/paper/swarm_drift/out/<model>/<backend>/` directory through the production runtime.
Use the same math settings for all convergence and reference runs.

## Model-local changes

The existing `MODEL_PARENT` mechanism shares these files across all 35 models:

- `const_defs.cuh`: common physical configuration and output schedule.
- `particle_init.cu`: initial velocities from Eqs. (42)-(43), as printed, including
  Eq. (42)'s `1 + L/2`. These expressions are approximate, not an exact trajectory.
- `dyn_rate_calc.cu`: fixes the requested timestep instead of applying the production
  CFL controller. The production runtime still clips steps at output boundaries.

The production SSA transport kernels, gas functions, boundaries, and runtime are
unchanged. `CONST_ST` means constant Stokes number, with stopping time St/Omega_K(R).

The gas rotation is matched to v_phi,g/v_K = sqrt(1 - 0.05^2). This code uses
p + q/2 - 3/2 in its midplane pressure-support closure, so p=1, q=-1, h=0.05
matches the test's forces. This differs from the paper's stated surface-density
exponent; with prescribed St and no other processes, the density normalization
and exponent have no further dynamical role. The mesh covers a full azimuth and
0.5 < R < 1.5; it supplies geometry and boundaries, not a gas interpolation error.

## Accuracy reference

The 28 runs alone do not supply an exact ground truth. Figure 4 uses a separate
SSA reference at dt=1e-5 for each St. Those seven reference models are included with names ending in `_dt1em5`.
Read each reference's final radial velocity at output 10 and compare directly
with the other runs at the same St and final time; do not rescale for their
slightly different final radii. These are numerical references, not exact solutions. Eqs. (42)-(43) can instead be used
for an approximate analytical overlay, explicitly labeled as such. A reference
must use identical initial conditions, gas coefficients, and final time.

The local Makefile reuses root build rules without changing them. Models live in
`mod/` and shared files in `src/`. Objects are in
`val/paper/swarm_drift/obj/<model>/`, and executables in
`val/paper/swarm_drift/obj/<model>/<backend>/gamedev`. Outputs and temporary files are ignored.
Previously downloaded output files were moved here unchanged. Rebuild any executable
made before migration: its compiled output path still points to the old location.

## Run the full 35-model set

From the repository root:

```bash
(
  set -e
  for model_dir in val/paper/swarm_drift/mod/swarm_drift_st*; do
    model_name="${model_dir##*/}"
    make -C val/paper/swarm_drift -j8 MODEL="$model_name" \
      GPU_BACKEND=rocm GPU_TARGET=gfx942
    "val/paper/swarm_drift/obj/$model_name/rocm/gamedev"
  done
)
```

To run only the seven new references, use the glob `swarm_drift_st*_dt1em5`.
The full loop also reruns the existing 28 cases and writes to their usual outputs.
Reference nominal step counts, in increasing St order, are
1,000,000; 1,000,000; 1,000,000; 1,000,000; 10,000,000;
100,000,000; and 1,000,000,000. No reference GPU runs have been performed locally.

## Source and artifacts

`mod/` selects model parameters; `src/` holds only test-specific overrides.
Builds reuse root source and write executables to `obj/<model>/<backend>/gamedev`
and results to `out/<model>/<backend>/`. Old downloaded results remain directly
under `out/<model>/` and are not overwritten by new runs.
