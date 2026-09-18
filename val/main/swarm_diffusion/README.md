# Radial diffusion ring pilot

One model, `ring_pilot`, isolates the production radial diffusion operator.
It is a pilot for selecting a later convergence matrix, not a complete accuracy campaign.

| Setting | Value |
|---|---|
| Particles | 1,048,576, equal represented mass |
| Initial distribution | ln(R) is normal with mean 0 and standard deviation 0.05 |
| Diffusivity | D_R = 1e-4 R^2 |
| Domain | exp(-3) < R < exp(3), axisymmetric midplane |
| Mesh | N_X=1, N_Y=128, N_Z=1 |
| Nominal timestep | 1 |
| Duration | 1,000, or A*t=0.1 |
| Output | initial state plus ten intervals of 100 |
| Initialization seed | 17, independent per-particle streams |
| Evolution seed | production default 1 |

ALPHA=0.04, ASPR_0=0.05, IDX_Q=0.5, and SCHMIDT_R=1 give A=1e-4.
STOKES_0=0 selects the tracer limit, preserving this analytical coefficient after
root diffusion gained the factor 1/(1+St²). Dynamics are disabled in this model.
The initial velocities are zero. A model-local no-op ssa_transport.cu disables
deterministic motion; dyn_rate_calc.cu fixes the timestep. The production runtime
still makes two diffusion calls per step, each of duration 0.5. The diffusion
and evolution RNG kernels are unchanged. A local initializer draws the ring using
the backend's RNG library; CUDA and ROCm draws need not match pathwise.

The normal host initialization is still called before the model-local initializer;
its sampled positions are overwritten. Its total represented mass sets an overall
normalization only. The analysis compares surface density divided by total mass.

## ROCm run

From the repository root:

```bash
make -C val/main/swarm_diffusion -j8 MODEL=ring_pilot \
  GPU_BACKEND=rocm GPU_TARGET=gfx942
./val/temp/swarm_diffusion/bin/ring_pilot/gamedev
```

Use your GPU target. CUDA routing also exists via GPU_BACKEND=cuda and a matching
GPU_TARGET. Objects and executables live under val/temp/swarm_diffusion/{obj,bin}.
Results live under val/main/swarm_diffusion/outputs/ring_pilot/.
Each particle snapshot is 48 MiB; the runtime additionally saves RNG checkpoints.
Copy variables.txt and particle_00000.dat through particle_00010.dat for analysis.
RNG files are needed only for restart, not analysis.

## Analytical comparison

For u=ln(R), the exact mean is 2At and variance is 0.05^2+2At.
The reference is the unbounded-domain solution of density diffusion with D_R=AR^2,
including both D_R/R and dD_R/dR drift terms. It is not dust-to-gas-ratio diffusion.
At the final time the exact logarithmic mean is 0.2 and variance is 0.2025.

With Python and NumPy:

```bash
python3 val/main/swarm_diffusion/analyze.py
```

An optional positional argument selects a downloaded output directory.
The script requires all eleven snapshots and writes analysis.json in that directory:
CDF errors, logarithmic moments and sampling standard errors, velocity residuals,
radial extrema, and simulated/analytical annular-bin surface densities per unit mass.
It checks snapshot shape, finite active particles, and pilot parameters. It does
not impose a scientific pass threshold or generate a plot. Missing outputs are errors.

The finite boundaries reflect. The wide domain is chosen to make encounters rare,
but saved extrema cannot exclude crossings between outputs. Boundary effects must
remain below the errors being interpreted. A plateau at this particle count is
not by itself evidence of timestep error; independent seeds and timestep/particle
refinement are still needed. Pilot values and initial width are fixed in this
analysis script; update the reference deliberately if the physical setup changes.

## Verification

```bash
python3 val/main/swarm_diffusion/test_reference.py
```

This checks normalization, moments, the cylindrical diffusion equation numerically,
and analysis of synthetic binary snapshots. It is not GPU execution. CUDA/ROCm
Makefile routing was checked; native compilation, initialization statistics, and
numerical diffusion accuracy remain unverified locally.
