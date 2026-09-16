# Particle orbital-motion test

Four single-particle, planar eccentric-orbit models: GM=a=1, e=0.5, P=2*pi.
The initial state is pericentre (R=0.5, phi=0), v_R=0, and stored
azimuthal angular momentum sqrt(3)/2. The full-azimuth domain spans 0.3<R<2.
Radiation, diffusion, collisions, and gas drag are absent.

| Model | Timestep | Duration | Nominal steps |
|---|---|---|---:|
| orbit_dt_p80 | P/80 | 100P | 8,000 |
| orbit_dt_p160 | P/160 | 100P | 16,000 |
| orbit_dt_p320 | P/320 | 100P | 32,000 |
| orbit_dt_p640 | P/640 | 100P | 64,000 |

Each run saves the initial state plus 2,000 snapshots separated by P/20.
The last is particle_02000.dat. Output intervals contain an integer number
of nominal steps, although floating-point accumulation may cause tiny remainder
steps in the production runtime.

## Build and run

From the repository root, on a CUDA machine:

```bash
make -C val/main/swarm_orbit -j8 MODEL=orbit_dt_p80 \
  GPU_BACKEND=cuda GPU_TARGET=sm_80 CUDA_MATH=precise
./val/main/swarm_orbit/models/orbit_dt_p80/gamedev
```

Use your actual GPU architecture. For ROCm use GPU_BACKEND=rocm and the matching
GPU_TARGET. All four models can be run sequentially:

```bash
(
  set -e
  for model_dir in val/main/swarm_orbit/models/orbit_dt_p*; do
    model_name="${model_dir##*/}"
    make -C val/main/swarm_orbit -j8 MODEL="$model_name" \
      GPU_BACKEND=cuda GPU_TARGET=sm_80 CUDA_MATH=precise
    "$model_dir/gamedev"
  done
)
```

Outputs go to val/main/swarm_orbit/outputs/<model>/; intermediates go to build/.
Both are ignored by Git. The root Makefile is reused without modification.

## Scope of the overrides

Shared constants prescribe the test. particle_init.cu sets the orbit, and
dyn_rate_calc.cu imposes a fixed step rather than the production CFL policy.
The local _transport.cuh includes the existing validation zero-drag specialization
in val/comm/swarm/test_orbit_ecc_2d/_transport.cuh. That specialization imports
the production force and drift helpers. Keep that dependency when transferring
the suite. This tests the zero-drag SSA specialization, not the finite-drag branch.
The production runtime, ssa_transport kernel, and boundaries remain in use.

Compare the trajectories with the exact Kepler solution and monitor energy
E=-1/2 and angular momentum sqrt(3)/2. Energy error alone does not measure phase
accuracy. The reference equations and paper-facing design are in Section 2 of
doc/publication/numerical_tests.md. No postprocessing or reference solver has
been added here.

## Local setup check

With a working C++ compiler, from the repository root:

```bash
c++ -std=c++17 -DTRANSPORT \
  -I val/main/swarm_orbit/models/orbit_dt_p80 -I inc/cuda/swarm \
  val/main/swarm_orbit/common/check.cpp -o /tmp/orbit-check
/tmp/orbit-check
```

This checks initialization, timing, and the fixed-rate kernel on the host.
It does not run the transport update or qualify GPU accuracy.
