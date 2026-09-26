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
make -C val/paper/ecc_orbit -j8 MODEL=orbit_dt_p80 \
  GPU_BACKEND=cuda GPU_TARGET=sm_80
./val/paper/ecc_orbit/obj/orbit_dt_p80/cuda/gamedev
```

Use your actual GPU architecture. For ROCm use GPU_BACKEND=rocm and the matching
GPU_TARGET. All four models can be run sequentially:

```bash
(
  set -e
  for model_dir in val/paper/ecc_orbit/mod/orbit_dt_p*; do
    model_name="${model_dir##*/}"
    make -C val/paper/ecc_orbit -j8 MODEL="$model_name" \
      GPU_BACKEND=cuda GPU_TARGET=sm_80
    "val/paper/ecc_orbit/obj/$model_name/cuda/gamedev"
  done
)
```

Outputs go to `out/<model>/<backend>/`; build products go to `obj/<model>/`.
Both are ignored by Git. The root Makefile is reused without modification.

## Scope of the overrides

Shared constants prescribe the test. particle_init.cu sets the orbit, and
dyn_rate_calc.cu imposes a fixed step rather than the production CFL policy.
The local `src/_transport.cuh` calls the current production `_ssa_advance<true>`
helper for the exact zero-drag limit. It contains no copied integration stages.
This tests the zero-drag SSA specialization; finite-drag accuracy is a separate test.
The production runtime, ssa_transport kernel, and boundaries remain in use.

Compare the trajectories with the exact Kepler solution and monitor energy
E=-1/2 and angular momentum sqrt(3)/2. Energy error alone does not measure phase
accuracy. For the analytical trajectory, solve u - e*sin(u) = 2*pi*t/P for the eccentric
anomaly u, then use Cartesian X=a*(cos(u)-e), Y=a*sqrt(1-e*e)*sin(u).
The exact specific energy is -GM/(2*a). Compare at matching output times.
No postprocessing or reference solver is included in this model directory.


## Source and artifacts

`mod/` selects model parameters; `src/` holds only test-specific overrides.
Builds reuse root source and write executables to `obj/<model>/<backend>/gamedev`
and results to `out/<model>/<backend>/`. Historical outputs, if retained in `val_stale/`, are not evidence for the current source.
CUDA/ROCm build routing is checked locally; fresh native GPU runs are required
before claiming accuracy for the current source.
