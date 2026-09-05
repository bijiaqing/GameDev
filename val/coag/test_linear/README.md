# Linear-kernel coagulation accuracy tests

This is a standalone CUDA swarm validation family for comparing the frozen-bath
collision routine with the continuous Smoluchowski solution for the additive
kernel

```text
K(m_i, m_j) = Lambda_0 * (m_i + m_j).
```

It uses fixed representative-particle positions and coagulation-only evolution.
Each case-directory name records `N_P`, `N_K`, and `COL_BATH_EPS`; all cases
inherit the implementation in `test_common` through `MODEL_PARENT`.

## Campaign grid

The campaign fixes the representative-particle count and evaluates the full
`N_K` by `COL_BATH_EPS` grid:

```text
N_P = 1e6

N_K = 10, 20, 50, 100, 200
COL_BATH_EPS = 5e-3, 1e-2, 2e-2, 4e-2, 8e-2
```

The 25 case directories contain only a `flags.mk`. The first nine runs covered
the two center lines, `COL_BATH_EPS = 1e-2` and `N_K = 200`. The remaining 16
off-axis cases are divided among four approximately equal-runtime batches.
The shared constants accept `SWEEP_N_P`, `SWEEP_N_K`, and
`SWEEP_COL_BATH_EPS` compile definitions.

For example, `1e+6n_2e+2k_8e-2b` means `N_P = 1e6`, `N_K = 200`, and
`COL_BATH_EPS = 0.08`.

## Active configuration

`test_common/common_flags.mk` selects:

- the CUDA swarm representation and KD-tree collision search;
- the default frozen-bath collision integrator;
- `MULTISIZE` and `CODE_UNIT`;
- `COLLISION_UNIT_VOLUME`, so collision rates contain no spatial-volume factor.

No transport, diffusion, radiation, imported gas, opacity output, or
dust-density output is enabled. Fixed positions allow the first KD-tree and
physical-neighbor cache to be reused for the entire run.

The linear campaign differs from the constant campaign in three places:

- `COAG_KERNEL = 1` selects the additive kernel;
- `rand_linear_size` initializes the required mass-weighted exponential
  distribution;
- output timing is linear with `DT_OUT = 1`, producing the requested times
  0, 1, 2, 3, and 4.

All other collision-controller and spatial-test overrides are inherited from
the constant campaign copy.

## Initial mass and representative-particle distribution

The physical number density at time zero is

```text
f(m, 0) = N_0 * exp(-m / m_bar) / m_bar,
```

with `m_bar = 1`. Equal-mass representative swarms must sample the
mass-weighted density

```text
f_R(m, 0) = m * exp(-m),
```

which is a gamma distribution with shape 2 and scale 1. The shared host sampler
therefore uses `std::gamma_distribution<real>(2, 1)` and returns

```text
size = cbrt(mass)
```

because the model-local grain-mass convention is `m = size^3`. This avoids the
Lambert-W branch-point failure possible in the former `rand_4_linear`
implementation while sampling the same distribution.

With `N_Z = 1`, the domain-mass factor is independent of grain size. The common
normalization consequently assigns every representative swarm exactly the same
mass,

```text
M_i = total_dust_mass / N_P,
par_numr_i = M_i / m_i.
```

The unweighted representative-particle mass histogram follows `f_R`; the
physical number histogram must be weighted by `par_numr` to recover `f`.

## Collision normalization

The runtime passes

```text
lambda_0 = N_P / (N_K * total_dust_mass)
```

to the sampled-neighbor kernel. The `N_P/N_K` factor corrects the KNN
subsampling, giving an effective full-ensemble coefficient

```text
Lambda_0 = 1 / total_dust_mass.
```

The analytical dimensionless time is therefore

```text
tau = total_dust_mass * Lambda_0 * t = t.
```

Unlike the constant-kernel campaign, the initial rate depends on owner mass;
it is not intended to equal one for every representative particle.

Relative velocity remains zero for the synthetic kernel. Since `V_FRAG = 1`,
all accepted events use coagulation and fragmentation is inactive.

## Output schedule

`SAVE_MAX = 4`, `DT_OUT = 1`, and `LIN_BASE = 1` retain five particle
snapshots:

| File | Time |
| --- | ---: |
| `particle_00000.dat` | 0 |
| `particle_00001.dat` | 1 |
| `particle_00002.dat` | 2 |
| `particle_00003.dat` | 3 |
| `particle_00004.dat` | 4 |

CUDA RNG states are omitted, so these are analysis snapshots rather than full
restart checkpoints.

## Shared source overrides

- `_collision.cuh`: unit collision-search measure and production synthetic
  kernels.
- `param_phys.cuh`: unit-density test mass convention `m = size^3`.
- `swarm_host.cuh`: mass-weighted linear initializer, prescribed jittered
  collision-test positions, output helpers, and controller metadata.
- `swarm_runtime.cu`: fixed total dust mass, linear initializer, fixed-position
  search reuse, and dynamic controller size ranges.
- `_col_chain.cuh`: rate-adaptive frozen baths, dynamic size bins, and bath
  diagnostics.
- `swarm_kern.cuh`: compile-time checks for the model-local controller.
- `const_defs.cuh`: campaign constants and output schedule.

## Build and run

Build one Ampere case with:

```sh
make -C val/coag/test_linear MODEL=1e+6n_1e+1k_1e-2b GPU_TARGET=sm_80
```

Build and run all 25 models sequentially with:

```sh
python3 val/coag/test_linear/run_models.py
```

Run one of the four off-axis batches with:

```sh
python3 val/coag/test_linear/run_models.py --batch 1
```

Each batch writes a separate `wall_time_batch_<batch>.json`, so the four
runners can operate concurrently without modifying the same timing summary.
Submit all four as a Slurm array from the project root with:

```sh
sbatch val/coag/test_linear/job_submit_batches.sh
```

Each array task requests one A100 for 24 hours and runs its assigned models
sequentially. Estimated batch runtimes are 6.89, 6.62, 6.65, and 6.68 hours.

Executables and objects are written under `val/temp/coag/test_linear/`.
Scientific output is written under
`val/logs/coagulation/test_linear/<case>/`. The runner atomically updates
its timing JSON after each model; compilation time is excluded from the
recorded process wall time.

This validation family is not registered with `val/tool/run_all.py`.
