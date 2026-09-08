# Constant-kernel coagulation accuracy tests

This is a standalone CUDA swarm validation family for an accuracy and stability check of the
frozen-bath collision routine against the constant-kernel Smoluchowski solution.
It evolves fixed representative particles by coagulation only. Each case-directory
name records `N_P`, `N_K`, and `COL_BATH_EPS`; all cases under `models/`
inherit the implementation in `models/models_common` through
`MODEL_PARENT := models_common`.

## Active build configuration

`models/models_common/common_flags.mk` selects:

- `DUST_REPR := swarm`
- the KD-tree collision search
- `COLLISION` with the default frozen-bath algorithm; `BERNOULLI` is not set
- `MULTISIZE`, so collisions evolve particle size and represented grain number
- `CODE_UNIT`
- `COLLISION_UNIT_VOLUME`, which makes the collision-search measure equal to one
- `LOGTIMING`, which gives logarithmically spaced physical output times

No transport, diffusion, radiation, imported gas, opacity output, or dust-density
output is enabled. Frozen-bath collisions retain their physical-neighbor cache
without defining `KNN_CACHE`; that flag is only legal with the Bernoulli method.

## Parameter differences from CUDA production defaults

The shared `models/models_common/const_defs.cuh` changes the following production values:

| Parameter | CUDA production | Reference value | Purpose |
| --- | ---: | ---: | --- |
| `N_P` | `1e7` | `1e6` | fixed particle count for the parameter campaign |
| `N_X` | `100` | `2` | activate a two-dimensional collision geometry without a useful mesh |
| `N_K` | `200` | `10` | first neighbor-count case |
| `H_SEARCH` | `1` | `128` | make the search cutoff non-limiting for the prescribed positions |
| `COL_BIN_S` | `8` | `64` | resolve the evolving size range before sparse-bin merging |
| `COL_BATH_EPS` | `0.06` | `0.01` | first frozen-bath tolerance case |
| `SAVE_MAX` | `100` | `9` | evolve through logarithmic output frame 9 at time `1e8` |
| `DT_OUT` | `1` | `0.1` | shift the base-10 logarithmic schedule so frame 1 is at time 1 |

The production absolute bath cap `COL_BATH_MAX = 0.05` is removed. Bath length
is instead limited by the remaining operator interval and the rate-based
`COL_BATH_EPS` controller.

`COL_BATH_EPS`, `COL_BATH_ALPHA`, and the two size-range factors are declared
`constexpr` in the shared constants so the `models_common` compile-time validity checks can
use these floating-point values in `static_assert` expressions.

The fixed controller range `COL_SIZE_MIN = 0.5*INIT_SMIN` through
`COL_SIZE_MAX = 8*INIT_SMAX` is replaced by per-bath factors:

```text
COL_SIZE_MIN_FACTOR = 0.5
COL_SIZE_MAX_FACTOR = 8.0
```

At the beginning of every bath, the controller range is therefore
`[0.5*minimum_current_size, 8*maximum_current_size]`.

The unchanged parameters important to all cases are `COAG_KERNEL = 0`,
`INIT_SMIN = INIT_SMAX = 1`, `V_FRAG = 1`, `COL_BIN_MIN = 64`, and
`COL_BATH_ALPHA = 1e-3`.

## Parameter-search models

The campaign fixes the representative-particle count and evaluates the complete
`N_K`--`COL_BATH_EPS` Cartesian grid:

```text
N_P          = 1e6
N_K          = 10, 20, 50, 100, 200
COL_BATH_EPS = 5e-3, 1e-2, 2e-2, 4e-2, 8e-2
```

The resulting 25 models provide every `N_K` sweep at fixed bath tolerance,
every bath-tolerance sweep at fixed `N_K`, and the full interaction surface
at `N_P = 1e6`.

The shared `const_defs.cuh` accepts `SWEEP_N_P`, `SWEEP_N_K`, and
`SWEEP_COL_BATH_EPS` compile definitions. Every case directory contains only a
`flags.mk` that names `models_common` as its `MODEL_PARENT`, includes the shared
flags, and supplies the three parameter values. This prevents the CUDA overrides
from drifting between otherwise identical cases.

The model name uses `n`, `k`, and `b` suffixes. For example,
`1e+6n_2e+2k_8e-2b` means `N_P = 1e6`, `N_K = 200`, and
`COL_BATH_EPS = 0.08`.

## Source overrides

### `_collision.cuh`

Adds the `COLLISION_UNIT_VOLUME` branch to `_get_ball_measure`. A positive search
radius returns exactly `1.0`, so spatial volume does not enter collision rates.
The existing zero-radius guard remains active.

### `param_phys.cuh`

Changes the compact-grain mass from
`pi*RHO_0*size^3/6` to `size^3`. Consequently, a grain of size one has unit mass.

### `swarm_host.cuh`

Adds `rand_collision_test_pos`, which places particles on a reproducibly
jittered two-dimensional annular grid. It chooses radial and azimuthal counts
from the domain aspect ratio and distributes incomplete rows evenly. The
quarter-cell jitter avoids systematic equal-distance KNN ties while keeping
every particle inside its cell. The runtime reads the position and collision
seeds from `COAG_POSITION_SEED` and `COAG_COLLISION_SEED`, both defaulting to
one, and records them in `variables.txt`.

The saved configuration metadata reports the bath cap as `rate_adaptive` and
records the two controller size-range factors instead of fixed size limits.

### `swarm_runtime.cu`

- Prescribes `total_dust_mass = 1e30` instead of deriving it from the analytic disk.
- Uses `rand_collision_test_pos` instead of the production disk sampler.
- Takes the minimum and maximum frozen particle sizes at the start of every bath.
- Passes the resulting dynamic size limits through the bin-count, rate-bin, and
  post-bath audit kernels.
- Records the actual size limits in every collision-controller bath record.

Because no position-changing operator is enabled, the first collision call builds
the KD-tree and neighbor cache and all later baths and output intervals reuse them.

### `_col_chain.cuh`

- Adds `size_min` and `size_max` to each recorded bath.
- Makes size-bin selection and its three callers use runtime size limits.
- Removes the absolute `COL_BATH_MAX` restriction from `_choose_col_bath`.
- Writes the runtime size limits into `collision_chain_*.json`.

### `swarm_kern.cuh`

Removes the obsolete assertion for `COL_BATH_MAX` and replaces the assertions for
fixed `COL_SIZE_MIN/MAX` with an assertion that
`COL_SIZE_MAX_FACTOR > COL_SIZE_MIN_FACTOR > 0`.

## Initial normalization and collision outcome

All sampled sizes are exactly one. The mass normalization therefore assigns each
representative

```text
par_numr = total_dust_mass / N_P.
```

physical grains. The runtime kernel normalization inherited from current
production is

```text
lambda_0 = N_P / (N_K * total_dust_mass).
```

Each cached slot, including the representative's own swarm, initially contributes
`lambda_0*par_numr = 1/N_K`. Thus every case starts with total collision rate one,
independently of `N_P` and `N_K`. At `N_P = 1e6`, `par_numr = 1e24`;
for the `N_K = 10` reference, `lambda_0 = 1e-25`, and each slot contributes
`0.1`.

For the three synthetic kernels, relative velocity remains zero. With
`COAG_KERNEL = 0` and `V_FRAG = 1`, all accepted events take the coagulation branch;
fragmentation is inactive without an additional model flag.

Self-swarm inclusion and the corresponding `N_K` normalization are current
production behavior, not model-local differences.

## Output schedule

With `LOGTIMING`, `LOG_BASE = 10`, and `DT_OUT = 0.1`, output frame `i >= 1`
has physical time

```text
t_i = DT_OUT * LOG_BASE^i.
```

The complete run therefore writes:

| File | Time |
| --- | ---: |
| `particle_00000.dat` | `0` |
| `particle_00001.dat` | `1` |
| `particle_00002.dat` | `10` |
| `particle_00003.dat` | `100` |
| `particle_00004.dat` | `1000` |
| `particle_00005.dat` | `10000` |
| `particle_00006.dat` | `100000` |
| `particle_00007.dat` | `1000000` |
| `particle_00008.dat` | `10000000` |
| `particle_00009.dat` | `100000000` |

Manual runs write all ten particle snapshots from `particle_00000.dat` through
`particle_00009.dat`. CUDA RNG states are intentionally omitted, so these are
analysis snapshots rather than complete restart checkpoints; a collision run
cannot resume from one without the corresponding RNG-state file. The multiseed
runner scores the final snapshot and then deletes these bulk files as described
below.

The collision-only path applies one full collision operator over each output
interval. There is no transport operator requiring two Strang-split half steps.

## Build and validation status

The validation family is intentionally CUDA-specific because its runtime and supporting
headers were copied from the CUDA production branch. A dry run resolves all seven
shared source/header overrides, and repository static checks pass. The complete
25-model, 10-seed CUDA campaign has finished; its compact retained results are
described below.

An example build for an Ampere target is:

```sh
make -C val/coag/test_const MODEL=1e+6n_1e+1k_1e-2b GPU_TARGET=sm_80
```

Use the wrapper's `run` target to build and launch one case. Executables and object
files are written under `val/temp/coag/test_const/`. More specifically, a model's
objects are stored under
`val/temp/coag/test_const/obj/<case>/swarm/cuda/kdtree/fast/<GPU_TARGET>/`, and
its executable is `val/temp/coag/test_const/bin/<case>/gamedev`. A manual run
writes scientific output under `out/test_const/seed_000/<case>/`; `out/` is
ignored by Git. This validation family is not registered with
`val/tool/run_all.py`.

Run the independent 10-seed campaign with:

```sh
python3 val/coag/test_const/run_models.py
```

The script builds all 25 models once and then runs 10 deterministic replicates
of the full grid. Replicate zero uses the previous baseline position and
collision seeds `(1, 1)`; replicate `r` uses `(2r+1, 2r+1)`. Compilation is
excluded from the process wall times. The completed standalone seed-0 process
times are retained in `wall_time.json`.

The campaign writes its active model output under
`out/test_const/multiseed/<case>/`. Each completed model is scored at
`particle_00009.dat` against the exact
constant-kernel mass distribution. The atomically written seed JSON contains
TV, Jensen--Shannon, log-mass Wasserstein and CDF errors; mass, number and
second-moment checks; fixed-bin mass histograms; controller summaries; wall
time; and the actual saved seeds. After replicate zero is scored, its complete
model directory is moved to `out/test_const/seed_000/<case>/` before the record
is marked passed. For later replicates, the runner deletes `particle_*.dat` and
`collision_chain_*.json` only after the score record is durable. Failed-model
raw output is retained. An interrupted campaign resumes already recorded
models and refuses to combine seed files with changed campaign sources.

Download the compact result directory:

```text
val/coag/test_const/multiseed/
```

It contains `manifest.json`, `seed_000.json` through `seed_009.json`, and the
final across-seed `summary.json`. Seed JSON files are retained locally but
ignored by Git; the manifest and summary remain trackable.
