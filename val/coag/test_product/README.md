# Product-kernel coagulation accuracy tests

This standalone CUDA swarm validation family compares the frozen-bath collision
routine with the discrete Smoluchowski solution for

```text
K(m_i, m_j) = Lambda_0 * m_i * m_j.
```

Representative-particle positions are fixed and the only evolution process is
coagulation. Each case-directory under `models/` records `N_P`, `N_K`, and
`COL_BATH_EPS`; all cases inherit the implementation in
`models/models_common` through `MODEL_PARENT := models_common`.

## Campaign grid

The campaign fixes the representative-particle count and evaluates the complete
neighbor-count--bath-tolerance grid:

```text
N_P          = 1e6
N_K          = 10, 20, 50, 100, 200
COL_BATH_EPS = 5e-3, 1e-2, 2e-2, 4e-2, 8e-2
```

The 25 case directories contain only a `flags.mk`. The shared constants accept
`SWEEP_N_P`, `SWEEP_N_K`, and `SWEEP_COL_BATH_EPS` compile definitions.

For example, `1e+6n_2e+2k_8e-2b` means `N_P = 1e6`, `N_K = 200`, and
`COL_BATH_EPS = 0.08`.

## Active configuration

`models/models_common/common_flags.mk` selects:

- the CUDA swarm representation and KD-tree collision search;
- the default frozen-bath collision integrator;
- `MULTISIZE` and `CODE_UNIT`;
- `COLLISION_UNIT_VOLUME`, so collision rates contain no spatial-volume factor.

No transport, diffusion, radiation, imported gas, opacity output, or
dust-density output is enabled. Fixed positions allow the first KD-tree and
neighbor cache to be reused for the entire run. Before every frozen bath, a
new reproducible random permutation relabels the cached neighbor indices. Its
seed is read from `COAG_PARTNER_SEED`, defaulting to `COL_PARTNER_SEED = 2`.
The position and collision seeds likewise come from `COAG_POSITION_SEED` and
`COAG_COLLISION_SEED`, both defaulting to one. All three are recorded in
`variables.txt`. The cached entries therefore act as
`N_K` well-mixed bath-sampling slots instead of a permanently coupled local
network.

The product campaign differs from the constant campaign only in the selected
kernel and physical output schedule:

- `COAG_KERNEL = 2` selects the product kernel;
- output time is linear, with `DT_OUT = 0.1` and no `LOGTIMING` definition.

## Initial state

The product-kernel solution assumes a monodisperse initial population with

```text
m_0 = 1.
```

The copied constant-test initializer has `INIT_SMIN = INIT_SMAX = 1`. Since the
model-local grain-mass convention is `m = size^3`, every physical grain starts
with `size = 1` and `mass = 1`.

The common normalization assigns every representative swarm

```text
par_numr = total_dust_mass / N_P,
```

so all representative swarms initially carry equal mass. With
`total_dust_mass = 1e30` and `N_P = 1e6`, each representative initially stands
for `1e24` physical grains.

## Collision and time normalization

The runtime passes

```text
lambda_0 = N_P / (N_K * total_dust_mass)
```

to the `N_K` randomly relabeled bath slots. The `N_P/N_K` factor corrects KNN
subsampling, giving an effective full-ensemble coefficient

```text
Lambda_0 = 1 / total_dust_mass.
```

Each relabeled slot selects every representative with equal probability. A
self-partner is therefore allowed, but appears with probability `1/N_P` per
slot rather than occupying one slot deterministically. Consequently,
`N_K = 10` means ten bath samples.

Initially `N_0 = total_dust_mass` because `m_0 = 1`, so the analytical product-
kernel time coordinate is

```text
eta = Lambda_0 * N_0 * t = t.
```

The snapshots stop at `eta = 0.9`, before the gelation point `eta = 1`.

Relative velocity remains zero for the synthetic kernel. Since `V_FRAG = 1`,
all accepted events use coagulation and fragmentation is inactive.

## Output schedule

`SAVE_MAX = 9`, `DT_OUT = 0.1`, and `LIN_BASE = 1` retain:

| File | Time |
| --- | ---: |
| `particle_00000.dat` | 0.0 |
| `particle_00001.dat` | 0.1 |
| `particle_00002.dat` | 0.2 |
| `particle_00003.dat` | 0.3 |
| `particle_00004.dat` | 0.4 |
| `particle_00005.dat` | 0.5 |
| `particle_00006.dat` | 0.6 |
| `particle_00007.dat` | 0.7 |
| `particle_00008.dat` | 0.8 |
| `particle_00009.dat` | 0.9 |

CUDA RNG states are omitted, so these are analysis snapshots rather than full
restart checkpoints.

## Shared source overrides

- `_collision.cuh`: unit collision-search measure and the product kernel.
- `param_phys.cuh`: test mass convention `m = size^3`.
- `swarm_host.cuh`: monodisperse size sampling, prescribed jittered positions,
  output helpers, and controller metadata.
- `swarm_runtime.cu`: fixed total dust mass, fixed-position search reuse,
  bath-slot randomization, and dynamic controller size ranges.
- `_col_chain.cuh`: rate-adaptive frozen baths, dynamic size bins, and bath
  diagnostics.
- `swarm_kern.cuh`: compile-time checks for the model-local controller.
- `const_defs.cuh`: product-kernel and campaign constants.

## Build and run

Build one Ampere case with:

```sh
make -C val/coag/test_product MODEL=1e+6n_1e+1k_1e-2b GPU_TARGET=sm_80
```

Run the independent 15-seed campaign with:

```sh
python3 val/coag/test_product/run_models.py
```

Executables and objects are written under `val/temp/coag/test_product/`.
The campaign writes its active scientific output under
`out/test_product/multiseed/<case>/`. The runner builds all 25 models once,
then runs 15 deterministic replicates of the full grid. Replicate zero
uses the previous baseline streams `(position, collision, partner) = (1, 1,
2)`; replicate `r` uses `(2r+1, 2r+1, 2r+2)`. Compilation is excluded from
the process wall times. The completed standalone seed-0 process times are
retained in `wall_time.json`.

Each completed model is scored at the final pre-gelation snapshot against the
exact Borel mass distribution. The atomically written seed JSON contains TV,
Jensen--Shannon, log-mass Wasserstein and CDF errors; mass, number and
second-moment checks; fixed-bin mass histograms; controller summaries; wall
time; and the actual saved seeds. After replicate zero is scored, its complete
model directory is moved to `out/test_product/seed_000/<case>/` before the
record is marked passed. For later replicates, the runner deletes
`particle_*.dat` and `collision_chain_*.json` only after the score record is
durable. Failed-model raw output is retained. An interrupted campaign resumes
already recorded models and refuses to combine seed files with changed
campaign sources.

Download the compact result directory:

```text
val/coag/test_product/multiseed/
```

It contains `manifest.json`, `seed_000.json` through `seed_014.json`, and the
final across-seed `summary.json`. Seed JSON files are retained locally but
ignored by Git; the manifest and summary remain trackable.

This validation family is not registered with `val/tool/run_all.py`.
