# CUDA verification models

These model directories implement the analytical tests summarized in
`doc/testset_fluid.md` as cluster-runnable CUDA cases without changing `inc`, `src`,
or production `mod`. Model-local `fluid_runtime.cu` files include the shared driver in this
directory. The Makefile still selects production kernels from `src/` unless a test explicitly
supplies a model-local replacement.

See `TEST_CASES.md` for each model's exact initialization, analytical solution, production kernels
exercised, finite-volume comparison, and expected result.

## Prepared cases

| Model | CUDA code exercised | Analytical reference |
|---|---|---|
| `test_x_transport_2d` | Production FARGO/PPM/HLL X sweep | Periodic rotating Fourier mode after one revolution |
| `test_y_transport_cyl` | Production radial PPM/HLL sweep with cylindrical volume | Compact homologous expansion with d=2 |
| `test_y_transport_sph` | Production radial PPM/HLL sweep with spherical volume | Extruded compact homologous expansion with d=3 |
| `test_z_transport_3d` | Production polar PPM/HLL sweep | Constant-polar-angular-momentum translation of rho sin(z) |
| `test_x_diffusion_2d` | Production cyclic CN X diffusion and momentum flux | Fourier decay on each ring |
| `test_y_diffusion_cyl` | Production radial CN diffusion and momentum flux | d=2 Neumann shell eigenmode |
| `test_y_diffusion_sph` | Production radial CN diffusion and momentum flux | d=3 Neumann shell eigenmode |
| `test_z_diffusion_3d` | Production polar CN diffusion and momentum flux | Even l=2 Legendre decay on a hemisphere |
| `test_source_drag` | Test-only source kernel using the production exponential weights | Exact linear-force drag relaxation over dt/ts from 1e-6 to 1e6 |
| `test_optdepth` | Production optical-depth cell quadrature and radial cumulative sum | Power-law radial integral |
| `test_ring_transport_2d` | Production symmetric X/Y transport and source composition | Force-balanced Keplerian Fourier rings |
| `test_ring_diffusion_2d` | Above plus production X/Y diffusion | Rotating Fourier rings with ring-dependent exponential damping |
| `test_ring_radiation_2d` | Above plus production optical depth and radiation source | Optically thin reduced-gravity Fourier rings |
| `test_ring_all_2d` | Transport, diffusion, radiation, drag, and gravity | Damped optically thin reduced-gravity Fourier rings |

The constant-diffusivity operator cases use `test_common/param_phys.cuh`. It wraps the
production helper header and replaces only `_get_nu` and `_get_alpha`; PPM, HLL,
geometry, vacuum recovery, and all other helpers remain the production definitions. The source
test necessarily replaces `source_update.cu`, because the production interface cannot prescribe
arbitrary constant gas velocities and force endpoints. Its scope is the exponential quadrature,
not the disk force calculation.

## Running on a CUDA cluster

Run one complete refinement sequence from the repository root, for example:

```bash
python3 qav/fluid/test_x_transport_2d/run.py --res 32 64 128 256
python3 qav/fluid/test_y_transport_cyl/run.py --cfl 0.05 --res 64 128 256 512
python3 qav/fluid/test_y_transport_cyl/run.py --cfl 0.5 --res 64 128 256 512
python3 qav/fluid/test_optdepth/run.py --power -1 --res 32 64 128 256
```

Exercise the FARGO integer/fractional shift cases separately with `--shift 3`, `3.25`, `3.5`, and
`3.75`. The value is the nominal number of azimuthal cells traversed per full operator step; the
last step is shortened to land exactly at one revolution.

Each wrapper performs `make clean`, builds with the requested `RES`, runs the executable, compares
the binary output with independently evaluated finite-volume averages, writes tagged
`qav/fluid/out/SWEEP/MODEL/metrics_N*.json` records, and prints an error/order table. `SWEEP` is
`thread` by default and `block` when selected through `FLUID_SWEEP`. Raw data from
CFL, FARGO-shift, and optical-depth-power sweeps are separated into correspondingly tagged
subdirectories. Cleaning between resolutions is required because changing a Make variable alone
does not invalidate existing object files.

Run the entire prepared matrix, or a selected group, with:

```bash
python3 qav/fluid/test_common/run_suite.py --quick
python3 qav/fluid/test_common/run_suite.py --group diffusion --res 32 64 128 256
python3 qav/fluid/test_common/run_suite.py --group all --res 32 64 128 256
```

Prefix any suite command with `FLUID_SWEEP=block` to validate the block sweep; omitting it selects
the reference thread sweep. Their artifacts are stored independently under `out/block/` and
`out/thread/`, so a block run cannot overwrite the thread baseline.

CUDA builds use `--use_fast_math` by default.  Use `--math-mode precise` for a matched-arithmetic
backend check; this omits `--use_fast_math` and writes results below `out/thread_precise/` or
`out/block_precise/` without replacing the ordinary archive.  For example, rerun only polar
transport with:

```bash
python3 qav/fluid/test_z_transport_3d/run.py \
    --math-mode precise \
    --cfl 0.05 \
    --res 32 64 128 256
python3 qav/fluid/test_z_transport_3d/run.py \
    --math-mode precise \
    --cfl 0.5 \
    --res 32 64 128 256
```

The separate implementation-comparison branch builds both methods and checks their outputs directly:

```bash
python3 qav/fluid/test_common/run_suite.py --group sweep --quick
python3 qav/fluid/test_common/run_suite.py --group sweep --sweep-dim 2d
python3 qav/fluid/test_common/run_suite.py --group sweep --sweep-dim 3d
```

Full defaults are a transport-only `1024^2` pair and a diffusion-enabled `128^3` pair, both without
radiation. Override the resolutions with `--sweep-res-2d` and `--sweep-res-3d`. This fixed-work
branch is intentionally not included in `--group all`.

Only NumPy is required by the validator. The d=2/d=3 shell references are independently integrated
with a fine-grid RK4 solve; they do not use the CUDA helpers or the production diffusion matrix.

Each test model owns a local `const_defs.cuh` that selects its analytical case before including the
shared verification constants. Its `flags.mk` contains only include routing, sweep selection, and
active-physics flags. The runner passes resolution, CFL, profile power, and shift as `TEST_*`
controls through the root Makefile rather than defining physical constants in `flags.mk`.

For the optical-depth test, repeat powers `0`, `-1`, and another non-degenerate value such as `1`:

```bash
python3 qav/fluid/test_optdepth/run.py --power 0  --res 32 64 128 256
python3 qav/fluid/test_optdepth/run.py --power -1 --res 32 64 128 256
python3 qav/fluid/test_optdepth/run.py --power 1  --res 32 64 128 256
```

## Expected outcomes

- Smooth diffusion and optical-depth errors should approach second order.
- X transport should be at least second order and conserve periodic mass to roundoff.
- Y and Z transport use SSPRK(3,3), whose native CFL=0.5 conservative-field convergence passes on
  the finer intervals. Judge formal order from the complete sequence rather than the under-resolved
  $N=32$ interval. Also inspect step counts to confirm that the conservative invariant-domain
  limiter prevents nearly empty cells from collapsing the global CFL timestep.
- The source test should agree with the analytical result near roundoff over all stiffness ratios.
- Ring cases should approach second order in density and stored momenta. Mass should remain at
  roundoff for transport-only/radiation-only rings; diffusion conservation may accumulate ordinary
  floating-point reduction error.

Do not assign a universal pass threshold before inspecting where each case enters its asymptotic
range on the target GPU. Preserve the raw metrics and compiler/GPU information used for a paper.

## Deliberately not claimed by this preparation

These directories do not yet implement the full 3D manufactured-solution matrix. That test needs:

1. an independent symbolic derivation of density and all three momentum residuals;
2. test-only density and momentum forcing kernels;
3. symmetric, at-least-second-order forcing insertion in the split main loop;
4. exact manufactured boundary data; and
5. an independently integrated optical-depth reference when radiation is enabled.

Creating nominal `flags.mk` files without those pieces would produce ordinary disk simulations,
not MMS verification. The detailed implementation contract is recorded in
`qav/fluid/test_mms_3d/README.md`.
