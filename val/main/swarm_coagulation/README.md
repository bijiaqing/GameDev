# Radial–vertical coagulation experiments

Fresh weak/strong turbulence runs using the validated `erosion_cached_rates`
implementation promoted from `lab/collision_fix`. The two model folders select
alpha and output count; `common/` contains the shared local collision scheduler,
64-thread event kernel, erosion outcomes, narrow grouping, and query-environment /
initial-rate caches. This suite has no dependency on `lab/`. Root `inc/` and
`src/` are unchanged. The original lab models/results remain available.

## Four requested CUDA runs

| Model | Alpha | Search | Outputs after initial 0 | End time |
|---|---:|---|---:|---:|
| weak_turbulence | 1e-4 | KD-tree | 250 | 25000 yr |
| weak_turbulence | 1e-4 | Morton | 250 | 25000 yr |
| strong_turbulence | 1e-3 | KD-tree | 125 | 12500 yr |
| strong_turbulence | 1e-3 | Morton | 125 | 12500 yr |

Output spacing is 100 yr. CUDA is the suite default; specify search explicitly.
All commands below start at time zero: **no checkpoint argument**.

Transfer the entire `val/main/swarm_coagulation/` directory plus the root Makefile,
inc/ and src/. Before the first fresh launch, preserve any server outputs from
older implementations (do this once, before starting any of the four jobs):

```sh
if [ -d val/main/swarm_coagulation/outputs ]; then
    mv val/main/swarm_coagulation/outputs "val/main/swarm_coagulation/outputs_before_cached_$(date +%Y%m%d_%H%M%S)"
fi
```

Run independently, or sequentially on the same GPU for useful timings:

```sh
make -C val/main/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/temp/swarm_coagulation/bin/weak_turbulence/cuda/kdtree/gamedev

make -C val/main/swarm_coagulation -j8 MODEL=weak_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/temp/swarm_coagulation/bin/weak_turbulence/cuda/morton/gamedev

make -C val/main/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=kdtree &&
val/temp/swarm_coagulation/bin/strong_turbulence/cuda/kdtree/gamedev

make -C val/main/swarm_coagulation -j8 MODEL=strong_turbulence GPU_BACKEND=cuda GPU_TARGET=sm_80 COLLISION_SEARCH=morton &&
val/temp/swarm_coagulation/bin/strong_turbulence/cuda/morton/gamedev
```

Outputs: `val/main/swarm_coagulation/outputs/<model>/<backend>/<search>/`.
Objects: `val/temp/swarm_coagulation/obj/`.
Executables: `val/temp/swarm_coagulation/bin/<model>/<backend>/<search>/gamedev`.
ROCm remains selectable with `GPU_BACKEND=rocm GPU_TARGET=gfx942`; it is not one
of these four requested runs. CUDA math defaults to precise.

## Collision implementation

- Query-local gas: both sizes use the owner's R,Z. Relative speeds combine
  differential radial/azimuthal drift, settling capped at St=0.5, Brownian motion,
  and the existing Ormel–Cuzzi turbulent regimes. Epstein stopping times remain.
- Low-speed collisions stick. Narrow packet grouping remains q<=1e-6 with a
  maximum 1e-4 target-mass increment; the wider-grouping experiment is not included.
- High-speed collisions at v>=100 cm/s use erosion for q=m_projectile/m_target<=0.1.
  Remnant packets and debris transitions retain their separate rates/probabilities.
  Other high-speed events fragment with the old target mass as the upper bound,
  truncated at the monomer floor. See `common/erosion_outcome.cuh`.
- The local change-based adaptive scheduler, neighbor compatibility, audit feedback,
  compact continuations and 32-event continuation cap are inherited unchanged.
  N_K=200 and COL_BATH_EPS=0.02 remain fixed. The event cap is not a refresh cap.
- Eight query-environment coefficients are cached per collision half-step. Fresh
  rate totals skip the repeated pair pass for no-event intervals. No stale rates
  are reused after events or reservoir publication. Persistent extra memory is
  96 MiB for 2^20 representatives.

This replaces the former production-controller-only suite and its old cancelled
experimental override. Do **not** follow the obsolete instruction to delete
common/_col_chain.cuh: that file is now required. The promoted implementation is
shared by both search methods; neither search is silently replaced by the other.

## Model parameters


All dimensional quantities in these models and their binary outputs are **CGS**.
`CODE_UNIT` is disabled so Brownian collision velocities are included.

| Parameter | Both models |
|---|---|
| Star | 1 solar mass |
| Spherical radius | 5–50 au |
| Polar angle | pi/2 +/- 0.2 radians; both hemispheres |
| Mesh | axisymmetric: (N_X,N_Y,N_Z) = (1,128,64) |
| Representative particles | 2^20 = 1048576 |
| Gas surface-density normalization | 1410.4014065096128 g/cm^2 at 1 au; radial exponent -1 |
| Temperature | 209.7926358245702 K at 1 au; radial exponent -1/2 |
| Mean molecular mass | 2.34 proton masses |
| Initial dust/gas volume-density ratio | 0.01 throughout the finite domain |
| Initial grain radius | 0.5–1 micron, MRN number spectrum dN/da proportional to a^-3.5 |
| Grain internal density | 1 g/cm^3 |
| Fragmentation threshold | 100 cm/s = 1 m/s |
| Diffusion | D_R = D_Z = alpha c_s H, unit Schmidt numbers |
| Collision neighbors | 200 slots, including the owner |
| Duration | Strong: 12500 years; weak: 25000 years |
| Output interval | 100 years; strong: 125 intervals (126 snapshots); weak: 250 intervals (251 snapshots) |
| Maximum dynamics/bath interval | 1 year; production controllers can shorten it |

The code stores **diameter**: `INIT_SMIN=1e-4 cm` and `INIT_SMAX=2e-4 cm`.
The reference Stokes number is pi*rho_s*S_0/(4*Sigma_0); its local value follows
production Epstein scaling with grain diameter and gas density. Gas density and
temperature are static. Gas radial accretion, radiation, and P-R drag are disabled.

The saved particle record contains eight doubles: phi, spherical r, theta,
physical v_phi, v_r, v_theta, grain diameter, and represented grain count.
For analysis use radius a=par_size/2, cylindrical R=r sin(theta), Z=r cos(theta),
and mass weights par_numr*pi*rho_s*par_size^3/6. Do not use unweighted particle
counts as a physical grain-number distribution. Particle snapshots alone occupy
about 7.9 GiB for strong turbulence and 15.7 GiB for weak turbulence; RNG checkpoints and other outputs add to this.

## Initialization and comparison scope

The existing `common/swarm_host.cuh` and `initial_profile.hpp` retain the same
finite-domain, well-mixed spherical hydrostatic dust initialization and equal
represented masses as the lab run. Root transport and density diffusion remain.
These are Eriksson-inspired experiments, not exact reproductions or analytical
accuracy tests. MCDUST's downloaded run starts with monomers rather than our MRN
range; gas evaluation, diffusion, grouping and domain/boundary details also differ.

## Verification and provenance

The promoted implementation files match `lab/collision_fix/models/erosion_cached_rates`
byte-for-byte. Shared constants match its header after removing the two model
selector defines; weak/strong model headers select alpha/output count separately.
The existing initializer files are unchanged. Diagnostics retain method name
`erosion_cached_rates`; model/backend/search are identified by their output path.

```sh
python3 val/main/swarm_coagulation/check_setup.py
```

The host check covers initialization, normalization, cached versus original
relative velocities at both alphas, fixed controller parameters/narrow grouping,
and all eight available model/backend/search build routes, including shared runtime
and header overrides. It does not execute GPU kernels.

Prior downloaded weak/CUDA/KD-tree lab evidence at 2500 yr: 46.49 min versus
66.44 min without caching, with particle hashes identical at outputs
0,5,10,15,20,25. This is evidence for that lab configuration only. New publication
runs, strong turbulence, Morton and ROCm are not qualified by those timings.
Do not extrapolate full-run time linearly: collision work changes as grains evolve.
