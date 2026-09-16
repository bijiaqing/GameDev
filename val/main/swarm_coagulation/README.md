# Radial–vertical coagulation experiments

Two **Eriksson-inspired** particle-model experiments: strong turbulence
(`strong_turbulence`, alpha = 1e-3) and weak turbulence
(`weak_turbulence`, alpha = 1e-4). These use the current production transport,
density diffusion, a query-local physical collision kernel, and the production
sticking/fragmentation prescription.
They are coupled-physics experiments, not analytical accuracy oracles or exact
reproductions of another code's equations.

Both models enable `COLLISION_QUERY_LOCAL`. Both grain sizes use the query
particle's cylindrical R,Z and gas properties, never the partner position or a
pair average. Collision velocities use the mcdust local prescriptions:

- radial: v_R(s) = 2 vn St/(1+St^2), with zero gas radial velocity;
- azimuthal difference: vn [1/(1+St_i^2) - 1/(1+St_j^2)];
- settling difference: Z Omega [min(St_i,0.5) - min(St_j,0.5)];
- vn = -eta R Omega, using the analytic pressure derivative of our gas profile;
- local turbulent Re = sqrt(pi/2) alpha rho_g c_s X_SEC/(Omega M_MOL).

Brownian motion and the existing Ormel–Cuzzi formula are retained. Both stopping
times use our existing Epstein prescription. The supplied effective column
`Sigma_g * gas_strat` gives the local Reynolds number through the existing helper;
it does not change the gas surface density. Particle dynamics and diffusion are
unchanged. Rates and event outcomes both call the same local pair-speed function,
including updated owner grain sizes during a collision chain. This directed
query-environment estimate need not equal the reverse pair estimate.

This replaces `COLLISION_NO_AZIMUTHAL`: physical differential azimuthal drift is
restored without subtracting orbital velocities across the neighbor separation.
Other models retain their full resolved collision velocities. Transfer the root
`inc/comm/swarm/_collision.cuh` and both model `flags.mk` files, then rebuild.
CPU closure checks and eight build-route checks pass; GPU execution and timings
for this prescription have not been verified.

## Build and run (ROCm)

Each model can run independently in any of these four configurations:

| Backend | KD-tree | Morton |
|---|---|---|
| CUDA | `GPU_BACKEND=cuda COLLISION_SEARCH=kdtree` | `GPU_BACKEND=cuda COLLISION_SEARCH=morton` |
| ROCm | `GPU_BACKEND=rocm COLLISION_SEARCH=kdtree` | `GPU_BACKEND=rocm COLLISION_SEARCH=morton` |

Thus there are eight runs total. From the repository root, select **one** run:

```bash
model=strong_turbulence  # or weak_turbulence
backend=rocm            # or cuda
search=morton           # or kdtree
target=gfx942           # use the actual architecture, e.g. sm_80 for CUDA
make -C val/main/swarm_coagulation -j8 MODEL="$model" \
  GPU_BACKEND="$backend" GPU_TARGET="$target" COLLISION_SEARCH="$search" CUDA_MATH=precise
./val/temp/swarm_coagulation/bin/$model/$backend/$search/gamedev
```

All runs use `N_K=200` and `COL_BATH_EPS=0.02` (the bath epsilon requested for
this benchmark), with the same production frozen-bath controller. Compile-time
assertions enforce these values. CUDA defaults to precise math for this suite.
No combined runner is required. Avoid overlapping runs on the same GPU when
measuring performance. Record GPU model, compiler/version, architecture, and
math settings: CUDA-versus-ROCm timings also compare different hardware.

Transfer the suite together with the root `Makefile`, `inc/`, and `src/` dependencies.

- Models: `val/main/swarm_coagulation/models/<model>/`
- Outputs: `val/main/swarm_coagulation/outputs/<model>/<backend>/<search>/`
- Objects: `val/temp/swarm_coagulation/obj/`
- Executables: `val/temp/swarm_coagulation/bin/<model>/<backend>/<search>/gamedev`

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
| Duration | 200000 years |
| Output interval | 1000 years, plus initial state |
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
about 12.6 GiB per completed model; RNG checkpoints and other outputs add to this.

## Initialization and local changes

`common/const_defs.cuh` supplies the physical/numerical constants. The two model
headers differ only in alpha. `common/swarm_host.cuh` imports the appropriate
production backend header and overrides only the spatial sampling and its mass
integral. Collision events use the production `_col_chain.cuh`; there is no
model-local erosion or small-projectile grouping.

The default production initializer uses a smoothed radial profile and an initially
settled vertical distribution. Here `initial_profile.hpp` instead samples dust
mixed with the **production spherical hydrostatic gas profile**, independent of
grain size. With r in au its spherical mass-density weight is

    w(r,theta) = r^(-1/4) sin(theta)^(-5/4)
                 exp[(sin(theta)-1)/(h0^2 sqrt(r sin(theta)))].

Rejection sampling uses a bounded proposal over the spherical domain; Simpson
quadrature supplies its finite-domain mass. The existing runtime samples the MRN
mass spectrum and assigns equal represented masses. The production particle
initializer supplies steady radial drift and terminal settling velocities;
this initially well-mixed density is not assumed to be a settled equilibrium.

## Scope relative to Eriksson et al.

Source: [Eriksson et al. (2026), Section 5](https://arxiv.org/abs/2603.22550).
Setup scripts inspected at
[dustComparison commit 6fd92fd](https://github.com/astrolinn/dustComparison/tree/6fd92fd78b765fb03df6d21f95be2b1efd5a7d74/2Dsims),
particularly the mcdust setup and cuDisc power-law disk model. Cite their work
when presenting these adapted experiments.

Important retained differences:

- Production diffusion acts on dust density, not dust/gas concentration. Vertical
  equilibrium therefore need not match cuDisc or mcdust.
- Production collision outcomes use a sharp sticking/fragmentation threshold,
  a sampled fragment with a minimum-diameter floor, and no erosion channel.
- Production stochastic boundaries reflect; deterministic transport has outflow
  at the radial and polar domain edges. These do not match cuDisc's open boundaries.
- Collision velocities use the query-local closure described above, with
  Epstein-only drag. Grain-size-dependent
  Stokes-I drag and another code's collision-velocity distribution are not added.
- Density normalization uses the Gaussian midplane convention with exact spherical
  stratification. Its finite-domain integral is computed explicitly; it is not
  silently equated to an infinite Gaussian column mass.
- Snapshots are on a 1000-year grid, not exactly the paper's 11133/23809-year times.
  Use 11000/24000 years for nearby qualitative views, with actual times labeled.

## Reverting the erosion/grouping experiment

The model-local mcdust fragmentation, erosion, and grouping experiment was removed.
Query-local collision velocities and the production completed-particle early
return remain enabled. When updating a server copy, also delete the obsolete
`common/_col_chain.cuh` and `common/collision_rules.hpp` files; an ordinary rsync
without deletion will leave the old override active. Force a rebuild with
`make -B` using the same model/backend/search arguments. Preserve existing outputs
separately before starting a fresh run.

## Diagnostics and verification

Useful comparisons: mass per logarithmic grain-radius bin versus cylindrical
radius and height, mass-weighted grain radius, total dust mass remaining, and
vertical dust-density profiles. Integrate over R<15 au and R>25 au for the inner
and outer samples. At high altitude, report particle sampling adequacy and check
sensitivity to particle count/neighborhood size before interpreting fine details.

Run the small host check:

```bash
python3 val/main/swarm_coagulation/check_setup.py
```

It tests the actual spatial sampler and production grain initializer, CGS
normalizations, finite-domain mass quadrature, sampling moments, mass weights,
settling direction, and all eight CUDA/ROCm x KD-tree/Morton model build routes. Output is JSON.
It does **not** execute GPU collision/diffusion kernels or establish scientific
convergence. Native GPU compilation and execution remain to be checked on the
ROCm machine; no simulation output or runtime estimate is claimed.


### Runtime estimation

No timings exist yet for these production-like models. The one-year maximum
step requires at least 200000 dynamics steps over the full run; orbital,
settling and diffusion constraints can shorten them, and collisions can require
additional bath intervals. The polar-acceleration limit near the inner domain
can be about 0.3 year for material at large polar offsets. This is a local scale,
not a prediction of the evolving global timestep.

Estimate elapsed runtime from output timestamp differences on the actual GPU.
Each output interval is 1000 simulated years; 200 times a measured interval gives
a provisional full-run estimate. Use several intervals and revise the estimate
as grains grow and settle. Initialization/build time should be separated, and
output I/O should be included consistently. Fixed-position synthetic-kernel
benchmarks cannot predict the cost of rebuilding neighbors during these runs.
