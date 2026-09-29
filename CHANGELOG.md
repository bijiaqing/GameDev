# Changelog

This file lists what each GameDev release contains. Versions follow
[semantic versioning](https://semver.org): the major number changes when output formats, feature
flags, or model configuration change incompatibly, the minor number when capabilities are added, and
the patch number for fixes. Record the release you used, together with the model directory and
build settings listed under [Reproducibility](README.md#reproducibility).

## [1.0.0] - unreleased

The first public release.

### Dust representations

- A Lagrangian **dust swarm** of weighted representative particles: semi-analytic drag and gravity
  trajectories, attenuated radiation pressure with optional Poynting–Robertson drag, Itô stochastic
  diffusion of density or concentration, and optional coagulation and fragmentation.
- Collisions through a frozen-bath continuous-time event chain with an adaptive bath controller,
  using exact top-$`K`$ neighbors from a KD tree or an adaptive Morton hierarchy, with periodic
  images across azimuthal wedges.
- An Eulerian, pressureless **dust fluid**: finite-volume PPM reconstruction with HLL fluxes and
  invariant-domain limiting, integer FARGO orbital shifts, SSPRK(3,3) sweeps, exponentially
  weighted drag and force sources, and Crank–Nicolson diffusion with a conservative donor-momentum
  closure, with a thread-per-line and a block-per-line sweep.
- A shared prescribed gas disk with one-way gas-to-dust coupling, optional viscous radial flow,
  and, for the swarm, imported gas fields.

### Build and platforms

- One `Makefile` builds any model for NVIDIA CUDA or AMD HIP/ROCm; each build selects one
  representation, fluid sweep, or collision search through compile-time settings in the model
  directory.
- Supported geometries: radial-only, radial–azimuthal, radial–polar, and full 3D for the swarm;
  radial–azimuthal and full 3D for the fluid.

### Validation and documentation

- Validation suites for both representations against analytical, statistical, and brute-force
  references, a one-command native campaign per backend (`val/run_all.py`), archive checks, and a
  CUDA–ROCm comparison.
- Scientific campaigns under `val/paper/` for the accompanying code paper.
- User guide, shared disk model, numerical guides for both representations, and the validation
  guide under `doc/`.

[1.0.0]: https://github.com/bijiaqing/GameDev/releases/tag/v1.0.0
