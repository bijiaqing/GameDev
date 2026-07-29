# GameDev documentation

This directory is the canonical documentation for the current GameDev dust solvers as of
2026-07-27. It replaces the historical audits, correction diaries, merge proposals, and duplicated
method notes that previously accumulated under `.github/`.

## Canonical documents

- [`numerics_fluid.md`](numerics_fluid.md) describes the Eulerian dust-fluid equations,
  discretization, operator ordering, boundaries, and current limitations
- [`testset_fluid.md`](testset_fluid.md) defines the analytical CUDA verification suite,
  records the available native results, and separates verified claims from unfinished tests
- [`numerics_swarm.md`](numerics_swarm.md) describes the Lagrangian representative-particle
  solver, including transport, stochastic diffusion, radiation, collisions, and outstanding
  validation
- [`testset_swarm.md`](testset_swarm.md) defines the analytical and statistical CUDA swarm test
  suite, its expected results, current evidence status, and unfinished coverage
- [`ref_file_architecture.md`](ref_file_architecture.md) records the fluid–swarm consistency contract, the current
  migration state, what can be shared, and what must remain representation-specific
- [`ref_naming_variables.md`](ref_naming_variables.md) records the canonical naming rules and
  intentional differences between the two representations

## Current repository state

The merged project is active at the repository root:

- `inc/fluid/` and `src/fluid/` contain the Eulerian fluid implementation
- `inc/swarm/` and `src/swarm/` contain the Lagrangian swarm implementation
- `inc/share/` and `src/share/` are reserved for future representation-independent infrastructure
- `qav/fluid/` contains the fluid CUDA verification suite and its recorded outputs
- `qav/swarm/` contains the prepared swarm CUDA verification suite
- `legacy/swarm_before_audit/` and `legacy/swarm_after_audit/` are frozen recovery snapshots

The root build selects exactly one representation through `DUST_REPR` in the chosen model's
`flags.mk`. An optional model-local `const_defs.cuh` has include priority over `inc/fluid/` or
`inc/swarm/`; models without one inherit the representation defaults.

## Authority and maintenance

When statements disagree, use this order of authority:

1. current source, model constants, and model flags
2. recorded native test metrics under the sweep-specific directories in `qav/fluid/out/`
3. these documents
4. Git history and the historical swarm patch

Resolved bug narratives are intentionally omitted unless they explain a current invariant or
regression test. New numerical changes should update the appropriate canonical document and add or
update a test; they should not create another standalone audit diary.

## Review and evidence baseline

The 2026-07-26 merged-tree review independently re-derived the production fluid and swarm formulas
and found no additional production correctness defect in its inspected scope. Its resolved
documentation, flag-combination, and Python-reference findings have been incorporated into the
canonical documents above. The review inspected the vendored cuKD library only through GameDev's
call sites and could not perform a local native CUDA build.

The authoritative native fluid evidence remains the 85 analytical records and 22 environment
records under `qav/fluid/out/thread/`. Block analytical results are written independently under
`qav/fluid/out/block/`. The automated thread/block comparison branch exists, but its
benchmark JSON and profiler artifacts are not currently archived in the repository. The swarm
branch now has a runnable analytical and statistical suite under `qav/swarm/`, but no native result
set has yet been archived, so its current evidence remains source review plus test preparation.

The source uses spherical coordinates

$$
x=\phi,\qquad y=r,\qquad z=\theta,
$$

with cylindrical coordinates

$$
R=y\sin z,\qquad Z=y\cos z.
$$

Both representations store internal angular variables but write physical linear velocity to
science files. The detailed state contract is in
[`ref_file_architecture.md`](ref_file_architecture.md).
