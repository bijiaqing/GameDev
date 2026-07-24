# Graffiti documentation

This directory is the canonical documentation for the current Graffiti dust solvers as of
2026-07-24. It replaces the historical audits, correction diaries, merge proposals, and duplicated
method notes that previously accumulated under `.github/`.

## Canonical documents

- [`fluid_numerics.md`](fluid_numerics.md) describes the Eulerian dust-fluid equations,
  discretization, operator ordering, boundaries, and current limitations
- [`fluid_verification.md`](fluid_verification.md) defines the analytical CUDA verification suite,
  records the available native results, and separates verified claims from unfinished tests
- [`swarm_numerics.md`](swarm_numerics.md) describes the Lagrangian representative-particle
  solver, including transport, stochastic diffusion, radiation, collisions, and outstanding
  validation
- [`architecture.md`](architecture.md) records the fluid–swarm consistency contract, the current
  migration state, what can be shared, and what must remain representation-specific

## Current repository state

The repository is in a staged migration:

- `inc/`, `src/`, and the root `Makefile` contain the active swarm reference implementation
- `new/inc/fluid/` and `new/src/fluid/` contain the migrated fluid implementation
- `new/tst/fluid/` contains the fluid CUDA verification suite and its recorded outputs
- `new/inc/swarm/` and `new/src/swarm/` are reserved for the future swarm migration
- `legacy/` is a frozen recovery copy and is not an active implementation

The `new/` build already selects a representation through `DUST_REPR`, but only
`DUST_REPR=fluid` is implemented there. The root build remains the way to compile the swarm.

## Authority and maintenance

When statements disagree, use this order of authority:

1. current source and model flags
2. recorded native test metrics under `new/tst/fluid/out/`
3. these documents
4. Git history and the historical swarm patch

Resolved bug narratives are intentionally omitted unless they explain a current invariant or
regression test. New numerical changes should update the appropriate canonical document and add or
update a test; they should not create another standalone audit diary.

The source uses spherical coordinates

$$
x=\phi,\qquad y=r,\qquad z=\theta,
$$

with cylindrical coordinates

$$
R=y\sin z,\qquad Z=y\cos z.
$$

Both representations store internal angular variables but write physical linear velocity to
science files. The detailed state contract is in [`architecture.md`](architecture.md).
