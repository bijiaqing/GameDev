# GameDev documentation

This directory is the canonical documentation for the current GameDev dust solvers as of
2026-07-26. It replaces the historical audits, correction diaries, merge proposals, and duplicated
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

The merged project is active at the repository root:

- `inc/fluid/` and `src/fluid/` contain the Eulerian fluid implementation
- `inc/swarm/` and `src/swarm/` contain the Lagrangian swarm implementation
- `inc/share/` and `src/share/` contain the currently shared infrastructure
- `tst/fluid/` contains the fluid CUDA verification suite and its recorded outputs
- `legacy/swarm_before_audit/` and `legacy/swarm_after_audit/` are frozen recovery snapshots

The root build selects exactly one representation through `DUST_REPR` in the chosen model's
`flags.mk`. An optional model-local `const_defs.cuh` has include priority over `inc/fluid/` or
`inc/swarm/`; models without one inherit the representation defaults.

## Authority and maintenance

When statements disagree, use this order of authority:

1. current source, model constants, and model flags
2. recorded native test metrics under `tst/fluid/out/`
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
