# Swarm verification case index

The canonical model inventory, physical and statistical references, acceptance criteria, current
results, and coverage limits are maintained in
[`doc/swarm_testset.md`](../../../../doc/swarm_testset.md). The executable matrix and evidence tiers
are defined in `qav/tool/qav_config.py`.

When `--group radial` selects `test_knn`, the runner links only the physical one-dimensional
collision model and executes the collinear radial matrix plus radial-line and inactive-particle edge
checks. Its records remain isolated below
`qav/logs/swarm/BACKEND/groups/radial/test_knn/`.

## QAV source overrides

Test instrumentation is intentionally absent from production `inc/` and `src/`. For a QAV model,
the root Makefile searches the model directory, `qav/BACKEND/swarm/test_common/`, and
`qav/comm/swarm/test_common/` before the production trees. The common QAV runtime and header copies
therefore provide deterministic stochastic seeds, collision-geometry accounting, timing, analytic
collision initialization, and reduced cache diagnostics without adding test branches to production
files. `test_knn/morton/morton_index.cuh` similarly owns the standalone query and digest kernels
used only by the KNN test executables.

Because these are complete overrides, changes to the corresponding production runtime or host,
cache, and Morton headers must be mirrored into the QAV copies while preserving the test-only
sections. `qav/tool/static_check.py` enforces the file boundary and backend parity; native builds
remain responsible for compile- and runtime-level validation.
