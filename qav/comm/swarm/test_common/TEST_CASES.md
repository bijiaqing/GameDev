# Swarm verification case index

The canonical model inventory, physical and statistical references, acceptance criteria, current
results, and coverage limits are maintained in
[`doc/swarm_testset.md`](../../../../doc/swarm_testset.md). The executable matrix and evidence tiers
are defined in `qav/tool/qav_config.py`.

When `--group radial` selects `test_knn`, the runner links only the physical one-dimensional
collision model and executes the collinear radial matrix plus radial-line and inactive-particle edge
checks. Its records remain isolated below
`qav/logs/swarm/BACKEND/groups/radial/test_knn/`.
