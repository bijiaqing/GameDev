# Swarm validation source layout

This directory owns backend-neutral swarm model flags, analytical definitions, validators, and KNN
reference analysis. Native wrappers and runtime drivers live under `val/cuda/swarm/` and
`val/rocm/swarm/`.

Executables, objects, dependency files, KNN build stamps, and numerical results are all generated
below `val/logs/`.

The common dispatcher writes the canonical archive below `val/logs/swarm/BACKEND/`. Focused groups
write below `groups/GROUP/`, direct wrappers use `groups/manual/`, and KNN records remain grouped
under `test_knn/`; these paths prevent exploratory runs from replacing the complete archive.

The retained publication inventory, mathematical references, statistical criteria, commands, and
coverage limits are maintained in
[`doc/swarm_testset.md`](../../../doc/swarm_testset.md).
