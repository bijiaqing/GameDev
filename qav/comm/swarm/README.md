# Swarm QAV source layout

This directory owns backend-neutral swarm model flags, analytical definitions, validators, and KNN
reference analysis. Native wrappers and runtime drivers live under `qav/cuda/swarm/` and
`qav/rocm/swarm/`.

Each common test executable is generated beside its model's `flags.mk` and is ignored by Git.

The common dispatcher writes the canonical archive below `qav/logs/swarm/BACKEND/`. Focused groups
write below `groups/GROUP/`, direct wrappers use `groups/manual/`, and KNN records remain grouped
under `test_knn/`; these paths prevent exploratory runs from replacing the complete archive.

The complete model inventory, mathematical references, statistical criteria, commands, current
evidence, and remaining coverage are maintained in
[`doc/swarm_testset.md`](../../../doc/swarm_testset.md).
