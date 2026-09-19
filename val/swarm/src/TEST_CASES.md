# Swarm publication case index

The canonical inventory, equations, statistical criteria, and coverage limits are maintained in
[`doc/swarm_testset.md`](../../../../doc/swarm_testset.md). The executable matrix is defined once in
`val/val_config.py` and consumed by both backend runners.

The common matrix retains deterministic orbit and drag trajectories, stochastic diffusion,
continuous polydisperse initialization, physical collision-rate prescriptions, and the standalone
KNN correctness matrix. Backend-native `chain` groups additionally exercise the production
frozen-bath collision runtime with KD-tree and Morton searches.

Test instrumentation stays outside production `inc/` and `src/`. A model may provide a complete
same-named source or header override only when the analytical setup cannot be expressed through the
production interface. Any such override must preserve the production algorithm being tested and
explain the test-specific change in its source comments.
