# Attribution

This experimental ROCm tree reproduces the numerical and physical implementation of GameDev while translating its GPU backend from CUDA to HIP

The KD-tree implementation under `inc/swarm/kdtree/` is a modified HIP port of [cudaKDTree](https://github.com/ingowald/cudaKDTree), copyright 2018-2023 Ingo Wald, and retains the upstream Apache License 2.0 notices in its source files

The bitonic-sort material under `inc/swarm/kdtree/cubit/` derives from [cudaBitonic](https://github.com/ingowald/cudaBitonic), copyright 2018-2023 Ingo Wald, and retains its original copyright and license notices

Both components are distributed under the [Apache License 2.0](inc/swarm/kdtree/Apache-2.0.txt)

No claim of authorship is made over those upstream algorithms; the work in this tree is limited to integration, HIP portability changes, and GameDev-specific verification
