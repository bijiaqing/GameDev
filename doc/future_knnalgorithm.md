# Future exact KNN backends

## Scope and present decision

GameDev currently uses cuKD to recover the fixed-\(N_K\) neighborhoods required by the swarm collision estimator. The leading alternative is a pointer-free adaptive Morton cell list with cooperative exact top-\(K\) selection.

The laboratory experiment has succeeded as an exact-search prototype:

- all ordinary and periodic adversarial cases passed
- top-200 neighborhoods matched cuKD and independent brute force
- adaptive refinement remained usable in smooth disks, rings, and strong clumps
- periodic boundary ghosts reduced the persistent search memory substantially
- million-particle performance was comparable to or modestly better than cuKD

This is not yet a production-backend claim. The implementation remains under `lab/` and has not been connected to the production collision-rate and collision-event kernels. The current policy is therefore:

- keep cuKD as the production and regression backend
- continue Morton as an optional exact backend candidate
- preserve the existing collision estimator and physics for both methods
- promote Morton only after production collision and memory tests pass

The recorded numerical tables below were produced on the cluster and transcribed into this document. The current local `lab/results/` directory is empty, so these measurements should be rerun and archived before they are used as release evidence.

## Collision-search contract

During one collision half-interval, the swarm runtime:

1. converts representative positions to Cartesian coordinates
2. creates periodic entries when the azimuthal domain is a wedge
3. constructs the search structure
4. freezes grain size and represented-grain number for one collision batch
5. queries up to \(N_K\) neighbors for every representative
6. calculates each total collision propensity
7. reduces the maximum propensity to determine the batch timestep
8. reconstructs the neighborhood for representatives that undergo an event
9. samples one partner from the pair propensities

Collisions change grain properties but not positions, so the spatial structure remains valid until transport moves the particles.

The location-dependent query cap is

$$
q_i = H_{\mathrm{SEARCH}}H_g(R_i)R_i,
$$

where \(H_g\) is dimensionless and \(H_gR\) is the gas scale height. If \(S_i\) is the returned neighborhood, the unnormalized propensity is

$$
\widetilde{\lambda}_i
=
\sum_{j\in S_i,\ j\ne i}N_jK_{ij},
$$

and the final rate is

$$
\lambda_i
=
\frac{\widetilde{\lambda}_i}
{V_{\mathrm{accessible}}(i,d_{K,i})}.
$$

Here \(N_j\) is the number of grains represented by particle \(j\), \(K_{ij}\) is the physical pair kernel, and \(d_{K,i}\) is the farthest valid returned distance. The host normalization also contains \(N_P/[(N_K-1)M_{\mathrm{dust}}]\), so fixed-\(N_K\) sampling is part of the estimator, not merely a search preference.

A second backend must preserve:

- \(N_K\) and the cap \(q_i\)
- stable physical identifiers and deterministic tie ordering
- accessible-ball normalization
- pair-rate physics
- maximum-rate timestep selection
- event probability and partner sampling
- coagulation and fragmentation outcomes

A fixed-radius or approximate-neighbor estimator would require a separate mathematical derivation and validation.

## Validated Morton design

### Structure construction

The laboratory implementation:

1. maps two- or three-dimensional Cartesian positions to integer cell coordinates
2. interleaves their bits into 64-bit Morton keys
3. radix-sorts records by key and stable physical identifier
4. identifies occupied key ranges
5. recursively refines cells above a target population
6. stores compact node, leaf, and record arrays without pointers

The stable secondary identifier makes construction deterministic when several representatives have the same cell key. A leaf target of 128 records was the best common choice in the completed tuning matrix.

The current hierarchy is assembled on the host after GPU key sorting. This is adequate for algorithm validation but should eventually be replaced by GPU-native leaf construction.

### Exact traversal

For a query point \(\boldsymbol{x}_i\) and a Cartesian node with bounds \([\boldsymbol{b}_{\min},\boldsymbol{b}_{\max}]\), the conservative lower bound is

$$
d_{\min,\mathrm{node}}^2
=
\sum_\alpha
\left[
\max\left(
b_{\min,\alpha}-x_{i,\alpha},
0,
x_{i,\alpha}-b_{\max,\alpha}
\right)
\right]^2.
$$

Nodes are visited in increasing lower-bound distance. Candidate records are retained in the lexicographic order

$$
(d_i^2,\mathrm{id}_i)<(d_j^2,\mathrm{id}_j).
$$

Traversal stops only after every unvisited node satisfies

$$
d_{\min,\mathrm{unvisited}}^2\ge d_{K,i}^2
$$

or lies outside \(q_i\). If fewer than \(N_K\) records exist within \(q_i\), every intersecting node is examined and the smaller valid count is returned.

Repeated subdivision in single precision required a small scale-aware expansion of node bounds. Without it, roundoff could make the nominal lower bound too large and incorrectly prune a valid neighbor.

### Cooperative top-\(K\) selection

The cuKD query instantiates a private `HeapCandidateList<N_K>`. At \(N_K=200\), 32-bit distances and identifiers alone require approximately

$$
8N_K=1600\ \mathrm{bytes}
$$

per active query thread, before traversal state. This can create local-memory traffic and occupancy pressure even when total VRAM is sufficient.

The Morton query assigns one CUDA block to a query. Threads cooperatively evaluate candidate tiles and retain the nearest \(N_K\) pairs in explicitly sized shared memory. No global \(N_PN_K\) neighbor table is allocated; rate queries consume their result immediately and event neighborhoods are reconstructed only when needed.

An early buffered selector contained a race because thread 0 advanced a shared batch offset before all warps had finished using it. Adding a block-wide barrier restored exact results. This is now covered by the adversarial tests.

### Clumps and degeneracy

A uniform Morton list was exact but extremely slow in dense clumps. Adaptive refinement bounds ordinary leaf occupancy and removed this failure mode. It does not eliminate the intrinsic cost of coincident particles: no geometric hierarchy can separate records with identical coordinates. A production backend must therefore retain occupancy, depth, candidate-count, and overflow diagnostics.

## Periodic wedges

Two exact policies were tested:

- **query images:** store only physical records, rotate boundary queries by a wedge width, search again, then merge and deduplicate by physical identifier
- **boundary ghosts:** copy only source records whose search halo intersects a wedge face, rotate them across the seam, then perform one ordinary query

Query images minimize persistent storage but repeat traversal. Boundary ghosts use more records near a seam but were substantially faster for seam-dominated clumps and map naturally to future multi-GPU halos.

Ghost coordinates must be generated with arithmetic consistent with the reference path. Host `std::sin` and `std::cos` introduced near-tie differences relative to GPU `sincosf`; GPU generation removed all observed disagreements.

Every physical representative may contribute at most once to a query. Periodic candidates are therefore deduplicated by stable identifier after applying the minimum-image geometry.

## Laboratory evidence

### Correctness

The completed prototype passed:

- 10 of 10 ordinary adversarial cases in both 2D and 3D
- 10 of 10 periodic adversarial cases in both 2D and 3D
- a twelve-case smooth, ring, and clump matrix at \(N_P=10^5\) and \(10^6\)
- sixteen periodic-wedge cases at \(N_P=10^5\) and \(10^6\)

The cases cover equal-distance ties, coincident particles spanning multiple candidate tiles, an inclusive cutoff boundary, fewer than \(K\) valid neighbors, Morton split planes, both wedge faces, narrow-wedge image overlap, full-\(2\pi\) geometry, and seam-centered clumps.

The validator checks a configured subset against exhaustive CPU search. When GPU backends disagree outside that subset, every disagreement is also checked by brute force and attributed to the incorrect backend.

### Nonperiodic million-particle results

The table reports \(t_{\mathrm{KD}}/t_{\mathrm{Morton}}\), so values above one favor Morton.

| Distribution | Dimension | Query speed ratio | Persistent-memory ratio |
| --- | ---: | ---: | ---: |
| smooth | 2D | 1.090 | 0.706 |
| ring | 2D | 1.063 | 0.733 |
| clump | 2D | 1.075 | 0.723 |
| smooth | 3D | 1.129 | 0.716 |
| ring | 3D | 1.079 | 0.752 |
| clump | 3D | 0.986 | 0.745 |

Morton was faster in five of six cases and 1.4 percent slower in the remaining 3D clump, while using about 25-29 percent less persistent search memory.

### Periodic million-particle results

| Case | cuKD query, ms | Query-image, ms | Boundary-ghost, ms | Ghost speed ratio | Query-image memory ratio | Ghost memory ratio | Records per particle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth 2D | 274.056 | 309.259 | 247.388 | 1.108 | 0.248 | 0.276 | 1.128 |
| ring 2D | 278.378 | 327.188 | 257.273 | 1.082 | 0.240 | 0.269 | 1.128 |
| interior clump 2D | 286.077 | 299.580 | 268.659 | 1.065 | 0.243 | 0.249 | 1.026 |
| seam clump 2D | 305.986 | 776.406 | 269.886 | 1.134 | 0.243 | 0.441 | 1.825 |
| smooth 3D | 504.115 | 577.332 | 466.637 | 1.080 | 0.254 | 0.290 | 1.128 |
| ring 3D | 491.590 | 614.558 | 487.570 | 1.008 | 0.242 | 0.279 | 1.127 |
| interior clump 3D | 503.339 | 554.206 | 493.359 | 1.020 | 0.250 | 0.255 | 1.025 |
| seam clump 3D | 525.594 | 1387.227 | 495.593 | 1.061 | 0.251 | 0.455 | 1.826 |

Boundary-ghost queries were 0.8-13.4 percent faster than cuKD in all eight cases. Their persistent structure used 24.9-45.5 percent of the three-image cuKD structure, a 54.5-75.1 percent reduction. Including construction and one all-particle query, the measured total-cycle speed ratio was approximately 1.046-1.164.

At \(N_P=10^5\), cuKD queries were about 13-30 percent faster. Faster Morton construction still made build plus query faster in seven of eight cases; the artificial 3D seam clump was about 3 percent slower overall.

These timings include host hierarchy assembly and therefore do not assume an unmeasured GPU-native construction benefit.

## Choosing a backend

| Regime | Preferred method | Reason |
| --- | --- | --- |
| small or established single-NVIDIA-GPU run | cuKD | mature reference and often faster at small \(N_P\) |
| memory-sufficient single-GPU production | cuKD initially | present Morton speedup is modest |
| \(N_P\gtrsim10^6\) | benchmark both | measured crossover is distribution dependent |
| narrow periodic wedge | boundary-ghost Morton | avoids three complete image populations |
| minimum persistent wedge memory | query-image Morton | stores only physical records |
| memory-constrained run | Morton | compact arrays and bounded query scratch |
| ROCm target | Morton | built from portable radix and cooperative primitives |
| multi-GPU run | Morton | cells directly define ownership and halos |
| extreme coincident clump | neither automatically | geometry cannot separate identical points |
| scientific regression | both plus brute force | independent exact implementations expose defects |

Even when cuKD memory is tolerable, Morton retains benefits:

- explicit spatial ownership for multi-GPU partitioning
- contiguous structures suitable for CUDA and ROCm
- predictable cooperative scratch instead of a large private heap
- boundary-only periodic and distributed halos
- direct occupancy and workload diagnostics for dust clumps
- stable tie behavior across construction order and GPU platform
- project control over exact halo certification and boundary policy

These advantages must be balanced against maintaining custom construction, traversal, selection, and validation code.

## Selectable production interface

Use one build option:

```make
COLLISION_SEARCH := kdtree
```

or:

```make
COLLISION_SEARCH := morton
```

Exactly one backend should be compiled when `COLLISION` is enabled. Backend-specific types, allocation, construction, and queries should remain inside separate swarm headers and source files; collision mathematics remains in the existing swarm collision implementation. No fluid–swarm file sharing is planned.

Both backends should return the same logical data:

- stable physical identifier
- squared distance
- valid result count
- farthest valid distance
- deterministic distance-and-identifier order

The backend is rebuilt from particle positions, so it does not belong in particle checkpoints. Reproducibility metadata should nevertheless record the backend, \(N_K\), \(H_{\mathrm{SEARCH}}\), Morton leaf target and maximum level, periodic policy, GPU count, decomposition policy, and halo policy.

## Location-dependent periodic halos

Because \(q_i\) depends only on particle position, it remains fixed through collision batches and changes only after transport. A conservative global halo is

$$
h_{\mathrm{halo}}=\max_i q_i.
$$

This is exact but can duplicate too many particles when \(H_gR\) varies strongly with radius.

### Radially binned halo

Divide cylindrical radius into intervals

$$
I_b=[R_b,R_{b+1})
$$

and calculate

$$
c_b=\max_{i:R_i\in I_b}q_i.
$$

For source bin \(s\) and query bin \(b\), the minimum radial separation is

$$
D_{sb}
=
\max\left(0,R_s-R_{b+1},R_b-R_{s+1}\right).
$$

The certified source halo is

$$
h_s
=
\max_{\{b:D_{sb}\le c_b\}}c_b.
$$

A source in bin \(s\) is copied across an azimuthal seam when

$$
d_{\mathrm{seam}}\le h_s+\epsilon.
$$

The bins only select ghost records. They do not create separate trees, and each query still uses its own \(q_i\). A first implementation should compare 32, 64, and 128 logarithmic radial bins against the global-halo result and require identical neighbor sets.

## Multi-GPU extension

### Ownership and records

Each GPU should own coarse cells or contiguous Morton-key ranges and store:

- complete live state for owned representatives
- frozen search records for owned and ghost representatives
- compact node and leaf arrays
- boundary send and receive buffers

Collision events update only representative \(i\) and read \(j\) from the frozen snapshot. Remote candidates can therefore be read-only ghosts without remote atomics.

### Exact halo certification

Let \(d_{\mathrm{remote}}\) be a conservative lower bound to every unimported domain. A local result is globally exact when either

$$
d_{K,i}\le d_{\mathrm{remote}}
$$

or

$$
q_i\le d_{\mathrm{remote}}
$$

after every imported cell intersecting \(q_i\) has been examined. A query that fails this condition must request another halo layer or remote cell range.

### Collision synchronization

Positions remain fixed during a collision half-interval, but size and represented-grain number change after every batch. A distributed implementation must:

1. exchange updated boundary collision properties
2. freeze owned and ghost properties
3. calculate local rates
4. globally reduce the maximum rate for one common batch timestep
5. update owned representatives
6. repeat until the half-interval is complete

The property exchange and global reduction may dominate strong scaling. That cost belongs to the globally synchronized collision integrator, not to Morton search alone.

## Remaining work and promotion criteria

### Production integration

1. add selectable cuKD and Morton owners inside the copied swarm implementation in `lab/`
2. implement the location-dependent cutoff and global boundary-ghost halo
3. compare exact neighbor sets, per-particle rates, maximum rates, and batch timesteps
4. compare partner sampling and size-distribution evolution statistically
5. verify representative-mass conservation and accessible-ball normalization
6. measure persistent and peak VRAM, local-memory traffic, occupancy, and construction/query timings
7. add radially binned halos and require equality with the global policy
8. promote tests to `qav/swarm/` only after the laboratory backend is stable

### Later engineering

- GPU-native hierarchy construction
- CUDA and ROCm performance tuning
- multi-GPU ownership, migration, and exact halo exchange
- batch-wise ghost-property refresh and global timestep coordination
- workload balancing for evolving clumps
- communication/computation overlap

Promotion requires exact neighbors outside defined equal-distance ties, floating-point-level rate agreement, statistically consistent collision evolution, no conservation regression, and a measured capacity or performance advantage in the intended production regime.

## References

- Johnson, Douze, and Jégou (2017), [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- García et al. (2012), [Multi-GPU SPH through spatial decomposition, radix sorting, and halo exchange](https://arxiv.org/abs/1210.1017)
- NVIDIA, [CCCL/CUB block-wide CUDA primitives](https://github.com/NVIDIA/cccl)
- AMD, [rocPRIM block radix sort](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.0.2/block_ops/ops_classes/sort.html)
- NVIDIA cuVS, [multi-GPU nearest-neighbor distribution and result merging](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/multi-gpu)
- Howard et al. (2019), [Quantized BVH neighbor search](https://arxiv.org/abs/1901.08088)
- Bayraktar et al. (2009), [GPU-based neighbor search](https://repository.bilkent.edu.tr/items/c7fc4617-b45d-4976-bde1-130c68fb09e8)
