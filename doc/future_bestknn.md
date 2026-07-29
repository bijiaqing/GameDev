# Future memory-efficient and distributed exact KNN

## Assessment

The recommended replacement for the current cuKD collision search is a **distributed adaptive Morton cell list with cooperative exact top-\(K\) selection**

This method is a better production target than a flat spatial hash for GameDev because it:

- preserves the existing fixed-\(N_K\) collision estimator
- removes the pointer-like KD traversal and large thread-local candidate heaps
- stores particles and cells in compact contiguous arrays
- refines only cells that become overpopulated in rings or dust clumps
- represents spatial ownership directly through cell keys
- exchanges only boundary ghosts in a multi-GPU calculation
- maps onto CUDA and ROCm radix-sort and block-selection primitives

The method still belongs to the broader family of cell-list searches, but it is not merely a uniform spatial hash. Morton ordering supplies spatially coherent keys, adaptive levels control occupancy, and exact lower-bound termination guarantees the same nearest-neighbor problem as the KD implementation

The safest development sequence is:

1. retain cuKD as the validated reference backend
2. implement a single-GPU Morton cell list with exact \(N_K\)-neighbor results
3. replace per-thread KNN storage with warp- or block-cooperative selection
4. add adaptive refinement for overfull cells
5. add spatial multi-GPU decomposition and ghost exchange
6. compare neighbor sets, collision rates, VRAM, and timings
7. make the Morton backend the default only after the complete validation suite passes

Approximate graph indexes, compressed indexes, and fixed-radius collision estimators are not recommended as the first replacement because they change which particles contribute to the collision rate

## Current KD-tree collision method

During each collision half-interval, the swarm runtime currently:

1. converts every representative position to Cartesian coordinates
2. constructs periodic-image entries when the azimuthal domain is a wedge
3. builds a cuKD tree
4. freezes particle size and represented-grain number at the beginning of each collision batch
5. queries up to \(N_K\) nearest entries for every representative
6. calculates the total collision propensity
7. reduces the maximum propensity to select the collision-batch timestep
8. repeats the same KNN query for representatives that undergo an event
9. samples one collision partner from the pair propensities

The tree remains valid over a collision half-interval because collision events change grain properties but not representative positions

The current query is limited by

$$
q_i = H_{\mathrm{SEARCH}} H_g(R_i)R_i,
$$

where \(H_g\) is dimensionless and \(H_gR\) is the physical gas scale height. The candidate list contains at most \(N_K\) entries within \(q_i\), normally including the query representative itself

For representative \(i\), the unnormalized local propensity is

$$
\widetilde{\lambda}_i
=
\sum_{j\in S_i,\ j\ne i}
N_jK_{ij},
$$

where:

- \(S_i\) is the returned fixed-\(K\) neighborhood
- \(N_j\) is the number of physical grains represented by particle \(j\)
- \(K_{ij}\) is the selected physical collision kernel

If \(d_{K,i}\) is the farthest valid returned distance, the code divides by the accessible neighborhood measure:

$$
\lambda_i
=
\frac{\widetilde{\lambda}_i}
{V_{\mathrm{accessible}}(i,d_{K,i})}.
$$

The host-side representative normalization also contains

$$
\frac{N_P}{(N_K-1)M_{\mathrm{dust}}},
$$

so the numerical estimator is explicitly based on a fixed number of neighboring representatives. A replacement search must preserve that neighborhood definition unless the collision estimator is independently re-derived and revalidated

## Recommended data structure

### Overview

The proposed structure has three logical layers:

1. a coarse spatial decomposition used for GPU ownership and halo exchange
2. a local Morton-ordered cell list on each GPU
3. finer Morton subcells created only inside overfull cells

The structure is pointer-free. Particle records belonging to the same leaf cell are contiguous, and every occupied leaf is described by a compact range

A possible persistent point record is:

```cpp
struct col_point
{
    float3 cartesian;
    uint32_t index_old;
};
```

A possible leaf descriptor is:

```cpp
struct col_cell
{
    uint64_t key;
    uint32_t begin;
    uint32_t count;
    uint16_t level;
};
```

The Cartesian cell bounds can normally be reconstructed from `key` and `level` rather than stored for every cell

### Morton keys

For integer Cartesian cell coordinates \((c_x,c_y,c_z)\), construct a Morton key by interleaving coordinate bits:

$$
\mathcal{M}(c_x,c_y,c_z)
=
\operatorname{interleave}(c_x,c_y,c_z).
$$

In two active dimensions, use the corresponding two-coordinate Morton code. Refinement appends child bits to the parent prefix, so parents and descendants retain useful ordering locality

The key must also encode or accompany:

- refinement level
- periodic-image identity when an explicit boundary ghost is used
- owning GPU or owner rank only when needed outside the local structure

Keys must be collision-free within the configured domain. Hashing a Morton code into a smaller key is unnecessary and would require collision resolution

### Construction

For every collision half-interval:

1. convert owned and ghost representative positions to Cartesian coordinates
2. calculate the coarse cell coordinate
3. generate a Morton key and stable particle identifier
4. radix-sort point records by `(key,index_old)`
5. run-length encode equal keys into occupied-cell ranges
6. mark cells whose occupancy exceeds the refinement target
7. generate child keys only for particles in marked cells
8. repeat sorting and range construction for the marked subsets
9. build a compact lookup from leaf key to leaf descriptor

The secondary `index_old` ordering makes construction deterministic when multiple representatives share one cell

The first implementation should support a fixed maximum refinement level. If a group of coincident or nearly coincident representatives remains overfull at that level, the query kernel must process the dense leaf directly rather than silently dropping candidates

### Cell lookup

A sorted array of leaf keys permits binary search with minimal metadata, but repeated binary searches can be expensive. Two practical choices are:

- a dense coarse-cell offset table with compact child arrays
- an open-addressed lookup table containing only occupied leaf keys

The dense coarse table is attractive for compact domains with moderate cell counts. The sparse lookup is safer for vertically thin disks embedded in a much larger Cartesian bounding volume

The lookup method is an implementation detail and must not change the neighbor ordering or collision estimator

## Exact KNN query

### Geometric lower bound

For query point \(\boldsymbol{x}_i\) and a Cartesian cell with bounds
\([\boldsymbol{b}_{\min},\boldsymbol{b}_{\max}]\), calculate the squared minimum possible distance:

$$
d_{\min,\mathrm{cell}}^2
=
\sum_{\alpha}
\left[
\max\left(
b_{\min,\alpha}-x_{i,\alpha},
0,
x_{i,\alpha}-b_{\max,\alpha}
\right)
\right]^2.
$$

No point in that cell can be closer than this bound

### Search sequence

For every representative \(i\):

1. locate the leaf containing \(\boldsymbol{x}_i\)
2. visit intersecting leaves in increasing lower-bound distance
3. scan their particle records in cooperative tiles
4. retain the best \(N_K\) pairs `(distance_squared,index_old)`
5. update the current farthest retained distance \(d_{K,i}^2\)
6. terminate when all unvisited cells satisfy

$$
d_{\min,\mathrm{unvisited}}^2
\ge
d_{K,i}^2
$$

7. also terminate at the configured radius \(q_i\)

If fewer than \(N_K\) valid entries exist inside \(q_i\), the query must visit every leaf intersecting that sphere and return the smaller valid count. It must reproduce the current capped-query behavior rather than importing more distant representatives

Equal-distance candidates should be ordered by stable particle identifier:

$$
(d_i^2,\mathrm{index}_i)
<
(d_j^2,\mathrm{index}_j).
$$

This produces deterministic results across cell construction orders and makes single- and multi-GPU comparisons much easier

### Minimum-image periodic distance

Every physical representative should contribute at most once to one query neighborhood. Periodic candidates should therefore be compared using the minimum-image distance and deduplicated by stable particle identifier

For a narrow periodic wedge, the search radius should remain smaller than half the periodic physical span at the query radius. If larger radii are permitted, more than one periodic image of the same physical representative can intersect the query sphere and the intended physical sampling measure must be defined explicitly

### Cooperative top-\(K\) selection

The current `cukd::HeapCandidateList<N_K>` is instantiated inside one query thread. For \(N_K=200\), storing a 32-bit distance and 32-bit index costs approximately

$$
8N_K = 1600\ \mathrm{bytes}
$$

before traversal state and compiler overhead. Large thread-local arrays are likely to spill into device local memory and reduce occupancy

The Morton query should assign one warp, or occasionally one complete block, to one query:

1. lanes load a tile of cell candidates
2. lanes calculate squared distances in parallel
3. warp collectives reject candidates beyond \(q_i\) or the current threshold
4. a cooperative partial sort or merge retains the nearest \(N_K\)
5. the retained list resides in shared memory

One shared top-\(K\) list still costs about 1600 bytes, but it is allocated per active cooperative query rather than as a large private array for every query thread. Four warps per block would require about 6.4 KB for neighbor pairs; eight warps would require about 12.8 KB before other scratch storage

The best number of warps per block must be selected by profiling. Register use, shared-memory capacity, candidate-cell occupancy, and GPU architecture can all change the optimum

CUDA CUB and AMD rocPRIM both provide block-level sort, scan, and reduction primitives. Backend-specific primitives should be hidden behind a small portability layer rather than embedded in collision physics

### Lower-memory selection mode

If shared memory becomes the limiting resource, an alternative exact mode can find the \(K\)-th distance through multipass radix selection:

1. count candidate distance prefixes
2. retain the prefix containing the \(K\)-th value
3. repeat for lower bits
4. rescan candidates and sum the selected pair propensities

This avoids retaining all \(N_K\) pairs simultaneously, but it scans candidates several times. It should be treated as a memory-minimizing alternative rather than the expected fastest implementation

## Adaptive refinement

### Why refinement is required

A uniform cell list performs well only while cell occupancy remains controlled. Dust rings, vortices, and clumps can produce cells with

$$
n_c \gg N_K.
$$

Scanning all entries in such a cell for every resident representative can approach

$$
O(n_c^2)
$$

work even though only \(N_K\) entries are retained

Adaptive refinement splits only overfull cells and preserves the compact representation elsewhere

### Refinement policy

Useful initial profiling targets are:

- desired leaf occupancy of approximately 32–64 representatives
- refinement trigger near \(2N_K\)
- a configurable maximum level
- direct dense-leaf processing when the maximum level is reached

These are performance starting points, not mathematical requirements. The final values must be measured for the production particle distribution and GPU architecture

Refinement should be based on query cost as well as raw occupancy. A cell may contain many representatives but still be cheap if most candidates are rejected by the radius cap; conversely, several moderately occupied nearby cells can create a large candidate set

### Degenerate clumps

No spatial hierarchy can separate perfectly coincident representatives geometrically. In that case:

- retain deterministic ordering by particle identifier
- process the dense leaf cooperatively
- report the maximum leaf occupancy
- avoid imposing an unvalidated candidate cap

An approximate truncation would bias collision rates precisely in the scientifically important clumping regime

## Geometry and periodic boundaries

### Radial-azimuthal disk

For a full radial-azimuthal disk, construct Cartesian coordinates

$$
X = R\cos x,
\qquad
Y = R\sin x.
$$

The cell list is two-dimensional and ordinary Cartesian distance gives the correct physical separation. A full \(2\pi\) disk has no artificial azimuthal seam in Cartesian space

### Full 3D disk

For spherical coordinates \((x,y,z)\), use

$$
X = y\sin z\cos x,
\qquad
Y = y\sin z\sin x,
\qquad
Z = y\cos z.
$$

The cell list is three-dimensional. The collision ball and its lower-bound tests are evaluated in this physical Cartesian space

### Azimuthal wedge

The present KD implementation constructs two rotated copies of every representative for a periodic wedge. That can increase the search structure from \(N_P\) to \(3N_P\)

The Morton implementation should instead:

1. identify owned particles within the search halo of either azimuthal face
2. rotate only those boundary records by one wedge width
3. insert the rotated records as local periodic ghosts
4. preserve the original stable particle identifier
5. deduplicate query results by that identifier

The memory overhead then scales with boundary population rather than the entire simulation population

### Axisymmetric models

If an axisymmetric reduced-dimensional model is supported, neighbor distances are still calculated in the represented radial-polar plane and the existing ring-revolution factor remains part of `_get_ball_measure`. The search data structure must not silently change this collision normalization

### Physical boundaries

The existing accessible-ball correction at radial and polar boundaries remains shared collision mathematics. Morton-cell clipping is used only to enumerate candidate records; it does not replace `_get_ball_measure`

## Multi-GPU design

### Spatial ownership

Assign each coarse cell to exactly one GPU. Ownership can initially use regular Cartesian bricks because they have simple neighboring faces and compact halos

Later, strongly imbalanced disks can use:

- weighted cell-plane cuts
- recursive coordinate bisection
- contiguous weighted Morton ranges

Pure equal-particle partitioning is insufficient when clumps make candidate-search cost highly nonuniform. A useful weight combines owned query count, candidates examined, and collision-event frequency

### Owned and ghost records

Each GPU stores:

- complete live state for owned representatives
- compact frozen search records for local owned representatives
- read-only frozen records for neighboring ghosts
- its local coarse and refined cell tables
- send and receive buffers for adjacent GPU domains

The current event kernel updates only representative \(i\) while representative \(j\) is read from the frozen snapshot. This is favorable for domain decomposition: queries for owned \(i\) can consume remote \(j\) as read-only ghosts without remote atomics

### Halo width

A simple conservative halo width is

$$
h_{\mathrm{halo}}
=
\max_i\left[H_{\mathrm{SEARCH}}H_g(R_i)R_i\right].
$$

Every particle within this distance of a neighboring subdomain is exported as a ghost

This is straightforward and exact, but it may communicate too much data when \(H_gR\) varies strongly with radius. A more efficient implementation can exchange coarse cell layers adaptively

### Exact halo certification

For a query owned by one GPU, let \(d_{\mathrm{remote}}\) be the lower-bound distance to every unimported neighboring domain. The local result is globally exact when either:

$$
d_{K,i}
\le
d_{\mathrm{remote}},
$$

or the configured cap satisfies:

$$
q_i
\le
d_{\mathrm{remote}}
$$

after all cells intersecting \(q_i\) have been examined

Queries that fail this certificate must request another halo layer or additional remote cells. They must not accept the local \(K\)-th result as a global result

### Collision-batch synchronization

Positions and velocities remain fixed over one collision half-interval, so their ghost records need to be rebuilt only when the spatial search structure is rebuilt

Grain size and represented-grain number change after every collision batch. To reproduce the single-GPU frozen-snapshot algorithm:

1. exchange updated boundary size and multiplicity fields
2. freeze the local and ghost collision properties
3. calculate local collision rates
4. perform a global maximum reduction for the common batch timestep
5. update owned representatives
6. repeat until the half-interval is complete

The global maximum reduction and property exchange may eventually dominate strong scaling. This is a property of the globally synchronized collision integrator, not of Morton search itself

### Communication backend

A portable baseline should use GPU-aware MPI or a communication abstraction that can support both CUDA and ROCm

Possible later optimizations include:

- CUDA peer access or NCCL send/receive within one node
- NVSHMEM for NVIDIA GPU-initiated exchange
- ROC_SHMEM or an equivalent ROCm path
- overlap of interior queries with boundary packing and halo transfer

The search interface should not depend directly on one communication library

### Scaling expectation

With balanced spatial decomposition, persistent search storage per GPU is approximately

$$
M_{\mathrm{GPU}}
\approx
\frac{M_{\mathrm{owned}}}{G}
+
M_{\mathrm{halo}}
+
M_{\mathrm{cells}}
+
M_{\mathrm{communication}}.
$$

Weak scaling should be favorable while the halo-to-owned ratio remains small. Strong scaling eventually saturates when:

- subdomains become comparable to the halo width
- global reductions dominate computation
- clumps reside predominantly on one GPU
- boundary property exchanges become more expensive than local queries

## VRAM comparison

Assume:

- \(N_P=10^7\)
- \(N_K=200\)
- `sizeof(tree)` is approximately 24 bytes
- one compact point record is approximately 16 bytes

### Existing KD search

| Allocation | Full disk | Wedge with three complete images |
|---|---:|---:|
| KD nodes | approximately 240 MB | approximately 720 MB |
| builder peak | implementation-dependent and substantially larger | scales with the image-expanded input |
| query candidates | approximately 1600 bytes per active thread | approximately 1600 bytes per active thread |

The bundled cuKD builder has additional sorting and construction workspace, so the peak exceeds the persistent tree size

### Adaptive Morton search

| Allocation | Approximate scaling |
|---|---:|
| sorted point records | \(16N_{\mathrm{local}}\) bytes |
| occupied leaf descriptors | roughly 16–24 bytes per leaf |
| lookup table | depends on dense or sparse lookup |
| construction keys | normally 8 bytes per local record |
| sort ping-pong and scratch | implementation-dependent temporary storage |
| cooperative candidates | \(8N_K\) bytes per active warp or block |
| periodic or remote ghosts | boundary population only |

For \(10^7\) records on one GPU, the persistent point array is roughly 160 MB before leaf metadata. The construction peak will be higher because radix sorting requires keys, alternate storage, and temporary workspace, but those buffers can be released or reused after construction

For \(G\) balanced GPUs, the dominant owned arrays scale approximately as \(1/G\), unlike a replicated global tree. Halo records prevent perfect \(1/G\) scaling but should remain a surface term while subdomains are large compared with the search radius

No implementation should allocate an \(N_PN_K\) global neighbor table. Rates are calculated immediately from one cooperative query, and event neighborhoods are reconstructed only for representatives that collide

## Expected performance

### Likely gains

The adaptive Morton method should improve:

- construction throughput through parallel key generation and radix sorting
- memory coalescing because cell records are contiguous
- query occupancy by removing large private candidate heaps
- branch coherence within cell scans
- periodic-wedge memory by creating boundary ghosts only
- multi-GPU capacity through distributed ownership
- CUDA and ROCm portability

GPU cooperative \(k\)-selection has demonstrated high throughput in other nearest-neighbor workloads, although GameDev's low-dimensional, spatially local search must be benchmarked independently

### Costs and risks

Potential costs include:

- repeated candidate scans for exact lower-bound termination
- shared-memory pressure for \(N_K=200\)
- extra refinement passes during construction
- sparse-cell lookup overhead
- difficult load balancing for evolving dust clumps
- halo exchange and global maximum reductions
- degenerate dense leaves that cannot be refined geometrically

The method is expected to outperform the KD implementation in smooth and moderately clumped large runs, but no universal speedup should be claimed before profiling

### Important metrics

Record at least:

- mean, median, 95th-, and 99th-percentile leaf occupancy
- maximum leaf occupancy
- candidates examined per query
- leaf cells visited per query
- fraction of queries requiring halo expansion
- refinement-level distribution
- structure-build time
- rate-query time
- event-query time
- halo packing, communication, and unpacking time
- global-reduction time
- peak and persistent VRAM
- achieved occupancy and local-memory transactions

## Regimes favoring each method

| Regime | Favored backend | Reason |
|---|---|---|
| smooth radial-azimuthal disk | adaptive Morton | coherent 2D cells and controlled occupancy |
| smooth full 3D disk | adaptive Morton | contiguous local scans when cell width is well tuned |
| \(N_P\gtrsim10^6\) | adaptive Morton | compact arrays and parallel construction become valuable |
| frequent structure rebuilds | adaptive Morton | radix construction is highly parallel |
| narrow periodic wedge | adaptive Morton | only boundary ghosts are duplicated |
| multi-GPU calculation | adaptive Morton | cells provide ownership and halo units |
| CUDA and ROCm portability | adaptive Morton | relies on common radix and cooperative primitives |
| VRAM-constrained calculation | adaptive Morton | no persistent global tree or complete periodic images |
| moderate rings and clumps | adaptive Morton | local refinement controls occupancy |
| extreme coincident clump | neither is automatically cheap | geometric structures cannot separate coincident points |
| small validation problem | KD tree | already implemented and useful as a reference |
| scientifically critical regression | both | compare exact neighborhoods and collision statistics |
| approximate similarity search | graph or IVF index | not recommended for the collision estimator |

The KD backend should remain available until the Morton implementation has passed smooth, sparse, boundary, and strong-clumping validation

## Keeping both methods

### Model-facing selection

Use one explicit build setting:

```make
COLLISION_SEARCH := kdtree
```

or:

```make
COLLISION_SEARCH := morton
```

The Makefile can define one internal macro:

```make
COLLISION_SEARCH ?= kdtree

ifeq ($(COLLISION_SEARCH),kdtree)
NVCC += -DCOLLISION_KDTREE
else ifeq ($(COLLISION_SEARCH),morton)
NVCC += -DCOLLISION_MORTON
else
$(error COLLISION_SEARCH must be kdtree or morton)
endif
```

When `COLLISION` is enabled, compile-time validation should require exactly one backend:

```cpp
#if defined(COLLISION_KDTREE) == defined(COLLISION_MORTON)
#error "COLLISION requires exactly one collision-search backend"
#endif
```

Only the selected backend should be compiled, allocated, and initialized

### Search structure versus collision estimator

`COLLISION_SEARCH` selects only the data structure used to recover neighbors. It must not change:

- \(N_K\)
- the search cap \(q_i\)
- pair-rate physics
- accessible-ball normalization
- collision-batch timestep
- Bernoulli event sampling
- partner selection
- coagulation or fragmentation outcomes

If a fixed-radius collision estimator is ever introduced, it needs a separate setting and a separate derivation. It must not be hidden behind `COLLISION_SEARCH=morton`

### Common query result

Both backends should expose the same logical result:

- stable particle identifier
- squared distance
- number of valid returned entries
- farthest valid distance
- deterministic tie ordering

The rate and event kernels should share collision mathematics and call a backend-specific query routine

### Conditional compilation

Under `COLLISION_KDTREE`, retain:

- cuKD types and traits
- `dev_col_tree`
- bounding box
- KD construction
- KD query implementation

Under `COLLISION_MORTON`, allocate:

- sorted local point records
- leaf descriptors
- leaf lookup
- construction key and scratch buffers
- ghost records
- optional multi-GPU communication buffers

cuKD headers must not be included by Morton-only translation units. This separation is important for a future ROCm build

### Reproducibility metadata

Every collision run should record:

```text
COLLISION_SEARCH
N_K
H_SEARCH
Morton base-cell width
Morton target occupancy
Morton refinement trigger
Morton maximum level
top-K selection mode
number of GPUs
domain-decomposition mode
halo policy
```

The search backend does not alter the particle checkpoint format because the structure is rebuilt from particle positions, but recording these settings is necessary for reproducibility and performance diagnosis

## Required production-code changes

### `inc/swarm/const_defs.cuh`

Move KD-only structures and constants under `COLLISION_KDTREE`. Add Morton point and leaf types under `COLLISION_MORTON`

Keep physical collision constants independent of the selected backend:

- `N_K`
- `H_SEARCH`
- `CFL_COL`
- collision-kernel selection

Morton tuning values should initially be model-overridable constants rather than hard-coded kernel literals

### `inc/swarm/_collision.cuh`

Keep pair physics, boundary measure, and collision outcomes in the common collision header

Move direct `cukd/knn.h` and `HeapCandidateList` dependencies into a KD-specific query header. Add a Morton-specific query header containing:

- Cartesian cell-distance lower bound
- leaf lookup
- cooperative exact top-\(K\) selection
- periodic minimum-image handling
- halo-exactness certification

### `inc/swarm/swarm_kern.cuh`

Expose only backend-relevant construction kernels and descriptors. Common rate and event kernels should receive one selected search-view type rather than unconditional KD pointers

### `src/swarm/col_tree_init.cu`

Retain this file only for `COLLISION_KDTREE`

Add separate Morton construction kernels for:

- Cartesian record generation
- coarse key generation
- occupied-cell range construction
- overfull-cell marking
- child-key generation
- ghost-record insertion

Global radix sorting and run-length encoding can initially use Thrust/CUB on CUDA and rocThrust/rocPRIM on ROCm, behind a host-side portability wrapper

### `src/swarm/col_rate_calc.cu`

Keep the pair-propensity summation and accessible-volume normalization unchanged. Replace the direct cuKD call with the selected exact-query interface

The Morton path should calculate the rate directly from the cooperative top-\(K\) result and store only the total rate and farthest valid distance

### `src/swarm/col_event_run.cu`

Reconstruct the same exact neighborhood and sample the collision partner from the same pair propensities. Keep the frozen-state logic and update-only-\(i\) behavior unchanged

### `src/swarm/swarm_runtime.cu`

Introduce a backend owner responsible for allocation, construction, and release

For a future multi-GPU path, also add:

- rank and device ownership
- particle migration
- ghost packing and exchange
- ghost property refresh after each collision batch
- global maximum collision-rate reduction
- local and global timing diagnostics

The first Morton implementation should remain single-GPU so search correctness can be isolated before communication is introduced

### `Makefile`

Select backend-specific objects and dependencies from `COLLISION_SEARCH`. A Morton-only build must not compile cuKD

The CUDA implementation should use CUB/Thrust where useful, while the interface should permit rocPRIM/rocThrust equivalents

## Validation requirements

### Search correctness

1. compare against brute force for small random particle sets
2. compare exact neighbor identifiers and distances against cuKD
3. test fewer than \(N_K\) neighbors inside \(q_i\)
4. test equal-distance ties
5. test coincident representatives
6. test particles exactly on cell and refinement boundaries
7. test radial and polar physical boundaries
8. test full-\(2\pi\) and wedge periodicity
9. verify every periodic physical identifier appears at most once
10. test smooth, sparse, ring, vortex, and compact-clump distributions

### Collision correctness

1. compare per-particle collision rates
2. compare maximum rate and batch timestep
3. compare selected partner probability distributions
4. rerun constant-, linear-, product-, and custom-kernel tests
5. compare size-distribution evolution statistically
6. verify representative mass conservation
7. confirm no change to accessible-ball boundary normalization

### Multi-GPU correctness

1. compare one-GPU and two-GPU neighbor sets
2. place queries and neighbors on opposite sides of every decomposition face
3. test adaptive halo expansion and certification
4. test periodic wedges crossing both a physical GPU boundary and an azimuthal seam
5. compare globally reduced batch timesteps
6. verify ghost size and multiplicity refresh after every batch
7. test particle migration between owners
8. compare two-, four-, and eight-GPU statistical collision evolution

Floating-point summation order may prevent bitwise equality across backends or GPU counts. Neighbor identifiers should nevertheless agree exactly outside equal-distance ties, and rate differences should remain at floating-point ordering level

### Performance validation

Measure:

- persistent and peak VRAM
- build time
- rate-query time
- event-query time
- candidates per query
- local-memory loads and stores
- shared-memory use
- achieved occupancy
- halo volume
- communication time
- global-reduction time
- strong and weak multi-GPU scaling

Do not infer a speedup only from asymptotic complexity

## Difficulty and implementation order

### Moderate work

- compact Morton record and key definitions
- deterministic key generation
- radix-sort construction
- occupied-cell range construction
- uniform-cell exact search
- backend selection in the Makefile

### Substantial work

- efficient cooperative \(N_K=200\) selection
- exact mixed-level traversal
- sparse leaf lookup
- periodic minimum-image deduplication
- adaptive refinement without excessive rebuild cost
- performance portability between CUDA and ROCm

### High-complexity work

- multi-GPU ownership and migration
- exact adaptive halo exchange
- per-batch ghost-property synchronization
- global collision-timestep coordination
- dynamic workload balancing under dust clumping
- communication and computation overlap

The change is architecturally contained within search construction, query, and future distributed-runtime layers. Pair collision physics, coagulation and fragmentation outcomes, frozen-rate integration, and particle checkpoint contents should not require redesign

## References

- Johnson, Douze, and Jégou (2017), [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), for cooperative GPU \(k\)-selection
- García et al. (2012), [Multi-GPU SPH through spatial decomposition, radix sorting, and halo exchange](https://arxiv.org/abs/1210.1017)
- NVIDIA, [CCCL/CUB block-wide CUDA primitives](https://github.com/NVIDIA/cccl)
- AMD, [rocPRIM block radix sort](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.0.2/block_ops/ops_classes/sort.html)
- NVIDIA cuVS, [multi-GPU nearest-neighbor distribution and result merging](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/multi-gpu), useful as a comparison with generic sharded indexes
- Howard et al. (2019), [Quantized BVH neighbor search](https://arxiv.org/abs/1901.08088), showing that hierarchical alternatives can outperform uniform grids in some particle regimes
- Bayraktar et al. (2009), [GPU-based neighbor search](https://repository.bilkent.edu.tr/items/c7fc4617-b45d-4976-bde1-130c68fb09e8)

## Recommendation

Keep both backends:

```make
COLLISION_SEARCH := kdtree
COLLISION_SEARCH := morton
```

Use cuKD as the reference while developing a pointer-free adaptive Morton cell list. Preserve the current \(N_K\)-neighbor estimator, radius cap, accessible-ball measure, and collision-event mathematics

The first production milestone should be a single-GPU exact Morton search that matches brute force and cuKD. The second should add refinement for clumped disks. Only after those are stable should the code introduce multi-GPU spatial ownership, ghost exchange, and global collision-batch synchronization

The adaptive Morton method is the preferred long-term backend for large, memory-constrained, CUDA/ROCm-portable, and multi-GPU runs. The KD tree remains valuable for regression tests and for diagnosing distributions in which Morton occupancy or refinement becomes unfavorable
