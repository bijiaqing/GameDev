# Future spatial hashing for swarm collisions

## Assessment

Replacing the cuKD tree with spatial hashing is feasible and probably worthwhile, especially for ROCm portability and the current `N_P = 10^7`, `N_K = 200` configuration. It is not a trivial substitution

The safest implementation is a sorted spatial cell list that still returns the exact `N_K` nearest representatives. That preserves the existing collision-rate mathematics and limits the change to the neighbor-search layer. A faster fixed-radius hash can eliminate almost all per-thread neighbor storage, but it changes the numerical estimator and therefore requires a separate convergence study

The recommended path is:

1. implement exact-K spatial hashing alongside cuKD
2. compare neighbor sets, collision rates, VRAM, and timings
3. investigate fixed-radius streaming only after exact-K validation
4. retain an adaptive or fallback treatment for strongly clumped cells

## Current KD-tree collision method

During every collision half-step, the swarm runtime:

1. converts every representative position to Cartesian coordinates
2. adds periodic azimuthal images when needed
3. constructs a cuKD tree
4. freezes grain size and represented grain number
5. queries `N_K = 200` nearest entries for every representative
6. calculates its total collision rate
7. uses the maximum rate to choose a collision substep
8. queries the same neighborhood again for representatives that actually collide
9. samples a partner from the pair propensities

The Cartesian tree node contains:

```cpp
float3 cartesian;
int    index_old;
int    split_dim;
int    image;
```

This is expected to occupy 24 bytes per tree node under the current CUDA layout

The rate kernel effectively computes

$$
\lambda_i
=
\frac{1}{V_i(r_i)}
\sum_{j\in S_i,\ j\ne i}
w_{ij},
$$

where:

- $S_i$ is the returned `N_K`-entry neighborhood, including $i$
- $r_i$ is the distance to the most distant returned entry
- $V_i(r_i)$ is the boundary-corrected local measure
- $w_{ij}$ is the pair propensity produced by `_get_col_rate_ij`

The frozen-rate event probability is

$$
P_i=1-\exp(-\lambda_i\Delta t_{\rm col}),
$$

and the conditional probability of selecting partner $j$ is

$$
P(j\mid i)
=
\frac{w_{ij}}{\sum_{k\in S_i,\ k\ne i}w_{ik}}.
$$

Spatial hashing must preserve these quantities if it is intended as a purely computational replacement

## Recommended data structure

A sorted uniform-grid cell list is preferable to an atomic linked-list hash

For every physical or periodic-image entry $a$, calculate a Cartesian position $\boldsymbol p_a$ and integer cell coordinates

$$
b_\alpha
=
\left\lfloor
\frac{p_\alpha-p_{\alpha,\min}}{\ell}
\right\rfloor,
$$

where $\ell$ is the spatial cell width

Construct a collision-free 64-bit cell key such as

$$
h_a
=
b_x+N_{bx}(b_y+N_{by}b_z).
$$

The principal GPU arrays are then

```cpp
uint64_t cell_key[N_H];
uint32_t particle_id[N_H];
```

where $N_H$ is the number of physical entries plus any required periodic images

The build consists of:

1. calculate `(cell_key, particle_id)` pairs
2. radix-sort them by `cell_key`
3. determine the start and count of each occupied cell
4. optionally construct a sparse key-to-range lookup table

This is the standard GPU uniform-grid design used in NVIDIA's particle sample, which supports both atomic and radix-sort construction. Sorted construction should provide more coherent neighbor reads than linked lists. The same design is portable to ROCm through rocThrust or a thin CUB/rocPRIM radix-sort wrapper

References:

- [CUDA particle sample](https://docs.nvidia.com/cuda/archive/10.0/cuda-samples/index.html)
- [CUDA uniform-grid particle white paper](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf)
- [rocPRIM sorting documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html)

## Exact-K spatial-hash query

An exact-K implementation can reproduce the present KD result

For representative $i$:

1. find its hash cell
2. visit the cell containing $i$
3. visit successive shells of surrounding cells
4. calculate the exact Cartesian distance to every candidate
5. retain the nearest `N_K` entries in a maximum heap
6. stop when no unvisited cell can contain a closer entry

For an axis-aligned Cartesian grid, after visiting a rectangular collection of cells, calculate the minimum distance $d_{\rm out}$ from $\boldsymbol p_i$ to the exterior of that visited region

The query is complete when either

$$
r_K \le d_{\rm out},
$$

where $r_K$ is the current K-th distance, or

$$
d_{\rm out}\ge q_i,
$$

where

$$
q_i=H_{\rm SEARCH}H_g(R_i)
$$

is the configured maximum search radius

This produces the exact `N_K` nearest entries inside $q_i$, apart from roundoff-level differences and ambiguous equal-distance ties

The existing calculations can then remain unchanged:

- `N_K`
- `H_SEARCH`
- `_get_ball_measure`
- `lambda_0`
- `_get_col_rate_ij`
- `dev_col_dist`
- the frozen-rate timestep
- the Bernoulli event probability
- partner weighting
- coagulation and fragmentation updates

This is the appropriate first implementation

### Per-thread storage

The current cuKD query uses:

- $200\times8=1600$ bytes for the candidate heap
- approximately $30\times16=480$ bytes for the closest-corner traversal stack

One query thread may therefore require approximately 2.08 KiB of local state before other variables are considered

Exact-K hashing still needs the 1600-byte heap, but removes the approximately 480-byte tree traversal stack and its irregular node accesses. It improves local-memory pressure but does not eliminate its largest component

## Fixed-radius spatial hashing

A more aggressive design uses every representative inside a prescribed radius $q_i$:

$$
S_i(q_i)=
\left\{
j:\lVert\boldsymbol p_j-\boldsymbol p_i\rVert<q_i
\right\}.
$$

The rate would have the form

$$
\lambda_i
=
\frac{1}{V_i(q_i)}
\sum_{j\in S_i(q_i),\,j\ne i}w_{ij}.
$$

This requires no nearest-neighbor heap. The rate kernel can stream through the cell contents using scalar accumulators. The event kernel can stream through the same cells again while accumulating partner weights until the sampled threshold is crossed

This removes both the 1600-byte heap and the cuKD traversal stack

However, it is not a drop-in numerical replacement because the neighbor count

$$
M_i=\lvert S_i(q_i)\rvert-1
$$

varies with position and time. The current normalization contains `N_P/(N_K - 1)/M_dust`, which is explicitly tied to a fixed `N_K - 1` neighbor sample. It cannot simply be retained for variable $M_i$, nor is replacing `N_K - 1` with $M_i$ automatically correct without re-deriving the intended representative-particle sampling measure

Fixed-radius behavior also introduces:

- noisy rates where $M_i$ is small
- expensive queries where $M_i$ is large
- a new physical or numerical smoothing scale
- abrupt changes when representatives enter or leave the support
- potentially smaller global timesteps when a dense support produces a large rate

It may ultimately be faster, but it should be treated as a new collision estimator rather than only a data-structure optimization

## Geometry and periodic boundaries

### Full radial-azimuthal disk

A Cartesian 2D hash uses

$$
p_x=y\sin z\cos x,
\qquad
p_y=y\sin z\sin x.
$$

For a complete $2\pi$ domain, no periodic images are needed because $x=-\pi$ and $x=+\pi$ map continuously in Cartesian space

### Full 3D disk

Use the complete Cartesian position

$$
\begin{aligned}
p_x &= y\sin z\cos x,\\
p_y &= y\sin z\sin x,\\
p_z &= y\cos z.
\end{aligned}
$$

Distance rejection and boundary-volume calculations remain physically Cartesian

### Azimuthal wedge

The lowest-risk initial implementation retains the existing $\pm\Delta x$ images. It keeps the current $N_H=3N_P$ worst case but isolates the first change to the search algorithm

A later implementation can hash native azimuthal cells periodically and evaluate distance with

$$
\Delta x
=
\operatorname{remainder}(x_j-x_i,\ X_{\max}-X_{\min}).
$$

The exact spherical distance is then

$$
d_{ij}^2
=
y_i^2+y_j^2
-
2y_iy_j
\left[
\cos z_i\cos z_j+
\sin z_i\sin z_j\cos(\Delta x)
\right].
$$

This removes all periodic images but makes conservative cell enumeration more complicated. It should not be combined with the first hash prototype

### Axisymmetric models

For `N_X == 1`, the search is intrinsically radial or radial-polar, while `_get_ball_measure` restores the missing azimuthal measure through $2\pi R$. A dimension-specialized 1D or 2D hash is preferable to hashing degenerate Cartesian coordinates

## Cell-width selection

Cell width is the main performance parameter

Using the simulation mesh directly is not advisable. With the current defaults,

$$
\frac{N_P}{N_XN_YN_Z}
=
\frac{10^7}{10^4}
=
1000
$$

representatives occupy each mesh cell on average. Scanning a $3\times3$ 2D stencil would then examine approximately 9000 candidates per representative even though only 200 are needed

A better initial target is approximately 16 to 64 representatives per occupied hash cell. For a uniform distribution in $d$ dimensions,

$$
\ell
\approx
\left(
\frac{n_{\rm target}V_{\rm domain}}{N_P}
\right)^{1/d}.
$$

The exact-K query can expand over cell shells until 200 entries are certified

Dust clumping is the principal difficulty. If one hash cell contains $n_c\gg N_K$ representatives, every representative in that cell must inspect all $n_c$ candidates before identifying its nearest 200. The local work approaches $O(n_c^2)$, whereas a balanced KD tree is less sensitive to local occupancy

A production hash should eventually provide one of:

- recursive subdivision of cells above an occupancy threshold
- a two-level or hierarchical hash
- a smaller globally selected cell width
- a fallback KD or BVH search for overfull cells

A simple uniform hash must be stress-tested against concentrated rings and clumps before replacing cuKD

## VRAM comparison

The estimates below use:

- `N_P = 10^7`
- `sizeof(tree) ≈ 24` bytes
- 64-bit hash keys
- 32-bit particle indices
- decimal MB

### Existing KD-tree search

| Allocation | Full $2\pi$ | Wedge with three copies |
| --- | ---: | ---: |
| `dev_col_tree` | 240 MB | 720 MB |
| four collision scalar arrays | 320 MB | 320 MB |
| cuKD tree-related build peak | approximately 720 MB | approximately 2.16 GB |
| candidate heap per active query thread | 1600 B | 1600 B |
| traversal stack per active query thread | approximately 480 B | approximately 480 B |

The cuKD build estimate comes from its bundled `builder.h`, which describes the Thrust builder as using total memory of approximately three times the input data

Thread-local storage should not be multiplied by all `N_P` and interpreted as a permanent allocation, but it strongly affects compiler spilling, local-memory traffic, occupancy, and runtime backing storage

### Sorted spatial hash

One sortable entry needs

$$
8\ {\rm bytes\ for\ key}
+
4\ {\rm bytes\ for\ index}
=
12\ {\rm bytes}.
$$

A double-buffer radix sort therefore needs approximately

$$
24N_H\ {\rm bytes}.
$$

For $N_H=N_P=10^7$, that is approximately 240 MB before library-specific radix-sort scratch storage

The cell lookup costs approximately:

- dense start/count table: $8C$ bytes
- sparse open-addressed table: approximately $16C/\alpha$ bytes

where $C$ is the number of occupied cells and $\alpha$ is the hash-table load factor

For about 32 representatives per occupied cell,

$$
C\approx312\,500.
$$

At $\alpha=0.7$, a sparse 16-byte cell table occupies approximately 7.1 MB

After construction, a minimal query representation needs only the sorted particle-index array and the cell-range table. The key and alternate sort buffers can be retained as reusable build workspace or repurposed between builds

### Likely memory result

For a full $2\pi$ domain:

- persistent search memory can fall from about 240 MB of tree nodes to roughly 40 MB of sorted indices plus a small cell table
- reusable build workspace raises the actual retained allocation
- peak build memory should be lower than cuKD, but radix-sort scratch must be measured

For an azimuthal wedge, wrapped periodic hashing offers the largest gain because it can remove the current threefold image expansion. Retaining images during the first prototype gives a smaller memory improvement

Fixed-radius streaming additionally removes approximately 1.6 KiB of candidate storage per active query thread. Exact-K hashing does not

## Expected performance

There is no reliable universal speedup estimate without profiling a collision-enabled model. The current `swarm_fiducial` configuration does not enable `COLLISION`, so the repository does not yet contain representative timing evidence

The cost models are approximately

$$
T_{\rm KD}
=
T_{\rm build,KD}
+
N_PT_{\rm traverse}
+
N_PT_{\rm heap}
+
P_{\rm event}N_PT_{\rm query},
$$

and

$$
T_{\rm hash}
=
T_{\rm key}
+
T_{\rm sort}
+
N_P
\left(
N_{\rm cells}T_{\rm lookup}
+
N_{\rm cand}T_{\rm distance}
+
T_{\rm selection}
\right).
$$

### Likely gains

Spatial hashing provides:

- one radix sort rather than hierarchical KD construction
- contiguous cell ranges
- simpler control flow
- fewer divergent tree branches
- no KD traversal stack
- easier CUDA and ROCm portability
- potential removal of all candidate storage in fixed-radius mode

For a smooth disk with 16 to 64 entries per cell and a few hundred candidates examined per particle, exact-K hashing could plausibly outperform the current query substantially

### Likely losses

Spatial hashing becomes unfavorable when:

- cell occupancy greatly exceeds `N_K`
- the distribution spans very different density scales
- $H_gR$ varies strongly across the domain
- sparse regions require searching many empty cell shells
- cell size is selected from the global minimum spacing
- the full search radius contains thousands of representatives

Because dust clumping is a scientific objective, the overfull-cell case is not pathological and must be supported deliberately

Published results also demonstrate that grid-based search is not universally superior. A quantized BVH study reported roughly two to four times higher neighbor-search performance than uniform grids for its molecular-dynamics benchmarks. This does not predict GameDev's result, but it warns against assuming that hashing is automatically faster

References:

- Teschner et al. (2003), [optimized spatial hashing](https://cgl.ethz.ch/Downloads/Publications/Papers/2003/Tes03/Tes03.pdf)
- Bayraktar et al. (2009), [GPU-based neighbor search](https://repository.bilkent.edu.tr/items/c7fc4617-b45d-4976-bde1-130c68fb09e8)
- Howard et al. (2019), [quantized BVH neighbor search](https://arxiv.org/abs/1901.08088)

## Required production-code changes

### `inc/swarm/const_defs.cuh`

Remove or replace:

- cuKD builder include
- `bbox`
- `tree`
- `tree_traits`
- `N_T` and `NB_T`

Add:

- cell-key type
- packed particle/image identifier
- hash cell-width or occupancy configuration
- hash-table capacity and load parameters
- optional overfull-cell threshold

### `inc/swarm/_collision.cuh`

Remove:

- `cukd/knn.h`
- `candidatelist`

Add:

- Cartesian-distance helper
- cell-key encoder
- cell-range lookup
- exact-K cell-shell iterator
- periodic-image or wrapped-coordinate handling

The physical pair functions such as `_get_col_rate_ij` can remain unchanged

### `src/swarm/col_tree_init.cu`

Replace this with kernels that:

1. calculate cell keys and encoded particle identifiers
2. identify starts and counts after sorting
3. optionally populate the sparse cell table

### `src/swarm/col_rate_calc.cu`

Replace the cuKD query with the shared cell-neighborhood iterator. In exact-K mode, the subsequent summation and measure calculation should remain unchanged

Launch one thread per physical representative with `NB_P`, rather than one thread per tree entry followed by image-node rejection

### `src/swarm/col_event_run.cu`

Use the same neighbor iterator to reconstruct the identical neighborhood. Keep the frozen-rate probability and pair-weight selection unchanged

### `src/swarm/swarm_runtime.cu`

Replace:

- `dev_boundbox`
- `dev_col_tree`
- `cukd::buildTree`

Add:

- key and index buffers
- sort workspace
- occupied-cell ranges
- optional sparse lookup table
- sort and range-build sequence

The hash is still built once per collision half-interval because positions remain fixed inside `evolve_collisions`

### `inc/swarm/swarm_kern.cuh`

Replace tree and bounding-box parameters with sorted indices and cell-table parameters

### `Makefile`

Replace `col_tree_init.o` with the hash-construction objects. After validation, the cuKD dependency can be removed from the swarm branch, eliminating the largest third-party obstacle to ROCm support

## Validation requirements

Before using the hash in production, require:

1. brute-force small-`N` neighbor comparisons
2. exact neighbor-set comparisons against cuKD
3. complete-$2\pi$ seam tests
4. wedge-periodicity tests
5. radial and polar boundary tests
6. inactive-particle tests
7. 2D and 3D tests
8. uniform, power-law, ring, and strongly clumped distributions
9. pair-rate comparisons for constant, additive, product, and physical kernels
10. repeated partner-sampling distribution tests
11. total coagulation convergence with `CFL_COL`
12. convergence with `N_P`, `N_K`, and hash cell width
13. peak-VRAM and kernel profiling
14. CUDA and ROCm compilation and comparison

For exact-K hashing, rates should agree with cuKD to floating-point tolerance except for equal-distance ties

For fixed-radius hashing, equality is not expected and independent convergence criteria are required

## Difficulty and effort

| Goal | Difficulty | Rough effort |
| --- | --- | --- |
| minimal fixed-radius prototype | moderate coding, high numerical risk | several days |
| exact-K sorted hash prototype | moderate | roughly one week |
| exact-K implementation with complete geometry tests | moderate to high | around one to two weeks |
| clump-safe hierarchical hash | high | several additional weeks |
| fully validated CUDA and ROCm replacement | high | project measured in weeks |

The change is relatively well isolated architecturally: the physical collision kernels, frozen snapshots, timestep logic, Bernoulli sampling, coagulation, and fragmentation do not need redesign. Most changes lie in construction and traversal interfaces

The hard parts are:

- certifying exact K-nearest completion
- choosing cell width
- preserving wedge periodicity
- handling dense clumps
- controlling GPU local memory
- validating collision-rate normalization if fixed-radius neighborhoods are introduced

## Recommendation

Implement an optional exact-K sorted spatial hash first, keeping cuKD as a reference:

```text
COLLISION_SEARCH = kdtree
COLLISION_SEARCH = cellhash
```

Use 64-bit collision-free cell keys, sorted particle identifiers, and a sparse occupied-cell lookup. Initially preserve periodic images and the existing `N_K`-neighbor rate formula. Once this matches cuKD, add wrapped azimuthal hashing and overfull-cell refinement

Do not initially replace the adaptive KNN estimator with a fixed-radius estimator. That version is potentially faster and substantially lighter in per-thread memory, but it is a numerical-method change rather than only a data-structure optimization
