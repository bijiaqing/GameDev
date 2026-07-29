# Collision-search laboratory

## Scope

`lab/` is an isolated development copy for the collision-search work proposed in
`doc/future_knnalgorithm.md`. Production files under `inc/swarm/` and `src/swarm/` and canonical tests
under `qav/swarm/` are not modified by this experiment.

The copied swarm snapshot is stored under:

```text
lab/inc/swarm/
lab/src/swarm/
```

The laboratory retains cuKD as the reference and adds a single-GPU exact adaptive Morton cell-list
under `lab/knn/`. The standalone benchmarks compare complete fixed-radius top-$K$ lists. The copied
swarm runtime now also selects either backend while preserving one collision-rate and event
implementation.

## Implemented baseline

The Morton prototype currently provides:

- 64-bit collision-free Morton keys with 21 bits per Cartesian axis
- stable radix-style ordering by cell key and original particle identifier
- compact linear adaptive nodes and contiguous leaf-point ranges
- recursive subdivision until a leaf reaches the target occupancy or maximum level
- nearest-child-first traversal with exact Cartesian node-distance rejection
- one CUDA block per query
- shared-memory top-$K$ merging with deterministic `(distance_squared,index_old)` ordering
- no global $N_PN_K$ neighbor table in the timed full-query path
- correctness comparison against cuKD for a sampled set of queries
- an independent CPU brute-force oracle for a smaller subset of those queries
- build time, full-query time, persistent-memory, leaf-occupancy, cell-visit, and candidate-visit metrics

## Copied swarm integration

The copied runtime supports:

- `COLLISION_SEARCH=kdtree` with the existing one-thread cuKD queries
- `COLLISION_SEARCH=morton` with one cooperative 256-thread block per representative
- the same location-dependent cutoff \(H_{\rm SEARCH}H_g(R)R\)
- exact periodic query images and stable-identifier deduplication for wedges
- the same pair kernel, accessible-ball normalization, frozen collision snapshot, Bernoulli event
  probability, partner weighting, and collision outcome code
- explicit traversal-overflow detection after rate and event queries
- backend build, rate, event, and persistent-memory diagnostics in the run log

The Morton hierarchy contains only physical records. It is rebuilt once for each fixed-position
collision interval and remains valid through its frozen-property collision batches. The current
host-assisted adaptive builder is retained so this stage tests integration correctness before a
GPU-native builder is attempted.

Four matched models are under `lab/mod/`:

| Model | Geometry and purpose |
| --- | --- |
| `collision_disk_2d` | full-\(2\pi\), vertically integrated disk |
| `collision_wedge_2d` | periodic azimuthal wedge, exercising query images |
| `collision_disk_3d` | full 3D disk with transport and diffusion enabled |
| `collision_disk_3d_1m` | the same full 3D integration test with \(10^6\) representatives |

The first three use $10^5$ representatives and the final model uses $10^6$. All use $N_K=200$,
radiation disabled, one short output interval, and the same deterministic $0.5$-to-$2.0$ grain-size
initialization for both backends. The constant-kernel baseline still has neighbor-dependent
propensities through represented grain multiplicity and neighbor-dependent coagulation outcomes.

Build one copied runtime with:

```bash
make -C lab swarm \
    SWARM_MODEL=collision_disk_2d \
    COLLISION_SEARCH=morton \
    ARCH=sm_80
```

Run the complete matched comparison with:

```bash
python3 lab/run_swarm_comparison.py \
    --models collision_disk_2d collision_wedge_2d collision_disk_3d \
    --arch sm_80
```

The runner cleans and rebuilds both backends in separate object directories, retains each complete
`ptxas -v` build log, runs them into separate result directories, and checks:

- byte-identical initial particle states
- exact first-batch neighbor counts and order-independent 64-bit neighbor-set fingerprints
- tightly matching first-batch per-particle collision rates and KNN radii before stochastic events
- finite final fields
- Bonferroni-controlled two-sample Kolmogorov-Smirnov tests of evolved field distributions
- represented-mass drift and final mass mismatch
- componentwise pointwise errors and final RNG-state equality as diagnostics only
- backend wall time and reported search memory

Simulation output is streamed live while being copied to `run.txt`. Lines beginning with
`[PROGRESS]` report the completed simulation time and collision timestep after every collision
batch; dynamics-enabled models also report the current dynamics timestep and its target time before
the composed step starts.

Results are written to `lab/out/swarm/comparison_MODEL.json`. A failed equality test is a
diagnostic boundary, not permission to loosen tolerances: inspect neighbor ordering, rate sums, and
event selection first.

Every accepted neighbor contributes a mixed stable-particle ID to a commutative 64-bit fingerprint,
and the accepted-neighbor count is stored separately. Equal fingerprints and counts provide a
high-confidence order-independent check that the backends selected the same physical neighbors;
the standalone full-list and brute-force suites remain the collision-free oracle. When fingerprints
differ, the analyzer recomputes every disputed query with two independent metrics: the original
double-precision spherical geometry and double-precision distances between the float Cartesian
coordinates actually supplied to both GPU searches. A substitution is accepted only when one
returned set matches the latter search-space reference and its relative KNN-boundary gap is no
larger than eight float epsilons. The physical-space result, raw mismatch count, and both hashes
remain in the JSON even when such a single-precision tie is classified as equivalent.

Collision rates are required to satisfy relative-L2 and maximum scaled-error limits of `1e-5` and
`1e-3`, respectively, on every query for which the two backends selected identical physical
neighbors. All-query rate differences are retained as diagnostics because two grains at an
unresolved KNN boundary can have different collision properties. The limits can be overridden with
`--probe-l2-tolerance`, `--probe-max-tolerance`, and `--knn-tie-tolerance`.

The cuKD wedge reference constructs its periodic ghosts by rotating the already-rounded physical
Cartesian record with `sincosf`, rather than rounding a shifted azimuth and reevaluating the
spherical-to-Cartesian map. This matches the Cartesian isometry used by Morton query images and
removes the extra shifted-angle rounding responsible for the observed $K$-boundary substitution.
The copied cuKD path also uses a small candidate-heap wrapper that retains cuKD's tree-node index
for lookup while ordering equal-distance candidates by stable physical particle ID. The stock cuKD
heap uses its mutable tree slot as the secondary key, whereas Morton uses the stable ID; leaving
that difference in place makes exact float-distance ties depend on backend construction order.

After a run has produced both backend directories, repeat only the Python analysis with
`--analyze-only`. This preserves the existing outputs and timings and avoids another CUDA build or
simulation. If the topology fingerprints disagree, the analyzer reports the independent
double-precision adjudication and exact K-boundary gap for every disputed query.

The final representative-by-representative state and raw RNG stream are not pass criteria. cuKD's
heap and Morton return the same categorical neighbors in different orders, so inverse-CDF partner
sampling can map the same uniform variate to different valid partners. Exact trajectory equality is
therefore neither expected nor a valid statistical requirement. The pre-event probes test the
deterministic numerical estimator; the KS tests assess the evolved distributions.

The million-particle model is deliberately excluded from the runner's default model list. Run it
explicitly after the three \(10^5\)-particle correctness cases have passed:

```bash
python3 lab/run_swarm_comparison.py \
    --models collision_disk_3d_1m \
    --arch sm_80 \
    --timeout 7200
```

Its separate model name keeps all objects, executables, backend outputs, and comparison metrics
apart from `collision_disk_3d`. Expect roughly 64 MB per particle checkpoint and allow several
hundred megabytes of filesystem space for both backends and their probes.

The test distributions are:

- `smooth`: a smooth annulus with a vertically thin 3D extension
- `ring`: a narrow radial ring
- `clump`: an 80-percent compact clump embedded in a smooth background

The clump case verifies that refinement prevents the nearly quadratic dense-cell behavior exposed
by the initial uniform prototype. Maximum leaf occupancy remains recorded so unresolved or
coincident clumps cannot silently bypass the configured maximum level.

### Discarded uniform baseline

The first native A100 run confirmed exact neighbor agreement but rejected the uniform design on
performance grounds. With $K=200$ and radius 0.1, the Morton-to-KD query-time speedups were 0.285
and 0.126 for the smooth $10^5$ and $10^6$ cases, 0.255 and 0.088 for the corresponding ring
cases, and 0.013 for the $10^5$ clump. The $10^6$ clump entered the predicted dense-cell
near-quadratic regime and was interrupted. These measurements describe the removed uniform
prototype, not the adaptive implementation now compiled by this laboratory.

## Running on a CUDA cluster

Build and run one small comparison with:

```bash
make -C lab ARCH=sm_80 K=200

lab/bin/knn_benchmark \
    --particles 100000 \
    --queries 4096 \
    --dim 2 \
    --distribution smooth \
    --radius 0.1 \
    --leaf-target 128 \
    --max-level 20 \
    --max-leaf-scan 4096 \
    --output lab/out/smooth_2d_N100000.json
```

Run the default 12-case matrix with:

```bash
python3 lab/run_benchmarks.py \
    --particles 100000 1000000 \
    --dim 2 3 \
    --distribution smooth ring clump
```

Summarize downloaded or locally generated results with:

```bash
python3 lab/analyze_results.py
```

By default the analyzer reads only the files named by the latest matrix manifest, preventing old
exploratory failures from contaminating the current table. Use `python3 lab/analyze_results.py
--all` when the historical experiments are intentionally wanted.

Build and run the deterministic edge-case suite with:

```bash
make -C lab edge ARCH=sm_80
lab/bin/knn_edge_tests
```

The edge suite checks deterministic equal-distance ordering, more-than-one-block coincident
particles, inclusive search-radius boundaries, fewer than $K$ neighbors, and points lying on Morton
subdivision planes in both 2D and 3D. Its expected lists come from an independent exhaustive CPU
search rather than cuKD, so cuKD tie-order choices cannot mask or create failures.

Build and run the periodic-query suite with:

```bash
make -C lab periodic ARCH=sm_80
lab/bin/knn_periodic_tests
```

The periodic prototype stores only physical particles. A boundary query performs the necessary
rotation by one wedge width, merges at most three local top-$K$ lists in shared memory, groups them
by stable particle identifier, retains the minimum-image distance, and then restores deterministic
distance ordering. Interior queries use one traversal; full-$2\pi$ queries use ordinary Cartesian
search. The suite compares lower and upper seams, interior queries, narrow-wedge three-image
deduplication, and full-disk behavior against exhaustive periodic references in 2D and 3D.

When two periodic query balls are geometrically disjoint, duplicate physical identifiers are
impossible and their sorted lists are merged directly in linear time. Overlapping two- or
three-image searches retain the general identifier-sort and minimum-distance deduplication path.
The benchmark records host-versus-GPU near-tie substitutions separately while still requiring the
two GPU backends to return exactly identical identifiers for every sampled query.

Run the periodic-wedge performance matrix with:

```bash
python3 lab/run_wedge_benchmarks.py \
    --particles 100000 1000000 \
    --dim 2 3 \
    --distribution smooth ring interior_clump seam_clump

python3 lab/analyze_wedge.py
```

The benchmark compares three backends. Reference cuKD contains all three complete wedge images,
matching the current swarm implementation. Query-image Morton stores only physical particles and
performs fused boundary query images. Boundary-ghost Morton stores physical particles plus only
the records whose search-radius halos cross a wedge face, preserving their stable physical IDs.
The host selects the required source particles, while a CUDA kernel rotates their ghost records
with the same `sincosf` arithmetic as the cuKD reference to avoid artificial neighbor reordering
from host-versus-device trigonometric rounding.
The timed paths store one checksum per physical query rather than a global neighbor table. Reported
persistent-memory ratios therefore compare the search structures themselves; temporary quality
arrays are excluded from every backend.

The executable exits unsuccessfully if either sampled Morton neighbor list differs from cuKD. The
first `--brute-queries` lists are checked independently against a CPU exhaustive search. If any GPU
backends disagree elsewhere, every disagreeing query is also checked exhaustively and the
backend-specific disagreement counts are written to JSON. Start with $10^5$ particles before
attempting the $10^6$ cases because the reference KD query retains a large private candidate heap
per thread.

The executable also aborts before launching queries if an unresolved leaf contains more than
`--max-leaf-scan` particles. This turns coincident-particle or insufficient-refinement pathologies
into an immediate diagnostic instead of an apparently hung quadratic scan.

The cooperative selector buffers candidates across several leaves and performs one exact
shared-memory merge only after accumulating at least $K$ candidates. This avoids paying a complete
512-entry bitonic sort for every leaf. A block-wide barrier separates
candidate writes from the thread-0 update of the shared batch offset, so every warp writes against
the same immutable batch range.

The default target is 128 particles per leaf. Recorded $10^6$-particle tuning over smooth, ring,
and clumped 2D/3D distributions found that target 128 gave the best common balance: target 64 paid
more tree-traversal and merge overhead, while target 256 examined too many candidates in structured
and clumped regions. Because those records span development revisions, reconfirm this ranking from
the final source snapshot before treating it as publication evidence.

Node-distance pruning expands each nominal Morton cell by a small scale-aware floating-point pad.
This preserves a conservative lower bound despite roundoff accumulated while subdividing the root
cube in single precision; the pad can add candidates but cannot remove a valid neighbor.

## Validation boundary

The copied swarm integration does not yet establish:

- direct production-kernel neighbor-list dumps for every representative
- production equivalence for axisymmetric, imported-gas, and strongly polydisperse configurations
- the faster periodic boundary-ghost path in the copied runtime
- location-dependent global or radially binned ghost halos
- multi-GPU ownership, halo exchange, or exact halo certification
- ROCm portability

The current hierarchy is assembled on the host from GPU-sorted maximum-level Morton keys. This is
useful for validating adaptive query mathematics, but it is not the final construction path or a
fair multi-GPU build benchmark. The production candidate must eventually construct compact leaves
on the GPU and distribute coarse ownership before collision-rate and event integration.

## Promotion criteria

Nothing in `lab/` should be promoted to production until:

1. every KD-versus-Morton neighbor comparison passes in 2D and 3D
2. boundary and equal-distance tie cases pass dedicated comparisons
3. adaptive refinement controls p99 and maximum leaf occupancy in clumps
4. collision rates and sampled partners agree with the KD backend under frozen snapshots
5. peak VRAM and kernel profiling confirm the intended reduction in local-memory traffic
6. repeated production-like distributions demonstrate a measured speed or capacity advantage
7. multi-GPU exactness and halo behavior have separate tests before distributed use

Only after those checks should the selected backend interface be moved into `inc/swarm/` and
`src/swarm/`, and the stable tests be copied into `qav/swarm/`.
