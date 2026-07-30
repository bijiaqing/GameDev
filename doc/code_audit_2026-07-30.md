# KNN rework audit: Morton backend, KD-tree backend, and the KNN test family (2026-07-30)

Date: 2026-07-30

Scope: the newly added exact neighbor-search backend — `doc/numerics_swarm.md` and
`doc/testset_swarm.md` (search-related sections), the new `inc/swarm/morton/` module
(`morton_types.cuh`, `morton_index.cuh`, `morton_query.cuh`, `morton_ghost.cuh`), the renamed
`inc/swarm/kdtree/` vendored library, `inc/swarm/_collision.cuh` (`stable_heap`), the collision
kernels (`col_tree_init.cu`, `col_rate_calc.cu`, `col_event_run.cu`), `swarm_runtime.cu`,
`inc/swarm/{const_defs,swarm_kern}.cuh`, the `COLLISION_SEARCH` Makefile routing, and the
complete `qav/swarm/test_knn/` family with its archived outputs. The review asked: (1) whether
the documents propose the right algorithm; (2) whether the Morton and KD-tree codes are
mathematically, physically, and numerically correct; (3) whether code and documents are
consistent. The requester suspected problems exist.

Method: line-by-line source inspection with independent algebraic re-derivation of every
geometric and selection formula (Morton key layout, hierarchy partitioning, AABB pruning with
padding, top-K merge, ghost criteria, dedup sufficiency, heap invariants), plus consistency
checks of the archived native results (`qav/swarm/test_knn/out/`, `qav/swarm/out/`). No CUDA
compilation or execution was performed on this machine.

## Executive summary

The rework is **correct**: no defect was found that changes a neighbor set, a collision rate, or
any physical quantity. The proposed algorithm is the right one and is honestly documented. The
archived native evidence agrees with it: byte-equal neighbor hashes in all four
collision-runtime backend comparisons, KS-equivalent distributions, zero mass drift, and
10 + 10 + 12 + 16 passing topology/benchmark cases.

The original audit identified **P1**, one-thread Morton pair-rate evaluation, and **P2**, a latent
few-ulp shortfall in deep-node pruning padding, as the two actionable findings. It also recorded a
documentation nuance, a duplicated QA heap, and a suspected polar-cap coverage gap. Subsequent
inspection showed that the polar-cap test already existed; the other four findings have now been
corrected as recorded below.

## Implementation update after the audit

The following corrections are now present in the source; native CUDA revalidation remains
required before replacing the archived baseline:

- **P1 implemented:** Morton threads evaluate individual pair propensities cooperatively into
  shared memory; thread zero still accumulates them and selects the partner in the original order,
  preserving the floating-point sum and RNG interpretation
- **P2 implemented:** `morton_view` carries `max_level`, and node pruning uses the conservative
  level-aware padding
  $2(L_{\max}+2)\epsilon_{\rm float}\max(1,|b_{\min}|,|b_{\max}|)$
- **P3 resolved:** the numerical documentation specifies the lexicographic selection rule without
  claiming that both backends expose the same physical heap order
- **P4 resolved:** production and standalone QA instantiate one generic `index_old` heap
  from `inc/swarm/kdtree/index_heap.cuh`
- **P5 was outdated:** `test_collision_3d` already places a query near the polar boundary and
  checks the corresponding polar-cap correction analytically

A subsequent clean KNN run passed the adversarial tests and ordinary matrix but exposed one
ghost-Morton/brute-force disagreement in `ring_3d_N100000`. The level-aware P2 correction was
applied after that run, so the failed case and then the complete matrix must be rerun.

## 1. Does the document propose the right algorithm?

**Yes.** Assessment of the proposal in `numerics_swarm.md`:

- The exact-KNN-within-cutoff estimator with lexicographic `(d², id)` tie-breaking is the right
  contract for the collision normalization: rates divide by the population of a deterministic
  neighbor ball, so the neighbor set must be deterministic under every tie and traversal order.
  Both backends implement exactly this contract.
- The adaptive pointer-free Morton hierarchy (cell mapping → bit interleave → stable sort →
  recursive refinement above `MORTON_LEAF_TARGET`) with block-cooperative top-K selection and an
  explicit traversal-overflow report is a sound, mainstream GPU design. Host-side hierarchy
  assembly is correctly classified as a performance (not correctness) improvement for later.
- Boundary-only periodic ghosts with the global `q_max = max_i q_i` halo are mathematically
  sufficient and much cheaper than the KD-tree's three full copies. The criterion is the right
  one: a source needs an image iff its distance to the seam is ≤ the largest query radius.
- The `duplicate_safe` criterion — image chord `2·R_min·sin(width/2) > 2·q_max` implies no query
  ball can hold two images of one particle — is the correct global test, and the narrow-wedge
  fallback (retain 3N_K records, deduplicate by original particle index, select the nearest N_K
  physical) is provably sufficient: each physical id contributes at most three records, so the
  K-th distinct id is always within the first 3K records.
- Keeping the KD-tree as an independent mature reference, and validating both against exhaustive
  brute force, is the right verification strategy. The documents are appropriately honest about
  memory (full-disk Morton storage can exceed the single-array KD-tree) and about not claiming
  universal speed; backend choice is correctly framed as a measured configuration, not a physics
  change.
- The future sketches (certified multi-GPU Morton ownership with conservative remote-distance
  certification, frozen-bath local continuous-time event chains with exact waiting times) are
  mathematically sound and correctly parked behind convergence evidence.

## 2. Are the Morton and KD-tree codes correct?

### Verified correct (independently re-derived)

- **Key generation and hierarchy** (`morton_index.cuh`): the 21-bit `_expand_morton_3d`
  interleave and `ix | iy<<1 | iz<<2` key layout are standard and correct; cell indices clamp to
  the root box; `thrust::stable_sort_by_key` gives a stable key ordering; the host recursion
  partitions children by the correct 3-bit level code (`shift = 3·(max_level − level − 1)`),
  splits only when `count > leaf_target` and `level < max_level` (so no negative shift), and
  records begin/count ranges into the sorted point array. Node bounds derive from origin plus
  power-of-two halvings (exact) with one float addition per level.
- **Pruning bound** (`_get_node_dist_sq`): `max(lower − q − pad, 0, q − upper − pad)` has the
  correct distance-to-AABB form. The audited fixed padding had insufficient worst-case margin at
  the deepest supported level; the current level-aware padding covers the accumulated
  bound-construction roundoff (see P2).
- **Top-K merge** (`_morton_topk` / `_morton_pair_sort`): the bitonic network over the
  power-of-two merge array with the `(d², id)` comparator is the standard construction; the
  batch accumulation (`batch_capacity = min(K, SORT_SIZE − K)`) fills, INF-pads, and sorts
  exactly when full and at the final flush, so slots `0..K−1` always hold the merged top-K
  within radius; the `min(radius², best_dist[K−1])` cutoff is a valid pruning bound at all
  times; `dist_sq <= radius_sq` boundary semantics match the KD-tree heap's
  `candidate < (cutoff², id_max)` acceptance; stack overflow sets a flag instead of silently
  truncating.
- **Ghosts** (`morton_ghost.cuh`): the seam distance `R·|sin Δ|` (or `R` past 90°) is the
  correct distance-to-ray; ghost rotation directions (+width for lower-face sources, −width for
  upper-face sources) place images where cross-seam queries need them; the exclusive-scan offset
  layout is exact; `source_id` keeps the original particle index on every record; the
  no-ghost fast paths (`N_X == 1`, full `2π`) are correct.
- **Deduplication** (`morton_query.cuh`): for `!duplicate_safe`, retaining 3K records, grouping
  by `(id, dist)` (nearest image first within each id group), dropping later duplicates, and
  re-sorting by `(d², id)` yields the exact nearest K physical particles (sufficiency proven in
  §1 above).
- **KD-tree heap** (`kdtree/index_heap.cuh::index_old_heap`): encoding `(dist² << 32) | index_old` makes
  the heap order lexicographic in `(d², id)` (the documented tie-breaker); `expandedCullDist2 =
  nextafterf(maxRadius2, +inf)` is the documented one-ulp cull expansion. The heap retains the
  original particle index directly rather than a second shuffled-slot array. When periodic query balls
  can overlap, the deduplicate path keeps at most one record per physical id (induction: a
  duplicate is either rejected or replaced, so the scan never sees two); sift-down replacement
  preserves the max-heap invariant. Disjoint image neighborhoods bypass this scan.
- **Backend wiring**: both `col_rate_calc` and `col_event_run` use the same per-particle cutoff
  `q_i = H_SEARCH·h_g·R`, the same `_get_ball_measure`, the same valid-neighbor filtering
  (active, non-self), the same farthest-valid-distance radius, the same pair propensities, and
  the same Bernoulli/event algebra. Morton traversal overflow zeroes the rate and throws via the
  host (`std::runtime_error`), i.e., fails loudly; `col_event_run` on overflow restores the RNG
  state before aborting. RNG consumption per particle follows the same sequence in both backends
  (probability draw, then partner target, then optional fragment sample).
- **Runtime** (`swarm_runtime.cu`): the ghost index is rebuilt once per collision half-interval
  (positions are frozen inside `evolve_collisions`), with the halo from the global maximum
  cutoff; `duplicate_safe` is computed per build from the documented chord criterion; overflow
  reductions follow both rate and event kernels.
- **Archived native evidence** (`qav/swarm/test_knn/out/`): all four collision-runtime
  comparisons (full-disk 2D, wedge 2D, full-disk 3D, full-disk 3D at 10⁶) record
  `initial_byte_equal: true`, `neighbor_hash: byte_equal` with zero mismatches, KS-equivalent
  distributions, zero mass drift in both backends, and identical maximum rates; the manifests
  record 10/10 ordinary adversarial, 10/10 periodic adversarial, 12 ordinary benchmark, and 16
  periodic-wedge cases passed. The analytical suite archive (`qav/swarm/out/`) contains all 25
  records with `"passed": true` (spot-checked `test_grid_2d` N256 against the doc's table).

### Problems found

- **P1 (implemented; native performance validation pending).** In the audited Morton collision kernels, after the
  cooperative 256-thread traversal, **only thread 0** iterates the `N_K = 200` neighbors and
  evaluates `_get_col_rate_ij` (in `col_event_run`, also the cumulative partner selection); the
  other 255 threads exit. The archived timings isolate the cost: Morton `rate_ms ≈ 29 ms/batch`
  versus KD-tree `≈ 14 ms/batch` (~2× slower), which is what drives the recorded wall ratios of
  ≈0.92–0.93 *against* Morton. For `CUSTOM_KERNEL` the serialization is worse because each pair
  rate evaluates `_get_vrel`. This is a performance defect, not a correctness one — but the
  documents report the wall ratios without identifying this serialization as their cause, and it
  largely erases the traversal's parallelism in production. The current implementation evaluates
  pair propensities across the block and retains ordered serial accumulation and partner selection.
- **P2 (implemented; native correctness validation pending).** The audited `_get_node_dist_sq` padded the pruning box by
  `8·FLT_EPSILON·scale`. Worst-case coherent accumulation of the subdivided lower corners is
  about `0.5 ulp` per level per axis, i.e. up to ~10 eps at `max_level = 20`, so at the deepest
  levels the bound can overestimate the true AABB distance by ~2 ulp·scale and prune a node that
  could contain the answer (an extremely narrow adversarial window). At production depths
  (~8–10 levels for `N_P = 10⁷` with `MORTON_LEAF_TARGET = 128`) the margin is adequate, so
  this is latent rather than active. The current view carries `max_level` and scales the pad as
  $2(L_{\max}+2)\epsilon_{\rm float}$ times the coordinate scale.
- **P3 (resolved doc–code nuance).** The audited contract said both backends return results "in lexicographic
  `(d², id)` order". Morton returns them sorted; the KD-tree `stable_heap` returns the same
  *set* in heap order. Rates and neighbor sets are identical (byte-equal in the archived
  comparisons), but partner sampling in `col_event_run` iterates in different orders, so
  identical RNG streams can select different partners across backends — statistically
  equivalent, not trajectory-equivalent. The archives record this honestly
  (`rng_byte_equal: false`, `trajectory_close: false`); the contract sentence was literally true
  only for Morton. The current wording describes a common selection rule and backend-specific
  internal order.
- **P4 (resolved drift hazard).** The hand copy was removed. Production and QA instantiate the
  generic `index_old_heap` with their respective node types.
- **P5 (outdated finding).** `test_collision_3d` already evaluates `_get_ball_measure` at a point
  one quarter-radius from the lower polar boundary and compares the result with the analytical
  spherical-cap correction.

## 3. Are code and documents consistent?

Checked claims and found consistent:

- Builder description (keygen → stable sort → host refinement above `MORTON_LEAF_TARGET`,
  pointer-free record/node arrays) — matches `morton_index.cuh`.
- Conservative padded pruning bound and overflow reporting — matches `_get_node_dist_sq` and the
  `stack_overflow` flag plus host-side throw in `swarm_runtime.cu`.
- Ghost policy (seam distance ≤ `q_max`, `sincosf` isometry, +width lower / −width upper) —
  matches `morton_ghost.cuh`; the cheaper disjoint-image path for ordinary wedges and the 3N_K
  dedup path for narrow wedges match `_morton_ghost_topk` and the `duplicate_safe_` computation.
- KD-tree heap tie-break and one-ulp cull expansion — matches `stable_heap`.
- Separate object directories per backend — matches the Makefile's
  `OBJ_DIR = .../$(COLLISION_SEARCH)` and the `COLLISION_SEARCH` validation/`?= kdtree`
  default; the "exactly one backend" compile-time guard in `swarm_kern.cuh` matches the
  documented contract.
- Recorded case counts and passes (10/10, 10/10, 12, 16, all four runtime comparisons), the
  memory ratios (1.40 full-disk Morton/KD-tree, 0.47 wedge — consistent with "full-disk Morton
  can exceed the KD-tree; partial wedges benefit more"), and the 2026-07-29 analytical native
  suite (25/25 passed) — all verified against the archived JSON/manifests/logs.
- `numerics_swarm.md`'s limitations list (multi-GPU ownership, halo bins, GPU-native
  construction, continuous-time chains as future work) matches the code state.

The audited P3 wording and missing P1 explanation have now been corrected. Updated native timing
evidence is still needed for the cooperative pair-rate implementation.

## Recommendations (priority order)

1. Rerun the failed `ring_3d_N100000` wedge case with the level-aware pruning pad.
2. If it passes, rerun the complete promoted KNN suite and all production backend comparisons.
3. Compare Morton `rate_ms` before and after cooperative pair evaluation; do not use the archived
   timings as a current production claim.
4. Record compiler resource reports and update the native result archive only after the complete
   clean matrix passes.

## Review limitations

- No CUDA compilation or execution was performed here; all conclusions are from source
  inspection and from the archived native artifacts, which were checked for internal consistency
  but not regenerated.
- `inc/swarm/kdtree/` internals were confirmed to be the unchanged vendored library (namespace
  rename only); only call-site consistency was re-audited.
- The fluid branch and the non-KNN swarm tests were not re-reviewed in this pass.
