# Future exact KNN backends

## Scope and present decision

GameDev currently uses cuKD to recover the fixed-$N_K$ neighborhoods required by the swarm
collision estimator. The leading alternative is a pointer-free adaptive Morton cell list with
cooperative exact top-$K$ selection.

The laboratory experiment has succeeded as an exact-search prototype:

- all ordinary and periodic adversarial cases passed
- top-200 neighborhoods matched cuKD and independent brute force
- adaptive refinement remained usable in smooth disks, rings, and strong clumps
- periodic boundary ghosts reduced the persistent search memory substantially
- standalone million-particle query performance was comparable to or modestly better than cuKD
- the clean copied-runtime matrix passed in full through one million particles

This is not yet a production-backend claim. A selectable Morton path is connected to the copied
collision-rate and collision-event kernels under `lab/`, while the active `src/swarm/`
implementation remains cuKD-only. The current policy is therefore:

- keep cuKD as the production and regression backend
- continue Morton as an optional exact backend candidate
- preserve the existing collision estimator and physics for both methods
- promote Morton only after its remaining runtime and memory work demonstrates a practical benefit

The numerical tables below record both the standalone cluster benchmarks and the matched
copied-runtime results archived under `lab/out/`. The clean matrix was generated on an
NVIDIA A100-SXM4-40GB with CUDA 12.1 and records base revision
`115aa791f9db149bf3d53838df2efc86134994b7`. The accompanying `worktree.txt` records uncommitted
laboratory changes, so the evidence is internally coherent but should be frozen into an immutable
commit before publication or production promotion.

## Collision-search contract

During one collision half-interval, the swarm runtime:

1. converts representative positions to Cartesian coordinates
2. creates periodic entries when the azimuthal domain is a wedge
3. constructs the search structure
4. freezes grain size and represented-grain number for one collision batch
5. queries up to $N_K$ neighbors for every representative
6. calculates each total collision propensity
7. reduces the maximum propensity to determine the batch timestep
8. reconstructs the neighborhood for representatives that undergo an event
9. samples one partner from the pair propensities

Collisions change grain properties but not positions, so the spatial structure remains valid until
transport moves the particles.

The location-dependent query cap is

$$
q_i = H_{\mathrm{SEARCH}}\,H_g(R_i)\,R_i,
$$

where $H_g$ is dimensionless and $H_g R$ is the gas scale height. If $S_i$ is the returned
neighborhood, the unnormalized propensity is

$$
\widetilde{\lambda}_i
=
\sum_{j\in S_i,\ j\ne i}N_j\,K_{ij},
$$

and the final rate is

$$
\lambda_i
=
\frac{\widetilde{\lambda}_i}
{V_{\mathrm{accessible}}(i,d_{K,i})}.
$$

Here $N_j$ is the number of grains represented by particle $j$, $K_{ij}$ is the physical pair
kernel, and $d_{K,i}$ is the farthest valid returned distance. The host normalization also
contains

$$
\frac{N_P}{(N_K-1)M_{\mathrm{dust}}},
$$

so fixed-$N_K$ sampling is part of the estimator, not merely a search preference.

A second backend must preserve:

- $N_K$ and the cap $q_i$
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

The stable secondary identifier makes construction deterministic when several representatives have
the same cell key. A leaf target of 128 records was the best common choice in the tuning matrix and
was retained throughout the clean validation snapshot.

The current hierarchy is assembled on the host after GPU key sorting. This is adequate for
algorithm validation but should eventually be replaced by GPU-native leaf construction.

### Exact traversal

For a query point $\boldsymbol{x}_i$ and a Cartesian node with bounds
$[\boldsymbol{b}_{\min},\boldsymbol{b}_{\max}]$, the conservative lower bound is

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

Nodes are visited in increasing lower-bound distance. Candidate records are retained in the
lexicographic order

$$
(d_i^2,\mathrm{id}_i) < (d_j^2,\mathrm{id}_j).
$$

Traversal stops only after every unvisited node satisfies

$$
d_{\min,\mathrm{unvisited}}^2 \ge d_{K,i}^2
$$

or lies outside $q_i$. If fewer than $N_K$ records exist within $q_i$, every intersecting node
is examined and the smaller valid count is returned.

Repeated subdivision in single precision required a small scale-aware expansion of node bounds.
Without it, roundoff could make the nominal lower bound too large and incorrectly prune a valid
neighbor.

The copied-runtime comparison exposed a separate reference-ordering issue at one million particles.
Morton ordered equal-distance candidates by stable physical particle identifier, whereas the stock
cuKD heap used its mutable tree-storage slot as the secondary key and could prune candidates at an
equal distance. The laboratory cuKD wrapper now stores the tree node for retrieval but orders its
heap by `(distance, stable identifier)`. Its culling radius is advanced to the next representable
floating-point value so equal-distance candidates remain searchable. After this correction, all six
previously mismatched one-million-particle neighborhoods became exactly equal. This changes only
the laboratory reference's deterministic tie policy; the active production cuKD path is untouched.

### Cooperative top-$K$ selection

The cuKD query instantiates a private `HeapCandidateList<N_K>`. At $N_K=200$, 32-bit distances and
identifiers alone require approximately

$$
8N_K = 1600\ \mathrm{bytes}
$$

per active query thread, before traversal state. This can create local-memory traffic and occupancy
pressure even when total VRAM is sufficient.

The Morton query assigns one CUDA block to a query. Threads cooperatively evaluate candidate tiles
and retain the nearest $N_K$ pairs in explicitly sized shared memory. No global $N_P N_K$
neighbor table is allocated; rate queries consume their result immediately and event neighborhoods
are reconstructed only when needed.

An early buffered selector contained a race because thread 0 advanced a shared batch offset before
all warps had finished using it. Adding a block-wide barrier restored exact results. This is now
covered by the adversarial tests.

### Clumps and degeneracy

A uniform Morton list was exact but extremely slow in dense clumps. Adaptive refinement bounds
ordinary leaf occupancy and removed this failure mode. It does not eliminate the intrinsic cost of
coincident particles: no geometric hierarchy can separate records with identical coordinates. A
production backend must therefore retain occupancy, depth, candidate-count, and overflow
diagnostics.

## Periodic wedges

Two exact policies were tested:

- **query images:** store only physical records, rotate boundary queries by a wedge width, search
  again, then merge and deduplicate by physical identifier
- **boundary ghosts:** copy only source records whose search halo intersects a wedge face, rotate
  them across the seam, then perform one ordinary query

Query images minimize persistent storage but repeat traversal. Boundary ghosts use more records near
a seam but were substantially faster for seam-dominated clumps and map naturally to future
multi-GPU halos.

Ghost coordinates must be generated with arithmetic consistent with the reference path. Host
`std::sin` and `std::cos` introduced near-tie differences relative to GPU `sincosf`; GPU generation
removed all observed disagreements.

Every physical representative may contribute at most once to a query. Periodic candidates are
therefore deduplicated by stable identifier after applying the minimum-image geometry.

## Laboratory evidence

### Correctness

The clean validation snapshot passed:

- 10 of 10 ordinary adversarial cases across 2D and 3D
- 10 of 10 periodic adversarial cases across 2D and 3D
- a twelve-case smooth, ring, and clump matrix at $N_P=10^5$ and $10^6$
- sixteen periodic-wedge cases at $N_P=10^5$ and $10^6$
- four copied-runtime collision comparisons: full-disk 2D, wedge 2D, full-disk 3D, and
  full-disk 3D at $N_P=10^6$

The cases cover equal-distance ties, coincident particles spanning multiple candidate tiles, an
inclusive cutoff boundary, fewer than $K$ valid neighbors, Morton split planes, both wedge faces,
narrow-wedge image overlap, full-$2\pi$ geometry, and seam-centered clumps.

The validator checks a configured subset against exhaustive CPU search. When GPU backends disagree
outside that subset, every disagreement is also checked by brute force and attributed to the
incorrect backend.

### Copied-runtime collision results

The copied swarm runtime was exercised from the clean validation snapshot with the same initialized
representatives and collision physics for both backends. The comparison requires finite outputs,
byte-identical initialization, exact neighbor identifiers and valid counts, collision-rate and KNN
radius agreement, statistically consistent evolved distributions, and mass conservation. Raw final
particle arrays and RNG states are diagnostics rather than equality requirements because different
valid partner orderings advance stochastic streams differently after the first collision event.

Correctness results:

| Case | $N_P$ | Neighbor mismatches | Rate relative L2 | Radius relative L2 | Size KS / limit | Mass mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full disk 2D | $10^5$ | $0$ | $4.27\times10^{-16}$ | $0$ | $4.80\times10^{-4}/8.59\times10^{-3}$ | $0$ |
| wedge 2D | $10^5$ | $0$ | $8.88\times10^{-8}$ | $4.36\times10^{-8}$ | $6.60\times10^{-4}/8.59\times10^{-3}$ | $0$ |
| full disk 3D | $10^5$ | $0$ | $3.60\times10^{-16}$ | $0$ | $1.61\times10^{-3}/8.59\times10^{-3}$ | $0$ |
| full disk 3D | $10^6$ | $0$ | $5.55\times10^{-16}$ | $0$ | $1.21\times10^{-3}/2.72\times10^{-3}$ | $0$ |

Every copied-runtime case has byte-identical neighbor hashes and counts. The wedge differences come
from periodic coordinate arithmetic and remain far below the configured rate tolerances; its
neighbor identifiers and counts are nevertheless exactly equal.

Integrated timing and persistent allocation results:

| Case | Morton build speed ratio | Morton rate speed ratio | Morton event speed ratio | Total wall speed ratio | Morton memory ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| full disk 2D, $10^5$ | 4.293 | 0.511 | 1.728 | 0.925 | 1.404 |
| wedge 2D, $10^5$ | 7.783 | 0.782 | 2.226 | 0.928 | 0.468 |
| full disk 3D, $10^5$ | 4.549 | 0.595 | 1.446 | 0.631 | 1.379 |
| full disk 3D, $10^6$ | 2.309 | 0.935 | 1.463 | 0.942 | 1.446 |

Every speed ratio is $t_{\mathrm{KD}}/t_{\mathrm{Morton}}$, so values above one favor Morton.
Morton construction and event queries are consistently faster, but repeated all-particle rate
queries dominate the current collision loop. Consequently, the copied Morton runtime is 6.2 percent
slower than cuKD at one million particles and allocates 44.6 percent more persistent memory in that
full-disk configuration. The 2D wedge uses only 46.8 percent of the three-image cuKD persistent
memory, but its current integrated traversal remains slower. These measurements support retaining
cuKD as the production default while developing boundary ghosts and reducing the Morton runtime
owner.

### Nonperiodic million-particle results

The table reports $t_{\mathrm{KD}}/t_{\mathrm{Morton}}$, so values above one favor Morton.

| Distribution | Dimension | Query speed ratio | Persistent-memory ratio |
| --- | ---: | ---: | ---: |
| smooth | 2D | 1.092 | 0.706 |
| ring | 2D | 1.066 | 0.733 |
| clump | 2D | 1.076 | 0.723 |
| smooth | 3D | 1.136 | 0.716 |
| ring | 3D | 1.077 | 0.752 |
| clump | 3D | 0.991 | 0.745 |

Morton was faster in five of six cases and 0.9 percent slower in the remaining 3D clump, while
using about 25-29 percent less persistent search memory.

### Periodic million-particle results

Performance:

| Case | cuKD, ms | Query-image, ms | Boundary-ghost, ms | Ghost speed ratio |
| --- | ---: | ---: | ---: | ---: |
| smooth 2D | 273.899 | 309.197 | 247.326 | 1.107 |
| ring 2D | 278.037 | 327.061 | 257.211 | 1.081 |
| interior clump 2D | 285.510 | 299.506 | 268.599 | 1.063 |
| seam clump 2D | 305.856 | 776.495 | 269.838 | 1.133 |
| smooth 3D | 504.694 | 577.245 | 466.552 | 1.082 |
| ring 3D | 491.118 | 614.430 | 487.485 | 1.007 |
| interior clump 3D | 500.415 | 554.130 | 493.233 | 1.015 |
| seam clump 3D | 526.245 | 1387.306 | 495.546 | 1.062 |

Persistent-memory use relative to cuKD:

| Case | Query-image ratio | Boundary-ghost ratio | Records per particle |
| --- | ---: | ---: | ---: |
| smooth 2D | 0.248 | 0.276 | 1.128 |
| ring 2D | 0.240 | 0.269 | 1.128 |
| interior clump 2D | 0.243 | 0.249 | 1.026 |
| seam clump 2D | 0.243 | 0.441 | 1.825 |
| smooth 3D | 0.254 | 0.290 | 1.128 |
| ring 3D | 0.242 | 0.279 | 1.127 |
| interior clump 3D | 0.250 | 0.255 | 1.025 |
| seam clump 3D | 0.251 | 0.455 | 1.826 |

Boundary-ghost queries were 0.7-13.3 percent faster than cuKD in all eight million-particle cases.
Their persistent structure used 24.9-45.5 percent of the three-image cuKD structure, a 54.5-75.1
percent reduction.
Including construction and one all-particle query, the measured total-cycle speed ratio was
approximately 1.046-1.164.

At $N_P=10^5$, cuKD queries were about 13-30 percent faster. Faster Morton construction still made
build plus query faster in seven of eight cases; the artificial 3D seam clump was about 3 percent
slower overall.

These timings include host hierarchy assembly and therefore do not assume an unmeasured GPU-native construction benefit.

## Choosing a backend

| Regime | Preferred method | Reason |
| --- | --- | --- |
| small or established single-NVIDIA-GPU run | cuKD | mature reference and often faster at small $N_P$ |
| memory-sufficient single-GPU production | cuKD initially | present Morton speedup is modest |
| $N_P\gtrsim10^6$ | benchmark both | measured crossover is distribution dependent |
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

These advantages must be balanced against maintaining custom construction, traversal, selection,
and validation code.

## Selectable production interface

Use one build option:

```make
COLLISION_SEARCH := kdtree
```

or:

```make
COLLISION_SEARCH := morton
```

Exactly one backend should be compiled when `COLLISION` is enabled. Backend-specific types,
allocation, construction, and queries should remain inside separate swarm headers and source files;
collision mathematics remains in the existing swarm collision implementation. No fluid–swarm file
sharing is planned.

Both backends should return the same logical data:

- stable physical identifier
- squared distance
- valid result count
- farthest valid distance
- deterministic distance-and-identifier order

The backend is rebuilt from particle positions, so it does not belong in particle checkpoints.
Reproducibility metadata should nevertheless record the backend, $N_K$,
$H_{\mathrm{SEARCH}}$, Morton leaf target and maximum level, periodic policy, GPU count,
decomposition policy, and halo policy.

## Location-dependent periodic halos

Because $q_i$ depends only on particle position, it remains fixed through collision batches and
changes only after transport. A conservative global halo is

$$
h_{\mathrm{halo}} = \max_i q_i.
$$

This is exact but can duplicate too many particles when $H_g R$ varies strongly with radius.

### Radially binned halo

Divide cylindrical radius into intervals

$$
I_b = [R_b,R_{b+1})
$$

and calculate

$$
c_b = \max_{i:R_i\in I_b}q_i.
$$

For source bin $s$ and query bin $b$, the minimum radial separation is

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

A source in bin $s$ is copied across an azimuthal seam when

$$
d_{\mathrm{seam}} \le h_s+\epsilon.
$$

The bins only select ghost records. They do not create separate trees, and each query still uses its
own $q_i$. A first implementation should compare 32, 64, and 128 logarithmic radial bins against
the global-halo result and require identical neighbor sets.

## Removing the global collision-timestep bottleneck

### Why the current timestep becomes very small

The present collision integrator is a parallel frozen-rate Bernoulli leap. In every collision batch
it freezes the grain properties, recalculates all representative collision propensities, reduces
their global maximum, and uses

$$
\Delta t_{\mathrm{col}}
=
\min\left(
\frac{\mathrm{CFL}_{\mathrm{COL}}}{\max_i\lambda_i},
\Delta t_{\mathrm{remaining}}
\right).
$$

Representative $i$ then undergoes at most one event, with probability

$$
P_i = 1-\exp(-\lambda_i\,\Delta t_{\mathrm{col}}).
$$

This restriction is numerically controlled when
$\mu_i=\lambda_i\,\Delta t_{\mathrm{col}}\ll 1$, because the probability of two or more events is

$$
P(N_i\ge 2) = 1-\exp(-\mu_i)(1+\mu_i)
\simeq \frac{\mu_i^2}{2}.
$$

It is nevertheless expensive because the single fastest representative determines the timestep of
the complete swarm. For example, $\mathrm{CFL}_{\mathrm{COL}}=0.02$ and
$\Delta t_{\mathrm{col}}=10^{-6}$ imply

$$
\lambda_{\max}\simeq 2\times 10^4.
$$

Over a dynamical interval $\Delta t_{\mathrm{dyn}}=10^{-3}$, that representative physically
expects

$$
\lambda_{\max}\,\Delta t_{\mathrm{dyn}}\simeq 20
$$

events. Raising `CFL_COL` enough to cover the dynamical interval would collapse approximately
twenty expected events into one Bernoulli decision, so it would hide rather than solve the problem.
The physical waiting times cannot be made longer; their unnecessary role as a globally synchronized
timestep can be removed.

### Recommended method: local continuous-time event chains

Use a larger **bath-refresh interval** $\tau_{\mathrm{bath}}$, while allowing every representative
to process an independent sequence of zero, one, or many events inside that interval.

At the start of one bath interval:

1. freeze all partner properties in a read-only snapshot
2. build or reuse the spatial search structure because collisions do not change positions
3. assign every representative a local clock $t_i=0$
4. query its KNN identifiers once and retain them only for the lifetime of its block

For representative $i$, repeatedly calculate its current pair propensities against the frozen
partners and total rate

$$
\lambda_i = \sum_{j\in S_i}\lambda_{ij}.
$$

Draw the next exact conditional waiting time

$$
\delta t_i = -\frac{\ln U}{\lambda_i},
\qquad U\sim\mathcal U(0,1).
$$

If $t_i+\delta t_i>\tau_{\mathrm{bath}}$, finish that representative. Otherwise choose a partner
with

$$
P(j\mid i) = \frac{\lambda_{ij}}{\lambda_i},
$$

apply the collision outcome to representative $i$, advance its local clock, recompute the weights
and rate using its new properties, and continue. The updated state is written once when the chain
finishes. Partner $j$ is always read from the immutable bath snapshot, so simultaneous chains are
race-free and retain the present representative-mass conservation rule.

This construction is an exact continuous-time event sequence **conditional on the frozen bath**.
The fastest particle in the example performs approximately twenty local events instead of forcing
roughly one thousand full-population KNN/rate sweeps. Representatives with small rates normally
exit after drawing no event.

### Accuracy control for the frozen bath

The method removes the event-probability CFL restriction but introduces a bath-freezing error.
Consequently, $\tau_{\mathrm{bath}}$, rather than the fastest individual waiting time, becomes the
collision accuracy control. It should limit changes in the propensities caused by evolution of the
partner population, for example through

$$
\max_i
\frac{
\left|\lambda_i(X_{\mathrm{new}})-\lambda_i(X_{\mathrm{frozen}})\right|
}{
\max(\lambda_i(X_{\mathrm{frozen}}),\lambda_{\mathrm{floor}})
}
\le \epsilon_{\mathrm{bath}}.
$$

A first implementation should use one bath interval per collision operator and establish convergence
with

$$
\tau_{\mathrm{bath}},\qquad
\frac{\tau_{\mathrm{bath}}}{2},\qquad
\frac{\tau_{\mathrm{bath}}}{4}.
$$

If distributions or moments do not converge at the required accuracy, use two or four refreshes per
dynamical collision interval. This remains much cheaper than selecting every global batch from
$\max_i\lambda_i$. A later adaptive implementation should use a conservative pre-leap propensity
change estimate. Naive outcome-dependent rollback can bias stochastic paths and should not be added
without a coupled post-leap construction.

### GPU scheduling

Long local chains cause warp and block workload imbalance but do not require a global timestep.
Before executing chains, bin representatives by their predicted event count

$$
\mu_i = \lambda_i\,\tau_{\mathrm{bath}},
$$

using queues such as

- $\mu_i<0.1$
- $0.1\le \mu_i<1$
- $1\le \mu_i<10$
- $\mu_i\ge 10$

and launch similar workloads together. A persistent work queue is a later alternative. A watchdog
limit such as `MAX_EVENTS_PER_LAUNCH` may split a long chain across launches, but an unfinished
representative must retain its local clock, current state, and RNG state in a continuation queue;
the chain must never be truncated.

No global $N_P N_K$ neighbor table is required. One block can query a neighborhood once, keep its
identifiers in shared memory, recalculate only the property-dependent pair weights after each event,
and discard the neighborhood when the local chain finishes. The existing rate helper must be
refactored to accept the evolving properties of $i$ explicitly while continuing to read $j$ from
the frozen arrays.

### Alternatives

**Dyadic multirate Bernoulli scheduling** assigns each representative a local level based on
$\mathrm{CFL}_{\mathrm{COL}}/\lambda_i$ and processes only representatives due at each fine tick.
It prevents sparse fast outliers from subcycling the full swarm, but it retains the one-event
approximation and needs a policy for partner-property refresh. It is a lower-risk intermediate
optimization rather than the preferred final integrator.

**Poisson tau-leaping** freezes $\lambda_i$ and draws
$N_i\sim\operatorname{Poisson}(\lambda_i\tau)$. It is appropriate for a genuinely state-independent
kernel, but direct batch application is not rigorous when each collision changes the size, number,
kernel, or fragmentation probabilities of $i$. Sequential local event chains handle that feedback
naturally.

**A global Gillespie or next-reaction method** is statistically exact for the fully coupled process,
but its global waiting time scales with the sum of all propensities and requires serial or globally
synchronized event selection. It is therefore a poor GPU replacement for the present method.

**Rate clipping, smoothing, or a quantile timestep** must not discard a real clump or fast tail. A
quantile policy is acceptable only if the excluded fast representatives are explicitly placed in a
separate subcycled or event-chain queue.

### Development and validation sequence

1. record the distribution of $\mu_i=\lambda_i\Delta t_{\mathrm{dyn}}$, including the mean, maximum,
   p99, p99.9, and logarithmic population bins
2. implement frozen-bath local chains for constant-kernel coagulation in `lab/`
3. verify exponential waiting times and Poisson event-count statistics for fixed rates
4. compare size moments and distributions with the current very-small-`CFL_COL` reference and the
   analytical constant-kernel Smoluchowski solution
5. demonstrate convergence under bath-interval halving
6. extend to additive and product kernels, then the custom physical kernel and fragmentation
7. add rate-binned work queues and continuation launches without changing the stochastic sequence
8. benchmark KNN construction, query reuse, event execution, VRAM, and total operator time

The legacy Bernoulli integrator should remain selectable throughout laboratory validation. Only
after these tests pass should `CFL_COL` be retired or reinterpreted as a legacy-backend control.

For multi-GPU evolution, local event chains are particularly valuable: frozen boundary properties
need to be exchanged only at bath refreshes, and the global reduction of $\max_i\lambda_i$ is no
longer required after every collision microbatch.

## Multi-GPU extension

### Ownership and records

Each GPU should own coarse cells or contiguous Morton-key ranges and store:

- complete live state for owned representatives
- frozen search records for owned and ghost representatives
- compact node and leaf arrays
- boundary send and receive buffers

Collision events update only representative $i$ and read $j$ from the frozen snapshot. Remote
candidates can therefore be read-only ghosts without remote atomics.

### Exact halo certification

Let $d_{\mathrm{remote}}$ be a conservative lower bound to every unimported domain. A local result
is globally exact when either

$$
d_{K,i} \le d_{\mathrm{remote}}
$$

or

$$
q_i \le d_{\mathrm{remote}}
$$

after every imported cell intersecting $q_i$ has been examined. A query that fails this condition
must request another halo layer or remote cell range.

### Collision synchronization

With the legacy globally synchronized Bernoulli integrator, positions remain fixed during a collision
half-interval, but size and represented-grain number change after every batch. A distributed
implementation must:

1. exchange updated boundary collision properties
2. freeze owned and ghost properties
3. calculate local rates
4. globally reduce the maximum rate for one common batch timestep
5. update owned representatives
6. repeat until the half-interval is complete

The property exchange and global reduction may dominate strong scaling. That cost belongs to the
globally synchronized collision integrator, not to Morton search alone. The proposed local event-chain
integrator instead exchanges and freezes ghost properties once per bath-refresh interval, evolves
owned representatives against that snapshot, and synchronizes again only at the next refresh.

## Remaining work and promotion criteria

### Production integration

Completed in the copied runtime:

- selectable cuKD and Morton owners
- location-dependent per-query cutoffs
- periodic query-image traversal
- shared collision-rate, accessible-volume, event, and outcome physics
- traversal-overflow guards and backend timing/memory logs
- matched full-runtime 2D, wedge, and 3D model definitions plus an automated comparison runner
- exact stable-identifier tie ordering in the laboratory cuKD reference
- one clean ordinary, edge, periodic, wedge, and copied-runtime validation matrix
- exact copied-runtime neighbor topology through $N_P=10^6$
- statistically consistent collision evolution and zero mass mismatch in every copied-runtime case

Next:

1. freeze the validated laboratory source and artifacts in one immutable revision
2. measure peak VRAM, local-memory traffic, occupancy, and kernel timings
3. reduce the full-disk Morton owner allocation and all-particle rate-query cost
4. integrate the faster boundary-ghost policy into the copied runtime
5. add radially binned halos and require equality with the global policy
6. extend the copied-runtime matrix to boundary ghosts and longer statistical evolution
7. promote stable tests to `qav/swarm/` only after the integrated backend demonstrates a useful
   production regime

### Later engineering

- GPU-native hierarchy construction
- CUDA and ROCm performance tuning
- multi-GPU ownership, migration, and exact halo exchange
- batch-wise ghost-property refresh and global timestep coordination
- workload balancing for evolving clumps
- communication/computation overlap

The clean laboratory matrix now satisfies the correctness, statistical-distribution, and
conservation requirements. Promotion still requires an immutable source snapshot and a measured
capacity, portability, multi-GPU, or performance advantage in the intended production regime.

## References

- Johnson, Douze, and Jégou (2017), [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- García et al. (2012), [Multi-GPU SPH through spatial decomposition, radix sorting, and halo exchange](https://arxiv.org/abs/1210.1017)
- NVIDIA, [CCCL/CUB block-wide CUDA primitives](https://github.com/NVIDIA/cccl)
- AMD, [rocPRIM block radix sort](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.0.2/block_ops/ops_classes/sort.html)
- NVIDIA cuVS, [multi-GPU nearest-neighbor distribution and result merging](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/multi-gpu)
- Howard et al. (2019), [Quantized BVH neighbor search](https://arxiv.org/abs/1901.08088)
- Bayraktar et al. (2009), [GPU-based neighbor search](https://repository.bilkent.edu.tr/items/c7fc4617-b45d-4976-bde1-130c68fb09e8)
- Gillespie (1976), [A general method for numerically simulating the stochastic time evolution of coupled chemical reactions](https://doi.org/10.1016/0021-9991(76)90041-3)
- Zsom and Dullemond (2008), [A representative particle approach to coagulation and fragmentation of dust aggregates and fluid droplets](https://arxiv.org/abs/0807.5052)
- Cao, Gillespie, and Petzold (2005), [Avoiding negative populations in explicit Poisson tau-leaping](https://people.cs.vt.edu/~ycao/publication/JChemPhys_123_054104.pdf)
- Anderson (2008), [Incorporating postleap checks in tau-leaping](https://doi.org/10.1063/1.2819665)
